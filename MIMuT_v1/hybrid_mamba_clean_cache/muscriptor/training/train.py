"""Distributed supervised training entry point."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from muscriptor.data.dataset import (
    BalancedWindowBatchSampler,
    ContinuousChunkDataset,
    NUM_INSTRUMENT_GROUPS,
    NUM_MIDI_PITCHES,
    REENTRY_NONE_CLASS,
    TrainingBatch,
    collate_training_examples,
)
from muscriptor.data.augmentation import (
    augmentation_catalog_fingerprint,
    read_augmentation_catalog,
    validate_augmentation_catalog,
)
from muscriptor.data.manifest import (
    manifest_fingerprint,
    read_manifest,
)
from muscriptor.models.lm import LMModel
from muscriptor.models.training import (
    IGNORE_INDEX,
    _condition_attributes,
    pack_training_batch,
)
from muscriptor.modules.streaming import (
    clone_model_state,
    increment_steps,
    init_states,
)
from muscriptor.modules.conditioners import MelSpectrogramConditioner
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.training.config import (
    BoundaryStateSupervisionConfig,
    CleanCacheTrainingConfig,
    ExperimentConfig,
    StageConfig,
)
from muscriptor.transcription_model import TranscriptionModel, _build_model


_EMPTY_SOURCE_SEQUENCE_SHA256 = hashlib.sha256(b"").hexdigest()


class LongContextTrainingModel(nn.Module):
    def __init__(
        self,
        model: LMModel,
        boundary_config: BoundaryStateSupervisionConfig | None = None,
        tokenizer: MT3Tokenizer | None = None,
        clean_cache_config: CleanCacheTrainingConfig | None = None,
    ):
        super().__init__()
        self.model = model
        boundary_config = boundary_config or BoundaryStateSupervisionConfig()
        self.clean_cache_config = clean_cache_config or CleanCacheTrainingConfig()
        self.tokenizer = tokenizer
        clean = self.clean_cache_config.enabled
        if clean and tokenizer is None:
            raise ValueError("clean-cache training requires the MT3 tokenizer")
        if clean and not model.model_config.clean_acoustic_cache:
            raise ValueError("model does not implement Clean Acoustic Cache")
        self.active_note_head = (
            nn.Linear(model.dim, NUM_INSTRUMENT_GROUPS * NUM_MIDI_PITCHES)
            if (
                not clean
                and boundary_config.enabled
                and boundary_config.active_weight > 0
            )
            else None
        )
        self.reentry_head = (
            nn.Linear(
                model.dim,
                NUM_INSTRUMENT_GROUPS * (REENTRY_NONE_CLASS + 1),
            )
            if (
                not clean
                and boundary_config.enabled
                and boundary_config.reentry_weight > 0
            )
            else None
        )

    @staticmethod
    def symbolic_state_probabilities(global_step: int) -> tuple[float, float, float]:
        """Return the frozen GT/corrupt/predicted curriculum."""

        if global_step < 15_000:
            return 1.0, 0.0, 0.0
        if global_step < 25_000:
            return 0.70, 0.20, 0.10
        if global_step < 40_000:
            return 0.40, 0.30, 0.30
        return 0.20, 0.30, 0.50

    def _predicted_active_state(self, logits: torch.Tensor) -> torch.Tensor:
        config = self.clean_cache_config
        scores = torch.sigmoid(logits.detach()).flatten(1)
        k = min(config.max_predicted_active_notes, scores.shape[1])
        values, indices = torch.topk(scores, k=k, dim=1)
        selected = values >= config.prediction_threshold
        flat = torch.zeros_like(scores, dtype=torch.bool)
        flat.scatter_(1, indices, selected)
        return flat.view_as(logits)

    @staticmethod
    def _corrupt_symbolic_state(
        active: torch.Tensor,
        reentry: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply one of six state faults independently to every sequence."""

        active = active.clone()
        reentry = reentry.clone()
        valid = valid.clone()
        batch, groups, pitches = active.shape
        operations = torch.randint(0, 6, (batch,), device=active.device)
        random_group = torch.randint(0, groups - 1, (batch,), device=active.device)
        random_pitch = torch.randint(0, pitches, (batch,), device=active.device)
        rows = torch.arange(batch, device=active.device)
        flat = active.view(batch, -1)
        has_active = flat.any(dim=1)
        first = flat.to(torch.int8).argmax(dim=1)
        first_group = first // pitches
        first_pitch = first % pitches

        drop = operations.eq(0) & has_active
        flat[rows[drop], first[drop]] = False

        add = operations.eq(1)
        active[rows[add], random_group[add], random_pitch[add]] = True

        wrong_program = operations.eq(2) & has_active
        flat[rows[wrong_program], first[wrong_program]] = False
        active[
            rows[wrong_program],
            random_group[wrong_program],
            first_pitch[wrong_program],
        ] = True

        wrong_pitch = operations.eq(3) & has_active
        flat[rows[wrong_pitch], first[wrong_pitch]] = False
        active[
            rows[wrong_pitch],
            first_group[wrong_pitch],
            random_pitch[wrong_pitch],
        ] = True

        stale = operations.eq(4)
        active[stale] = False
        active[rows[stale], random_group[stale], random_pitch[stale]] = True

        wrong_reentry = operations.eq(5)
        selected_group = random_group[wrong_reentry]
        selected_rows = rows[wrong_reentry]
        reentry[selected_rows, selected_group] = (
            reentry[selected_rows, selected_group] + 1
        ) % (REENTRY_NONE_CLASS + 1)
        valid[selected_rows, selected_group] = True
        return active, reentry, valid

    def _forward_clean_cache(
        self,
        batch: TrainingBatch,
        *,
        audio_condition_dropout: float,
        instrument_condition_dropout: float,
        dataset_condition_dropout: float,
        dropout_scope: str,
        global_step: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.tokenizer is None:
            raise RuntimeError("clean-cache tokenizer is missing")
        if (
            audio_condition_dropout != 0.0
            or instrument_condition_dropout != 1.0
            or dataset_condition_dropout != 1.0
        ):
            raise RuntimeError("clean-cache training requires true audio-only conditions")
        if batch.active_note_targets is None:
            raise RuntimeError("clean-cache batch is missing active-note targets")
        if batch.reentry_targets is None or batch.reentry_valid is None:
            raise RuntimeError("clean-cache batch is missing re-entry targets")

        bsz, chunks, max_target = batch.target_ids.shape
        attributes = _condition_attributes(
            batch,
            sample_rate=16_000,
            audio_condition_dropout=audio_condition_dropout,
            instrument_condition_dropout=instrument_condition_dropout,
            dataset_condition_dropout=dataset_condition_dropout,
            dropout_scope=dropout_scope,
        )
        if any(
            item.text.get("instrument_group") is not None
            or item.text.get("dataset_name") is not None
            for item in attributes
        ):
            raise RuntimeError("metadata leaked into an audio-only training batch")
        tokenized = self.model.condition_provider.tokenize(attributes)
        conditions = self.model.condition_provider(tokenized)
        prefix_flat = self.model.condition_prefix_embeddings(conditions)
        prefix_length = prefix_flat.shape[1]
        prefixes = prefix_flat.view(bsz, chunks, prefix_length, self.model.dim)

        clean_state = init_states(
            self.model,
            batch_size=bsz,
            sequence_length=chunks * prefix_length + max_target + 8,
        )
        selected_logits: list[torch.Tensor] = []
        selected_labels: list[torch.Tensor] = []
        active_outputs: list[torch.Tensor] = []
        reentry_outputs: list[torch.Tensor] = []
        mode_counts = torch.zeros(3, dtype=torch.long, device=batch.waveform.device)
        previous_active_logits = previous_reentry_logits = None

        if self.clean_cache_config.symbolic_state_curriculum:
            gt_probability, corrupt_probability, _ = (
                self.symbolic_state_probabilities(global_step)
            )
        else:
            gt_probability, corrupt_probability = 1.0, 0.0
        for chunk_index in range(chunks):
            prefix = prefixes[:, chunk_index]
            if chunk_index == 0:
                gt_active = torch.zeros_like(batch.active_note_targets[:, 0])
                gt_reentry = torch.full_like(
                    batch.reentry_targets[:, 0], REENTRY_NONE_CLASS
                )
                gt_valid = torch.zeros_like(batch.reentry_valid[:, 0])
            else:
                gt_active = batch.active_note_targets[:, chunk_index - 1]
                gt_reentry = batch.reentry_targets[:, chunk_index - 1]
                gt_valid = batch.reentry_valid[:, chunk_index - 1]

            corrupt_active, corrupt_reentry, corrupt_valid = (
                self._corrupt_symbolic_state(gt_active, gt_reentry, gt_valid)
            )
            if previous_active_logits is None:
                predicted_active = gt_active
                predicted_reentry = gt_reentry
                predicted_valid = gt_valid
            else:
                predicted_active = self._predicted_active_state(
                    previous_active_logits
                )
                predicted_reentry = previous_reentry_logits.detach().argmax(dim=-1)
                predicted_valid = torch.ones_like(gt_valid)

            draws = torch.rand(bsz, device=batch.waveform.device)
            modes = torch.where(
                draws < gt_probability,
                torch.zeros_like(draws, dtype=torch.long),
                torch.where(
                    draws < gt_probability + corrupt_probability,
                    torch.ones_like(draws, dtype=torch.long),
                    torch.full_like(draws, 2, dtype=torch.long),
                ),
            )
            if chunk_index == 0:
                # There is no prior committed state at a fresh window.
                modes.zero_()
            mode_counts += torch.bincount(modes, minlength=3)
            active_state = torch.where(
                modes[:, None, None] == 0,
                gt_active,
                torch.where(
                    modes[:, None, None] == 1,
                    corrupt_active,
                    predicted_active,
                ),
            )
            reentry_state = torch.where(
                modes[:, None] == 0,
                gt_reentry,
                torch.where(
                    modes[:, None] == 1,
                    corrupt_reentry,
                    predicted_reentry,
                ),
            )
            reentry_valid = torch.where(
                modes[:, None] == 0,
                gt_valid,
                torch.where(
                    modes[:, None] == 1,
                    corrupt_valid,
                    predicted_valid,
                ),
            )
            symbolic = self.model.symbolic_state_embeddings(
                active_state,
                reentry=reentry_state,
                reentry_valid=reentry_valid,
            )

            targets = batch.target_ids[:, chunk_index]
            lengths = batch.target_lengths[:, chunk_index]
            event_inputs = torch.zeros_like(targets)
            event_inputs[:, 0] = self.model.initial_token_id
            if max_target > 1:
                event_inputs[:, 1:] = targets[:, :-1]
            event_embeddings = self.model.emb(event_inputs)
            event_embeddings = (
                event_embeddings + self.model.type_embedding.weight[3]
            )
            combined = torch.cat([prefix, symbolic, event_embeddings], dim=1)

            labels = torch.full(
                (bsz, combined.shape[1]),
                IGNORE_INDEX,
                dtype=torch.long,
                device=targets.device,
            )
            valid_positions = (
                torch.arange(max_target, device=targets.device)[None, :]
                < lengths[:, None]
            )
            if batch.target_loss_mask is not None:
                valid_positions &= batch.target_loss_mask[:, chunk_index]
            event_labels = targets.masked_fill(~valid_positions, IGNORE_INDEX)
            labels[:, prefix_length + 1 :] = event_labels

            decode_state = clone_model_state(clean_state, detach=True)
            hidden = self.model.encode_embeddings(
                combined, model_state=decode_state
            )
            logits = self.model.linear(hidden)
            supervised = labels.ne(IGNORE_INDEX)
            selected_logits.append(logits[supervised])
            selected_labels.append(labels[supervised])

            # This location is causally audio-only: symbolic/event positions
            # occur strictly after it in the same forward pass.
            audio_boundary_hidden = hidden[:, prefix_length - 1]
            current_active, current_reentry = self.model.boundary_state_logits(
                audio_boundary_hidden
            )
            active_outputs.append(current_active)
            reentry_outputs.append(current_reentry)
            previous_active_logits = current_active
            previous_reentry_logits = current_reentry

            # Persistent state advances on a separate no-grad audio-only path.
            with torch.no_grad():
                self.model.encode_embeddings(prefix, model_state=clean_state)
                increment_steps(
                    self.model.transformer,
                    clean_state,
                    increment=prefix_length,
                )
            del decode_state

        flat_logits = torch.cat(selected_logits, dim=0)
        flat_labels = torch.cat(selected_labels, dim=0)
        return (
            flat_logits,
            flat_labels,
            int(flat_labels.numel()),
            torch.stack(active_outputs, dim=1),
            torch.stack(reentry_outputs, dim=1),
            mode_counts,
        )

    def forward(
        self,
        batch: TrainingBatch,
        audio_condition_dropout: float,
        instrument_condition_dro×m|öÚ$z{-®éÜj×6öçFW‡Eöf÷%÷7FvU÷7FW€¢7FvRÂ6VVCÖ6öæf–rç6VVBÂ7FvU÷7FWÖÆö6Åö–æFW€¢¢G'“ ¢&F6‚ÒæW‡B†ÆöFW%ö—FW&F÷'5¶6öçFW‡Eö6‡Væ·5Ò¢W†6WB„¶W”W'&÷"Â7F÷—FW&F–öâ’2W†3 ¢&—6R'VçF–ÖTW'&÷"€¢b'7FvR·7FvRææÖR'ÒW††W7FVB6öçFW‡C×¶6öçFW‡Eö6‡Væ·7Ò ¢b&B7FvU÷7FW×¶Æö6Åö–æFW‡Ò ¢’g&öÒW†0¢7FvUö6öçFW‡Eö6÷VçG5¶6öçFW‡Eö6‡Væ·5Ò³Ò¢6÷W&6U÷6WVVæ6Uö†6‚Ò÷WFFU÷6÷W&6U÷6WVVæ6Uö†6‚€¢6÷W&6U÷6WVVæ6Uö†6‚À¢&F6‚À¢7FvUö–æFWƒ×7FvUö–æFW‚À¢7FvU÷7FWÖÆö6Åö–æFW‚À¢6öçFW‡Eö6‡Væ·3Ö6öçFW‡Eö6‡Væ·2À¢¢&F6‚Ò&F6‚çFò†FWf–6R¢÷F–Ö—¦W"ç¦W&õöw&B‡6WE÷FõöæöæSÕG'VR¢v—F‚öWFö67B†FWf–6RÂ6öæf–rç&V6—6–öâ“ ¢G&÷÷WBÒ6öæf–ræ6öæF—F–öåöG&÷÷W@¢€¢Æöv—G2À¢Æ&VÇ2À¢Fö¶Våö6÷VçBÀ¢7F—fUöÆöv—G2À¢&VVçG'•öÆöv—G2À¢7–Ö&öÆ–5öÖöFUö6÷VçG2À¢’Òw&VB€¢&F6‚À¢G&÷÷WBæVF–òÀ¢G&÷÷WBæ–ç7G'VÖVçBÀ¢G&÷÷WBæFF6WBÀ¢6öæf–ræ6öæF—F–öæ–æræG&÷÷WE÷66÷RÀ¢vÆö&Å÷7FWÀ¢¢Fö¶VåöÆ÷76W2Òbæ7&÷75öVçG&÷’€¢Æöv—G2ç&W6†R‚ÓÂÆöv—G2ç6†U²ÓÒ’æfÆöB‚’À¢Æ&VÇ2ç&W6†R‚Ó’À¢–væ÷&Uö–æFWƒÔ”täõ$Uô”äDU‚À¢&VGV7F–öãÒ&æöæR"À¢¢fÆ–E÷Fö¶VâÒÆ&VÇ2ç&W6†R‚Ó’ææR„”täõ$Uô”äDU‚¢Fö¶Vå÷vV–v‡G2ÒF÷&6‚æöæW5öÆ–¶R‡Fö¶VåöÆ÷76W2¢Fö¶Vå÷vV–v‡G2ÒF÷&6‚çv†W&R€¢Æ&VÇ2ç&W6†R‚Ó’æW‡7GVFVçE÷Fö¶Væ—¦W"æV÷5ö–B’À¢F÷&6‚ægVÆÅöÆ–¶R‡Fö¶Vå÷vV–v‡G2Â6öæf–rçG'VUöV÷5öÆ÷75÷vV–v‡B’À¢Fö¶Vå÷vV–v‡G2À¢¢6RÒ‡Fö¶VåöÆ÷76W5·fÆ–E÷Fö¶VåÒ¢Fö¶Vå÷vV–v‡G5·fÆ–E÷Fö¶VåÒ’ç7VÒ‚¢6RÒ6RòFö¶Vå÷vV–v‡G5·fÆ–E÷Fö¶VåÒç7VÒ‚’æ6Æ×öÖ–âƒã¢Æ÷72Ò6P¢7F—fUöÆ÷72ÒF÷&6‚ç¦W&÷2‚‚’ÂFWf–6SÖFWf–6R¢&VVçG'•öÆ÷72ÒF÷&6‚ç¦W&÷2‚‚’ÂFWf–6SÖFWf–6R¢&÷VæF'•ö6öæf–rÒ6öæf–ræ&÷VæF'•÷7FFU÷7WW'f—6–öà¢–b7F—fUöÆöv—G2—2æ÷BæöæS ¢–b&F6‚æ7F—fUöæ÷FU÷F&vWG2—2æöæS ¢&—6R'VçF–ÖTW'&÷"‚&&F6‚—2Ö—76–ær7F—fRÖæ÷FRF&vWG2"¢7F—fUöÆ÷72Òö&Ææ6VEö7F—fUöæ÷FUöÆ÷72€¢7F—fUöÆöv—G2æfÆöB‚’Â&F6‚æ7F—fUöæ÷FU÷F&vWG0¢¢Æ÷72ÒÆ÷72²&÷VæF'•ö6öæf–ræ7F—fU÷vV–v‡B¢7F—fUöÆ÷70¢–b&VVçG'•öÆöv—G2—2æ÷BæöæS ¢–b&F6‚ç&VVçG'•÷F&vWG2—2æöæR÷"&F6‚ç&VVçG'•÷fÆ–B—2æöæS ¢&—6R'VçF–ÖTW'&÷"‚&&F6‚—2Ö—76–ær&RÖVçG'’F&vWG2"¢fÆ–BÒ&F6‚ç&VVçG'•÷fÆ–@¢–bfÆ–Bæç’‚“ ¢&VVçG'•öÆ÷72Òbæ7&÷75öVçG&÷’€¢&VVçG'•öÆöv—G5·fÆ–EÒæfÆöB‚’À¢&F6‚ç&VVçG'•÷F&vWG5·fÆ–EÒÀ¢¢VÇ6S ¢2&W6W'fR¦W&òw&F–VçBf÷"DEôe4E&F6†W2v†÷6P¢26×ÆVBv–æF÷w26öçF–âæòVæ6Vç6÷&VB&RÖVçG'’Æ&VÂà¢&VVçG'•öÆ÷72Ò&VVçG'•öÆöv—G2ç7VÒ‚’¢ã ¢Æ÷72ÒÆ÷72²&÷VæF'•ö6öæf–rç&VVçG'•÷vV–v‡B¢&VVçG'•öÆ÷70¢¶BÒF÷&6‚ç¦W&÷2‚‚’ÂFWf–6SÖFWf–6R¢–bFV6†W"—2æ÷BæöæS ¢v—F‚F÷&6‚ææõöw&B‚“ ¢FV6†W%öÆöv—G2Ò÷FV6†W%öÆöv—G2€¢FV6†W"À¢&F6‚À¢Ö–7&ö&F6…÷6—¦SÒ€¢6öæf–ræF—7F–ÆÆF–öâçFV6†W%öÖ–7&ö&F6…÷6—¦P¢’À¢¢¶BÒöF—7F–ÆÆF–öåöÆ÷72€¢Æöv—G2À¢Æ&VÇ2À¢FV6†W%öÆöv—G2À¢FV×W&GW&SÖ6öæf–ræF—7F–ÆÆF–öâçFV×W&GW&RÀ¢6†&VE÷fö6%÷6—¦S×6†&VE÷fö6%÷6—¦RÀ¢¢Æ÷72ÒÆ÷72²6öæf–ræF—7F–ÆÆF–öâçvV–v‡B¢¶@¢Æ÷72æ&6·v&B‚¢w&Eöæ÷&ÒÒF÷&6‚ææâçWF–Ç2æ6Æ—öw&Eöæ÷&Õò€¢w&VBç&ÖWFW'2‚’Â6öæf–ræ÷F–Ö—¦W"æw&F–VçEö6Æ— ¢¢†VÇF…ö6†V6¶VBÒvÆö&Å÷7FWÂ÷"€¢†vÆö&Å÷7FW²’R6öæf–ræÆöuöWfW'’ÓÒ ¢¢FV6†W%ög&÷¦VâÒG'VP¢F—7F–ÆÅ÷÷6—F—fRÒG'VP¢–b†VÇF…ö6†V6¶VC ¢FV6†W%ög&÷¦VâÒFV6†W"—2æöæR÷"ÆÂ€¢&ÖWFW"æw&B—2æöæRf÷"&ÖWFW"–âFV6†W"ç&ÖWFW'2‚¢¢F—7F–ÆÅ÷÷6—F—fRÒFV6†W"—2æöæR÷"&ööÂ†¶BæFWF6‚‚’â¢f–æ—FRÒÆÂ€¢&ööÂ‡F÷&6‚æ—6f–æ—FR‡fÇVRæFWF6‚‚’’¢f÷"fÇVR–â†Æ÷72Â6RÂ¶BÂw&Eöæ÷&Ò¢¢†VÇF‡’Òf–æ—FRæBF—7F–ÆÅ÷÷6—F—fRæBFV6†W%ög&÷¦Và¢†VÇF‚ÒF÷&6‚çFVç6÷"€¢–çB††VÇF‡’’ÂFWf–6SÖFWf–6RÂGG—S×F÷&6‚æ–çC3 ¢¢–bF—7Bæ—5ö–æ—F–Æ—¦VB‚“ ¢F—7BæÆÅ÷&VGV6R††VÇF‚Â÷ÖF—7Bå&VGV6T÷äÔ”â¢–bæ÷B&ööÂ††VÇF‚æ—FVÒ‚’“ ¢&—6RfÆöF–æuö–çDW'&÷"€¢'G&–æ–ær†VÇF‚6†V6²f–ÆVC¢&WV—&Rf–æ—FR4Rô´ÂöÆ÷72öw&BÂ ¢'÷6—F—fR´ÂÂæBg&÷¦VâFV6†W" ¢¢÷F–Ö—¦W"ç7FW‚¢66†VGVÆW"ç7FW‚¢vÆö&Å÷7FW³Ò ¢–bvÆö&Å÷7FWR6öæf–ræÆöuöWfW'’ÓÒ ¢æ÷rÒF–ÖRçW&eö6÷VçFW"‚¢–çFW'fÅ÷6V6öæG2Òæ÷rÒÆ7E÷F–ÖP¢&F6…öVF–õ÷6V6öæG2Ò€¢&F6‚çvfVf÷&Òç6†U³Ð¢¢&F6‚çvfVf÷&Òç6†U³Ð¢¢6öæf–ræÖöFVÂç6VvÖVçEöGW&F–öà¢¢v÷&ÆE÷6—¦P¢¢ÖWG&–73¢F–7E·7G"Âç•ÒÒ°¢'7FW#¢vÆö&Å÷7FWÀ¢'7FvR#¢7FvRææÖRÀ¢'7FvU÷7FW#¢Æö6Åö–æFW‚²À¢&÷F–Ö—¦W%÷7FW#¢vÆö&Å÷7FWÀ¢&6öçFW‡Eö6‡Væ·2#¢6öçFW‡Eö6‡Væ·2À¢&Æ÷72#¢fÆöB†Æ÷72æFWF6‚‚’’À¢&6R#¢fÆöB†6RæFWF6‚‚’’À¢&F—7F–ÆÂ#¢fÆöB†¶BæFWF6‚‚’’À¢&7F—fUöæ÷FUöÆ÷72#¢fÆöB†7F—fUöÆ÷72æFWF6‚‚’’À¢'&VVçG'•öÆ÷72#¢fÆöB‡&VVçG'•öÆ÷72æFWF6‚‚’’À¢&w&Eöæ÷&Ò#¢fÆöB†w&Eöæ÷&Ò’À¢&Ç"#¢66†VGVÆW"ævWEöÆ7EöÇ"‚•³ÒÀ¢'Fö¶Vç2#¢Fö¶Våö6÷VçBÀ¢'G'Væ6FVEö6‡Væµög&7F–öâ#¢€¢fÆöB†&F6‚çG'Væ6FVEö6‡Væ·2æfÆöB‚’æÖVâ‚’¢–b&F6‚çG'Væ6FVEö6‡Væ·2—2æ÷BæöæP¢VÇ6Rã ¢’À¢&FVç6Uö6‡Væµög&7F–öâ#¢€¢fÆöB€¢†&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2âS’æfÆöB‚’æÖVâ‚¢¢–b&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2—2æ÷BæöæP¢VÇ6Rã ¢’À¢'F&vWE÷Fö¶Vç5÷“R#¢€¢fÆöB€¢F÷&6‚çVçF–ÆR€¢&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2æfÆöB‚’Âã“P¢¢¢–b&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2—2æ÷BæöæP¢VÇ6Rã ¢’À¢'F&vWE÷Fö¶Vç5÷“’#¢€¢fÆöB€¢F÷&6‚çVçF–ÆR€¢&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2æfÆöB‚’Âã“¢¢¢–b&F6‚æ÷&–v–æÅ÷F&vWEöÆVæwF‡2—2æ÷BæöæP¢VÇ6Rã ¢’À¢&VvÖVçFF–öåög&7F–öâ#¢€¢fÆöB†&F6‚æVvÖVçFF–öåöÆ–VBæfÆöB‚’æÖVâ‚’¢–b&F6‚æVvÖVçFF–öåöÆ–VB—2æ÷BæöæP¢VÇ6Rã ¢’À¢'&VÖ—…ög&7F–öâ#¢€¢fÆöB†&F6‚ç&VÖ—†VBæfÆöB‚’æÖVâ‚’¢–b&F6‚ç&VÖ—†VB—2æ÷BæöæP¢VÇ6Rã ¢’À¢'—F6…÷6†–gEög&7F–öâ#¢€¢fÆöB†&F6‚ç—F6…÷6†–gE÷6VÖ—FöæW2ææRƒ’æfÆöB‚’æÖVâ‚’¢–b&F6‚ç—F6…÷6†–gE÷6VÖ—FöæW2—2æ÷BæöæP¢VÇ6Rã ¢’À¢&†VÇF…ö6†V6¶VB#¢†VÇF…ö6†V6¶VBÀ¢&F—7F–ÆÅ÷÷6—F—fR#¢F—7F–ÆÅ÷÷6—F—fRÀ¢'FV6†W%ög&÷¦Vâ#¢FV6†W%ög&÷¦VâÀ¢'6÷W&6U÷6WVVæ6U÷6†#Sb#¢6÷W&6U÷6WVVæ6Uö†6‚À¢'7FW5÷W%÷6V6öæB#¢6öæf–ræÆöuöWfW'’ò–çFW'fÅ÷6V6öæG2À¢&VF–õ÷6V6öæG5÷W%÷6V6öæB#¢€¢6öæf–ræÆöuöWfW'’¢&F6…öVF–õ÷6V6öæG2ò–çFW'fÅ÷6V6öæG0¢’À¢'VµöÖVÖ÷'•öv–"#¢€¢F÷&6‚æ7VFæÖ…öÖVÖ÷'•öÆÆö6FVB†FWf–6R’òƒ#B¢£2¢’À¢'7–Ö&öÆ–5÷7FFUöwEög&7F–öâ#¢fÆöB€¢7–Ö&öÆ–5öÖöFUö6÷VçG5³ÒæfÆöB‚¢ò7–Ö&öÆ–5öÖöFUö6÷VçG2ç7VÒ‚’æ6Æ×öÖ–âƒ¢’À¢'7–Ö&öÆ–5÷7FFUö6÷''WEög&7F–öâ#¢fÆöB€¢7–Ö&öÆ–5öÖöFUö6÷VçG5³ÒæfÆöB‚¢ò7–Ö&öÆ–5öÖöFUö6÷VçG2ç7VÒ‚’æ6Æ×öÖ–âƒ¢’À¢'7–Ö&öÆ–5÷7FFU÷&VF–7FVEög&7F–öâ#¢fÆöB€¢7–Ö&öÆ–5öÖöFUö6÷VçG5³%ÒæfÆöB‚¢ò7–Ö&öÆ–5öÖöFUö6÷VçG2ç7VÒ‚’æ6Æ×öÖ–âƒ¢’À¢Ð¢Æ7E÷F–ÖRÒæ÷p¢v—F‚Æöu÷F‚æ÷Vâ‚&"’27G&VÓ ¢7G&VÒçw&—FR†§6öâæGV×2†ÖWG&–72’²%Æâ"¢–b&æ²ÓÒ ¢&–çB†§6öâæGV×2†ÖWG&–72’ÂfÇW6ƒÕG'VR¢–bw&—FW"—2æ÷BæöæS ¢f÷"¶W’–â€¢&Æ÷72"À¢&6R"À¢&F—7F–ÆÂ"À¢&7F—fUöæ÷FUöÆ÷72"À¢'&VVçG'•öÆ÷72"À¢&w&Eöæ÷&Ò"À¢&Ç""À¢'G'Væ6FVEö6‡Væµög&7F–öâ"À¢&FVç6Uö6‡Væµög&7F–öâ"À¢'F&vWE÷Fö¶Vç5÷“R"À¢'F&vWE÷Fö¶Vç5÷“’"À¢&VvÖVçFF–öåög&7F–öâ"À¢'&VÖ—…ög&7F–öâ"À¢'—F6…÷6†–gEög&7F–öâ"À¢'7–Ö&öÆ–5÷7FFUöwEög&7F–öâ"À¢'7–Ö&öÆ–5÷7FFUö6÷''WEög&7F–öâ"À¢'7–Ö&öÆ–5÷7FFU÷&VF–7FVEög&7F–öâ"À¢“ ¢w&—FW"æFE÷66Æ"†¶W’ÂÖWG&–75¶¶W•ÒÂvÆö&Å÷7FW ¢–bvÆö&Å÷7FWR6öæf–rç6fUöWfW'’ÓÒ ¢÷6fUö6†V6·ö–çB€¢w&VC×w&VBÀ¢&u÷w&W#×&u÷w&W"À¢÷F–Ö—¦W#Ö÷F–Ö—¦W"À¢66†VGVÆW#×66†VGVÆW"À¢6öæf–sÖ6öæf–rÀ¢÷WGWEöF—#Ö÷WGWEöF—"À¢vÆö&Å÷7FWÖvÆö&Å÷7FWÀ¢7FvUö–æFWƒ×7FvUö–æFW‚À¢7FvU÷7FWÖÆö6Åö–æFW‚²À¢Öæ–fW7Eö†6ƒÖÖæ–fW7Eö†6‚À¢F†öæö×“×F†öæö×’À¢&æ³×&æ²À¢6öçFW‡EöG&uö6÷VçG3×7FvUö6öçFW‡Eö6÷VçG2À¢6÷W&6U÷6WVVæ6U÷6†#Sc×6÷W&6U÷6WVVæ6Uö†6‚À¢¢&W7VÖU÷7FvU÷7FWÒ  ¢÷6fUö6†V6·ö–çB€¢w&VC×w&VBÀ¢&u÷w&W#×&u÷w&W"À¢÷F–Ö—¦W#Ö÷F–Ö—¦W"À¢66†VGVÆW#×66†VGVÆW"À¢6öæf–sÖ6öæf–rÀ¢÷WGWEöF—#Ö÷WGWEöF—"À¢vÆö&Å÷7FWÖvÆö&Å÷7FWÀ¢7FvUö–æFWƒÖÆVâ†6öæf–rç7FvW2’À¢7FvU÷7FWÓÀ¢Öæ–fW7Eö†6ƒÖÖæ–fW7Eö†6‚À¢F†öæö×“×F†öæö×’À¢&æ³×&æ²À¢6öçFW‡EöG&uö6÷VçG3×·ÒÀ¢6÷W&6U÷6WVVæ6U÷6†#Sc×6÷W&6U÷6WVVæ6Uö†6‚À¢¢–bF—7Bæ—5ö–æ—F–Æ—¦VB‚“ ¢F—7Bæ&'&–W"‚¢F—7BæFW7G&÷•÷&ö6W75öw&÷W‚¢–bw&—FW"—2æ÷BæöæS ¢w&—FW"æ6Æ÷6R‚  ¦FVbÖ–â‚’ÓâæöæS ¢'6W"Ò&w'6Rä&wVÖVçE'6W"‚¢'6W"æFEö&wVÖVçB‚"ÒÖ6öæf–r"Â&WV—&VCÕG'VR¢'6W"æFEö&wVÖVçB‚"Ò×6VVB"ÂG—SÖ–çB¢'6W"æFEö&wVÖVçB‚"ÒÖ÷WGWBÖF—""¢'6W"æFEö&wVÖVçB‚"ÒÖæÖR"¢'6W"æFEö&wVÖVçB‚"Ò×&VfÆ–v‡B×7FW2"ÂG—SÖ–çB¢'6W"æFEö&wVÖVçB‚"ÒÖ6öçFW‡BÖ6‡Væ·2"ÂG—SÖ–çBÂ6†ö–6W3ÒƒÂ"ÂBÂ‚’¢'6W"æFEö&wVÖVçB‚"ÒÖvÆö&ÂÖ&F6‚ÖVF–ò×6V6öæG2"ÂG—SÖ–çB¢&w2Ò'6W"ç'6Uö&w2‚¢g&öÒ×W67&—F÷"çG&–æ–æræ6öæf–r–×÷'BÆöEöW‡W&–ÖVçEö6öæf–p ¢6öæf–rÒÆöEöW‡W&–ÖVçEö6öæf–r†&w2æ6öæf–r¢–b&w2ç6VVB—2æ÷BæöæRæB&w2ç6VVBÒ6öæf–rç6VVBæBæ÷B&w2æ÷WGWEöF—# ¢'6W"æW'&÷"‚"Ò×6VVB÷fW'&–FW2&WV—&RF—7F–æ7BÒÖ÷WGWBÖF—""¢÷fW'&–FW2Ò·Ð¢–b&w2ç6VVB—2æ÷BæöæS ¢÷fW'&–FW5²'6VVB%ÒÒ&w2ç6VV@¢–b&w2æ÷WGWEöF—"—2æ÷BæöæS ¢÷fW'&–FW5²&÷WGWEöF—"%ÒÒ&w2æ÷WGWEöF— ¢–b&w2ææÖR—2æ÷BæöæS ¢÷fW'&–FW5²&æÖR%ÒÒ&w2ææÖP¢–b&w2ævÆö&Åö&F6…öVF–õ÷6V6öæG2—2æ÷BæöæS ¢÷fW'&–FW5²&vÆö&Åö&F6…öVF–õ÷6V6öæG2%ÒÒ&w2ævÆö&Åö&F6…öVF–õ÷6V6öæG0¢6öæf–rÒ&WÆ6R†6öæf–rÂ¢¦÷fW'&–FW2¢–b&w2ç&VfÆ–v‡E÷7FW2—2æ÷BæöæR÷"&w2æ6öçFW‡Eö6‡Væ·2—2æ÷BæöæS ¢–b&w2ç&VfÆ–v‡E÷7FW2—2æöæR÷"&w2æ6öçFW‡Eö6‡Væ·2—2æöæS ¢'6W"æW'&÷"‚'&VfÆ–v‡B&WV—&W2&÷F‚Ò×&VfÆ–v‡B×7FW2æBÒÖ6öçFW‡BÖ6‡Væ·2"¢–b&w2ç&VfÆ–v‡E÷7FW2ÃÒ ¢'6W"æW'&÷"‚"Ò×&VfÆ–v‡B×7FW2×W7B&R÷6—F—fR"¢–b&w2æ÷WGWEöF—"—2æöæS ¢'6W"æW'&÷"‚'&VfÆ–v‡B&WV—&W2F—7F–æ7BÒÖ÷WGWBÖF—""¢6÷W&6U÷7FvRÒ6öæf–rç7FvW5³Ð¢&VfÆ–v‡E÷7FvRÒ&WÆ6R€¢6÷W&6U÷7FvRÀ¢æÖSÖb'&VfÆ–v‡E÷¶&w2æ6öçFW‡Eö6‡Væ·2¢W×2"À¢7FW3Ö&w2ç&VfÆ–v‡E÷7FW2À¢6öçFW‡Eö6‡Væ·3Ö&w2æ6öçFW‡Eö6‡Væ·2À¢6öçFW‡EöF—7G&–'WF–öã×·ÒÀ¢FF6WG3Õ²'6Æ¶ƒ#÷&VGW‚%ÒÀ¢FF6WE÷vV–v‡G3×·ÒÀ¢VvÖVçFF–öãÔfÇ6RÀ¢¢6öæf–rÒ&WÆ6R†6öæf–rÂ7FvW3Õ·&VfÆ–v‡E÷7FvUÒÂ&W7VÖSÔæöæR¢G&–â†6öæf–r  ¦–bõöæÖUõòÓÒ%õöÖ–åõò# ¢Ö–â‚