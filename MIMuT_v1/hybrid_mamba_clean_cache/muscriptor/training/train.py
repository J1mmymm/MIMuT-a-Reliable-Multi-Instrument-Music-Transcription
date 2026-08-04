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
        instrument_condition_dropout: float,
        dataset_condition_dropout: float,
        dropout_scope: str = "sequence",
        global_step: int = 0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        if self.clean_cache_config.enabled:
            return self._forward_clean_cache(
                batch,
                audio_condition_dropout=audio_condition_dropout,
                instrument_condition_dropout=instrument_condition_dropout,
                dataset_condition_dropout=dataset_condition_dropout,
                dropout_scope=dropout_scope,
                global_step=global_step,
            )
        packed = pack_training_batch(
            self.model,
            batch,
            audio_condition_dropout=audio_condition_dropout,
            instrument_condition_dropout=instrument_condition_dropout,
            dataset_condition_dropout=dataset_condition_dropout,
            dropout_scope=dropout_scope,
        )
        hidden = self.model.encode_embeddings(packed.embeddings)
        logits = self.model.linear(hidden)
        active_logits = reentry_logits = None
        if self.active_note_head is not None or self.reentry_head is not None:
            positions = (packed.chunk_end_positions - 1).clamp_min(0)
            gather_index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            boundary_hidden = hidden.gather(1, gather_index)
            if self.active_note_head is not None:
                active_logits = self.active_note_head(boundary_hidden).view(
                    hidden.shape[0],
                    positions.shape[1],
                    NUM_INSTRUMENT_GROUPS,
                    NUM_MIDI_PITCHES,
                )
            if self.reentry_head is not None:
                reentry_logits = self.reentry_head(boundary_hidden).view(
                    hidden.shape[0],
                    positions.shape[1],
                    NUM_INSTRUMENT_GROUPS,
                    REENTRY_NONE_CLASS + 1,
                )
        return (
            logits,
            packed.labels,
            packed.token_count,
            active_logits,
            reentry_logits,
            torch.zeros(3, dtype=torch.long, device=logits.device),
        )

    def load_auxiliary_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore training-only heads without polluting inference weights."""

        own = self.state_dict()
        expected = {key for key in own if not key.startswith("model.")}
        if set(state) != expected:
            raise RuntimeError(
                "boundary-state checkpoint mismatch: "
                f"expected {sorted(expected)}, received {sorted(state)}"
            )
        own.update(state)
        self.load_state_dict(own)


def _balanced_active_note_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Balanced BCE so sparse active notes are not swamped by null pairs."""

    targets = targets.to(dtype=logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets > 0.5
    negative = ~positive
    positive_loss = (
        elementwise[positive].mean()
        if positive.any()
        else torch.zeros((), device=logits.device, dtype=logits.dtype)
    )
    negative_loss = elementwise[negative].mean()
    return (0.5 * (positive_loss + negative_loss)) if positive.any() else negative_loss


def _training_signature(config: ExperimentConfig) -> dict[str, Any]:
    signature = asdict(config)
    signature.pop("resume", None)
    signature.pop("output_dir", None)
    return signature


def _update_source_sequence_hash(
    previous: str,
    batch: TrainingBatch,
    *,
    stage_index: int,
    stage_step: int,
    context_chunks: int,
) -> str:
    """Extend a rank-local audit chain without logging private source rows."""

    if len(previous) != 64:
        raise ValueError("invalid source-sequence hash")
    payload = {
        "stage_index": stage_index,
        "stage_step": stage_step,
        "context_chunks": context_chunks,
        "track_ids": list(batch.track_ids),
        "start_times": [float(value) for value in batch.start_times.tolist()],
        "augmentation_lineage": batch.augmentation_lineage or [],
    }
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _assert_new_run_output_is_empty(config: ExperimentConfig) -> None:
    """Refuse to mix a fresh v2 run with an existing output directory."""

    if config.resume:
        return
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"new training run refuses non-empty output directory: {output}"
        )


def _context_for_stage_step(
    stage: StageConfig, *, seed: int, stage_step: int
) -> int:
    """Draw one context length without touching process or worker RNG state."""

    digest = hashlib.sha256(
        f"{seed}:{stage.name}:{stage_step}".encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    probabilities = stage.context_probabilities()
    for chunks, probability in probabilities.items():
        cumulative += probability
        if draw < cumulative:
            return chunks
    return next(reversed(probabilities))


def _context_counts_before(
    stage: StageConfig, *, seed: int, stage_step: int
) -> dict[int, int]:
    counts = {chunks: 0 for chunks in stage.context_probabilities()}
    for step in range(stage_step):
        counts[_context_for_stage_step(stage, seed=seed, stage_step=step)] += 1
    return counts


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=getattr(torch, precision),
    )


def _unwrap_lm(wrapped: nn.Module, raw_wrapper: LongContextTrainingModel) -> LMModel:
    if isinstance(wrapped, DistributedDataParallel):
        return wrapped.module.model
    return raw_wrapper.model


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tokenizer_contract(tokenizer: MT3Tokenizer, model_config) -> dict[str, Any]:
    """Fingerprint the exact event and instrument mappings used by KD."""

    event_mapping = [
        [event.type, event.value] for event in tokenizer._vocab
    ]
    group_mapping = {
        str(group): list(programs)
        for group, programs in sorted(tokenizer.group_program_map.items())
    }

    def fingerprint(value: Any) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    return {
        "name": model_config.tokenizer_name,
        "max_shift_steps": model_config.tokenizer_max_shift_steps,
        "frame_rate": tokenizer.frame_rate,
        "vocab_size": tokenizer.num_tokens,
        "event_mapping_sha256": fingerprint(event_mapping),
        "instrument_mapping_sha256": fingerprint(group_mapping),
    }


def _verify_teacher_checkpoint(
    config: ExperimentConfig, *, rank: int
) -> dict[str, Any] | None:
    teacher_path = config.distillation.teacher_checkpoint
    if not teacher_path:
        return None
    path = Path(teacher_path).expanduser().resolve()
    payload: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            if not path.is_file() or path.suffix.lower() != ".safetensors":
                raise RuntimeError(
                    "teacher_checkpoint must be a pre-downloaded local "
                    "safetensors file"
                )
            config_path = path.with_name("config.json")
            if not config_path.is_file():
                raise RuntimeError(
                    "teacher snapshot must include config.json next to the weights"
                )
            actual = _sha256_file(path)
            expected = str(config.distillation.teacher_sha256).lower()
            if actual != expected:
                raise RuntimeError(
                    "teacher checkpoint SHA256 mismatch: "
                    f"expected {expected}, got {actual}"
                )
            payload[0] = {
                "path": str(path),
                "sha256": actual,
                "revision": config.distillation.teacher_revision,
                "config_sha256": _sha256_file(config_path),
            }
        except Exception as exc:
            payload[0] = {"error": str(exc)}
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        # Non-distributed nonzero rank is impossible; retain a fail-closed
        # guard for unusual launchers.
        raise RuntimeError("teacher verification result was not broadcast")
    if "error" in payload[0]:
        raise RuntimeError(payload[0]["error"])
    return payload[0]


def _teacher_logits(
    teacher: LMModel,
    batch: TrainingBatch,
    *,
    microbatch_size: int,
) -> torch.Tensor:
    bsz, chunks, max_length = batch.target_ids.shape
    flat = bsz * chunks
    attributes = _condition_attributes(
        batch,
        sample_rate=16_000,
        audio_condition_dropout=0.0,
        instrument_condition_dropout=1.0,
        dataset_condition_dropout=1.0,
    )
    # Any explicitly enabled teacher remains audio-only. This avoids leaking
    # oracle instrument or dataset metadata through its soft targets.
    targets = batch.target_ids.reshape(flat, max_length)
    lengths = batch.target_lengths.reshape(flat)
    loss_mask = (
        batch.target_loss_mask.reshape(flat, max_length)
        if batch.target_loss_mask is not None
        else torch.arange(max_length, device=lengths.device).unsqueeze(0)
        < lengths.unsqueeze(1)
    )
    inputs = torch.zeros_like(targets)
    inputs[:, 0] = teacher.initial_token_id
    inputs[:, 1:] = targets[:, :-1]
    pieces = []
    for start in range(0, flat, microbatch_size):
        end = min(flat, start + microbatch_size)
        local_attributes = attributes[start:end]
        conditions = teacher.condition_provider(
            teacher.condition_provider.tokenize(local_attributes)
        )
        logits = teacher(inputs[start:end], conditions, first_step=True)
        pieces.append(logits[loss_mask[start:end]])
    return torch.cat(pieces, dim=0)


def _distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
    shared_vocab_size: int,
) -> torch.Tensor:
    selected = student_logits[labels.ne(IGNORE_INDEX)]
    if selected.shape[0] != teacher_logits.shape[0]:
        raise RuntimeError("teacher/student token alignment mismatch")
    if (
        selected.shape[-1] < shared_vocab_size
        or teacher_logits.shape[-1] < shared_vocab_size
    ):
        raise RuntimeError(
            "teacher/student output cards do not cover the tokenizer vocabulary"
        )
    student = F.log_softmax(
        selected[:, :shared_vocab_size].float() / temperature, dim=-1
    )
    teacher = F.softmax(
        teacher_logits[:, :shared_vocab_size].float() / temperature, dim=-1
    )
    return F.kl_div(student, teacher, reduction="batchmean") * temperature * temperature


def _scheduler_lambda(
    step: int,
    *,
    warmup: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _save_checkpoint(
    *,
    wrapped: nn.Module,
    raw_wrapper: LongContextTrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    output_dir: Path,
    global_step: int,
    stage_index: int,
    stage_step: int,
    manifest_hash: str,
    taxonomy: dict[str, Any],
    rank: int,
    context_draw_counts: dict[int, int] | None = None,
    source_sequence_sha256: str = _EMPTY_SOURCE_SEQUENCE_SHA256,
) -> None:
    if config.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        with FSDP.state_dict_type(
            wrapped,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            wrapper_state = wrapped.state_dict()
            optimizer_state = FSDP.optim_state_dict(wrapped, optimizer)
        state = {
            key.removeprefix("model."): value.detach().cpu().contiguous()
            for key, value in wrapper_state.items()
            if key.startswith("model.")
        }
        auxiliary_state = {
            key: value.detach().cpu().contiguous()
            for key, value in wrapper_state.items()
            if not key.startswith("model.")
        }
    else:
        wrapper_state = raw_wrapper.state_dict()
        state = {
            key.removeprefix("model."): value.detach().cpu().contiguous()
            for key, value in wrapper_state.items()
            if key.startswith("model.")
        }
        auxiliary_state = {
            key: value.detach().cpu().contiguous()
            for key, value in wrapper_state.items()
            if not key.startswith("model.")
        }
        optimizer_state = optimizer.state_dict()

    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
    }
    if dist.is_initialized():
        rng_by_rank: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(rng_by_rank, local_rng)
        source_sequence_by_rank: list[str | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(
            source_sequence_by_rank, source_sequence_sha256
        )
    else:
        rng_by_rank = [local_rng]
        source_sequence_by_rank = [source_sequence_sha256]
    if rank != 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"checkpoint_{global_step:08d}"
    tokenizer_metadata = {
        "name": config.model.tokenizer_name,
        "max_shift_steps": config.model.tokenizer_max_shift_steps,
        "segment_duration": config.model.segment_duration,
    }
    save_file(
        state,
        output_dir / f"{stem}.safetensors",
        metadata={
            "model_config": json.dumps(config.model.to_dict(), sort_keys=True),
            "tokenizer": json.dumps(tokenizer_metadata, sort_keys=True),
            "taxonomy": json.dumps(taxonomy, sort_keys=True),
            "manifest_sha256": manifest_hash,
            "experiment": json.dumps(
                {
                    "name": config.name,
                    "seed": config.seed,
                    "teacher": taxonomy.get("teacher"),
                    "conditioning": asdict(config.conditioning),
                    "condition_dropout": asdict(config.condition_dropout),
                    "augmentation_catalog_sha256": taxonomy.get(
                        "augmentation_catalog_sha256"
                    ),
                    "evidence_status": "formal_result_not_yet_evaluated",
                },
                sort_keys=True,
            ),
        },
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.model.to_dict(), indent=2) + "\n"
    )
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_metadata, indent=2) + "\n"
    )
    (output_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, sort_keys=True) + "\n"
    )
    trainer_state = {
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "manifest_sha256": manifest_hash,
        "taxonomy": taxonomy,
        "world_size": len(rng_by_rank),
        "rng_by_rank": rng_by_rank,
        "training_signature": _training_signature(config),
        "context_draw_counts": dict(context_draw_counts or {}),
        "source_sequence_sha256_by_rank": source_sequence_by_rank,
        "auxiliary_state": auxiliary_state,
    }
    torch.save(trainer_state, output_dir / f"{stem}.trainer.pt")
    latest_payload = (
        json.dumps(
            {
                "weights": f"{stem}.safetensors",
                "trainer": f"{stem}.trainer.pt",
                "global_step": global_step,
            },
            indent=2,
        )
        + "\n"
    )
    latest_tmp = output_dir / f".latest.{os.getpid()}.json.tmp"
    latest_tmp.write_text(latest_payload)
    os.replace(latest_tmp, output_dir / "latest.json")


def _load_resume_files(
    config: ExperimentConfig,
    wrapper: LongContextTrainingModel,
    manifest_hash: str,
    taxonomy: dict[str, Any],
    device: torch.device,
    rank: int,
) -> dict[str, Any] | None:
    if not config.resume:
        return None
    resume = Path(config.resume)
    if resume.is_dir():
        latest = json.loads((resume / "latest.json").read_text())
        weights = resume / latest["weights"]
        trainer_path = resume / latest["trainer"]
    else:
        trainer_path = resume
        weights = Path(str(resume).replace(".trainer.pt", ".safetensors"))
    from safetensors.torch import load_file

    wrapper.model.load_state_dict(load_file(weights, device=str(device)))
    state = torch.load(trainer_path, map_location=device, weights_only=False)
    if state["manifest_sha256"] != manifest_hash:
        raise RuntimeError("manifest changed since the resumed checkpoint")
    if state.get("taxonomy") != taxonomy:
        raise RuntimeError("dataset taxonomy changed since the resumed checkpoint")
    requires_v2_signature = (
        config.model.position_encoding == "rope"
        or config.optimizer.min_lr_ratio > 0
        or any(stage.context_distribution for stage in config.stages)
        or config.augmentation.enabled
        or config.distillation.teacher_checkpoint is not None
    )
    if state.get("training_signature") is None and requires_v2_signature:
        raise RuntimeError(
            "v2 experiments cannot resume checkpoints without a training signature"
        )
    if (
        requires_v2_signature
        and state.get("source_sequence_sha256_by_rank") is None
    ):
        raise RuntimeError(
            "v2 experiments cannot resume checkpoints without a source-sequence audit"
        )
    if state.get("training_signature") is not None and state[
        "training_signature"
    ] != _training_signature(config):
        raise RuntimeError(
            "experiment configuration changed since the resumed checkpoint"
        )
    stage_index = int(state.get("stage_index", len(config.stages)))
    if "context_draw_counts" in state and stage_index < len(config.stages):
        expected_counts = _context_counts_before(
            config.stages[stage_index],
            seed=config.seed,
            stage_step=int(state["stage_step"]),
        )
        saved_counts = {
            int(chunks): int(count)
            for chunks, count in state["context_draw_counts"].items()
        }
        if saved_counts != expected_counts:
            raise RuntimeError("checkpoint context draw counts are inconsistent")
    auxiliary_state = state.get("auxiliary_state", {})
    if auxiliary_state or any(
        not key.startswith("model.") for key in wrapper.state_dict()
    ):
        wrapper.load_auxiliary_state_dict(auxiliary_state)
    if "rng_by_rank" in state:
        current_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if state.get("world_size") != current_world_size:
            raise RuntimeError(
                "deterministic resume requires the original world size "
                f"({state.get('world_size')} != {current_world_size})"
            )
        if rank >= len(state["rng_by_rank"]):
            raise RuntimeError("checkpoint world size is smaller than the resumed rank")
    return state


def _restore_resume_rng(
    state: dict[str, Any], *, rank: int, device: torch.device
) -> None:
    """Restore RNG only after wrappers/optimizer construction is complete."""

    if "rng_by_rank" in state:
        rank_rng = state["rng_by_rank"][rank]
        random.setstate(rank_rng["python"])
        np.random.set_state(rank_rng["numpy"])
        torch.set_rng_state(rank_rng["torch"].cpu())
        if torch.cuda.is_available() and rank_rng["cuda"] is not None:
            torch.cuda.set_rng_state(rank_rng["cuda"].cpu(), device=device)
    else:
        # Backward-compatible resume for checkpoints created before per-rank
        # RNG capture was introduced.
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])


def _context_loader(
    config: ExperimentConfig,
    stage: StageConfig,
    records,
    *,
    context_chunks: int,
    world_size: int,
    rank: int,
    consumed_batches: int,
    remaining_batches: int,
    dataset_ids: dict[str, int],
    augmentation_entries,
) -> DataLoader:
    selected = [
        record
        for record in records
        if not stage.datasets or record.dataset in stage.datasets
    ]
    stage_augmentation_enabled = (
        config.augmentation.enabled
        if stage.augmentation is None
        else stage.augmentation
    )
    if stage_augmentation_enabled and not config.augmentation.enabled:
        raise RuntimeError(
            f"stage {stage.name!r} enables augmentation but the experiment does not"
        )
    stage_augmentation = replace(
        config.augmentation, enabled=stage_augmentation_enabled
    )
    dataset = ContinuousChunkDataset(
        selected,
        split=stage.split,
        context_chunks=context_chunks,
        segment_duration=config.model.segment_duration,
        dataset_ids=dataset_ids,
        include_boundary_targets=config.boundary_state_supervision.enabled,
        augmentation_config=stage_augmentation,
        augmentation_entries=(
            augmentation_entries if stage_augmentation_enabled else []
        ),
    )
    if not len(dataset):
        raise RuntimeError(
            f"stage {stage.name!r} has no "
            f"{context_chunks * config.model.segment_duration:g}s "
            "continuous training windows"
        )
    global_examples = max(
        1,
        round(
            config.global_batch_audio_seconds
            / (config.model.segment_duration * context_chunks)
        ),
    )
    per_device = max(1, global_examples // world_size)
    sampler = BalancedWindowBatchSampler(
        dataset,
        batch_size=per_device,
        num_batches=remaining_batches,
        seed=(
            config.seed
            + 10_007 * context_chunks
            + int.from_bytes(
                hashlib.sha256(stage.name.encode("utf-8")).digest()[:4], "big"
            )
        ),
        rank=rank,
        world_size=world_size,
        start_batch=consumed_batches,
        dataset_probabilities=stage.dataset_weights or None,
    )
    worker_generator = torch.Generator()
    worker_generator.manual_seed(
        config.seed
        + 1_000_003 * rank
        + 10_007 * context_chunks
        + int.from_bytes(
            hashlib.sha256(stage.name.encode("utf-8")).digest()[:4], "big"
        )
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        generator=worker_generator,
        collate_fn=collate_training_examples,
    )


def _stage_loaders(
    config: ExperimentConfig,
    stage: StageConfig,
    records,
    *,
    world_size: int,
    rank: int,
    stage_step: int,
    dataset_ids: dict[str, int],
    augmentation_entries,
) -> dict[int, DataLoader]:
    consumed = _context_counts_before(stage, seed=config.seed, stage_step=stage_step)
    total = _context_counts_before(stage, seed=config.seed, stage_step=stage.steps)
    loaders: dict[int, DataLoader] = {}
    for chunks in stage.context_probabilities():
        remaining = total[chunks] - consumed[chunks]
        if remaining <= 0:
            continue
        loaders[chunks] = _context_loader(
            config,
            stage,
            records,
            context_chunks=chunks,
            world_size=world_size,
            rank=rank,
            consumed_batches=consumed[chunks],
            remaining_batches=remaining,
            dataset_ids=dataset_ids,
            augmentation_entries=augmentation_entries,
        )
    return loaders


def train(config: ExperimentConfig) -> None:
    rank, local_rank, world_size = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid-Mamba training requires a CUDA machine")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _seed_everything(config.seed, rank)

    output_validation: list[str | None] = [None]
    if rank == 0:
        try:
            _assert_new_run_output_is_empty(config)
        except Exception as exc:
            output_validation[0] = str(exc)
    if dist.is_initialized():
        dist.broadcast_object_list(output_validation, src=0)
    if output_validation[0] is not None:
        raise RuntimeError(output_validation[0])

    records = read_manifest(config.manifest)
    manifest_hash = manifest_fingerprint(config.manifest)
    augmentation_entries = []
    augmentation_catalog_hash = None
    if config.augmentation.enabled:
        augmentation_entries = read_augmentation_catalog(config.augmentation.catalog)
        maximum_context = max(
            chunks
            for stage in config.stages
            for chunks in stage.context_probabilities()
        )
        validation_payload: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                validate_augmentation_catalog(
                    augmentation_entries,
                    records,
                    allowed_datasets=config.augmentation.allowed_datasets,
                    required_duration=(
                        maximum_context * config.model.segment_duration
                    ),
                    check_files=True,
                )
                validation_payload[0] = {
                    "sha256": augmentation_catalog_fingerprint(
                        config.augmentation.catalog
                    )
                }
            except Exception as exc:
                validation_payload[0] = {"error": str(exc)}
        if dist.is_initialized():
            dist.broadcast_object_list(validation_payload, src=0)
        if validation_payload[0] is None:
            raise RuntimeError("augmentation validation result was not broadcast")
        if "error" in validation_payload[0]:
            raise RuntimeError(validation_payload[0]["error"])
        augmentation_catalog_hash = validation_payload[0]["sha256"]
    dataset_ids = {
        name: index
        for index, name in enumerate(sorted({record.dataset for record in records}))
    }
    if len(dataset_ids) > config.model.num_dataset_condition_classes:
        raise ValueError(
            f"manifest has {len(dataset_ids)} datasets but the model allows "
            f"{config.model.num_dataset_condition_classes}"
        )
    student_tokenizer = MT3Tokenizer(
        instrument_vocabulary=config.model.tokenizer_name,
        max_shift_steps=config.model.tokenizer_max_shift_steps,
    )
    student_tokenizer_contract = _tokenizer_contract(
        student_tokenizer, config.model
    )
    taxonomy: dict[str, Any] = {
        "datasets": dataset_ids,
        "instrument_vocabulary": config.model.tokenizer_name,
        "instrument_condition_classes": (config.model.num_instrument_condition_classes),
        "augmentation_catalog_sha256": augmentation_catalog_hash,
        "tokenizer_contract": student_tokenizer_contract,
    }
    teacher_provenance = _verify_teacher_checkpoint(config, rank=rank)
    taxonomy["teacher"] = teacher_provenance
    raw_model = _build_model(device, config.model)
    for conditioner in raw_model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
    if hasattr(raw_model.transformer, "set_gradient_checkpointing"):
        raw_model.transformer.set_gradient_checkpointing(config.gradient_checkpointing)
    raw_wrapper = LongContextTrainingModel(
        raw_model,
        config.boundary_state_supervision,
        tokenizer=student_tokenizer,
        clean_cache_config=config.clean_cache_training,
    ).to(device)
    teacher = None
    shared_vocab_size = student_tokenizer.num_tokens
    if config.distillation.teacher_checkpoint:
        loaded_teacher = TranscriptionModel.load_model(
            teacher_provenance["path"],
            device=device,
            dtype=config.precision,
        )
        teacher = loaded_teacher._model
        teacher_tokenizer_contract = _tokenizer_contract(
            loaded_teacher._tokenizer, teacher.model_config
        )
        if teacher_tokenizer_contract != student_tokenizer_contract:
            raise RuntimeError(
                "teacher/student tokenizer event or instrument mappings are not identical"
            )
        if teacher.card < shared_vocab_size or raw_model.card < shared_vocab_size:
            raise RuntimeError(
                "teacher/student output cards do not cover the tokenizer vocabulary"
            )
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("teacher must remain fully frozen")
        teacher_provenance.update(
            {
                "tokenizer_contract": teacher_tokenizer_contract,
                "conditioning": "audio_only",
                "chunk_policy": "independent_5s",
                "teacher_microbatch_size": (
                    config.distillation.teacher_microbatch_size
                ),
            }
        )
    resume_state = _load_resume_files(
        config, raw_wrapper, manifest_hash, taxonomy, device, rank
    )

    if config.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
        )

        wrapped: nn.Module = FSDP(
            raw_wrapper,
            device_id=device,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=getattr(torch, config.precision),
                reduce_dtype=torch.float32,
                buffer_dtype=getattr(torch, config.precision),
            ),
        )
    elif config.distributed == "ddp" and world_size > 1:
        wrapped = DistributedDataParallel(
            raw_wrapper,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        wrapped = raw_wrapper

    optimizer = torch.optim.AdamW(
        _optimizer_groups(wrapped, config.optimizer.weight_decay),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
    )
    total_steps = sum(stage.steps for stage in config.stages)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _scheduler_lambda(
            step,
            warmup=config.optimizer.warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.optimizer.min_lr_ratio,
        ),
    )
    if resume_state is not None:
        if config.distributed == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            optimizer_state = FSDP.optim_state_dict_to_load(
                wrapped, optimizer, resume_state["optimizer"]
            )
            optimizer.load_state_dict(optimizer_state)
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        resume_stage = int(resume_state["stage_index"])
        resume_stage_step = int(resume_state["stage_step"])
        source_hashes = resume_state.get("source_sequence_sha256_by_rank")
        source_sequence_hash = (
            str(source_hashes[rank])
            if source_hashes is not None
            else _EMPTY_SOURCE_SEQUENCE_SHA256
        )
    else:
        global_step = resume_stage = resume_stage_step = 0
        source_sequence_hash = _EMPTY_SOURCE_SEQUENCE_SHA256

    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "experiment.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )
    log_path = output_dir / f"metrics.rank{rank}.jsonl"
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output_dir / "tensorboard")
        except ImportError:
            writer = None
    if resume_state is not None:
        # Make the checkpoint RNG the final mutable setup state.  DataLoader
        # workers use dedicated generators below, so they cannot perturb the
        # condition-dropout sequence after this point.
        _restore_resume_rng(resume_state, rank=rank, device=device)
    last_time = time.perf_counter()

    for stage_index, stage in enumerate(config.stages):
        if stage_index < resume_stage:
            continue
        stage_start = resume_stage_step if stage_index == resume_stage else 0
        if stage_start >= stage.steps:
            resume_stage_step = 0
            continue
        loaders = _stage_loaders(
            config,
            stage,
            records,
            world_size=world_size,
            rank=rank,
            stage_step=stage_start,
            dataset_ids=dataset_ids,
            augmentation_entries=augmentation_entries,
        )
        loader_iterators = {
            chunks: iter(loader) for chunks, loader in loaders.items()
        }
        stage_context_counts = _context_counts_before(
            stage, seed=config.seed, stage_step=stage_start
        )
        wrapped.train()
        for local_index in range(stage_start, stage.steps):
            context_chunks = _context_for_stage_step(
                stage, seed=config.seed, stage_step=local_index
            )
            try:
                batch = next(loader_iterators[context_chunks])
            except (KeyError, StopIteration) as exc:
                raise RuntimeError(
                    f"stage {stage.name!r} exhausted context={context_chunks} "
                    f"at stage_step={local_index}"
                ) from exc
            stage_context_counts[context_chunks] += 1
            source_sequence_hash = _update_source_sequence_hash(
                source_sequence_hash,
                batch,
                stage_index=stage_index,
                stage_step=local_index,
                context_chunks=context_chunks,
            )
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, config.precision):
                dropout = config.condition_dropout
                (
                    logits,
                    labels,
                    token_count,
                    active_logits,
                    reentry_logits,
                    symbolic_mode_counts,
                ) = wrapped(
                    batch,
                    dropout.audio,
                    dropout.instrument,
                    dropout.dataset,
                    config.conditioning.dropout_scope,
                    global_step,
                )
                token_losses = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                )
                valid_token = labels.reshape(-1).ne(IGNORE_INDEX)
                token_weights = torch.ones_like(token_losses)
                token_weights = torch.where(
                    labels.reshape(-1).eq(student_tokenizer.eos_id),
                    torch.full_like(token_weights, config.true_eos_loss_weight),
                    token_weights,
                )
                ce = (token_losses[valid_token] * token_weights[valid_token]).sum()
                ce = ce / token_weights[valid_token].sum().clamp_min(1.0)
                loss = ce
                active_loss = torch.zeros((), device=device)
                reentry_loss = torch.zeros((), device=device)
                boundary_config = config.boundary_state_supervision
                if active_logits is not None:
                    if batch.active_note_targets is None:
                        raise RuntimeError("batch is missing active-note targets")
                    active_loss = _balanced_active_note_loss(
                        active_logits.float(), batch.active_note_targets
                    )
                    loss = loss + boundary_config.active_weight * active_loss
                if reentry_logits is not None:
                    if batch.reentry_targets is None or batch.reentry_valid is None:
                        raise RuntimeError("batch is missing re-entry targets")
                    valid = batch.reentry_valid
                    if valid.any():
                        reentry_loss = F.cross_entropy(
                            reentry_logits[valid].float(),
                            batch.reentry_targets[valid],
                        )
                    else:
                        # Preserve a zero gradient for DDP/FSDP batches whose
                        # sampled windows contain no uncensored re-entry label.
                        reentry_loss = reentry_logits.sum() * 0.0
                    loss = loss + boundary_config.reentry_weight * reentry_loss
                kd = torch.zeros((), device=device)
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = _teacher_logits(
                            teacher,
                            batch,
                            microbatch_size=(
                                config.distillation.teacher_microbatch_size
                            ),
                        )
                    kd = _distillation_loss(
                        logits,
                        labels,
                        teacher_logits,
                        temperature=config.distillation.temperature,
                        shared_vocab_size=shared_vocab_size,
                    )
                    loss = loss + config.distillation.weight * kd
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapped.parameters(), config.optimizer.gradient_clip
            )
            health_checked = global_step < 1000 or (
                (global_step + 1) % config.log_every == 0
            )
            teacher_frozen = True
            distill_positive = True
            if health_checked:
                teacher_frozen = teacher is None or all(
                    parameter.grad is None for parameter in teacher.parameters()
                )
                distill_positive = teacher is None or bool(kd.detach() > 0)
                finite = all(
                    bool(torch.isfinite(value.detach()))
                    for value in (loss, ce, kd, grad_norm)
                )
                healthy = finite and distill_positive and teacher_frozen
                health = torch.tensor(
                    int(healthy), device=device, dtype=torch.int32
                )
                if dist.is_initialized():
                    dist.all_reduce(health, op=dist.ReduceOp.MIN)
                if not bool(health.item()):
                    raise FloatingPointError(
                        "training health check failed: require finite CE/KL/loss/grad, "
                        "positive KL, and a frozen teacher"
                    )
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % config.log_every == 0:
                now = time.perf_counter()
                interval_seconds = now - last_time
                batch_audio_seconds = (
                    batch.waveform.shape[0]
                    * batch.waveform.shape[1]
                    * config.model.segment_duration
                    * world_size
                )
                metrics: dict[str, Any] = {
                    "step": global_step,
                    "stage": stage.name,
                    "stage_step": local_index + 1,
                    "optimizer_step": global_step,
                    "context_chunks": context_chunks,
                    "loss": float(loss.detach()),
                    "ce": float(ce.detach()),
                    "distill": float(kd.detach()),
                    "active_note_loss": float(active_loss.detach()),
                    "reentry_loss": float(reentry_loss.detach()),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": token_count,
                    "truncated_chunk_fraction": (
                        float(batch.truncated_chunks.float().mean())
                        if batch.truncated_chunks is not None
                        else 0.0
                    ),
                    "dense_chunk_fraction": (
                        float(
                            (batch.original_target_lengths > 1500).float().mean()
                        )
                        if batch.original_target_lengths is not None
                        else 0.0
                    ),
                    "target_tokens_p95": (
                        float(
                            torch.quantile(
                                batch.original_target_lengths.float(), 0.95
                            )
                        )
                        if batch.original_target_lengths is not None
                        else 0.0
                    ),
                    "target_tokens_p99": (
                        float(
                            torch.quantile(
                                batch.original_target_lengths.float(), 0.99
                            )
                        )
                        if batch.original_target_lengths is not None
                        else 0.0
                    ),
                    "augmentation_fraction": (
                        float(batch.augmentation_applied.float().mean())
                        if batch.augmentation_applied is not None
                        else 0.0
                    ),
                    "remix_fraction": (
                        float(batch.remixed.float().mean())
                        if batch.remixed is not None
                        else 0.0
                    ),
                    "pitch_shift_fraction": (
                        float(batch.pitch_shift_semitones.ne(0).float().mean())
                        if batch.pitch_shift_semitones is not None
                        else 0.0
                    ),
                    "health_checked": health_checked,
                    "distill_positive": distill_positive,
                    "teacher_frozen": teacher_frozen,
                    "source_sequence_sha256": source_sequence_hash,
                    "steps_per_second": config.log_every / interval_seconds,
                    "audio_seconds_per_second": (
                        config.log_every * batch_audio_seconds / interval_seconds
                    ),
                    "peak_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / (1024**3)
                    ),
                    "symbolic_state_gt_fraction": float(
                        symbolic_mode_counts[0].float()
                        / symbolic_mode_counts.sum().clamp_min(1)
                    ),
                    "symbolic_state_corrupt_fraction": float(
                        symbolic_mode_counts[1].float()
                        / symbolic_mode_counts.sum().clamp_min(1)
                    ),
                    "symbolic_state_predicted_fraction": float(
                        symbolic_mode_counts[2].float()
                        / symbolic_mode_counts.sum().clamp_min(1)
                    ),
                }
                last_time = now
                with log_path.open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if rank == 0:
                    print(json.dumps(metrics), flush=True)
                    if writer is not None:
                        for key in (
                            "loss",
                            "ce",
                            "distill",
                            "active_note_loss",
                            "reentry_loss",
                            "grad_norm",
                            "lr",
                            "truncated_chunk_fraction",
                            "dense_chunk_fraction",
                            "target_tokens_p95",
                            "target_tokens_p99",
                            "augmentation_fraction",
                            "remix_fraction",
                            "pitch_shift_fraction",
                            "symbolic_state_gt_fraction",
                            "symbolic_state_corrupt_fraction",
                            "symbolic_state_predicted_fraction",
                        ):
                            writer.add_scalar(key, metrics[key], global_step)

            if global_step % config.save_every == 0:
                _save_checkpoint(
                    wrapped=wrapped,
                    raw_wrapper=raw_wrapper,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    output_dir=output_dir,
                    global_step=global_step,
                    stage_index=stage_index,
                    stage_step=local_index + 1,
                    manifest_hash=manifest_hash,
                    taxonomy=taxonomy,
                    rank=rank,
                    context_draw_counts=stage_context_counts,
                    source_sequence_sha256=source_sequence_hash,
                )
        resume_stage_step = 0

    _save_checkpoint(
        wrapped=wrapped,
        raw_wrapper=raw_wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=output_dir,
        global_step=global_step,
        stage_index=len(config.stages),
        stage_step=0,
        manifest_hash=manifest_hash,
        taxonomy=taxonomy,
        rank=rank,
        context_draw_counts={},
        source_sequence_sha256=source_sequence_hash,
    )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--name")
    parser.add_argument("--preflight-steps", type=int)
    parser.add_argument("--context-chunks", type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--global-batch-audio-seconds", type=int)
    args = parser.parse_args()
    from muscriptor.training.config import load_experiment_config

    config = load_experiment_config(args.config)
    if args.seed is not None and args.seed != config.seed and not args.output_dir:
        parser.error("--seed overrides require a distinct --output-dir")
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.name is not None:
        overrides["name"] = args.name
    if args.global_batch_audio_seconds is not None:
        overrides["global_batch_audio_seconds"] = args.global_batch_audio_seconds
    config = replace(config, **overrides)
    if args.preflight_steps is not None or args.context_chunks is not None:
        if args.preflight_steps is None or args.context_chunks is None:
            parser.error("preflight requires both --preflight-steps and --context-chunks")
        if args.preflight_steps <= 0:
            parser.error("--preflight-steps must be positive")
        if args.output_dir is None:
            parser.error("preflight requires a distinct --output-dir")
        source_stage = config.stages[0]
        preflight_stage = replace(
            source_stage,
            name=f"preflight_{args.context_chunks * 5}s",
            steps=args.preflight_steps,
            context_chunks=args.context_chunks,
            context_distribution={},
            datasets=["slakh2100_redux"],
            dataset_weights={},
            augmentation=False,
        )
        config = replace(config, stages=[preflight_stage], resume=None)
    train(config)


if __name__ == "__main__":
    main()
