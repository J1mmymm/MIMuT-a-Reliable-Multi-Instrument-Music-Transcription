"""Pack repeated audio/MIDI chunks for long-context teacher forcing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from muscriptor.data.dataset import TrainingBatch
from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import (
    ConditioningAttributes,
    WavCondition,
)


IGNORE_INDEX = -100


@dataclass
class PackedBatch:
    embeddings: torch.Tensor  # [B, T, D]
    labels: torch.Tensor  # [B, T], IGNORE_INDEX outside MIDI positions
    token_count: int
    chunk_end_positions: torch.Tensor  # [B, C]


def _condition_attributes(
    batch: TrainingBatch,
    *,
    sample_rate: int,
    audio_condition_dropout: float = 0.0,
    instrument_condition_dropout: float = 1.0,
    dataset_condition_dropout: float = 1.0,
    dropout_scope: str = "sequence",
    condition_dropout: float | None = None,
) -> list[ConditioningAttributes]:
    """Build independently masked acoustic and metadata conditions.

    ``condition_dropout`` is retained only as a compatibility alias for old
    callers; when supplied it overrides all three explicit probabilities.
    A mask is sampled once per contiguous example and reused for every chunk,
    so the recurrent model never sees a randomly missing middle chunk.
    """

    if condition_dropout is not None:
        audio_condition_dropout = condition_dropout
        instrument_condition_dropout = condition_dropout
        dataset_condition_dropout = condition_dropout

    def keep(probability: float, device: torch.device) -> bool:
        if probability <= 0:
            return True
        if probability >= 1:
            return False
        return bool(torch.rand((), device=device) >= probability)

    if dropout_scope != "sequence":
        raise ValueError("dropout_scope must be sequence")

    bsz, chunks, _, samples = batch.waveform.shape
    result = []
    for batch_index in range(bsz):
        keep_audio = keep(audio_condition_dropout, batch.waveform.device)
        keep_instruments = keep(
            instrument_condition_dropout, batch.waveform.device
        )
        keep_dataset = keep(dataset_condition_dropout, batch.waveform.device)
        for chunk_index in range(chunks):
            waveform = batch.waveform[batch_index : batch_index + 1, chunk_index]
            if keep_audio:
                length = torch.tensor(
                    [samples], device=waveform.device, dtype=torch.long
                )
            else:
                waveform = torch.zeros_like(waveform[..., :1])
                length = torch.zeros(1, device=waveform.device, dtype=torch.long)
            wav = WavCondition(
                wav=waveform,
                length=length,
                sample_rate=[sample_rate],
                path=[None],
                seek_time=[
                    float(batch.start_times[batch_index])
                    + chunk_index * (samples / sample_rate)
                ],
            )
            result.append(
                ConditioningAttributes(
                    wav={"self_wav": wav},
                    text={
                        "instrument_group": (
                            batch.instrument_groups[batch_index]
                            if keep_instruments
                            else None
                        ),
                        "dataset_name": (
                            str(int(batch.dataset_ids[batch_index].item()))
                            if keep_dataset
                            else None
                        ),
                    },
                )
            )
    return result


def pack_training_batch(
    model: LMModel,
    batch: TrainingBatch,
    *,
    sample_rate: int = 16_000,
    audio_condition_dropout: float = 0.0,
    instrument_condition_dropout: float = 1.0,
    dataset_condition_dropout: float = 1.0,
    dropout_scope: str = "sequence",
    condition_dropout: float | None = None,
) -> PackedBatch:
    if model.type_embedding is None or model.chunk_start is None:
        raise ValueError("long-context training requires use_type_embeddings=True")
    bsz, chunks, _ = batch.target_ids.shape[:3]
    attributes = _condition_attributes(
        batch,
        sample_rate=sample_rate,
        audio_condition_dropout=audio_condition_dropout,
        instrument_condition_dropout=instrument_condition_dropout,
        dataset_condition_dropout=dataset_condition_dropout,
        dropout_scope=dropout_scope,
        condition_dropout=condition_dropout,
    )
    tokenized = model.condition_provider.tokenize(attributes)
    conditions = model.condition_provider(tokenized)
    preferred = ["self_wav", "instrument_group", "dataset_name"]
    ordered = [key for key in preferred if key in conditions]
    ordered.extend(key for key in conditions if key not in ordered)

    samples: list[torch.Tensor] = []
    labels_per_sample: list[torch.Tensor] = []
    chunk_ends = torch.zeros(
        bsz, chunks, dtype=torch.long, device=batch.target_ids.device
    )
    token_count = 0
    for batch_index in range(bsz):
        pieces: list[torch.Tensor] = []
        label_pieces: list[torch.Tensor] = []
        flat_offset = 0
        for chunk_index in range(chunks):
            flat_index = batch_index * chunks + chunk_index
            chunk = model.chunk_start[0]
            chunk = chunk + model.type_embedding.weight[0]
            pieces.append(chunk)
            label_pieces.append(
                torch.full(
                    (1,),
                    IGNORE_INDEX,
                    dtype=torch.long,
                    device=batch.target_ids.device,
                )
            )
            flat_offset += 1

            for key in ordered:
                cond, mask = conditions[key]
                value = cond[flat_index].to(model.emb.weight.dtype)
                kind = 1 if key == "self_wav" else 2
                value = value + model.type_embedding.weight[kind]
                value = value * mask[flat_index].unsqueeze(-1).to(value.dtype)
                pieces.append(value)
                label_pieces.append(
                    torch.full(
                        (value.shape[0],),
                        IGNORE_INDEX,
                        dtype=torch.long,
                        device=batch.target_ids.device,
                    )
                )
                flat_offset += value.shape[0]

            length = int(batch.target_lengths[batch_index, chunk_index])
            targets = batch.target_ids[batch_index, chunk_index, :length]
            inputs = torch.empty_like(targets)
            inputs[0] = model.initial_token_id
            if length > 1:
                inputs[1:] = targets[:-1]
            token_embeddings = model.emb(inputs)
            token_embeddings = token_embeddings + model.type_embedding.weight[3]
            pieces.append(token_embeddings)
            target_labels = targets
            if batch.target_loss_mask is not None:
                mask = batch.target_loss_mask[
                    batch_index, chunk_index, :length
                ]
                target_labels = targets.masked_fill(~mask, IGNORE_INDEX)
            label_pieces.append(target_labels)
            flat_offset += length
            token_count += int(target_labels.ne(IGNORE_INDEX).sum().item())
            chunk_ends[batch_index, chunk_index] = flat_offset

        samples.append(torch.cat(pieces, dim=0))
        labels_per_sample.append(torch.cat(label_pieces, dim=0))

    max_length = max(sample.shape[0] for sample in samples)
    embeddings = torch.stack(
        [F.pad(sample, (0, 0, 0, max_length - sample.shape[0])) for sample in samples]
    )
    labels = torch.stack(
        [
            F.pad(
                sample,
                (0, max_length - sample.shape[0]),
                value=IGNORE_INDEX,
            )
            for sample in labels_per_sample
        ]
    )
    return PackedBatch(
        embeddings=embeddings,
        labels=labels,
        token_count=token_count,
        chunk_end_positions=chunk_ends,
    )
