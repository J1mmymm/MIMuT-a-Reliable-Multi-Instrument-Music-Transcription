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


@dataclass
class DecoderTrainingBatch:
    """Per-chunk decoder inputs with long-context acoustic conditions."""

    inputs: torch.Tensor  # [B*C, L]
    labels: torch.Tensor  # [B*C, L], IGNORE_INDEX outside supervised tokens
    conditions: dict
    token_count: int


def _condition_attributes(
    batch: TrainingBatch,
    *,
    sample_rate: int,
    condition_dropout: float,
) -> list[ConditioningAttributes]:
    bsz, chunks, _, samples = batch.waveform.shape
    result = []
    for batch_index in range(bsz):
        # One draw per condition per SEQUENCE, applied to all of its chunks.
        # This matches classifier-free-guidance semantics at inference: a
        # whole stream is conditional or unconditional, never a mix in which
        # some chunks' audio silently disappears mid-song.  The dataset
        # condition is dropped too — inference always runs with
        # dataset_name=None, so the null dataset slot must be trained.
        device = batch.waveform.device
        keep_audio = (
            condition_dropout <= 0
            or torch.rand((), device=device) >= condition_dropout
        )
        keep_instruments = (
            condition_dropout <= 0
            or torch.rand((), device=device) >= condition_dropout
        )
        keep_dataset = (
            condition_dropout <= 0
            or torch.rand((), device=device) >= condition_dropout
        )
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
                seek_time=[float(batch.start_times[batch_index])],
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
    condition_dropout: float = 0.2,
) -> PackedBatch:
    if model.type_embedding is None or model.chunk_start is None:
        raise ValueError("long-context training requires use_type_embeddings=True")
    bsz, chunks, _ = batch.target_ids.shape[:3]
    attributes = _condition_attributes(
        batch,
        sample_rate=sample_rate,
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
            # The teacher-forced input stream may differ from the labels:
            # with tie corruption, the prologue tokens the model READS are
            # perturbed while the labels for that region are masked, so the
            # model learns to continue correctly from an imperfect forced
            # prelude (as happens at inference) without being taught to
            # reproduce the perturbation.
            input_source = (
                batch.input_ids[batch_index, chunk_index, :length]
                if batch.input_ids is not None
                else targets
            )
            inputs = torch.empty_like(targets)
            inputs[0] = model.initial_token_id
            if length > 1:
                inputs[1:] = input_source[:-1]
            token_embeddings = model.emb(inputs)
            token_embeddings = token_embeddings + model.type_embedding.weight[3]
            pieces.append(token_embeddings)
            labels = targets
            if (
                batch.tie_corrupt is not None
                and batch.tie_lengths is not None
                and bool(batch.tie_corrupt[batch_index, chunk_index])
            ):
                labels = targets.clone()
                tie_length = int(batch.tie_lengths[batch_index, chunk_index])
                labels[: min(tie_length, length)] = IGNORE_INDEX
            label_pieces.append(labels)
            flat_offset += length
            token_count += length
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


def prepare_decoder_training_batch(
    model: LMModel,
    batch: TrainingBatch,
    *,
    sample_rate: int = 16_000,
    condition_dropout: float = 0.2,
) -> DecoderTrainingBatch:
    """Encode audio continuously but build independent decoder sequences.

    Audio frames belonging to one training example are concatenated inside
    the acoustic conditioner, so Mamba receives 5/10/20/40 seconds in one
    causal forward pass.  The resulting conditions are split back to B*C
    five-second rows before the token decoder is called.  Every flattened row
    gets an independent decoder forward pass, so Hybrid-Mamba decoder state is
    reset at chunk boundaries just like Transformer KV state.
    """

    bsz, chunks, max_length = batch.target_ids.shape
    attributes = _condition_attributes(
        batch,
        sample_rate=sample_rate,
        condition_dropout=condition_dropout,
    )
    tokenized = model.condition_provider.tokenize(attributes)
    conditions = model.condition_provider(
        tokenized,
        acoustic_batch_shape=(bsz, chunks),
    )

    targets = batch.target_ids.reshape(bsz * chunks, max_length)
    sources = (
        batch.input_ids if batch.input_ids is not None else batch.target_ids
    ).reshape(bsz * chunks, max_length)
    lengths = batch.target_lengths.reshape(bsz * chunks)

    inputs = torch.zeros_like(targets)
    inputs[:, 0] = model.initial_token_id
    if max_length > 1:
        inputs[:, 1:] = sources[:, :-1]

    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    if batch.tie_corrupt is not None and batch.tie_lengths is not None:
        corrupt = batch.tie_corrupt.reshape(bsz * chunks).unsqueeze(1)
        tie_lengths = batch.tie_lengths.reshape(bsz * chunks).unsqueeze(1)
        valid = valid & ~(corrupt & (positions < tie_lengths))

    labels = targets.masked_fill(~valid, IGNORE_INDEX)
    return DecoderTrainingBatch(
        inputs=inputs,
        labels=labels,
        conditions=conditions,
        token_count=int(valid.sum().item()),
    )
