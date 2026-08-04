"""Contiguous multi-chunk training dataset."""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from functools import lru_cache

import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from muscriptor.data.manifest import ManifestRecord
from muscriptor.data.schema import load_standardized_track
from muscriptor.tokenizer.encode import (
    corrupt_tie_section,
    encode_contiguous_chunks,
    tie_section_length,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.audio import resample


@dataclass
class TrainingExample:
    waveform: torch.Tensor  # [C, 1, samples]
    target_chunks: list[list[int]]
    instrument_group: str | None
    dataset_id: int
    track_id: str
    start_time: float
    has_long_gap: bool
    # Teacher-forcing input stream: identical to target_chunks except for
    # chunks whose tie prologue was corrupted (see tie_corruption).
    input_chunks: list[list[int]] = field(default_factory=list)
    tie_lengths: list[int] = field(default_factory=list)
    tie_corrupt: list[bool] = field(default_factory=list)


@dataclass
class TrainingBatch:
    waveform: torch.Tensor  # [B, C, 1, samples]
    target_ids: torch.Tensor  # [B, C, L]
    target_lengths: torch.Tensor  # [B, C]
    instrument_groups: list[str | None]
    dataset_ids: torch.Tensor
    track_ids: list[str]
    start_times: torch.Tensor
    has_long_gap: torch.Tensor
    # Optional (None reproduces the historical inputs == shifted targets).
    input_ids: torch.Tensor | None = None  # [B, C, L]
    tie_lengths: torch.Tensor | None = None  # [B, C]
    tie_corrupt: torch.Tensor | None = None  # [B, C] bool

    def to(self, device: torch.device | str) -> "TrainingBatch":
        return TrainingBatch(
            waveform=self.waveform.to(device),
            target_ids=self.target_ids.to(device),
            target_lengths=self.target_lengths.to(device),
            instrument_groups=self.instrument_groups,
            dataset_ids=self.dataset_ids.to(device),
            track_ids=self.track_ids,
            start_times=self.start_times.to(device),
            has_long_gap=self.has_long_gap.to(device),
            input_ids=(
                self.input_ids.to(device) if self.input_ids is not None else None
            ),
            tie_lengths=(
                self.tie_lengths.to(device) if self.tie_lengths is not None else None
            ),
            tie_corrupt=(
                self.tie_corrupt.to(device) if self.tie_corrupt is not None else None
            ),
        )


@lru_cache(maxsize=256)
def _load_notes(path: str) -> tuple[Note, ...]:
    return tuple(load_standardized_track(path).notes)


def _record_window_gap_bins(
    notes: tuple[Note, ...],
    *,
    num_windows: int,
    context_duration: float,
    segment_duration: float,
) -> list[str]:
    """Assign gap strata in O(notes * context_chunks), not O(windows*notes)."""
    by_program: dict[int, list[Note]] = {}
    for note in notes:
        if note.is_drum:
            continue
        by_program.setdefault(note.program, []).append(note)
    maximum_gaps: list[float | None] = [None] * num_windows
    for program_notes in by_program.values():
        ordered = sorted(program_notes, key=lambda note: note.onset)
        for previous, current in zip(ordered, ordered[1:]):
            gap = max(0.0, current.onset - previous.offset)
            first = max(
                0,
                math.ceil((current.onset - context_duration) / segment_duration - 1e-9),
            )
            last = min(
                num_windows - 1,
                math.floor(previous.offset / segment_duration + 1e-9),
            )
            for window_index in range(first, last + 1):
                old = maximum_gaps[window_index]
                maximum_gaps[window_index] = gap if old is None else max(old, gap)

    def label(maximum_gap: float | None) -> str:
        if maximum_gap is None:
            return "none"
        if maximum_gap < 5:
            return "0-5"
        if maximum_gap < 10:
            return "5-10"
        if maximum_gap < 20:
            return "10-20"
        return "20+"

    return [label(gap) for gap in maximum_gaps]


class ContinuousChunkDataset(Dataset[TrainingExample]):
    def __init__(
        self,
        records: list[ManifestRecord],
        *,
        split: str = "train",
        context_chunks: int = 1,
        segment_duration: float = 5.0,
        sample_rate: int = 16_000,
        max_tokens_per_chunk: int = 4096,
        dataset_ids: dict[str, int] | None = None,
        tie_corruption: float = 0.0,
    ):
        if context_chunks not in {1, 2, 4, 8}:
            raise ValueError("context_chunks must be one of 1, 2, 4, 8")
        if not 0 <= tie_corruption < 1:
            raise ValueError("tie_corruption must be in [0, 1)")
        self.records = [record for record in records if record.split == split]
        self.context_chunks = context_chunks
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.tie_corruption = tie_corruption
        names = sorted({record.dataset for record in records})
        self.dataset_ids = dataset_ids or {
            name: index for index, name in enumerate(names)
        }
        self.tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
        )
        self.windows: list[tuple[int, int]] = []
        self.long_gap_flags: list[bool] = []
        self.long_gap_bins: list[str] = []
        for record_index, record in enumerate(self.records):
            chunks = max(1, math.ceil(record.duration / segment_duration))
            if context_chunks == 1:
                possible = chunks
            else:
                context_duration = context_chunks * segment_duration
                possible = max(
                    0,
                    math.floor((record.duration - context_duration) / segment_duration)
                    + 1,
                )
            notes = _load_notes(record.notes_path)
            gap_bins = _record_window_gap_bins(
                notes,
                num_windows=possible,
                context_duration=context_chunks * segment_duration,
                segment_duration=segment_duration,
            )
            for chunk_index, gap_bin in enumerate(gap_bins):
                self.windows.append((record_index, chunk_index))
                self.long_gap_bins.append(gap_bin)
                self.long_gap_flags.append(gap_bin in {"5-10", "10-20", "20+"})

    def __len__(self) -> int:
        return len(self.windows)

    def _load_audio(
        self, path: str, start_time: float, duration: float
    ) -> torch.Tensor:
        info = sf.info(path)
        start = round(start_time * info.samplerate)
        frames = round(duration * info.samplerate)
        array, source_rate = sf.read(
            path,
            start=start,
            frames=frames,
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(array).transpose(0, 1)
        waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            waveform = resample(waveform, source_rate, self.sample_rate)
        target_samples = round(duration * self.sample_rate)
        if waveform.shape[-1] < target_samples:
            waveform = F.pad(waveform, (0, target_samples - waveform.shape[-1]))
        return waveform[:, :target_samples]

    def __getitem__(self, index: int) -> TrainingExample:
        record_index, start_chunk = self.windows[index]
        record = self.records[record_index]
        start_time = start_chunk * self.segment_duration
        total_duration = self.context_chunks * self.segment_duration
        waveform = (
            self._load_audio(record.audio_path, start_time, total_duration)
            .view(
                1,
                self.context_chunks,
                round(self.segment_duration * self.sample_rate),
            )
            .transpose(0, 1)
        )
        notes = _load_notes(record.notes_path)
        chunks = encode_contiguous_chunks(
            self.tokenizer,
            list(notes),
            start_time=start_time,
            num_chunks=self.context_chunks,
            duration=self.segment_duration,
        )
        targets = []
        input_targets = []
        tie_lengths = []
        tie_corrupt = []
        for chunk_index, chunk in enumerate(chunks):
            ids = chunk.target_ids
            if len(ids) > self.max_tokens_per_chunk:
                # Truncate WITHOUT appending EOS: supervising EOS at an
                # arbitrary truncation point would teach the model to stop
                # early on dense chunks.  The label at the cut is masked in
                # pack_training_batch via the length alone; inference relies
                # on the generation budget for such pathological chunks.
                ids = ids[: self.max_tokens_per_chunk]
            tie_length = tie_section_length(self.tokenizer, ids)
            inputs = ids
            corrupted = False
            if self.tie_corruption > 0 and tie_length > 1:
                # Deterministic per (track, window, chunk) so resumed jobs
                # replay identical batches irrespective of worker scheduling.
                rng = random.Random(
                    zlib.crc32(
                        f"{record.track_id}:{start_chunk}:{chunk_index}".encode()
                    )
                )
                if rng.random() < self.tie_corruption:
                    inputs = corrupt_tie_section(self.tokenizer, ids, rng)
                    corrupted = inputs != ids
            targets.append(ids)
            input_targets.append(inputs)
            tie_lengths.append(tie_length)
            tie_corrupt.append(corrupted)
        group = " ".join(map(str, record.instrument_groups))
        return TrainingExample(
            waveform=waveform,
            target_chunks=targets,
            instrument_group=group or None,
            dataset_id=self.dataset_ids[record.dataset],
            track_id=record.track_id,
            start_time=start_time,
            has_long_gap=self.long_gap_flags[index],
            input_chunks=input_targets,
            tie_lengths=tie_lengths,
            tie_corrupt=tie_corrupt,
        )


def collate_training_examples(
    examples: list[TrainingExample],
) -> TrainingBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    batch = len(examples)
    chunks = examples[0].waveform.shape[0]
    max_length = max(len(ids) for example in examples for ids in example.target_chunks)
    target_ids = torch.zeros(batch, chunks, max_length, dtype=torch.long)
    input_ids = torch.zeros(batch, chunks, max_length, dtype=torch.long)
    lengths = torch.zeros(batch, chunks, dtype=torch.long)
    tie_lengths = torch.zeros(batch, chunks, dtype=torch.long)
    tie_corrupt = torch.zeros(batch, chunks, dtype=torch.bool)
    for batch_index, example in enumerate(examples):
        if example.waveform.shape[0] != chunks:
            raise ValueError("all examples must use the same context length")
        inputs_per_chunk = example.input_chunks or example.target_chunks
        for chunk_index, ids in enumerate(example.target_chunks):
            length = len(ids)
            target_ids[batch_index, chunk_index, :length] = torch.tensor(ids)
            input_ids[batch_index, chunk_index, :length] = torch.tensor(
                inputs_per_chunk[chunk_index]
            )
            lengths[batch_index, chunk_index] = length
        if example.tie_lengths:
            tie_lengths[batch_index] = torch.tensor(example.tie_lengths)
        if example.tie_corrupt:
            tie_corrupt[batch_index] = torch.tensor(example.tie_corrupt)
    return TrainingBatch(
        waveform=torch.stack([example.waveform for example in examples]),
        target_ids=target_ids,
        target_lengths=lengths,
        instrument_groups=[example.instrument_group for example in examples],
        dataset_ids=torch.tensor(
            [example.dataset_id for example in examples], dtype=torch.long
        ),
        track_ids=[example.track_id for example in examples],
        start_times=torch.tensor(
            [example.start_time for example in examples], dtype=torch.float
        ),
        has_long_gap=torch.tensor(
            [example.has_long_gap for example in examples], dtype=torch.bool
        ),
        input_ids=input_ids,
        tie_lengths=tie_lengths,
        tie_corrupt=tie_corrupt,
    )


def _capped_probabilities(
    values: dict[str, float], cap: float = 0.35
) -> dict[str, float]:
    if not values:
        return {}
    if len(values) * cap < 1.0:
        total = sum(values.values())
        return {key: value / total for key, value in values.items()}
    remaining = set(values)
    result: dict[str, float] = {}
    mass = 1.0
    while remaining:
        total = sum(values[key] for key in remaining)
        changed = False
        for key in list(remaining):
            proposed = mass * values[key] / total
            if proposed > cap:
                result[key] = cap
                mass -= cap
                remaining.remove(key)
                changed = True
        if not changed:
            for key in remaining:
                result[key] = mass * values[key] / total
            break
    return result


class BalancedWindowBatchSampler(Sampler[list[int]]):
    """Weighted batches with an explicit multi-instrument half-batch."""

    def __init__(
        self,
        dataset: ContinuousChunkDataset,
        *,
        batch_size: int,
        num_batches: int,
        seed: int = 0,
        rank: int = 0,
        start_batch: int = 0,
        dataset_probabilities: dict[str, float] | None = None,
    ):
        if batch_size <= 0 or num_batches <= 0:
            raise ValueError("batch_size and num_batches must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.rank = rank
        self.start_batch = start_batch
        hours: dict[str, float] = {}
        active_records = {record_index for record_index, _ in dataset.windows}
        for record_index in active_records:
            record = dataset.records[record_index]
            hours[record.dataset] = hours.get(record.dataset, 0.0) + (
                record.duration / 3600
            )
        if dataset_probabilities:
            provided = {
                name: max(0.0, dataset_probabilities.get(name, 0.0)) for name in hours
            }
            total = sum(provided.values())
            if total <= 0:
                raise ValueError("dataset probabilities sum to zero")
            probabilities = _capped_probabilities(provided)
        else:
            probabilities = _capped_probabilities(
                {name: max(value, 1e-6) ** 0.5 for name, value in hours.items()}
            )
        weights = []
        multi_indices = []
        gap_bins = getattr(
            dataset,
            "long_gap_bins",
            ["5-10" if flag else "0-5" for flag in dataset.long_gap_flags],
        )
        gap_counts: dict[tuple[str, str], int] = {}
        for window_index, (record_index, _) in enumerate(dataset.windows):
            record = dataset.records[record_index]
            key = (record.dataset, gap_bins[window_index])
            gap_counts[key] = gap_counts.get(key, 0) + 1
            if record.is_multi_instrument:
                multi_indices.append(window_index)
        target_gap_mass = {
            "none": 0.10,
            "0-5": 0.20,
            "5-10": 0.25,
            "10-20": 0.25,
            "20+": 0.20,
        }
        available_mass: dict[str, float] = {}
        for dataset_name in probabilities:
            available_mass[dataset_name] = sum(
                mass
                for gap_bin, mass in target_gap_mass.items()
                if gap_counts.get((dataset_name, gap_bin), 0)
            )
        for window_index, (record_index, _) in enumerate(dataset.windows):
            record = dataset.records[record_index]
            gap_bin = gap_bins[window_index]
            bin_count = gap_counts[(record.dataset, gap_bin)]
            weight = (
                probabilities[record.dataset]
                * target_gap_mass[gap_bin]
                / available_mass[record.dataset]
                / bin_count
            )
            weights.append(weight)
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.multi_indices = torch.tensor(multi_indices, dtype=torch.long)
        self.multi_weights = (
            self.weights[self.multi_indices]
            if len(multi_indices)
            else torch.empty(0, dtype=torch.double)
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        multi_count = (self.batch_size + 1) // 2 if len(self.multi_indices) else 0
        other_count = self.batch_size - multi_count
        # Seed every batch independently.  A resumed job beginning at
        # ``start_batch`` therefore samples the exact suffix that an
        # uninterrupted run would have produced, irrespective of DataLoader
        # prefetching or worker lifetime.
        for batch_index in range(self.start_batch, self.start_batch + self.num_batches):
            generator = torch.Generator()
            generator.manual_seed(
                self.seed + 1_000_003 * self.rank + 10_000_019 * batch_index
            )
            selected = []
            if multi_count:
                local = torch.multinomial(
                    self.multi_weights,
                    multi_count,
                    replacement=True,
                    generator=generator,
                )
                selected.extend(self.multi_indices[local].tolist())
            if other_count:
                selected.extend(
                    torch.multinomial(
                        self.weights,
                        other_count,
                        replacement=True,
                        generator=generator,
                    ).tolist()
                )
            permutation = torch.randperm(len(selected), generator=generator).tolist()
            yield [selected[index] for index in permutation]
