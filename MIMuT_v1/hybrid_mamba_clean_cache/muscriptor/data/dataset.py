"""Contiguous multi-chunk training dataset."""

from __future__ import annotations

import math
import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler

from muscriptor.data.manifest import ManifestRecord
from muscriptor.data.augmentation import AugmentationCatalogEntry
from muscriptor.data.schema import load_standardized_track
from muscriptor.tokenizer.encode import encode_contiguous_chunks, instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note
from muscriptor.utils.audio import resample


NUM_INSTRUMENT_GROUPS = 37
NUM_MIDI_PITCHES = 128
REENTRY_HORIZONS = (5.0, 10.0, 20.0, 40.0)
REENTRY_NONE_CLASS = len(REENTRY_HORIZONS)


@dataclass
class TrainingExample:
    waveform: torch.Tensor  # [C, 1, samples]
    target_chunks: list[list[int]]
    instrument_group: str | None
    dataset_id: int
    track_id: str
    start_time: float
    has_long_gap: bool
    target_loss_masks: list[list[bool]] | None = None
    original_target_lengths: list[int] | None = None
    truncated_chunks: list[bool] | None = None
    augmentation_applied: bool = False
    remixed: bool = False
    pitch_shift_semitones: int = 0
    augmentation_lineage: list[str] | None = None
    active_note_targets: torch.Tensor | None = None  # [C, 37, 128], bool
    reentry_targets: torch.Tensor | None = None  # [C, 37], long
    reentry_valid: torch.Tensor | None = None  # [C, 37], bool


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
    target_loss_mask: torch.Tensor | None = None
    original_target_lengths: torch.Tensor | None = None
    truncated_chunks: torch.Tensor | None = None
    augmentation_applied: torch.Tensor | None = None
    remixed: torch.Tensor | None = None
    pitch_shift_semitones: torch.Tensor | None = None
    augmentation_lineage: list[list[str]] | None = None
    active_note_targets: torch.Tensor | None = None
    reentry_targets: torch.Tensor | None = None
    reentry_valid: torch.Tensor | None = None

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
            target_loss_mask=(
                self.target_loss_mask.to(device)
                if self.target_loss_mask is not None
                else None
            ),
            original_target_lengths=(
                self.original_target_lengths.to(device)
                if self.original_target_lengths is not None
                else None
            ),
            truncated_chunks=(
                self.truncated_chunks.to(device)
                if self.truncated_chunks is not None
                else None
            ),
            augmentation_applied=(
                self.augmentation_applied.to(device)
                if self.augmentation_applied is not None
                else None
            ),
            remixed=(self.remixed.to(device) if self.remixed is not None else None),
            pitch_shift_semitones=(
                self.pitch_shift_semitones.to(device)
                if self.pitch_shift_semitones is not None
                else None
            ),
            augmentation_lineage=self.augmentation_lineage,
            active_note_targets=(
                self.active_note_targets.to(device)
                if self.active_note_targets is not None
                else None
            ),
            reentry_targets=(
                self.reentry_targets.to(device)
                if self.reentry_targets is not None
                else None
            ),
            reentry_valid=(
                self.reentry_valid.to(device)
                if self.reentry_valid is not None
                else None
            ),
        )


def boundary_state_targets(
    tokenizer: MT3Tokenizer,
    notes: tuple[Note, ...] | list[Note],
    *,
    boundaries: list[float],
    track_duration: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create leakage-free auxiliary labels at chunk boundaries.

    Active-note targets follow the tokenizer's boundary convention: a pitched
    note is active only when ``onset < boundary < offset``.  Re-entry targets
    are defined only for instrument groups already observed before a boundary
    and inactive at it.  A negative (``none``) target is valid only when a full
    40-second future horizon is annotated; shorter track tails are censored.
    """

    reverse = {
        program: group
        for group, programs in tokenizer.group_program_map.items()
        for program in programs
    }

    def group_id(note: Note) -> int | None:
        if note.is_drum:
            return NUM_INSTRUMENT_GROUPS - 1
        return reverse.get(note.program)

    grouped: dict[int, list[Note]] = {}
    for note in notes:
        group = group_id(note)
        if group is not None:
            grouped.setdefault(group, []).append(note)
    for group_notes in grouped.values():
        group_notes.sort(key=lambda note: (note.onset, note.offset, note.pitch))

    active = torch.zeros(
        len(boundaries), NUM_INSTRUMENT_GROUPS, NUM_MIDI_PITCHES, dtype=torch.bool
    )
    reentry = torch.full(
        (len(boundaries), NUM_INSTRUMENT_GROUPS),
        REENTRY_NONE_CLASS,
        dtype=torch.long,
    )
    valid = torch.zeros(
        len(boundaries), NUM_INSTRUMENT_GROUPS, dtype=torch.bool
    )

    for boundary_index, boundary in enumerate(boundaries):
        for group, group_notes in grouped.items():
            for note in group_notes:
                if not note.is_drum and note.onset < boundary < note.offset:
                    active[boundary_index, group, note.pitch] = True

            seen_before = any(note.onset < boundary for note in group_notes)
            group_active = any(
                not note.is_drum and note.onset < boundary < note.offset
                for note in group_notes
            )
            if not seen_before or group_active:
                continue

            future_onsets = [
                note.onset for note in group_notes if boundary <= note.onset
            ]
            if future_onsets:
                gap = min(future_onsets) - boundary
                for class_index, horizon in enumerate(REENTRY_HORIZONS):
                    is_last_horizon = class_index == len(REENTRY_HORIZONS) - 1
                    if gap < horizon or (is_last_horizon and gap <= horizon):
                        reentry[boundary_index, group] = class_index
                        valid[boundary_index, group] = True
                        break
            if not valid[boundary_index, group] and track_duration >= (
                boundary + REENTRY_HORIZONS[-1]
            ):
                reentry[boundary_index, group] = REENTRY_NONE_CLASS
                valid[boundary_index, group] = True

    return active, reentry, valid


@lru_cache(maxsize=256)
def _load_notes(path: str) -> tuple[Note, ...]:
    return tuple(load_standardized_track(path).notes)


def _record_window_gap_bins(
    notes: tuple[Note, ...],
    *,
    tokenizer: MT3Tokenizer,
    num_windows: int,
    context_duration: float,
    segment_duration: float,
) -> list[str]:
    """Assign instrument-group activity-gap strata to contiguous windows."""
    program_to_group = {
        program: group
        for group, programs in tokenizer.group_program_map.items()
        for program in programs
    }
    by_group: dict[int, list[Note]] = {}
    for note in notes:
        if note.is_drum:
            continue
        group = program_to_group.get(note.program)
        if group is not None:
            by_group.setdefault(group, []).append(note)
    maximum_gaps: list[float | None] = [None] * num_windows
    for group_notes in by_group.values():
        ordered = sorted(group_notes, key=lambda note: (note.onset, note.offset))
        if not ordered:
            continue
        active_until = ordered[0].offset
        for current in ordered[1:]:
            if current.onset <= active_until:
                active_until = max(active_until, current.offset)
                continue
            previous_activity_end = active_until
            gap = current.onset - previous_activity_end
            first = max(
                0,
                math.ceil((current.onset - context_duration) / segment_duration - 1e-9),
            )
            last = min(
                num_windows - 1,
                math.floor(previous_activity_end / segment_duration + 1e-9),
            )
            for window_index in range(first, last + 1):
                old = maximum_gaps[window_index]
                maximum_gaps[window_index] = gap if old is None else max(old, gap)
            active_until = current.offset

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
        max_tokens_per_chunk: int = 2000,
        dataset_ids: dict[str, int] | None = None,
        include_boundary_targets: bool = False,
        augmentation_config: Any | None = None,
        augmentation_entries: list[AugmentationCatalogEntry] | None = None,
    ):
        if context_chunks not in {1, 2, 4, 8}:
            raise ValueError("context_chunks must be one of 1, 2, 4, 8")
        self.records = [record for record in records if record.split == split]
        self.context_chunks = context_chunks
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.include_boundary_targets = include_boundary_targets
        self.augmentation_config = augmentation_config
        self.augmentation_entries = list(augmentation_entries or [])
        self.augmentation_enabled = bool(
            augmentation_config is not None
            and getattr(augmentation_config, "enabled", False)
        )
        if self.augmentation_enabled and not self.augmentation_entries:
            raise ValueError("enabled augmentation requires catalog entries")
        self.augmentation_by_audio = {
            str(Path(entry.audio_path).resolve()): entry
            for entry in self.augmentation_entries
        }
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
                tokenizer=self.tokenizer,
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

    @staticmethod
    def _shift_window_notes(
        notes: tuple[Note, ...], *, start_time: float, duration: float
    ) -> list[Note]:
        end_time = start_time + duration
        shifted = []
        for note in notes:
            if note.offset <= start_time or note.onset >= end_time:
                continue
            if note.is_drum and note.onset < start_time:
                continue
            shifted.append(
                Note(
                    is_drum=note.is_drum,
                    program=note.program,
                    onset=note.onset - start_time,
                    offset=note.offset - start_time,
                    pitch=note.pitch,
                )
            )
        return shifted

    def _source_count(self, rng: random.Random) -> int:
        probabilities = self.augmentation_config.source_count_probabilities
        total = sum(probabilities.values())
        draw = rng.random() * total
        cumulative = 0.0
        for count, weight in sorted(probabilities.items()):
            cumulative += weight
            if draw < cumulative:
                return int(count)
        return int(max(probabilities))

    def _select_remix_sources(
        self, rng: random.Random, *, duration: float
    ) -> list[AugmentationCatalogEntry]:
        eligible = [
            entry
            for entry in self.augmentation_entries
            if entry.duration >= duration
        ]
        by_dataset: dict[str, list[AugmentationCatalogEntry]] = {}
        for entry in eligible:
            by_dataset.setdefault(entry.dataset, []).append(entry)
        if len(by_dataset) < 2:
            raise RuntimeError(
                "augmentation catalog cannot satisfy a cross-dataset remix"
            )
        count = self._source_count(rng)
        first_datasets = rng.sample(sorted(by_dataset), 2)
        selected = [rng.choice(by_dataset[name]) for name in first_datasets]
        used_groups = {(entry.dataset, entry.group_id) for entry in selected}
        candidates = [
            entry
            for entry in eligible
            if (entry.dataset, entry.group_id) not in used_groups
        ]
        rng.shuffle(candidates)
        for entry in candidates:
            if len(selected) >= count:
                break
            key = (entry.dataset, entry.group_id)
            if key in used_groups:
                continue
            selected.append(entry)
            used_groups.add(key)
        if len(selected) != count:
            raise RuntimeError(
                f"augmentation catalog cannot provide {count} distinct groups"
            )
        return selected

    def _pitch_shift(
        self,
        waveform: torch.Tensor,
        notes: list[Note],
        rng: random.Random,
    ) -> tuple[torch.Tensor, list[Note], int]:
        pitched = [note for note in notes if not note.is_drum]
        if not pitched:
            return waveform, notes, 0
        feasible = [
            value
            for value in self.augmentation_config.semitone_choices
            if all(0 <= note.pitch + value <= 127 for note in pitched)
        ]
        if not feasible:
            return waveform, notes, 0
        semitones = int(rng.choice(feasible))
        try:
            from torchaudio.functional import pitch_shift
        except ImportError as exc:
            raise ImportError(
                "pitch augmentation requires torchaudio; install the train extra"
            ) from exc
        shifted_waveform = pitch_shift(
            waveform,
            self.sample_rate,
            n_steps=semitones,
        )
        shifted_notes = [
            Note(
                is_drum=note.is_drum,
                program=note.program,
                onset=note.onset,
                offset=note.offset,
                pitch=note.pitch if note.is_drum else note.pitch + semitones,
            )
            for note in notes
        ]
        return shifted_waveform, shifted_notes, semitones

    def _remix_window(
        self,
        rng: random.Random,
        *,
        duration: float,
        pitch_requested: bool,
    ) -> tuple[torch.Tensor, list[Note], list[str], int]:
        sources = self._select_remix_sources(rng, duration=duration)
        waveforms = []
        all_notes = []
        lineage = []
        semitones = 0
        # Choose one feasible shared transposition so pitched stems retain
        # their relative harmony; drum-only stems are never pitch-shifted.
        requested_semitones = None
        if pitch_requested:
            combined_notes = []
            for entry in sources:
                combined_notes.extend(_load_notes(entry.notes_path))
            feasible = [
                value
                for value in self.augmentation_config.semitone_choices
                if all(
                    note.is_drum or 0 <= note.pitch + value <= 127
                    for note in combined_notes
                )
            ]
            if feasible:
                requested_semitones = int(rng.choice(feasible))

        for entry in sources:
            maximum_start = max(0, math.floor((entry.duration - duration) / self.segment_duration))
            source_start = rng.randint(0, maximum_start) * self.segment_duration
            waveform = self._load_audio(entry.audio_path, source_start, duration)
            notes = self._shift_window_notes(
                _load_notes(entry.notes_path),
                start_time=source_start,
                duration=duration,
            )
            if requested_semitones is not None and not entry.is_drum_only:
                try:
                    from torchaudio.functional import pitch_shift
                except ImportError as exc:
                    raise ImportError(
                        "pitch augmentation requires torchaudio; install the train extra"
                    ) from exc
                waveform = pitch_shift(
                    waveform,
                    self.sample_rate,
                    n_steps=requested_semitones,
                )
                notes = [
                    Note(
                        is_drum=note.is_drum,
                        program=note.program,
                        onset=note.onset,
                        offset=note.offset,
                        pitch=(
                            note.pitch
                            if note.is_drum
                            else note.pitch + requested_semitones
                        ),
                    )
                    for note in notes
                ]
                semitones = requested_semitones
            gain_db = rng.uniform(
                self.augmentation_config.gain_db_min,
                self.augmentation_config.gain_db_max,
            )
            waveforms.append(waveform * (10.0 ** (gain_db / 20.0)))
            all_notes.extend(notes)
            lineage.extend(entry.lineage)

        mixed = torch.stack(waveforms).sum(dim=0)
        peak = float(mixed.abs().max())
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)
        all_notes.sort(key=lambda note: (note.onset, note.offset, note.pitch))
        return mixed, all_notes, sorted(set(lineage)), semitones

    def __getitem__(self, index: int | tuple[int, int]) -> TrainingExample:
        augmentation_seed = None
        if isinstance(index, tuple):
            index, augmentation_seed = int(index[0]), int(index[1])
        record_index, start_chunk = self.windows[index]
        record = self.records[record_index]
        start_time = start_chunk * self.segment_duration
        total_duration = self.context_chunks * self.segment_duration
        waveform = self._load_audio(record.audio_path, start_time, total_duration)
        notes: list[Note] | tuple[Note, ...] = _load_notes(record.notes_path)
        track_id = record.track_id
        group_ids = list(record.instrument_groups)
        has_long_gap = self.long_gap_flags[index]
        track_duration = record.duration
        remixed = False
        pitch_semitones = 0
        lineage: list[str] = []

        if self.augmentation_enabled and augmentation_seed is not None:
            rng = random.Random(augmentation_seed)
            remix_requested = (
                rng.random() < self.augmentation_config.remix_probability
            )
            pitch_requested = (
                rng.random() < self.augmentation_config.pitch_shift_probability
            )
            if remix_requested:
                waveform, notes, lineage, pitch_semitones = self._remix_window(
                    rng,
                    duration=total_duration,
                    pitch_requested=pitch_requested,
                )
                remixed = True
                start_time = 0.0
                track_duration = total_duration
                group_ids = instrument_group_ids(self.tokenizer, notes)
                gap_bin = _record_window_gap_bins(
                    tuple(notes),
                    tokenizer=self.tokenizer,
                    num_windows=1,
                    context_duration=total_duration,
                    segment_duration=self.segment_duration,
                )[0]
                has_long_gap = gap_bin in {"5-10", "10-20", "20+"}
                track_id = "aug:" + hashlib.sha1(
                    ("|".join(lineage) + f":{augmentation_seed}").encode("utf-8")
                ).hexdigest()[:16]
            elif pitch_requested:
                source = self.augmentation_by_audio.get(
                    str(Path(record.audio_path).resolve())
                )
                if source is not None and not source.is_drum_only:
                    waveform, shifted_notes, pitch_semitones = self._pitch_shift(
                        waveform, list(notes), rng
                    )
                    notes = shifted_notes
                    lineage = list(source.lineage)

        waveform = waveform.view(
            1,
            self.context_chunks,
            round(self.segment_duration * self.sample_rate),
        ).transpose(0, 1)
        chunks = encode_contiguous_chunks(
            self.tokenizer,
            list(notes),
            start_time=start_time,
            num_chunks=self.context_chunks,
            duration=self.segment_duration,
        )
        targets = []
        target_loss_masks = []
        original_target_lengths = []
        truncated_chunks = []
        for chunk in chunks:
            ids = list(chunk.target_ids)
            original_target_lengths.append(len(ids))
            truncated = len(ids) > self.max_tokens_per_chunk
            if len(ids) > self.max_tokens_per_chunk:
                ids = ids[: self.max_tokens_per_chunk - 1] + [self.tokenizer.eos_id]
            targets.append(ids)
            mask = [True] * len(ids)
            if truncated:
                # The synthetic EOS remains a structural delimiter/input but
                # is not a supervised early-stop target.
                mask[-1] = False
            target_loss_masks.append(mask)
            truncated_chunks.append(truncated)
        group = " ".join(map(str, group_ids))
        active_targets = reentry_targets = reentry_valid = None
        if self.include_boundary_targets:
            boundaries = [
                start_time + (index + 1) * self.segment_duration
                for index in range(self.context_chunks)
            ]
            active_targets, reentry_targets, reentry_valid = boundary_state_targets(
                self.tokenizer,
                notes,
                boundaries=boundaries,
                track_duration=track_duration,
            )
        return TrainingExample(
            waveform=waveform,
            target_chunks=targets,
            instrument_group=group or None,
            dataset_id=self.dataset_ids[record.dataset],
            track_id=track_id,
            start_time=start_time,
            has_long_gap=has_long_gap,
            target_loss_masks=target_loss_masks,
            original_target_lengths=original_target_lengths,
            truncated_chunks=truncated_chunks,
            augmentation_applied=remixed or bool(pitch_semitones),
            remixed=remixed,
            pitch_shift_semitones=pitch_semitones,
            augmentation_lineage=lineage,
            active_note_targets=active_targets,
            reentry_targets=reentry_targets,
            reentry_valid=reentry_valid,
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
    lengths = torch.zeros(batch, chunks, dtype=torch.long)
    loss_mask = torch.zeros(batch, chunks, max_length, dtype=torch.bool)
    original_lengths = torch.zeros(batch, chunks, dtype=torch.long)
    truncated = torch.zeros(batch, chunks, dtype=torch.bool)
    for batch_index, example in enumerate(examples):
        if example.waveform.shape[0] != chunks:
            raise ValueError("all examples must use the same context length")
        for chunk_index, ids in enumerate(example.target_chunks):
            length = len(ids)
            target_ids[batch_index, chunk_index, :length] = torch.tensor(ids)
            lengths[batch_index, chunk_index] = length
            masks = (
                example.target_loss_masks[chunk_index]
                if example.target_loss_masks is not None
                else [True] * length
            )
            if len(masks) != length:
                raise ValueError("target loss mask length does not match target ids")
            loss_mask[batch_index, chunk_index, :length] = torch.tensor(
                masks, dtype=torch.bool
            )
            original_lengths[batch_index, chunk_index] = (
                example.original_target_lengths[chunk_index]
                if example.original_target_lengths is not None
                else length
            )
            truncated[batch_index, chunk_index] = (
                example.truncated_chunks[chunk_index]
                if example.truncated_chunks is not None
                else False
            )
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
        target_loss_mask=loss_mask,
        original_target_lengths=original_lengths,
        truncated_chunks=truncated,
        augmentation_applied=torch.tensor(
            [example.augmentation_applied for example in examples],
            dtype=torch.bool,
        ),
        remixed=torch.tensor(
            [example.remixed for example in examples], dtype=torch.bool
        ),
        pitch_shift_semitones=torch.tensor(
            [example.pitch_shift_semitones for example in examples],
            dtype=torch.long,
        ),
        augmentation_lineage=[
            list(example.augmentation_lineage or []) for example in examples
        ],
        active_note_targets=(
            torch.stack([example.active_note_targets for example in examples])
            if all(example.active_note_targets is not None for example in examples)
            else None
        ),
        reentry_targets=(
            torch.stack([example.reentry_targets for example in examples])
            if all(example.reentry_targets is not None for example in examples)
            else None
        ),
        reentry_valid=(
            torch.stack([example.reentry_valid for example in examples])
            if all(example.reentry_valid is not None for example in examples)
            else None
        ),
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


class BalancedWindowBatchSampler(Sampler[list[int | tuple[int, int]]]):
    """Weighted batches with an explicit multi-instrument half-batch."""

    def __init__(
        self,
        dataset: ContinuousChunkDataset,
        *,
        batch_size: int,
        num_batches: int,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
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
        self.world_size = world_size
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
            ordered = [selected[index] for index in permutation]
            if getattr(self.dataset, "augmentation_enabled", False):
                augmentation_seeds = []
                for slot, window_index in enumerate(ordered):
                    record_index, _ = self.dataset.windows[window_index]
                    record = self.dataset.records[record_index]
                    absolute_position = (
                        (batch_index * self.world_size + self.rank)
                        * self.batch_size
                        + slot
                    )
                    source_group = getattr(record, "group_id", str(record_index))
                    digest = hashlib.sha256(
                        (
                            f"{self.seed}:{absolute_position}:"
                            f"{record.dataset}:{source_group}"
                        ).encode("utf-8")
                    ).digest()
                    augmentation_seeds.append(
                        int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
                    )
                yield list(zip(ordered, augmentation_seeds))
            else:
                yield ordered
