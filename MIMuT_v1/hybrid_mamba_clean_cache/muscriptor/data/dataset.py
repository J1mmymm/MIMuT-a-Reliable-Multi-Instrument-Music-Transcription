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
     ÛŽµ¶‰žËkºwµçE¹‘½´ ¤€ðÍ•±˜¹…Õµ•¹Ñ…Ñ¥½¹}½¹™¥œ¹Á¥Ñ¡}Í¡¥™Ñ}ÁÉ½‰…‰¥±¥Ñä(€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜É•µ¥á}É•ÅÕ•ÍÑ•è(€€€€€€€€€€€€€€€Ý…Ù•™½É´°¹½Ñ•Ì°±¥¹•…”°Á¥Ñ¡}Í•µ¥Ñ½¹•Ì€ôÍ•±˜¹}É•µ¥á}Ý¥¹‘½Ü (€€€€€€€€€€€€€€€€€€€É¹œ°(€€€€€€€€€€€€€€€€€€€‘ÕÉ…Ñ¥½¸õÑ½Ñ…±}‘ÕÉ…Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€Á¥Ñ¡}É•ÅÕ•ÍÑ•õÁ¥Ñ¡}É•ÅÕ•ÍÑ•°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€É•µ¥á•€ôQÉÕ”(€€€€€€€€€€€€€€€ÍÑ…ÉÑ}Ñ¥µ”€ô€À¸À(€€€€€€€€€€€€€€€ÑÉ…­}‘ÕÉ…Ñ¥½¸€ôÑ½Ñ…±}‘ÕÉ…Ñ¥½¸(€€€€€€€€€€€€€€€É½ÕÁ}¥‘Ì€ô¥¹ÍÑÉÕµ•¹Ñ}É½ÕÁ}¥‘Ì¡Í•±˜¹Ñ½­•¹¥é•È°¹½Ñ•Ì¤(€€€€€€€€€€€€€€€…Á}‰¥¸€ô}É•½É‘}Ý¥¹‘½Ý}…Á}‰¥¹Ì (€€€€€€€€€€€€€€€€€€€ÑÕÁ±”¡¹½Ñ•Ì¤°(€€€€€€€€€€€€€€€€€€€Ñ½­•¹¥é•ÈõÍ•±˜¹Ñ½­•¹¥é•È°(€€€€€€€€€€€€€€€€€€€¹Õµ}Ý¥¹‘½ÝÌôÄ°(€€€€€€€€€€€€€€€€€€€½¹Ñ•áÑ}‘ÕÉ…Ñ¥½¸õÑ½Ñ…±}‘ÕÉ…Ñ¥½¸°(€€€€€€€€€€€€€€€€€€€Í•µ•¹Ñ}‘ÕÉ…Ñ¥½¸õÍ•±˜¹Í•µ•¹Ñ}‘ÕÉ…Ñ¥½¸°(€€€€€€€€€€€€€€€€¥lÁt(€€€€€€€€€€€€€€€¡…Í}±½¹}…À€ô…Á}‰¥¸¥¸ìˆÔ´ÄÀˆ°€ˆÄÀ´ÈÀˆ°€ˆÈÀ¬‰ô(€€€€€€€€€€€€€€€ÑÉ…­}¥€ô€‰…Õœèˆ€¬¡…Í¡±¥ˆ¹Í¡„Ä (€€€€€€€€€€€€€€€€€€€€ ‰ðˆ¹©½¥¸¡±¥¹•…”¤€¬˜ˆéí…Õµ•¹Ñ…Ñ¥½¹}Í••‘ôˆ¤¹•¹½‘” ‰ÕÑ˜´àˆ¤(€€€€€€€€€€€€€€€€¤¹¡•á‘¥•ÍÐ ¥lèÄÙt(€€€€€€€€€€€•±¥˜Á¥Ñ¡}É•ÅÕ•ÍÑ•è(€€€€€€€€€€€€€€€Í½ÕÉ”€ôÍ•±˜¹…Õµ•¹Ñ…Ñ¥½¹}‰å}…Õ‘¥¼¹•Ð (€€€€€€€€€€€€€€€€€€€ÍÑÈ¡A…Ñ ¡É•½É¹…Õ‘¥½}Á…Ñ ¤¹É•Í½±Ù” ¤¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€¥˜Í½ÕÉ”¥Ì¹½Ð9½¹”…¹¹½ÐÍ½ÕÉ”¹¥Í}‘ÉÕµ}½¹±äè(€€€€€€€€€€€€€€€€€€€Ý…Ù•™½É´°Í¡¥™Ñ•‘}¹½Ñ•Ì°Á¥Ñ¡}Í•µ¥Ñ½¹•Ì€ôÍ•±˜¹}Á¥Ñ¡}Í¡¥™Ð (€€€€€€€€€€€€€€€€€€€€€€€Ý…Ù•™½É´°±¥ÍÐ¡¹½Ñ•Ì¤°É¹œ(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€¹½Ñ•Ì€ôÍ¡¥™Ñ•‘}¹½Ñ•Ì(€€€€€€€€€€€€€€€€€€€±¥¹•…”€ô±¥ÍÐ¡Í½ÕÉ”¹±¥¹•…”¤((€€€€€€€Ý…Ù•™½É´€ôÝ…Ù•™½É´¹Ù¥•Ü (€€€€€€€€€€€€Ä°(€€€€€€€€€€€Í•±˜¹½¹Ñ•áÑ}¡Õ¹­Ì°(€€€€€€€€€€€É½Õ¹¡Í•±˜¹Í•µ•¹Ñ}‘ÕÉ…Ñ¥½¸€¨Í•±˜¹Í…µÁ±•}É…Ñ”¤°(€€€€€€€€¤¹ÑÉ…¹ÍÁ½Í” À°€Ä¤(€€€€€€€¡Õ¹­Ì€ô•¹½‘•}½¹Ñ¥Õ½ÕÍ}¡Õ¹­Ì (€€€€€€€€€€€Í•±˜¹Ñ½­•¹¥é•È°(€€€€€€€€€€€±¥ÍÐ¡¹½Ñ•Ì¤°(€€€€€€€€€€€ÍÑ…ÉÑ}Ñ¥µ”õÍÑ…ÉÑ}Ñ¥µ”°(€€€€€€€€€€€¹Õµ}¡Õ¹­ÌõÍ•±˜¹½¹Ñ•áÑ}¡Õ¹­Ì°(€€€€€€€€€€€‘ÕÉ…Ñ¥½¸õÍ•±˜¹Í•µ•¹Ñ}‘ÕÉ…Ñ¥½¸°(€€€€€€€€¤(€€€€€€€Ñ…É•ÑÌ€ômt(€€€€€€€Ñ…É•Ñ}±½ÍÍ}µ…Í­Ì€ômt(€€€€€€€½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ì€ômt(€€€€€€€ÑÉÕ¹…Ñ•‘}¡Õ¹­Ì€ômt(€€€€€€€™½È¡Õ¹¬¥¸¡Õ¹­Ìè(€€€€€€€€€€€¥‘Ì€ô±¥ÍÐ¡¡Õ¹¬¹Ñ…É•Ñ}¥‘Ì¤(€€€€€€€€€€€½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ì¹…ÁÁ•¹¡±•¸¡¥‘Ì¤¤(€€€€€€€€€€€ÑÉÕ¹…Ñ•€ô±•¸¡¥‘Ì¤€øÍ•±˜¹µ…á}Ñ½­•¹Í}Á•É}¡Õ¹¬(€€€€€€€€€€€¥˜±•¸¡¥‘Ì¤€øÍ•±˜¹µ…á}Ñ½­•¹Í}Á•É}¡Õ¹¬è(€€€€€€€€€€€€€€€¥‘Ì€ô¥‘ÍlèÍ•±˜¹µ…á}Ñ½­•¹Í}Á•É}¡Õ¹¬€´€Åt€¬mÍ•±˜¹Ñ½­•¹¥é•È¹•½Í}¥‘t(€€€€€€€€€€€Ñ…É•ÑÌ¹…ÁÁ•¹¡¥‘Ì¤(€€€€€€€€€€€µ…Í¬€ômQÉÕ•t€¨±•¸¡¥‘Ì¤(€€€€€€€€€€€¥˜ÑÉÕ¹…Ñ•è(€€€€€€€€€€€€€€€€ŒQ¡”Íå¹Ñ¡•Ñ¥Œ=LÉ•µ…¥¹Ì„ÍÑÉÕÑÕÉ…°‘•±¥µ¥Ñ•È½¥¹ÁÕÐ‰ÕÐ(€€€€€€€€€€€€€€€€Œ¥Ì¹½Ð„ÍÕÁ•ÉÙ¥Í••…É±äµÍÑ½ÀÑ…É•Ð¸(€€€€€€€€€€€€€€€µ…Í­l´Åt€ô…±Í”(€€€€€€€€€€€Ñ…É•Ñ}±½ÍÍ}µ…Í­Ì¹…ÁÁ•¹¡µ…Í¬¤(€€€€€€€€€€€ÑÉÕ¹…Ñ•‘}¡Õ¹­Ì¹…ÁÁ•¹¡ÑÉÕ¹…Ñ•¤(€€€€€€€É½ÕÀ€ô€ˆ€ˆ¹©½¥¸¡µ…À¡ÍÑÈ°É½ÕÁ}¥‘Ì¤¤(€€€€€€€…Ñ¥Ù•}Ñ…É•ÑÌ€ôÉ••¹ÑÉå}Ñ…É•ÑÌ€ôÉ••¹ÑÉå}Ù…±¥€ô9½¹”(€€€€€€€¥˜Í•±˜¹¥¹±Õ‘•}‰½Õ¹‘…Éå}Ñ…É•ÑÌè(€€€€€€€€€€€‰½Õ¹‘…É¥•Ì€ôl(€€€€€€€€€€€€€€€ÍÑ…ÉÑ}Ñ¥µ”€¬€¡¥¹‘•à€¬€Ä¤€¨Í•±˜¹Í•µ•¹Ñ}‘ÕÉ…Ñ¥½¸(€€€€€€€€€€€€€€€™½È¥¹‘•à¥¸É…¹”¡Í•±˜¹½¹Ñ•áÑ}¡Õ¹­Ì¤(€€€€€€€€€€€t(€€€€€€€€€€€…Ñ¥Ù•}Ñ…É•ÑÌ°É••¹ÑÉå}Ñ…É•ÑÌ°É••¹ÑÉå}Ù…±¥€ô‰½Õ¹‘…Éå}ÍÑ…Ñ•}Ñ…É•ÑÌ (€€€€€€€€€€€€€€€Í•±˜¹Ñ½­•¹¥é•È°(€€€€€€€€€€€€€€€¹½Ñ•Ì°(€€€€€€€€€€€€€€€‰½Õ¹‘…É¥•Ìõ‰½Õ¹‘…É¥•Ì°(€€€€€€€€€€€€€€€ÑÉ…­}‘ÕÉ…Ñ¥½¸õÑÉ…­}‘ÕÉ…Ñ¥½¸°(€€€€€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸QÉ…¥¹¥¹á…µÁ±” (€€€€€€€€€€€Ý…Ù•™½É´õÝ…Ù•™½É´°(€€€€€€€€€€€Ñ…É•Ñ}¡Õ¹­ÌõÑ…É•ÑÌ°(€€€€€€€€€€€¥¹ÍÑÉÕµ•¹Ñ}É½ÕÀõÉ½ÕÀ½È9½¹”°(€€€€€€€€€€€‘…Ñ…Í•Ñ}¥õÍ•±˜¹‘…Ñ…Í•Ñ}¥‘ÍmÉ•½É¹‘…Ñ…Í•Ñt°(€€€€€€€€€€€ÑÉ…­}¥õÑÉ…­}¥°(€€€€€€€€€€€ÍÑ…ÉÑ}Ñ¥µ”õÍÑ…ÉÑ}Ñ¥µ”°(€€€€€€€€€€€¡…Í}±½¹}…Àõ¡…Í}±½¹}…À°(€€€€€€€€€€€Ñ…É•Ñ}±½ÍÍ}µ…Í­ÌõÑ…É•Ñ}±½ÍÍ}µ…Í­Ì°(€€€€€€€€€€€½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ìõ½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ì°(€€€€€€€€€€€ÑÉÕ¹…Ñ•‘}¡Õ¹­ÌõÑÉÕ¹…Ñ•‘}¡Õ¹­Ì°(€€€€€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}…ÁÁ±¥•õÉ•µ¥á•½È‰½½°¡Á¥Ñ¡}Í•µ¥Ñ½¹•Ì¤°(€€€€€€€€€€€É•µ¥á•õÉ•µ¥á•°(€€€€€€€€€€€Á¥Ñ¡}Í¡¥™Ñ}Í•µ¥Ñ½¹•ÌõÁ¥Ñ¡}Í•µ¥Ñ½¹•Ì°(€€€€€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}±¥¹•…”õ±¥¹•…”°(€€€€€€€€€€€…Ñ¥Ù•}¹½Ñ•}Ñ…É•ÑÌõ…Ñ¥Ù•}Ñ…É•ÑÌ°(€€€€€€€€€€€É••¹ÑÉå}Ñ…É•ÑÌõÉ••¹ÑÉå}Ñ…É•ÑÌ°(€€€€€€€€€€€É••¹ÑÉå}Ù…±¥õÉ••¹ÑÉå}Ù…±¥°(€€€€€€€€¤(()‘•˜½±±…Ñ•}ÑÉ…¥¹¥¹}•á…µÁ±•Ì (€€€•á…µÁ±•Ìè±¥ÍÑmQÉ…¥¹¥¹á…µÁ±•t°(¤€´øQÉ…¥¹¥¹	…Ñ è(€€€¥˜¹½Ð•á…µÁ±•Ìè(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰…¹¹½Ð½±±…Ñ”…¸•µÁÑä‰…Ñ ˆ¤(€€€‰…Ñ €ô±•¸¡•á…µÁ±•Ì¤(€€€¡Õ¹­Ì€ô•á…µÁ±•ÍlÁt¹Ý…Ù•™½É´¹Í¡…Á•lÁt(€€€µ…á}±•¹Ñ €ôµ…à¡±•¸¡¥‘Ì¤™½È•á…µÁ±”¥¸•á…µÁ±•Ì™½È¥‘Ì¥¸•á…µÁ±”¹Ñ…É•Ñ}¡Õ¹­Ì¤(€€€Ñ…É•Ñ}¥‘Ì€ôÑ½É ¹é•É½Ì¡‰…Ñ °¡Õ¹­Ì°µ…á}±•¹Ñ °‘ÑåÁ”õÑ½É ¹±½¹œ¤(€€€±•¹Ñ¡Ì€ôÑ½É ¹é•É½Ì¡‰…Ñ °¡Õ¹­Ì°‘ÑåÁ”õÑ½É ¹±½¹œ¤(€€€±½ÍÍ}µ…Í¬€ôÑ½É ¹é•É½Ì¡‰…Ñ °¡Õ¹­Ì°µ…á}±•¹Ñ °‘ÑåÁ”õÑ½É ¹‰½½°¤(€€€½É¥¥¹…±}±•¹Ñ¡Ì€ôÑ½É ¹é•É½Ì¡‰…Ñ °¡Õ¹­Ì°‘ÑåÁ”õÑ½É ¹±½¹œ¤(€€€ÑÉÕ¹…Ñ•€ôÑ½É ¹é•É½Ì¡‰…Ñ °¡Õ¹­Ì°‘ÑåÁ”õÑ½É ¹‰½½°¤(€€€™½È‰…Ñ¡}¥¹‘•à°•á…µÁ±”¥¸•¹Õµ•É…Ñ”¡•á…µÁ±•Ì¤è(€€€€€€€¥˜•á…µÁ±”¹Ý…Ù•™½É´¹Í¡…Á•lÁt€„ô¡Õ¹­Ìè(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰…±°•á…µÁ±•ÌµÕÍÐÕÍ”Ñ¡”Í…µ”½¹Ñ•áÐ±•¹Ñ ˆ¤(€€€€€€€™½È¡Õ¹­}¥¹‘•à°¥‘Ì¥¸•¹Õµ•É…Ñ”¡•á…µÁ±”¹Ñ…É•Ñ}¡Õ¹­Ì¤è(€€€€€€€€€€€±•¹Ñ €ô±•¸¡¥‘Ì¤(€€€€€€€€€€€Ñ…É•Ñ}¥‘Ím‰…Ñ¡}¥¹‘•à°¡Õ¹­}¥¹‘•à°€é±•¹Ñ¡t€ôÑ½É ¹Ñ•¹Í½È¡¥‘Ì¤(€€€€€€€€€€€±•¹Ñ¡Ím‰…Ñ¡}¥¹‘•à°¡Õ¹­}¥¹‘•át€ô±•¹Ñ (€€€€€€€€€€€µ…Í­Ì€ô€ (€€€€€€€€€€€€€€€•á…µÁ±”¹Ñ…É•Ñ}±½ÍÍ}µ…Í­Ím¡Õ¹­}¥¹‘•át(€€€€€€€€€€€€€€€¥˜•á…µÁ±”¹Ñ…É•Ñ}±½ÍÍ}µ…Í­Ì¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€•±Í”mQÉÕ•t€¨±•¹Ñ (€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜±•¸¡µ…Í­Ì¤€„ô±•¹Ñ è(€€€€€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰Ñ…É•Ð±½ÍÌµ…Í¬±•¹Ñ ‘½•Ì¹½Ðµ…Ñ Ñ…É•Ð¥‘Ìˆ¤(€€€€€€€€€€€±½ÍÍ}µ…Í­m‰…Ñ¡}¥¹‘•à°¡Õ¹­}¥¹‘•à°€é±•¹Ñ¡t€ôÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€€€€€µ…Í­Ì°‘ÑåÁ”õÑ½É ¹‰½½°(€€€€€€€€€€€€¤(€€€€€€€€€€€½É¥¥¹…±}±•¹Ñ¡Ím‰…Ñ¡}¥¹‘•à°¡Õ¹­}¥¹‘•át€ô€ (€€€€€€€€€€€€€€€•á…µÁ±”¹½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ím¡Õ¹­}¥¹‘•át(€€€€€€€€€€€€€€€¥˜•á…µÁ±”¹½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ì¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€•±Í”±•¹Ñ (€€€€€€€€€€€€¤(€€€€€€€€€€€ÑÉÕ¹…Ñ•‘m‰…Ñ¡}¥¹‘•à°¡Õ¹­}¥¹‘•át€ô€ (€€€€€€€€€€€€€€€•á…µÁ±”¹ÑÉÕ¹…Ñ•‘}¡Õ¹­Ím¡Õ¹­}¥¹‘•át(€€€€€€€€€€€€€€€¥˜•á…µÁ±”¹ÑÉÕ¹…Ñ•‘}¡Õ¹­Ì¥Ì¹½Ð9½¹”(€€€€€€€€€€€€€€€•±Í”…±Í”(€€€€€€€€€€€€¤(€€€É•ÑÕÉ¸QÉ…¥¹¥¹	…Ñ  (€€€€€€€Ý…Ù•™½É´õÑ½É ¹ÍÑ…¬¡m•á…µÁ±”¹Ý…Ù•™½É´™½È•á…µÁ±”¥¸•á…µÁ±•Ít¤°(€€€€€€€Ñ…É•Ñ}¥‘ÌõÑ…É•Ñ}¥‘Ì°(€€€€€€€Ñ…É•Ñ}±•¹Ñ¡Ìõ±•¹Ñ¡Ì°(€€€€€€€¥¹ÍÑÉÕµ•¹Ñ}É½ÕÁÌõm•á…µÁ±”¹¥¹ÍÑÉÕµ•¹Ñ}É½ÕÀ™½È•á…µÁ±”¥¸•á…µÁ±•Ít°(€€€€€€€‘…Ñ…Í•Ñ}¥‘ÌõÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹‘…Ñ…Í•Ñ}¥™½È•á…µÁ±”¥¸•á…µÁ±•Ít°‘ÑåÁ”õÑ½É ¹±½¹œ(€€€€€€€€¤°(€€€€€€€ÑÉ…­}¥‘Ìõm•á…µÁ±”¹ÑÉ…­}¥™½È•á…µÁ±”¥¸•á…µÁ±•Ít°(€€€€€€€ÍÑ…ÉÑ}Ñ¥µ•ÌõÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹ÍÑ…ÉÑ}Ñ¥µ”™½È•á…µÁ±”¥¸•á…µÁ±•Ít°‘ÑåÁ”õÑ½É ¹™±½…Ð(€€€€€€€€¤°(€€€€€€€¡…Í}±½¹}…ÀõÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹¡…Í}±½¹}…À™½È•á…µÁ±”¥¸•á…µÁ±•Ít°‘ÑåÁ”õÑ½É ¹‰½½°(€€€€€€€€¤°(€€€€€€€Ñ…É•Ñ}±½ÍÍ}µ…Í¬õ±½ÍÍ}µ…Í¬°(€€€€€€€½É¥¥¹…±}Ñ…É•Ñ}±•¹Ñ¡Ìõ½É¥¥¹…±}±•¹Ñ¡Ì°(€€€€€€€ÑÉÕ¹…Ñ•‘}¡Õ¹­ÌõÑÉÕ¹…Ñ•°(€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}…ÁÁ±¥•õÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹…Õµ•¹Ñ…Ñ¥½¹}…ÁÁ±¥•™½È•á…µÁ±”¥¸•á…µÁ±•Ít°(€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹‰½½°°(€€€€€€€€¤°(€€€€€€€É•µ¥á•õÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹É•µ¥á•™½È•á…µÁ±”¥¸•á…µÁ±•Ít°‘ÑåÁ”õÑ½É ¹‰½½°(€€€€€€€€¤°(€€€€€€€Á¥Ñ¡}Í¡¥™Ñ}Í•µ¥Ñ½¹•ÌõÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€m•á…µÁ±”¹Á¥Ñ¡}Í¡¥™Ñ}Í•µ¥Ñ½¹•Ì™½È•á…µÁ±”¥¸•á…µÁ±•Ít°(€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹±½¹œ°(€€€€€€€€¤°(€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}±¥¹•…”õl(€€€€€€€€€€€±¥ÍÐ¡•á…µÁ±”¹…Õµ•¹Ñ…Ñ¥½¹}±¥¹•…”½Èmt¤™½È•á…µÁ±”¥¸•á…µÁ±•Ì(€€€€€€€t°(€€€€€€€…Ñ¥Ù•}¹½Ñ•}Ñ…É•ÑÌô (€€€€€€€€€€€Ñ½É ¹ÍÑ…¬¡m•á…µÁ±”¹…Ñ¥Ù•}¹½Ñ•}Ñ…É•ÑÌ™½È•á…µÁ±”¥¸•á…µÁ±•Ít¤(€€€€€€€€€€€¥˜…±°¡•á…µÁ±”¹…Ñ¥Ù•}¹½Ñ•}Ñ…É•ÑÌ¥Ì¹½Ð9½¹”™½È•á…µÁ±”¥¸•á…µÁ±•Ì¤(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€É••¹ÑÉå}Ñ…É•ÑÌô (€€€€€€€€€€€Ñ½É ¹ÍÑ…¬¡m•á…µÁ±”¹É••¹ÑÉå}Ñ…É•ÑÌ™½È•á…µÁ±”¥¸•á…µÁ±•Ít¤(€€€€€€€€€€€¥˜…±°¡•á…µÁ±”¹É••¹ÑÉå}Ñ…É•ÑÌ¥Ì¹½Ð9½¹”™½È•á…µÁ±”¥¸•á…µÁ±•Ì¤(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€€€€É••¹ÑÉå}Ù…±¥ô (€€€€€€€€€€€Ñ½É ¹ÍÑ…¬¡m•á…µÁ±”¹É••¹ÑÉå}Ù…±¥™½È•á…µÁ±”¥¸•á…µÁ±•Ít¤(€€€€€€€€€€€¥˜…±°¡•á…µÁ±”¹É••¹ÑÉå}Ù…±¥¥Ì¹½Ð9½¹”™½È•á…µÁ±”¥¸•á…µÁ±•Ì¤(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤°(€€€€¤(()‘•˜}…ÁÁ•‘}ÁÉ½‰…‰¥±¥Ñ¥•Ì (€€€Ù…±Õ•Ìè‘¥ÑmÍÑÈ°™±½…Ñt°…Àè™±½…Ð€ô€À¸ÌÔ(¤€´ø‘¥ÑmÍÑÈ°™±½…Ñtè(€€€¥˜¹½ÐÙ…±Õ•Ìè(€€€€€€€É•ÑÕÉ¸íô(€€€¥˜±•¸¡Ù…±Õ•Ì¤€¨…À€ð€Ä¸Àè(€€€€€€€Ñ½Ñ…°€ôÍÕ´¡Ù…±Õ•Ì¹Ù…±Õ•Ì ¤¤(€€€€€€€É•ÑÕÉ¸í­•äèÙ…±Õ”€¼Ñ½Ñ…°™½È­•ä°Ù…±Õ”¥¸Ù…±Õ•Ì¹¥Ñ•µÌ ¥ô(€€€É•µ…¥¹¥¹œ€ôÍ•Ð¡Ù…±Õ•Ì¤(€€€É•ÍÕ±Ðè‘¥ÑmÍÑÈ°™±½…Ñt€ôíô(€€€µ…ÍÌ€ô€Ä¸À(€€€Ý¡¥±”É•µ…¥¹¥¹œè(€€€€€€€Ñ½Ñ…°€ôÍÕ´¡Ù…±Õ•Ím­•åt™½È­•ä¥¸É•µ…¥¹¥¹œ¤(€€€€€€€¡…¹•€ô…±Í”(€€€€€€€™½È­•ä¥¸±¥ÍÐ¡É•µ…¥¹¥¹œ¤è(€€€€€€€€€€€ÁÉ½Á½Í•€ôµ…ÍÌ€¨Ù…±Õ•Ím­•åt€¼Ñ½Ñ…°(€€€€€€€€€€€¥˜ÁÉ½Á½Í•€ø…Àè(€€€€€€€€€€€€€€€É•ÍÕ±Ñm­•åt€ô…À(€€€€€€€€€€€€€€€µ…ÍÌ€´ô…À(€€€€€€€€€€€€€€€É•µ…¥¹¥¹œ¹É•µ½Ù”¡­•ä¤(€€€€€€€€€€€€€€€¡…¹•€ôQÉÕ”(€€€€€€€¥˜¹½Ð¡…¹•è(€€€€€€€€€€€™½È­•ä¥¸É•µ…¥¹¥¹œè(€€€€€€€€€€€€€€€É•ÍÕ±Ñm­•åt€ôµ…ÍÌ€¨Ù…±Õ•Ím­•åt€¼Ñ½Ñ…°(€€€€€€€€€€€‰É•…¬(€€€É•ÑÕÉ¸É•ÍÕ±Ð(()±…ÍÌ	…±…¹•‘]¥¹‘½Ý	…Ñ¡M…µÁ±•È¡M…µÁ±•Ém±¥ÍÑm¥¹ÐðÑÕÁ±•m¥¹Ð°¥¹Ñuut¤è(€€€€ˆˆ‰]•¥¡Ñ•‰…Ñ¡•ÌÝ¥Ñ …¸•áÁ±¥¥ÐµÕ±Ñ¤µ¥¹ÍÑÉÕµ•¹Ð¡…±˜µ‰…Ñ ¸ˆˆˆ((€€€‘•˜}}¥¹¥Ñ}| (€€€€€€€Í•±˜°(€€€€€€€‘…Ñ…Í•Ðè½¹Ñ¥¹Õ½ÕÍ¡Õ¹­…Ñ…Í•Ð°(€€€€€€€€¨°(€€€€€€€‰…Ñ¡}Í¥é”è¥¹Ð°(€€€€€€€¹Õµ}‰…Ñ¡•Ìè¥¹Ð°(€€€€€€€Í••è¥¹Ð€ô€À°(€€€€€€€É…¹¬è¥¹Ð€ô€À°(€€€€€€€Ý½É±‘}Í¥é”è¥¹Ð€ô€Ä°(€€€€€€€ÍÑ…ÉÑ}‰…Ñ è¥¹Ð€ô€À°(€€€€€€€‘…Ñ…Í•Ñ}ÁÉ½‰…‰¥±¥Ñ¥•Ìè‘¥ÑmÍÑÈ°™±½…Ñtð9½¹”€ô9½¹”°(€€€€¤è(€€€€€€€¥˜‰…Ñ¡}Í¥é”€ðô€À½È¹Õµ}‰…Ñ¡•Ì€ðô€Àè(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰‰…Ñ¡}Í¥é”…¹¹Õµ}‰…Ñ¡•ÌµÕÍÐ‰”Á½Í¥Ñ¥Ù”ˆ¤(€€€€€€€Í•±˜¹‘…Ñ…Í•Ð€ô‘…Ñ…Í•Ð(€€€€€€€Í•±˜¹‰…Ñ¡}Í¥é”€ô‰…Ñ¡}Í¥é”(€€€€€€€Í•±˜¹¹Õµ}‰…Ñ¡•Ì€ô¹Õµ}‰…Ñ¡•Ì(€€€€€€€Í•±˜¹Í••€ôÍ••(€€€€€€€Í•±˜¹É…¹¬€ôÉ…¹¬(€€€€€€€Í•±˜¹Ý½É±‘}Í¥é”€ôÝ½É±‘}Í¥é”(€€€€€€€Í•±˜¹ÍÑ…ÉÑ}‰…Ñ €ôÍÑ…ÉÑ}‰…Ñ (€€€€€€€¡½ÕÉÌè‘¥ÑmÍÑÈ°™±½…Ñt€ôíô(€€€€€€€…Ñ¥Ù•}É•½É‘Ì€ôíÉ•½É‘}¥¹‘•à™½ÈÉ•½É‘}¥¹‘•à°|¥¸‘…Ñ…Í•Ð¹Ý¥¹‘½ÝÍô(€€€€€€€™½ÈÉ•½É‘}¥¹‘•à¥¸…Ñ¥Ù•}É•½É‘Ìè(€€€€€€€€€€€É•½É€ô‘…Ñ…Í•Ð¹É•½É‘ÍmÉ•½É‘}¥¹‘•át(€€€€€€€€€€€¡½ÕÉÍmÉ•½É¹‘…Ñ…Í•Ñt€ô¡½ÕÉÌ¹•Ð¡É•½É¹‘…Ñ…Í•Ð°€À¸À¤€¬€ (€€€€€€€€€€€€€€€É•½É¹‘ÕÉ…Ñ¥½¸€¼€ÌØÀÀ(€€€€€€€€€€€€¤(€€€€€€€¥˜‘…Ñ…Í•Ñ}ÁÉ½‰…‰¥±¥Ñ¥•Ìè(€€€€€€€€€€€ÁÉ½Ù¥‘•€ôì(€€€€€€€€€€€€€€€¹…µ”èµ…à À¸À°‘…Ñ…Í•Ñ}ÁÉ½‰…‰¥±¥Ñ¥•Ì¹•Ð¡¹…µ”°€À¸À¤¤™½È¹…µ”¥¸¡½ÕÉÌ(€€€€€€€€€€€ô(€€€€€€€€€€€Ñ½Ñ…°€ôÍÕ´¡ÁÉ½Ù¥‘•¹Ù…±Õ•Ì ¤¤(€€€€€€€€€€€¥˜Ñ½Ñ…°€ðô€Àè(€€€€€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰‘…Ñ…Í•ÐÁÉ½‰…‰¥±¥Ñ¥•ÌÍÕ´Ñ¼é•É¼ˆ¤(€€€€€€€€€€€ÁÉ½‰…‰¥±¥Ñ¥•Ì€ô}…ÁÁ•‘}ÁÉ½‰…‰¥±¥Ñ¥•Ì¡ÁÉ½Ù¥‘•¤(€€€€€€€•±Í”è(€€€€€€€€€€€ÁÉ½‰…‰¥±¥Ñ¥•Ì€ô}…ÁÁ•‘}ÁÉ½‰…‰¥±¥Ñ¥•Ì (€€€€€€€€€€€€€€€í¹…µ”èµ…à¡Ù…±Õ”°€Å”´Ø¤€¨¨€À¸Ô™½È¹…µ”°Ù…±Õ”¥¸¡½ÕÉÌ¹¥Ñ•µÌ ¥ô(€€€€€€€€€€€€¤(€€€€€€€Ý•¥¡ÑÌ€ômt(€€€€€€€µÕ±Ñ¥}¥¹‘¥•Ì€ômt(€€€€€€€…Á}‰¥¹Ì€ô•Ñ…ÑÑÈ (€€€€€€€€€€€‘…Ñ…Í•Ð°(€€€€€€€€€€€€‰±½¹}…Á}‰¥¹Ìˆ°(€€€€€€€€€€€lˆÔ´ÄÀˆ¥˜™±…œ•±Í”€ˆÀ´Ôˆ™½È™±…œ¥¸‘…Ñ…Í•Ð¹±½¹}…Á}™±…Ít°(€€€€€€€€¤(€€€€€€€…Á}½Õ¹ÑÌè‘¥ÑmÑÕÁ±•mÍÑÈ°ÍÑÉt°¥¹Ñt€ôíô(€€€€€€€™½ÈÝ¥¹‘½Ý}¥¹‘•à°€¡É•½É‘}¥¹‘•à°|¤¥¸•¹Õµ•É…Ñ”¡‘…Ñ…Í•Ð¹Ý¥¹‘½ÝÌ¤è(€€€€€€€€€€€É•½É€ô‘…Ñ…Í•Ð¹É•½É‘ÍmÉ•½É‘}¥¹‘•át(€€€€€€€€€€€­•ä€ô€¡É•½É¹‘…Ñ…Í•Ð°…Á}‰¥¹ÍmÝ¥¹‘½Ý}¥¹‘•át¤(€€€€€€€€€€€…Á}½Õ¹ÑÍm­•åt€ô…Á}½Õ¹ÑÌ¹•Ð¡­•ä°€À¤€¬€Ä(€€€€€€€€€€€¥˜É•½É¹¥Í}µÕ±Ñ¥}¥¹ÍÑÉÕµ•¹Ðè(€€€€€€€€€€€€€€€µÕ±Ñ¥}¥¹‘¥•Ì¹…ÁÁ•¹¡Ý¥¹‘½Ý}¥¹‘•à¤(€€€€€€€Ñ…É•Ñ}…Á}µ…ÍÌ€ôì(€€€€€€€€€€€€‰¹½¹”ˆè€À¸ÄÀ°(€€€€€€€€€€€€ˆÀ´Ôˆè€À¸ÈÀ°(€€€€€€€€€€€€ˆÔ´ÄÀˆè€À¸ÈÔ°(€€€€€€€€€€€€ˆÄÀ´ÈÀˆè€À¸ÈÔ°(€€€€€€€€€€€€ˆÈÀ¬ˆè€À¸ÈÀ°(€€€€€€€ô(€€€€€€€…Ù…¥±…‰±•}µ…ÍÌè‘¥ÑmÍÑÈ°™±½…Ñt€ôíô(€€€€€€€™½È‘…Ñ…Í•Ñ}¹…µ”¥¸ÁÉ½‰…‰¥±¥Ñ¥•Ìè(€€€€€€€€€€€…Ù…¥±…‰±•}µ…ÍÍm‘…Ñ…Í•Ñ}¹…µ•t€ôÍÕ´ (€€€€€€€€€€€€€€€µ…ÍÌ(€€€€€€€€€€€€€€€™½È…Á}‰¥¸°µ…ÍÌ¥¸Ñ…É•Ñ}…Á}µ…ÍÌ¹¥Ñ•µÌ ¤(€€€€€€€€€€€€€€€¥˜…Á}½Õ¹ÑÌ¹•Ð ¡‘…Ñ…Í•Ñ}¹…µ”°…Á}‰¥¸¤°€À¤(€€€€€€€€€€€€¤(€€€€€€€™½ÈÝ¥¹‘½Ý}¥¹‘•à°€¡É•½É‘}¥¹‘•à°|¤¥¸•¹Õµ•É…Ñ”¡‘…Ñ…Í•Ð¹Ý¥¹‘½ÝÌ¤è(€€€€€€€€€€€É•½É€ô‘…Ñ…Í•Ð¹É•½É‘ÍmÉ•½É‘}¥¹‘•át(€€€€€€€€€€€…Á}‰¥¸€ô…Á}‰¥¹ÍmÝ¥¹‘½Ý}¥¹‘•át(€€€€€€€€€€€‰¥¹}½Õ¹Ð€ô…Á}½Õ¹ÑÍl¡É•½É¹‘…Ñ…Í•Ð°…Á}‰¥¸¥t(€€€€€€€€€€€Ý•¥¡Ð€ô€ (€€€€€€€€€€€€€€€ÁÉ½‰…‰¥±¥Ñ¥•ÍmÉ•½É¹‘…Ñ…Í•Ñt(€€€€€€€€€€€€€€€€¨Ñ…É•Ñ}…Á}µ…ÍÍm…Á}‰¥¹t(€€€€€€€€€€€€€€€€¼…Ù…¥±…‰±•}µ…ÍÍmÉ•½É¹‘…Ñ…Í•Ñt(€€€€€€€€€€€€€€€€¼‰¥¹}½Õ¹Ð(€€€€€€€€€€€€¤(€€€€€€€€€€€Ý•¥¡ÑÌ¹…ÁÁ•¹¡Ý•¥¡Ð¤(€€€€€€€Í•±˜¹Ý•¥¡ÑÌ€ôÑ½É ¹Ñ•¹Í½È¡Ý•¥¡ÑÌ°‘ÑåÁ”õÑ½É ¹‘½Õ‰±”¤(€€€€€€€Í•±˜¹µÕ±Ñ¥}¥¹‘¥•Ì€ôÑ½É ¹Ñ•¹Í½È¡µÕ±Ñ¥}¥¹‘¥•Ì°‘ÑåÁ”õÑ½É ¹±½¹œ¤(€€€€€€€Í•±˜¹µÕ±Ñ¥}Ý•¥¡ÑÌ€ô€ (€€€€€€€€€€€Í•±˜¹Ý•¥¡ÑÍmÍ•±˜¹µÕ±Ñ¥}¥¹‘¥•Ít(€€€€€€€€€€€¥˜±•¸¡µÕ±Ñ¥}¥¹‘¥•Ì¤(€€€€€€€€€€€•±Í”Ñ½É ¹•µÁÑä À°‘ÑåÁ”õÑ½É ¹‘½Õ‰±”¤(€€€€€€€€¤((€€€‘•˜}}±•¹}|¡Í•±˜¤€´ø¥¹Ðè(€€€€€€€É•ÑÕÉ¸Í•±˜¹¹Õµ}‰…Ñ¡•Ì((€€€‘•˜}}¥Ñ•É}|¡Í•±˜¤è(€€€€€€€µÕ±Ñ¥}½Õ¹Ð€ô€¡Í•±˜¹‰…Ñ¡}Í¥é”€¬€Ä¤€¼¼€È¥˜±•¸¡Í•±˜¹µÕ±Ñ¥}¥¹‘¥•Ì¤•±Í”€À(€€€€€€€½Ñ¡•É}½Õ¹Ð€ôÍ•±˜¹‰…Ñ¡}Í¥é”€´µÕ±Ñ¥}½Õ¹Ð(€€€€€€€€ŒM•••Ù•Éä‰…Ñ ¥¹‘•Á•¹‘•¹Ñ±ä¸€É•ÍÕµ•©½ˆ‰•¥¹¹¥¹œ…Ð(€€€€€€€€ŒÍÑ…ÉÑ}‰…Ñ¡€Ñ¡•É•™½É”Í…µÁ±•ÌÑ¡”•á…ÐÍÕ™™¥àÑ¡…Ð…¸(€€€€€€€€ŒÕ¹¥¹Ñ•ÉÉÕÁÑ•ÉÕ¸Ý½Õ±¡…Ù”ÁÉ½‘Õ•°¥ÉÉ•ÍÁ•Ñ¥Ù”½˜…Ñ…1½…‘•È(€€€€€€€€ŒÁÉ•™•Ñ¡¥¹œ½ÈÝ½É­•È±¥™•Ñ¥µ”¸(€€€€€€€™½È‰…Ñ¡}¥¹‘•à¥¸É…¹”¡Í•±˜¹ÍÑ…ÉÑ}‰…Ñ °Í•±˜¹ÍÑ…ÉÑ}‰…Ñ €¬Í•±˜¹¹Õµ}‰…Ñ¡•Ì¤è(€€€€€€€€€€€•¹•É…Ñ½È€ôÑ½É ¹•¹•É…Ñ½È ¤(€€€€€€€€€€€•¹•É…Ñ½È¹µ…¹Õ…±}Í•• (€€€€€€€€€€€€€€€Í•±˜¹Í••€¬€Å|ÀÀÁ|ÀÀÌ€¨Í•±˜¹É…¹¬€¬€ÄÁ|ÀÀÁ|ÀÄä€¨‰…Ñ¡}¥¹‘•à(€€€€€€€€€€€€¤(€€€€€€€€€€€Í•±•Ñ•€ômt(€€€€€€€€€€€¥˜µÕ±Ñ¥}½Õ¹Ðè(€€€€€€€€€€€€€€€±½…°€ôÑ½É ¹µÕ±Ñ¥¹½µ¥…° (€€€€€€€€€€€€€€€€€€€Í•±˜¹µÕ±Ñ¥}Ý•¥¡ÑÌ°(€€€€€€€€€€€€€€€€€€€µÕ±Ñ¥}½Õ¹Ð°(€€€€€€€€€€€€€€€€€€€É•Á±…•µ•¹ÐõQÉÕ”°(€€€€€€€€€€€€€€€€€€€•¹•É…Ñ½Èõ•¹•É…Ñ½È°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€Í•±•Ñ•¹•áÑ•¹¡Í•±˜¹µÕ±Ñ¥}¥¹‘¥•Ím±½…±t¹Ñ½±¥ÍÐ ¤¤(€€€€€€€€€€€¥˜½Ñ¡•É}½Õ¹Ðè(€€€€€€€€€€€€€€€Í•±•Ñ•¹•áÑ•¹ (€€€€€€€€€€€€€€€€€€€Ñ½É ¹µÕ±Ñ¥¹½µ¥…° (€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹Ý•¥¡ÑÌ°(€€€€€€€€€€€€€€€€€€€€€€€½Ñ¡•É}½Õ¹Ð°(€€€€€€€€€€€€€€€€€€€€€€€É•Á±…•µ•¹ÐõQÉÕ”°(€€€€€€€€€€€€€€€€€€€€€€€•¹•É…Ñ½Èõ•¹•É…Ñ½È°(€€€€€€€€€€€€€€€€€€€€¤¹Ñ½±¥ÍÐ ¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€Á•ÉµÕÑ…Ñ¥½¸€ôÑ½É ¹É…¹‘Á•É´¡±•¸¡Í•±•Ñ•¤°•¹•É…Ñ½Èõ•¹•É…Ñ½È¤¹Ñ½±¥ÍÐ ¤(€€€€€€€€€€€½É‘•É•€ômÍ•±•Ñ•‘m¥¹‘•át™½È¥¹‘•à¥¸Á•ÉµÕÑ…Ñ¥½¹t(€€€€€€€€€€€¥˜•Ñ…ÑÑÈ¡Í•±˜¹‘…Ñ…Í•Ð°€‰…Õµ•¹Ñ…Ñ¥½¹}•¹…‰±•ˆ°…±Í”¤è(€€€€€€€€€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}Í••‘Ì€ômt(€€€€€€€€€€€€€€€™½ÈÍ±½Ð°Ý¥¹‘½Ý}¥¹‘•à¥¸•¹Õµ•É…Ñ”¡½É‘•É•¤è(€€€€€€€€€€€€€€€€€€€É•½É‘}¥¹‘•à°|€ôÍ•±˜¹‘…Ñ…Í•Ð¹Ý¥¹‘½ÝÍmÝ¥¹‘½Ý}¥¹‘•át(€€€€€€€€€€€€€€€€€€€É•½É€ôÍ•±˜¹‘…Ñ…Í•Ð¹É•½É‘ÍmÉ•½É‘}¥¹‘•át(€€€€€€€€€€€€€€€€€€€…‰Í½±ÕÑ•}Á½Í¥Ñ¥½¸€ô€ (€€€€€€€€€€€€€€€€€€€€€€€€¡‰…Ñ¡}¥¹‘•à€¨Í•±˜¹Ý½É±‘}Í¥é”€¬Í•±˜¹É…¹¬¤(€€€€€€€€€€€€€€€€€€€€€€€€¨Í•±˜¹‰…Ñ¡}Í¥é”(€€€€€€€€€€€€€€€€€€€€€€€€¬Í±½Ð(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€Í½ÕÉ•}É½ÕÀ€ô•Ñ…ÑÑÈ¡É•½É°€‰É½ÕÁ}¥ˆ°ÍÑÈ¡É•½É‘}¥¹‘•à¤¤(€€€€€€€€€€€€€€€€€€€‘¥•ÍÐ€ô¡…Í¡±¥ˆ¹Í¡„ÈÔØ (€€€€€€€€€€€€€€€€€€€€€€€€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰íÍ•±˜¹Í••‘ôéí…‰Í½±ÕÑ•}Á½Í¥Ñ¥½¹ôèˆ(€€€€€€€€€€€€€€€€€€€€€€€€€€€˜‰íÉ•½É¹‘…Ñ…Í•ÑôéíÍ½ÕÉ•}É½ÕÁôˆ(€€€€€€€€€€€€€€€€€€€€€€€€¤¹•¹½‘” ‰ÕÑ˜´àˆ¤(€€€€€€€€€€€€€€€€€€€€¤¹‘¥•ÍÐ ¤(€€€€€€€€€€€€€€€€€€€…Õµ•¹Ñ…Ñ¥½¹}Í••‘Ì¹…ÁÁ•¹ (€€€€€€€€€€€€€€€€€€€€€€€¥¹Ð¹™É½µ}‰åÑ•Ì¡‘¥•ÍÑlèát°€‰‰¥œˆ¤€˜€  Ä€ðð€ØÌ¤€´€Ä¤(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€å¥•±±¥ÍÐ¡é¥À¡½É‘•É•°…Õµ•¹Ñ…Ñ¥½¹}Í••‘Ì¤¤(€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€å¥•±½É‘•É•