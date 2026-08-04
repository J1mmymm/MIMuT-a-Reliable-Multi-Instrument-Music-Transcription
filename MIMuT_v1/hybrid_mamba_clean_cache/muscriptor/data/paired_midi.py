"""Adapter for preprocessed datasets with aligned WAV/MIDI pairs."""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from muscriptor.data.manifest import (
    AUDIO_EXTENSIONS,
    BuildIssue,
    ManifestRecord,
    _infer_split,
)
from muscriptor.evaluation.io import midi_to_notes
from muscriptor.tokenizer.encode import instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, trim_overlapping_notes


def _split_overrides(dataset_root: Path) -> dict[Path, str]:
    result = {}
    for path in dataset_root.rglob("stratified_split*.json"):
        try:
            raw = json.loads(path.read_text())
            for split, names in raw.items():
                if split not in {"train", "validation", "test"}:
                    continue
                for name in names:
                    result[(path.parent / name).resolve()] = split
        except Exception:
            continue
    return result


def _audio_index(dataset_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            result.setdefault(path.stem, []).append(path)
    return result


def discover_paired_midi(
    dataset_root: Path,
    cache_root: Path,
    *,
    dataset_name: str,
    force_drums: bool = False,
) -> tuple[list[ManifestRecord], list[BuildIssue]]:
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    index = _audio_index(dataset_root)
    overrides = _split_overrides(dataset_root)
    records = []
    issues = []
    for midi_path in sorted(
        [
            *dataset_root.rglob("*.mid"),
            *dataset_root.rglob("*.midi"),
        ]
    ):
        try:
            candidates = index.get(midi_path.stem, [])
            same_directory = [
                path for path in candidates if path.parent == midi_path.parent
            ]
            if len(same_directory) == 1:
                audio = same_directory[0]
            elif len(candidates) == 1:
                audio = candidates[0]
            else:
                raise ValueError("no unique same-stem audio file")
            notes = midi_to_notes(midi_path)
            if force_drums:
                notes = [
                    Note(
                        is_drum=True,
                        program=DRUM_PROGRAM,
                        onset=note.onset,
                        offset=note.onset + 0.01,
                        pitch=note.pitch,
                    )
                    for note in notes
                ]
            notes = trim_overlapping_notes(notes, sort=True)
            if not notes:
                raise ValueError("MIDI contains no notes")
            track_id = audio.stem
            split = overrides.get(audio.resolve())
            if split is None:
                split = _infer_split(audio, dataset_name, track_id)
            cache_path = cache_root / split / f"{track_id}.notes.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "onset": note.onset,
                                "offset": note.offset,
                                "pitch": note.pitch,
                                "program": (0 if note.is_drum else note.program),
                                "is_drum": note.is_drum,
                                "velocity": 64,
                            }
                            for note in notes
                        ],
                        "provenance": {
                            "dataset": dataset_name,
                            "source_track_id": track_id,
                        },
                    }
                )
                + "\n"
            )
            info = sf.info(audio)
            groups = instrument_group_ids(tokenizer, notes)
            records.append(
                ManifestRecord(
                    track_id=track_id,
                    dataset=dataset_name,
                    split=split,
                    audio_path=str(audio.resolve()),
                    notes_path=str(cache_path.resolve()),
                    duration=float(info.frames / info.samplerate),
                    instrument_groups=groups,
                    group_id=track_id,
                    is_multi_instrument=len(groups) > 1,
                )
            )
        except Exception as exc:
            issues.append(BuildIssue(dataset_name, str(midi_path), str(exc)))
    return records, issues
