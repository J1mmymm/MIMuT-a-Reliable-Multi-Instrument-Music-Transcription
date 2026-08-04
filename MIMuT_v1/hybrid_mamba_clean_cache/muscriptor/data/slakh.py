"""Slakh2100 Redux adapter.

Slakh stores one MIDI file per stem rather than the common standardized JSON
used by the other datasets.  The adapter merges those stems into a cache next
to the generated manifest, never modifying the source dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import yaml

from muscriptor.data.manifest import BuildIssue, ManifestRecord
from muscriptor.evaluation.io import midi_to_notes
from muscriptor.tokenizer.encode import instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, trim_overlapping_notes


def _metadata(track_dir: Path) -> dict:
    for name in ("metadata.yaml", "metadata.yml"):
        path = track_dir / name
        if path.exists():
            raw = yaml.safe_load(path.read_text())
            return raw if isinstance(raw, dict) else {}
    return {}


def _stem_properties(metadata: dict, stem: str) -> tuple[int | None, bool]:
    stems = metadata.get("stems") or {}
    item = stems.get(stem) or stems.get(stem.upper()) or {}
    if not isinstance(item, dict):
        return None, False
    program = item.get("program_num", item.get("program"))
    is_drum = bool(item.get("is_drum", item.get("drum", False)))
    return (int(program) if program is not None else None), is_drum


def _merge_stems(track_dir: Path) -> list[Note]:
    midi_dir = track_dir / "MIDI"
    metadata = _metadata(track_dir)
    notes: list[Note] = []
    for midi_path in sorted(midi_dir.glob("*.mid")):
        program, is_drum = _stem_properties(metadata, midi_path.stem)
        for note in midi_to_notes(midi_path):
            if is_drum:
                note.is_drum = True
                note.program = DRUM_PROGRAM
                note.offset = note.onset + 0.01
            elif program is not None:
                note.is_drum = False
                note.program = program
            notes.append(note)
    return trim_overlapping_notes(notes, sort=True)


def discover_slakh(
    dataset_root: Path, cache_root: Path
) -> tuple[list[ManifestRecord], list[BuildIssue]]:
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    records = []
    issues = []
    for track_dir in sorted(dataset_root.rglob("Track*")):
        if not track_dir.is_dir() or not (track_dir / "MIDI").is_dir():
            continue
        try:
            audio = next(
                path
                for name in ("mix.flac", "mix.wav", "mixture.flac", "mixture.wav")
                if (path := track_dir / name).exists()
            )
            notes = _merge_stems(track_dir)
            if not notes:
                raise ValueError("no MIDI note events found")
            relative = track_dir.relative_to(dataset_root)
            cache_path = cache_root / relative.with_suffix(".notes.json")
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
                            "dataset": "slakh2100_redux",
                            "source_track_id": track_dir.name,
                        },
                    }
                )
                + "\n"
            )
            info = sf.info(audio)
            split = "train"
            lower_parts = {part.lower() for part in track_dir.parts}
            if "validation" in lower_parts:
                split = "validation"
            elif "test" in lower_parts:
                split = "test"
            groups = instrument_group_ids(tokenizer, notes)
            records.append(
                ManifestRecord(
                    track_id=track_dir.name,
                    dataset="slakh2100_redux",
                    split=split,
                    audio_path=str(audio.resolve()),
                    notes_path=str(cache_path.resolve()),
                    duration=float(info.frames / info.samplerate),
                    instrument_groups=groups,
                    group_id=track_dir.name,
                    is_multi_instrument=len(groups) > 1,
                )
            )
        except Exception as exc:
            issues.append(BuildIssue("slakh2100_redux", str(track_dir), str(exc)))
    return records, issues
