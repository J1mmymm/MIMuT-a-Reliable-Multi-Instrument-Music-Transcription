"""Adapter for URMP's aligned per-note text annotations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import soundfile as sf

from muscriptor.data.manifest import BuildIssue, ManifestRecord
from muscriptor.tokenizer.encode import instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note, trim_overlapping_notes


PROGRAMS = {
    "vn": 40,
    "va": 41,
    "vc": 42,
    "db": 43,
    "fl": 73,
    "ob": 68,
    "cl": 71,
    "sax": 65,
    "bn": 70,
    "tpt": 56,
    "hn": 60,
    "tbn": 57,
    "tba": 58,
}


def _frequency_to_midi(frequency: float) -> int:
    if frequency <= 0:
        raise ValueError(f"invalid note frequency: {frequency}")
    return max(0, min(127, round(69 + 12 * math.log2(frequency / 440))))


def _piece_notes(piece: Path) -> tuple[list[Note], list[str]]:
    notes: list[Note] = []
    component_ids = []
    instrument_codes = piece.name.split("_")[2:]
    for path in sorted(piece.glob("Notes_*.txt")):
        parts = path.stem.split("_")
        track_number = int(parts[1])
        code = (
            instrument_codes[track_number - 1]
            if track_number <= len(instrument_codes)
            else parts[2]
        )
        if code not in PROGRAMS:
            raise ValueError(f"unsupported URMP instrument code: {code}")
        component_ids.append(path.stem)
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.replace(",", " ").split()
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected 3 columns")
            onset, frequency, duration = map(float, fields[:3])
            notes.append(
                Note(
                    is_drum=False,
                    program=PROGRAMS[code],
                    onset=onset,
                    offset=onset + max(0.01, duration),
                    pitch=_frequency_to_midi(frequency),
                )
            )
    return trim_overlapping_notes(notes, sort=True), component_ids


def discover_urmp(
    dataset_root: Path, cache_root: Path
) -> tuple[list[ManifestRecord], list[BuildIssue]]:
    """Convert all 44 URMP mixtures and mark them test-only."""
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    records = []
    issues = []
    for piece in sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_dir() and list(path.glob("AuMix*.wav"))
    ):
        try:
            audio = next(piece.glob("AuMix*.wav"))
            notes, components = _piece_notes(piece)
            if not notes:
                raise ValueError("no aligned Notes_*.txt annotations found")
            cache_path = cache_root / f"{piece.name}.notes.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "notes": [
                            {
                                "onset": note.onset,
                                "offset": note.offset,
                                "pitch": note.pitch,
                                "program": note.program,
                                "is_drum": False,
                                "velocity": 64,
                            }
                            for note in notes
                        ],
                        "provenance": {
                            "dataset": "urmp",
                            "source_track_id": piece.name,
                            "component_track_ids": components,
                        },
                    }
                )
                + "\n"
            )
            info = sf.info(audio)
            groups = instrument_group_ids(tokenizer, notes)
            records.append(
                ManifestRecord(
                    track_id=piece.name,
                    dataset="urmp",
                    split="test",
                    audio_path=str(audio.resolve()),
                    notes_path=str(cache_path.resolve()),
                    duration=float(info.frames / info.samplerate),
                    instrument_groups=groups,
                    group_id=piece.name,
                    is_multi_instrument=len(groups) > 1,
                )
            )
        except Exception as exc:
            issues.append(BuildIssue("urmp", str(piece), str(exc)))
    return records, issues
