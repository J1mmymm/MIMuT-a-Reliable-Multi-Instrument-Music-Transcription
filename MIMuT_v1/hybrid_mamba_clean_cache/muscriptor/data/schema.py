"""Parser for the repository's standardized ``*.notes.json`` format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    MINIMUM_NOTE_DURATION_SEC,
    Note,
    trim_overlapping_notes,
    validate_notes,
)


@dataclass(frozen=True)
class StandardizedTrack:
    notes: list[Note]
    provenance: dict[str, Any]


def load_standardized_track(
    path: str | Path, *, trim_overlaps: bool = True
) -> StandardizedTrack:
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("notes"), list):
        raise ValueError(f"{path}: expected an object containing a notes list")

    notes: list[Note] = []
    for index, item in enumerate(raw["notes"]):
        try:
            onset = float(item["onset"])
            offset = float(item["offset"])
            pitch = int(item["pitch"])
            program = int(item["program"])
            is_drum = bool(item["is_drum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid note at index {index}") from exc
        if onset < 0 or offset < 0:
            raise ValueError(f"{path}: negative note time at index {index}")
        if not 0 <= pitch <= 127:
            raise ValueError(f"{path}: pitch outside 0..127 at index {index}")
        if is_drum:
            program = DRUM_PROGRAM
            offset = max(offset, onset + MINIMUM_NOTE_DURATION_SEC)
        elif not 0 <= program <= 127:
            raise ValueError(f"{path}: program outside 0..127 at index {index}")
        notes.append(
            Note(
                is_drum=is_drum,
                program=program,
                onset=onset,
                offset=offset,
                pitch=pitch,
            )
        )

    notes = validate_notes(notes, fix=True)
    if trim_overlaps:
        notes = trim_overlapping_notes(notes, sort=True)
    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError(f"{path}: provenance must be an object")
    return StandardizedTrack(notes=notes, provenance=provenance)
