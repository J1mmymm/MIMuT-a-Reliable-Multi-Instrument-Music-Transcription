"""Strict isolated-source catalog for deterministic training augmentation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import soundfile as sf

from muscriptor.data.manifest import ManifestRecord
from muscriptor.data.schema import load_standardized_track
from muscriptor.data.slakh import _metadata, _stem_properties
from muscriptor.evaluation.io import midi_to_notes
from muscriptor.tokenizer.encode import instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, trim_overlapping_notes


STRICT_ALLOWED_DATASETS = {
    "slakh2100_redux",
    "choralebricks",
    "maestro",
    "gaps",
    "guitarset",
    "idmt_smt_bass",
}


@dataclass(frozen=True)
class AugmentationCatalogEntry:
    track_id: str
    dataset: str
    split: str
    audio_path: str
    notes_path: str
    duration: float
    instrument_groups: list[int]
    group_id: str
    role: str
    is_drum_only: bool
    lineage: list[str]

    @classmethod
    def from_dict(cls, raw: dict) -> "AugmentationCatalogEntry":
        return cls(**raw)


def augmentation_catalog_fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_augmentation_catalog(path: str | Path) -> list[AugmentationCatalogEntry]:
    entries = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(AugmentationCatalogEntry.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"invalid augmentation catalog row {line_number}") from exc
    return entries


def write_augmentation_catalog(
    entries: Iterable[AugmentationCatalogEntry], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda item: (item.dataset, item.track_id))
    path.write_text(
        "".join(json.dumps(asdict(entry), sort_keys=True) + "\n" for entry in ordered)
    )


def validate_augmentation_catalog(
    entries: list[AugmentationCatalogEntry],
    manifest_records: list[ManifestRecord],
    *,
    allowed_datasets: Iterable[str] = STRICT_ALLOWED_DATASETS,
    required_duration: float | None = None,
    check_files: bool = True,
) -> dict:
    """Fail closed on incomplete labels, non-train sources, or split leakage."""

    allowed = set(allowed_datasets)
    nontrain_groups = {
        (record.dataset, record.group_id)
        for record in manifest_records
        if record.split != "train"
    }
    train_groups = {
        (record.dataset, record.group_id)
        for record in manifest_records
        if record.split == "train"
    }
    seen = set()
    errors = []
    eligible_datasets = set()
    eligible_groups = set()
    for entry in entries:
        key = (entry.dataset, entry.track_id)
        if key in seen:
            errors.append(f"duplicate source {entry.dataset}/{entry.track_id}")
        seen.add(key)
        if entry.dataset not in allowed:
            errors.append(f"non-whitelisted dataset {entry.dataset}")
        if entry.split != "train":
            errors.append(f"non-train source {entry.dataset}/{entry.track_id}")
        if entry.role != "isolated_complete":
            errors.append(f"unverified source role {entry.dataset}/{entry.track_id}")
        if (entry.dataset, entry.group_id) in nontrain_groups:
            errors.append(f"split leakage for {entry.dataset}/{entry.group_id}")
        if (entry.dataset, entry.group_id) not in train_groups:
            errors.append(
                "source group is not registered as train: "
                f"{entry.dataset}/{entry.group_id}"
            )
        if not entry.lineage:
            errors.append(f"missing lineage for {entry.dataset}/{entry.track_id}")
        if entry.duration <= 0:
            errors.append(f"non-positive duration for {entry.dataset}/{entry.track_id}")
        if check_files:
            if not Path(entry.audio_path).is_file():
                errors.append(f"missing audio {entry.audio_path}")
            if not Path(entry.notes_path).is_file():
                errors.append(f"missing notes {entry.notes_path}")
            else:
                try:
                    load_standardized_track(entry.notes_path)
                except Exception as exc:
                    errors.append(f"invalid notes {entry.notes_path}: {exc}")
        if required_duration is None or entry.duration >= required_duration:
            eligible_datasets.add(entry.dataset)
            eligible_groups.add((entry.dataset, entry.group_id))
    if required_duration is not None and len(eligible_datasets) < 2:
        errors.append(
            "cross-dataset remix needs two whitelisted datasets with sources "
            f"at least {required_duration:g}s long"
        )
    if required_duration is not None and len(eligible_groups) < 4:
        errors.append(
            "remix policy needs four distinct train groups at the maximum context"
        )
    if not entries:
        errors.append("augmentation catalog is empty")
    if errors:
        raise ValueError("augmentation catalog rejected: " + "; ".join(errors[:20]))
    return {
        "entries": len(entries),
        "datasets": sorted({entry.dataset for entry in entries}),
        "groups": len({(entry.dataset, entry.group_id) for entry in entries}),
    }


def _manifest_entry(record: ManifestRecord) -> AugmentationCatalogEntry | None:
    if record.dataset not in STRICT_ALLOWED_DATASETS or record.split != "train":
        return None
    if record.dataset == "slakh2100_redux":
        return None  # mixture rows are never isolated sources
    if record.dataset == "choralebricks" and not (
        "::tracks::" in record.track_id
        or "tracks" in {part.lower() for part in Path(record.notes_path).parts}
    ):
        return None
    if record.dataset == "guitarset":
        try:
            if sf.info(record.audio_path).channels != 1:
                return None
        except (OSError, RuntimeError):
            return None
    if record.is_multi_instrument:
        return None
    notes = load_standardized_track(record.notes_path).notes
    if not notes:
        return None
    return AugmentationCatalogEntry(
        track_id=record.track_id,
        dataset=record.dataset,
        split="train",
        audio_path=record.audio_path,
        notes_path=record.notes_path,
        duration=record.duration,
        instrument_groups=record.instrument_groups,
        group_id=record.group_id,
        role="isolated_complete",
        is_drum_only=all(note.is_drum for note in notes),
        lineage=[f"{record.dataset}:{record.group_id}:{record.track_id}"],
    )


def _write_stem_notes(
    midi_path: Path,
    cache_path: Path,
    *,
    program: int | None,
    is_drum: bool,
    source_track_id: str,
) -> list[Note]:
    notes = midi_to_notes(midi_path)
    for note in notes:
        if is_drum:
            note.is_drum = True
            note.program = DRUM_PROGRAM
            note.offset = note.onset + 0.01
        elif program is not None:
            note.is_drum = False
            note.program = program
    notes = trim_overlapping_notes(notes, sort=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "notes": [
                    {
                        "onset": note.onset,
                        "offset": note.offset,
                        "pitch": note.pitch,
                        "program": 0 if note.is_drum else note.program,
                        "is_drum": note.is_drum,
                        "velocity": 64,
                    }
                    for note in notes
                ],
                "provenance": {
                    "dataset": "slakh2100_redux",
                    "source_track_id": source_track_id,
                    "source_stem": midi_path.stem,
                    "source_role": "isolated_complete",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    return notes


def discover_slakh_augmentation_sources(
    slakh_root: str | Path, cache_root: str | Path
) -> list[AugmentationCatalogEntry]:
    slakh_root = Path(slakh_root).resolve()
    cache_root = Path(cache_root).resolve()
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    entries = []
    for track_dir in sorted(slakh_root.rglob("Track*")):
        if not track_dir.is_dir() or not (track_dir / "MIDI").is_dir():
            continue
        lower_parts = {part.lower() for part in track_dir.parts}
        if "test" in lower_parts or "validation" in lower_parts or "val" in lower_parts:
            continue
        metadata = _metadata(track_dir)
        relative_track = track_dir.relative_to(slakh_root)
        group_id = track_dir.name
        for midi_path in sorted((track_dir / "MIDI").glob("*.mid")):
            audio = next(
                (
                    candidate
                    for directory in ("stems", "Stems")
                    for extension in (".flac", ".wav")
                    if (
                        candidate := track_dir / directory / f"{midi_path.stem}{extension}"
                    ).is_file()
                ),
                None,
            )
            if audio is None:
                continue
            program, is_drum = _stem_properties(metadata, midi_path.stem)
            cache_path = cache_root / relative_track / f"{midi_path.stem}.notes.json"
            notes = _write_stem_notes(
                midi_path,
                cache_path,
                program=program,
                is_drum=is_drum,
                source_track_id=group_id,
            )
            if not notes:
                continue
            info = sf.info(audio)
            entries.append(
                AugmentationCatalogEntry(
                    track_id=f"{relative_track.as_posix()}::{midi_path.stem}",
                    dataset="slakh2100_redux",
                    split="train",
                    audio_path=str(audio.resolve()),
                    notes_path=str(cache_path),
                    duration=float(info.frames / info.samplerate),
                    instrument_groups=instrument_group_ids(tokenizer, notes),
                    group_id=group_id,
                    role="isolated_complete",
                    is_drum_only=all(note.is_drum for note in notes),
                    lineage=[
                        "slakh2100_redux:"
                        f"{group_id}:{relative_track.as_posix()}::{midi_path.stem}"
                    ],
                )
            )
    return entries


def build_augmentation_catalog(
    manifest_records: list[ManifestRecord],
    output: str | Path,
    *,
    slakh_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> list[AugmentationCatalogEntry]:
    entries = [
        entry
        for record in manifest_records
        if (entry := _manifest_entry(record)) is not None
    ]
    if slakh_root is not None:
        if cache_root is None:
            cache_root = Path(output).parent / "augmentation_cache" / "slakh_stems"
        entries.extend(discover_slakh_augmentation_sources(slakh_root, cache_root))
    validate_augmentation_catalog(entries, manifest_records, check_files=True)
    write_augmentation_catalog(entries, output)
    return entries
