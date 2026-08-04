"""Dataset discovery, canonical manifests, and leakage checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import soundfile as sf

from muscriptor.data.schema import load_standardized_track
from muscriptor.tokenizer.encode import instrument_group_ids
from muscriptor.tokenizer.mt3 import MT3Tokenizer


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff", ".m4a"}
STRICT_TEST_DATASETS = {
    "rwc_pop",
    "urmp",
    "bach10",
    "dagstuhl_choirset",
    "phenicx",
}


@dataclass(frozen=True)
class ManifestRecord:
    track_id: str
    dataset: str
    split: str
    audio_path: str
    notes_path: str
    duration: float
    instrument_groups: list[int]
    group_id: str
    is_multi_instrument: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "ManifestRecord":
        raw = dict(raw)
        if "audio" in raw:
            raw["audio_path"] = raw.pop("audio")
        if "notes" in raw:
            raw["notes_path"] = raw.pop("notes")
        if "instrument_classes" in raw:
            raw["instrument_groups"] = raw.pop("instrument_classes")
        return cls(**raw)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "dataset": self.dataset,
            "split": self.split,
            "audio": self.audio_path,
            "notes": self.notes_path,
            "duration": self.duration,
            "instrument_classes": self.instrument_groups,
            "group_id": self.group_id,
            "is_multi_instrument": self.is_multi_instrument,
        }


@dataclass(frozen=True)
class BuildIssue:
    dataset: str
    notes_path: str
    reason: str


def _stable_split(group_id: str) -> str:
    value = int(hashlib.sha1(group_id.encode()).hexdigest()[:8], 16) % 10
    if value < 8:
        return "train"
    if value == 8:
        return "validation"
    return "test"


def _infer_split(path: Path | Iterable[Path], dataset: str, group_id: str) -> str:
    if dataset.lower() in STRICT_TEST_DATASETS:
        return "test"
    paths = [path] if isinstance(path, Path) else list(path)
    explicit = set()
    for item in paths:
        parts = {part.lower() for part in item.parts}
        if "test" in parts or "testing" in parts or "test_data" in parts:
            explicit.add("test")
        if "validation" in parts or "valid" in parts or "val" in parts:
            explicit.add("validation")
        if "train" in parts or "training" in parts or "train_data" in parts:
            explicit.add("train")
    if len(explicit) > 1:
        raise ValueError(
            f"conflicting explicit splits for {dataset}/{group_id}: {sorted(explicit)}"
        )
    if explicit:
        return explicit.pop()
    return _stable_split(group_id)


def _audio_index(dataset_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            result.setdefault(path.stem, []).append(path)
    return result


def _audio_candidates(note_path: Path, dataset_root: Path) -> Iterable[Path]:
    stem = note_path.name.removesuffix(".notes.json")
    relative = note_path.relative_to(dataset_root)
    parts = list(relative.parts)
    replacements = []
    for index, part in enumerate(parts[:-1]):
        if part == "standardized_notes":
            for audio_dir in ("audio", "wav", "Wavfile"):
                candidate = parts.copy()
                candidate[index] = audio_dir
                candidate[-1] = stem
                replacements.append(dataset_root.joinpath(*candidate))
    replacements.append(note_path.with_name(stem))
    for base in replacements:
        for extension in AUDIO_EXTENSIONS:
            yield base.with_suffix(extension)


def _resolve_audio(
    note_path: Path,
    dataset_root: Path,
    index: dict[str, list[Path]],
) -> Path | None:
    for candidate in _audio_candidates(note_path, dataset_root):
        if candidate.exists():
            return candidate
    stem = note_path.name.removesuffix(".notes.json")
    matches = index.get(stem, [])
    return matches[0] if len(matches) == 1 else None


def _group_id(note_path: Path, provenance: dict, dataset: str) -> str:
    components = provenance.get("component_track_ids")
    if isinstance(components, list) and components:
        return "|".join(sorted(map(str, components)))
    for key in ("source_track_id", "track_id", "piece_id", "id"):
        if provenance.get(key) is not None:
            return str(provenance[key])
    stem = note_path.name.removesuffix(".notes.json")
    dataset_lower = dataset.lower()
    if "guitarset" in dataset_lower:
        return f"performer:{stem.split('_', 1)[0]}"
    if "enst" in dataset_lower:
        drummer = next(
            (part for part in note_path.parts if part.lower().startswith("drummer_")),
            None,
        )
        if drummer:
            return f"performer:{drummer}"
    if "mir1k" in dataset_lower:
        return f"performer:{stem.split('_', 1)[0]}"
    return stem


def discover_standardized_dataset(
    dataset_root: str | Path,
    *,
    dataset_name: str | None = None,
    tokenizer: MT3Tokenizer | None = None,
) -> tuple[list[ManifestRecord], list[BuildIssue]]:
    dataset_root = Path(dataset_root).resolve()
    dataset = dataset_name or dataset_root.name
    tokenizer = tokenizer or MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    index = _audio_index(dataset_root)
    records: list[ManifestRecord] = []
    issues: list[BuildIssue] = []

    for notes_path in sorted(dataset_root.rglob("*.notes.json")):
        try:
            track = load_standardized_track(notes_path)
            audio_path = _resolve_audio(notes_path, dataset_root, index)
            if audio_path is None:
                raise ValueError("no unique matching audio file")
            info = sf.info(audio_path)
            duration = float(info.frames / info.samplerate)
            group_id = _group_id(notes_path, track.provenance, dataset)
            groups = instrument_group_ids(tokenizer, track.notes)
            records.append(
                ManifestRecord(
                    track_id=notes_path.name.removesuffix(".notes.json"),
                    dataset=dataset,
                    split=_infer_split([notes_path, audio_path], dataset, group_id),
                    audio_path=str(audio_path),
                    notes_path=str(notes_path.resolve()),
                    duration=duration,
                    instrument_groups=groups,
                    group_id=group_id,
                    is_multi_instrument=len(groups) > 1,
                )
            )
        except Exception as exc:
            issues.append(BuildIssue(dataset, str(notes_path), str(exc)))
    return records, issues


def build_manifest(
    data_root: str | Path,
    output_path: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[ManifestRecord], list[BuildIssue]]:
    data_root = Path(data_root).resolve()
    records: list[ManifestRecord] = []
    issues: list[BuildIssue] = []
    output_path = Path(output_path)
    cache_root = output_path.parent / "standardized_cache"
    for dataset_root in sorted(path for path in data_root.iterdir() if path.is_dir()):
        found, failed = discover_standardized_dataset(
            dataset_root, dataset_name=dataset_root.name
        )
        if "slakh" in dataset_root.name.lower():
            from muscriptor.data.slakh import discover_slakh

            slakh_found, slakh_failed = discover_slakh(
                dataset_root, cache_root / "slakh2100_redux"
            )
            # Prefer explicit standardized JSON if the source already has it.
            existing = {record.track_id for record in found}
            found.extend(
                record for record in slakh_found if record.track_id not in existing
            )
            failed.extend(slakh_failed)
        if dataset_root.name.lower() == "urmp":
            from muscriptor.data.urmp import discover_urmp

            urmp_found, urmp_failed = discover_urmp(dataset_root, cache_root / "urmp")
            existing = {record.track_id for record in found}
            found.extend(
                record for record in urmp_found if record.track_id not in existing
            )
            failed.extend(urmp_failed)
        if (
            "idmt_smt_bass" in dataset_root.name.lower()
            or "star_drums" in dataset_root.name.lower()
        ):
            from muscriptor.data.paired_midi import discover_paired_midi

            paired_found, paired_failed = discover_paired_midi(
                dataset_root,
                cache_root / dataset_root.name,
                dataset_name=dataset_root.name,
                force_drums="star_drums" in dataset_root.name.lower(),
            )
            existing = {record.track_id for record in found}
            found.extend(
                record for record in paired_found if record.track_id not in existing
            )
            failed.extend(paired_failed)
        if not found:
            failed.append(
                BuildIssue(
                    dataset_root.name,
                    str(dataset_root),
                    "no supported standardized JSON or aligned audio/MIDI pairs",
                )
            )
        records.extend(found)
        issues.extend(failed)
    records = canonicalize_provenance_groups(records)
    validate_no_leakage(records)
    if strict and issues:
        raise RuntimeError(
            f"manifest discovery produced {len(issues)} unresolved files"
        )
    write_manifest(records, output_path)
    issue_path = Path(output_path).with_suffix(".issues.jsonl")
    issue_path.write_text(
        "".join(
            json.dumps(asdict(issue), ensure_ascii=False) + "\n" for issue in issues
        )
    )
    return records, issues


def canonicalize_provenance_groups(
    records: list[ManifestRecord],
) -> list[ManifestRecord]:
    """Collapse overlapping component sets into leakage-safe groups."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(item: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for record in records:
        components = [
            (record.dataset, item) for item in record.group_id.split("|") if item
        ]
        for component in components[1:]:
            union(components[0], component)

    output = []
    for record in records:
        components = [
            (record.dataset, item) for item in record.group_id.split("|") if item
        ]
        canonical = find(components[0])[1]
        split = _infer_split(
            [Path(record.notes_path), Path(record.audio_path)],
            record.dataset,
            canonical,
        )
        output.append(replace(record, group_id=canonical, split=split))
    return output


def write_manifest(records: Iterable[ManifestRecord], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            for record in records
        )
    )


def read_manifest(path: str | Path) -> list[ManifestRecord]:
    records = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(ManifestRecord.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid manifest record"
                    ) from exc
    validate_no_leakage(records)
    return records


def validate_no_leakage(records: Iterable[ManifestRecord]) -> None:
    seen: dict[tuple[str, str], str] = {}
    track_ids: set[tuple[str, str]] = set()
    for record in records:
        track_key = (record.dataset, record.track_id)
        if track_key in track_ids:
            raise ValueError(f"duplicate track_id: {record.dataset}/{record.track_id}")
        track_ids.add(track_key)
        key = (record.dataset, record.group_id)
        previous = seen.setdefault(key, record.split)
        if previous != record.split:
            raise ValueError(
                f"split leakage: {record.dataset}/{record.group_id} appears "
                f"in both {previous} and {record.split}"
            )


def manifest_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
