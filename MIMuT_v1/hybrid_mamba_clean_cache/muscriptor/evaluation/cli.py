"""Leakage-safe evaluation, paired statistics, and efficiency benchmarks."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Annotated

import torch
import typer
from safetensors import safe_open

from muscriptor.data.manifest import manifest_fingerprint, read_manifest
from muscriptor.evaluation.io import load_notes
from muscriptor.evaluation.metrics import evaluate_notes
from muscriptor.evaluation.statistics import (
    macro_average,
    micro_average,
    paired_track_bootstrap,
)
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.utils.audio import load_audio


app = typer.Typer(add_completion=False, help="MIMuT whole-track evaluation tools")


def _oracle_names(group_ids: list[int]) -> list[str]:
    inverse = {group_id: name for name, group_id in MT3_FULL_PLUS_GROUP_NAMES.items()}
    return [inverse[group_id] for group_id in group_ids if group_id in inverse]


def _prediction_path(directory: Path, dataset: str, track_id: str) -> Path | None:
    for suffix in (".notes.json", ".mid", ".midi"):
        for path in (
            directory / dataset / f"{track_id}{suffix}",
            directory / f"{track_id}{suffix}",
        ):
            if path.exists():
                return path
    return None


def _metric_only(row: dict) -> dict:
    return {
        key: value for key, value in row.items() if key not in {"track_id", "dataset"}
    }


def _elapsed_strata_support(rows: list[dict]) -> dict[str, dict]:
    """Pool explicit support counts for elapsed-time diagnostics."""

    available = {
        label
        for row in rows
        for label in row.get("elapsed_time_strata", {})
    }
    preferred = ("0-40", "40-80", "80-160", "160+")
    labels = [label for label in preferred if label in available]
    labels.extend(sorted(available - set(preferred)))
    result = {}
    for label in labels:
        strata = [
            row["elapsed_time_strata"][label]
            for row in rows
            if label in row.get("elapsed_time_strata", {})
        ]
        gap_labels = sorted(
            {
                gap
                for item in strata
                for gap in item.get("long_gap_reidentification", {})
            }
        )
        result[label] = {
            "whole_tracks": len(strata),
            "defined_tracks": sum(
                int(
                    item.get("reference_notes", 0) > 0
                    or item.get("predicted_notes", 0) > 0
                )
                for item in strata
            ),
            "reference_notes": sum(
                int(item.get("reference_notes", 0)) for item in strata
            ),
            "predicted_notes": sum(
                int(item.get("predicted_notes", 0)) for item in strata
            ),
            "boundary_reference_notes": sum(
                int(item.get("boundary_multi", {}).get("reference", 0))
                for item in strata
            ),
            "instrument_switch_comparisons": sum(
                int(item.get("instrument_switch_error", {}).get("reference", 0))
                for item in strata
            ),
            "long_gap_reference_episodes": {
                gap: sum(
                    int(
                        item.get("long_gap_reidentification", {})
                        .get(gap, {})
                        .get("reference", 0)
                    )
                    for item in strata
                )
                for gap in gap_labels
            },
        }
    return result


def _macro(rows: list[dict]) -> dict:
    """Backward-compatible alias used by older callers and tests."""

    return macro_average(rows)


def _read_run_metadata(directory: Path) -> dict | None:
    path = directory / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def _local_checkpoint_audit(model_path: str) -> dict | None:
    path = Path(model_path)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        return None
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata() or {}
    raw_experiment = metadata.get("experiment")
    experiment = json.loads(raw_experiment) if raw_experiment else None
    raw_model_config = metadata.get("model_config")
    raw_taxonomy = metadata.get("taxonomy")
    taxonomy = json.loads(raw_taxonomy) if raw_taxonomy else None
    return {
        "experiment": experiment if isinstance(experiment, dict) else None,
        "taxonomy": taxonomy if isinstance(taxonomy, dict) else None,
        "model_config_sha256": (
            hashlib.sha256(raw_model_config.encode()).hexdigest()
            if raw_model_config
            else None
        ),
        "training_manifest_sha256": metadata.get("manifest_sha256"),
    }


def _validate_run_contract(
    metadata: dict | None,
    *,
    manifest: Path,
    split: str,
    expected_track_count: int,
    allow_missing: bool,
    allow_oracle: bool,
    allow_unverified_conditions: bool,
) -> None:
    if metadata is None:
        if allow_unverified_conditions:
            return
        raise typer.BadParameter(
            "predictions/run.json is required to verify the audio-only condition; "
            "use --allow-unverified-conditions only for clearly labelled diagnostics"
        )

    oracle = bool(metadata.get("oracle_instruments"))
    if oracle and not allow_oracle:
        raise typer.BadParameter(
            "oracle-instrument predictions cannot enter the main comparison; "
            "use --allow-oracle only for an upper-bound table"
        )
    if not allow_unverified_conditions:
        expected_mode = "oracle_upper_bound" if oracle else "audio_only"
        if metadata.get("condition_mode") != expected_mode:
            raise typer.BadParameter(
                f"prediction condition_mode must be {expected_mode!r}"
            )
        if metadata.get("dataset_condition") is not None:
            raise typer.BadParameter(
                "dataset-conditioned predictions cannot enter the formal comparison"
            )
        if metadata.get("instrument_condition") is not None and not allow_oracle:
            raise typer.BadParameter(
                "instrument-conditioned predictions require --allow-oracle"
            )

    expected_manifest = manifest_fingerprint(manifest)
    recorded_manifest = metadata.get("manifest_sha256")
    if recorded_manifest is None and not allow_unverified_conditions:
        raise typer.BadParameter("prediction run does not record a manifest SHA")
    if recorded_manifest is not None and recorded_manifest != expected_manifest:
        raise typer.BadParameter(
            "prediction run was generated from a different manifest SHA"
        )
    recorded_split = metadata.get("split")
    if recorded_split is None and not allow_unverified_conditions:
        raise typer.BadParameter("prediction run does not record its split")
    if recorded_split is not None and recorded_split != split:
        raise typer.BadParameter(
            f"prediction run split {recorded_split!r} does not match {split!r}"
        )
    recorded_count = metadata.get("expected_track_count")
    if recorded_count is None and not allow_unverified_conditions:
        raise typer.BadParameter("prediction run does not record expected_track_count")
    if recorded_count is not None and int(recorded_count) != expected_track_count:
        raise typer.BadParameter(
            "prediction run expected_track_count does not match the selected manifest"
        )
    if metadata.get("status") != "complete" and not allow_missing:
        raise typer.BadParameter(
            "prediction run is not marked complete; --allow-missing is diagnostic only"
        )


def _mean_nested(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {}
    keys = set.intersection(*(set(row) for row in rows))
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = _mean_nested(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            result[key] = statistics.fmean(float(value) for value in values)
    return result


def _seed_statistics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    result = {}
    keys = set.intersection(*(set(row) for row in rows))
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = _seed_statistics(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            numeric = [float(value) for value in values]
            result[key] = {
                "mean": statistics.fmean(numeric),
                "std": statistics.pstdev(numeric),
                "values": numeric,
            }
    return result


@app.command("register-external")
def register_external(
    predictions: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    model_name: Annotated[str, typer.Option("--model-name")],
    split: str = "test",
    source_url: Annotated[str | None, typer.Option("--source-url")] = None,
    experiment_id: Annotated[str | None, typer.Option("--experiment-id")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    training_overlap: Annotated[str, typer.Option("--training-overlap")] = "unknown",
    oracle_instruments: Annotated[
        bool, typer.Option("--oracle-instruments/--no-oracle-instruments")
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Register externally generated MIDI files under the same audit contract."""

    allowed_overlap = {"none", "possible", "known", "unknown"}
    if training_overlap not in allowed_overlap:
        raise typer.BadParameter(
            f"training-overlap must be one of {sorted(allowed_overlap)}"
        )
    run_path = predictions / "run.json"
    if run_path.exists() and not overwrite:
        raise typer.BadParameter(
            f"{run_path} already exists; use --overwrite only after auditing it"
        )
    selected = [record for record in read_manifest(manifest) if record.split == split]
    missing = [
        f"{record.dataset}/{record.track_id}"
        for record in selected
        if _prediction_path(predictions, record.dataset, record.track_id) is None
    ]
    metadata = {
        "status": "complete" if not missing else "incomplete",
        "producer": "external_checkpoint",
        "model": model_name,
        "experiment_id": experiment_id,
        "seed": seed,
        "source_url": source_url,
        "manifest_sha256": manifest_fingerprint(manifest),
        "split": split,
        "expected_track_count": len(selected),
        "completed_track_count": len(selected) - len(missing),
        "missing_tracks": missing,
        "oracle_instruments": oracle_instruments,
        "instrument_condition": "reference" if oracle_instruments else None,
        "dataset_condition": None,
        "condition_mode": "oracle_upper_bound" if oracle_instruments else "audio_only",
        "training_overlap": training_overlap,
        "note": (
            "Registration records declared conditions; it does not prove the "
            "external model's hidden preprocessing or training corpus."
        ),
    }
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        f"registered {len(selected) - len(missing)}/{len(selected)} tracks "
        f"for {model_name} ({metadata['status']})"
    )


@app.command("aggregate-seeds")
def aggregate_seeds(
    inputs: Annotated[str, typer.Option("--inputs")],
    output: Annotated[Path, typer.Option("--output", "-o")],
    expected_seeds: Annotated[str, typer.Option("--expected-seeds")] = "3407,3408,3409",
) -> None:
    """Average complete evaluation files while retaining seed variability."""

    paths = [Path(raw.strip()) for raw in inputs.split(",") if raw.strip()]
    try:
        expected_seed_set = {
            int(raw.strip()) for raw in expected_seeds.split(",") if raw.strip()
        }
    except ValueError as error:
        raise typer.BadParameter(
            "expected-seeds must be comma-separated integers"
        ) from error
    if len(expected_seed_set) < 2:
        raise typer.BadParameter("aggregate-seeds requires at least two expected seeds")
    if len(paths) != len(expected_seed_set):
        raise typer.BadParameter(
            f"expected {len(expected_seed_set)} seed files, received {len(paths)}"
        )
    if len({path.resolve() for path in paths}) != len(paths):
        raise typer.BadParameter("aggregate-seeds inputs must be distinct files")
    for path in paths:
        if not path.is_file():
            raise typer.BadParameter(f"evaluation file does not exist: {path}")
    payloads = [json.loads(path.read_text()) for path in paths]
    if any(not payload.get("summary", {}).get("complete") for payload in payloads):
        raise typer.BadParameter("every seed evaluation must be complete")
    manifest_hashes = {
        payload.get("protocol", {}).get("manifest_sha256") for payload in payloads
    }
    if None in manifest_hashes or len(manifest_hashes) != 1:
        raise typer.BadParameter("seed evaluations use different manifests")
    prediction_runs = [
        payload.get("protocol", {}).get("prediction_run") for payload in payloads
    ]
    if any(not isinstance(run, dict) for run in prediction_runs):
        raise typer.BadParameter(
            "every seed evaluation must retain its audited prediction_run"
        )
    declared_seeds = {run.get("seed") for run in prediction_runs}
    if None in declared_seeds or declared_seeds != expected_seed_set:
        raise typer.BadParameter(
            f"declared seeds {sorted(seed for seed in declared_seeds if seed is not None)} "
            f"do not match expected seeds {sorted(expected_seed_set)}"
        )
    experiment_ids = {run.get("experiment_id") for run in prediction_runs}
    if None in experiment_ids or len(experiment_ids) != 1:
        raise typer.BadParameter(
            "seed evaluations must share one non-null experiment_id"
        )
    if any(run.get("condition_mode") != "audio_only" for run in prediction_runs):
        raise typer.BadParameter("formal seed aggregation is audio-only")
    if any(run.get("seed_verified") is not True for run in prediction_runs):
        raise typer.BadParameter(
            "formal seed aggregation requires checkpoint-verified seed metadata"
        )
    model_config_hashes = {
        (run.get("checkpoint_audit") or {}).get("model_config_sha256")
        for run in prediction_runs
    }
    training_manifest_hashes = {
        (run.get("checkpoint_audit") or {}).get("training_manifest_sha256")
        for run in prediction_runs
    }
    if None in model_config_hashes or len(model_config_hashes) != 1:
        raise typer.BadParameter(
            "seed checkpoints do not share one verified model configuration"
        )
    if None in training_manifest_hashes or len(training_manifest_hashes) != 1:
        raise typer.BadParameter(
            "seed checkpoints do not share one verified training manifest"
        )

    def keyed(payload: dict) -> dict[tuple[str, str], dict]:
        return {
            (str(row["dataset"]), str(row["track_id"])): row
            for row in payload["tracks"]
        }

    keyed_payloads = [keyed(payload) for payload in payloads]
    if any(
        len(keyed_payload) != len(payload["tracks"])
        for keyed_payload, payload in zip(keyed_payloads, payloads)
    ):
        raise typer.BadParameter(
            "a seed evaluation contains duplicate track identifiers"
        )
    track_keys = [set(payload) for payload in keyed_payloads]
    if any(keys != track_keys[0] for keys in track_keys[1:]):
        raise typer.BadParameter("seed evaluations do not contain identical tracks")
    tracks = []
    for dataset, track_id in sorted(track_keys[0]):
        metric_rows = [
            _metric_only(payload[(dataset, track_id)]) for payload in keyed_payloads
        ]
        tracks.append(
            {
                "dataset": dataset,
                "track_id": track_id,
                **_mean_nested(metric_rows),
            }
        )

    macros = [payload["summary"]["macro"] for payload in payloads]
    micros = [payload["summary"]["micro"] for payload in payloads]
    result = {
        "protocol": {
            "type": "seed_aggregate",
            "manifest_sha256": next(iter(manifest_hashes)),
            "seed_count": len(paths),
            "seeds": sorted(expected_seed_set),
            "experiment_id": next(iter(experiment_ids)),
            "sources": [str(path) for path in paths],
            "note": (
                "Track metrics are averaged across seeds; paired bootstrap "
                "still resamples whole tracks."
            ),
        },
        "summary": {
            "complete": True,
            "expected_track_count": len(tracks),
            "evaluated_track_count": len(tracks),
            "macro": _mean_nested(macros),
            "micro": _mean_nested(micros),
            "macro_seed_statistics": _seed_statistics(macros),
            "micro_seed_statistics": _seed_statistics(micros),
        },
        "tracks": tracks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"aggregated {len(paths)} seeds over {len(tracks)} tracks")


@app.command("compare")
def compare(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    predictions: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: str = "test",
    allow_missing: Annotated[bool, typer.Option("--allow-missing")] = False,
    allow_oracle: Annotated[bool, typer.Option("--allow-oracle")] = False,
    allow_unverified_conditions: Annotated[
        bool, typer.Option("--allow-unverified-conditions")
    ] = False,
) -> None:
    """Evaluate complete whole-track predictions with macro and pooled metrics."""

    selected = [record for record in read_manifest(manifest) if record.split == split]
    run_metadata = _read_run_metadata(predictions)
    _validate_run_contract(
        run_metadata,
        manifest=manifest,
        split=split,
        expected_track_count=len(selected),
        allow_missing=allow_missing,
        allow_oracle=allow_oracle,
        allow_unverified_conditions=allow_unverified_conditions,
    )
    missing = []
    rows = []
    for record in selected:
        prediction_path = _prediction_path(predictions, record.dataset, record.track_id)
        if prediction_path is None:
            missing.append(f"{record.dataset}/{record.track_id}")
            continue
        result = evaluate_notes(
            load_notes(record.notes_path), load_notes(prediction_path)
        )
        rows.append(
            {
                "track_id": record.track_id,
                "dataset": record.dataset,
                **result.to_dict(),
            }
        )

    if missing and not allow_missing:
        preview = ", ".join(missing[:10])
        raise typer.BadParameter(
            f"missing {len(missing)} of {len(selected)} prediction files: {preview}"
        )

    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [_metric_only(row) for row in rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "track_count": len(dataset_rows),
            "macro": macro_average(dataset_rows),
            "micro": micro_average(dataset_rows),
            "elapsed_time_strata_support": _elapsed_strata_support(
                [row for row in rows if row["dataset"] == dataset]
            ),
        }
    metric_rows = [_metric_only(row) for row in rows]
    payload = {
        "protocol": {
            "split": split,
            "manifest_sha256": manifest_fingerprint(manifest),
            "prediction_run": run_metadata,
            "missing_allowed": allow_missing,
            "oracle_allowed": allow_oracle,
            "primary_metric": "macro.multi.f1",
            "bootstrap_unit": "whole_track",
        },
        "summary": {
            "expected_track_count": len(selected),
            "evaluated_track_count": len(rows),
            "complete": not missing,
            "missing_tracks": missing,
            "macro": macro_average(metric_rows),
            "micro": micro_average(metric_rows),
            "elapsed_time_strata_support": _elapsed_strata_support(rows),
            "by_dataset": by_dataset,
        },
        "tracks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"evaluated {len(rows)}/{len(selected)} complete tracks")


@app.command("predict")
def predict(
    model_path: Annotated[str, typer.Option("--model", "-m")],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: str = "test",
    device: str = "cuda",
    experiment_id: Annotated[str | None, typer.Option("--experiment-id")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    long_context: str = "carry",
    prelude_forcing: Annotated[
        bool, typer.Option("--prelude-forcing/--no-prelude-forcing")
    ] = False,
    committed_symbolic_state: Annotated[
        bool,
        typer.Option(
            "--committed-symbolic-state/--no-committed-symbolic-state"
        ),
    ] = False,
    oracle_instruments: Annotated[
        bool,
        typer.Option("--oracle-instruments/--no-oracle-instruments"),
    ] = False,
    condition_mode: Annotated[
        str, typer.Option("--condition-mode")
    ] = "audio_only",
) -> None:
    """Transcribe a manifest and persist enough metadata to audit conditions."""

    checkpoint_audit = _local_checkpoint_audit(model_path)
    if condition_mode not in {"audio_only", "conditioned"}:
        raise typer.BadParameter("condition-mode must be audio_only or conditioned")
    dataset_mapping = (
        (checkpoint_audit or {}).get("taxonomy") or {}
    ).get("datasets")
    if condition_mode == "conditioned" and not isinstance(dataset_mapping, dict):
        raise typer.BadParameter(
            "conditioned inference requires checkpoint taxonomy dataset ids"
        )
    checkpoint_experiment = (
        checkpoint_audit.get("experiment") if checkpoint_audit else None
    )
    declared_condition_mode = (
        (checkpoint_experiment or {}).get("conditioning") or {}
    ).get("mode")
    if condition_mode == "conditioned" and declared_condition_mode != "conditioned":
        raise typer.BadParameter(
            "conditioned inference requires a checkpoint trained in conditioned mode"
        )
    if condition_mode == "conditioned" and oracle_instruments:
        raise typer.BadParameter(
            "conditioned dataset and oracle-instrument diagnostics must be separate"
        )
    checkpoint_seed = (
        checkpoint_experiment.get("seed") if checkpoint_experiment else None
    )
    if seed is None and checkpoint_seed is not None:
        seed = int(checkpoint_seed)
    if (
        seed is not None
        and checkpoint_seed is not None
        and int(seed) != int(checkpoint_seed)
    ):
        raise typer.BadParameter(
            f"declared seed {seed} disagrees with checkpoint seed {checkpoint_seed}"
        )
    seed_verified = checkpoint_seed is not None and seed == int(checkpoint_seed)
    model = TranscriptionModel.load_model(model_path, device=device)
    selected = [record for record in read_manifest(manifest) if record.split == split]
    if condition_mode == "conditioned":
        missing_datasets = sorted(
            {record.dataset for record in selected} - set(dataset_mapping)
        )
        if missing_datasets:
            raise typer.BadParameter(
                "checkpoint taxonomy lacks dataset ids for: "
                + ", ".join(missing_datasets)
            )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "running",
        "model": model_path,
        "experiment_id": experiment_id,
        "seed": seed,
        "seed_verified": seed_verified,
        "checkpoint_audit": checkpoint_audit,
        "manifest_sha256": manifest_fingerprint(manifest),
        "split": split,
        "expected_track_count": len(selected),
        "completed_track_count": 0,
        "long_context": long_context,
        "prelude_forcing": prelude_forcing,
        "committed_symbolic_state": committed_symbolic_state,
        "oracle_instruments": oracle_instruments,
        "instrument_condition": "reference" if oracle_instruments else None,
        "dataset_condition": (
            "manifest_dataset_id" if condition_mode == "conditioned" else None
        ),
        "condition_mode": (
            "oracle_upper_bound" if oracle_instruments else condition_mode
        ),
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    for index, record in enumerate(selected, 1):
        instruments = (
            _oracle_names(record.instrument_groups) if oracle_instruments else None
        )
        transcription_kwargs = {}
        if condition_mode == "conditioned":
            transcription_kwargs["dataset_name"] = str(
                dataset_mapping[record.dataset]
            )
        midi = model.transcribe_to_midi(
            record.audio_path,
            instruments=instruments,
            long_context=long_context,
            prelude_forcing=prelude_forcing,
            committed_symbolic_state=committed_symbolic_state,
            **transcription_kwargs,
        )
        path = output / record.dataset / f"{record.track_id}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(midi)
        metadata["completed_track_count"] = index
        run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"[{index}/{len(selected)}] {record.dataset}/{record.track_id}")

    metadata["status"] = "complete"
    run_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


@app.command("bootstrap")
def bootstrap(
    baseline: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidate: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    metrics: str = (
        "multi.f1,boundary_multi.f1,"
        "long_gap_reidentification.10-20.f1,"
        "long_gap_reidentification.20+.f1,"
        "long_gap_reidentification.10+.f1,"
        "instrument_switch_error.rate"
    ),
    samples: int = 10_000,
    seed: int = 3407,
    include_elapsed_strata: Annotated[
        bool, typer.Option("--elapsed-strata/--no-elapsed-strata")
    ] = True,
) -> None:
    """Run paired whole-track bootstrap and Holm-adjust metric p-values."""

    baseline_payload = json.loads(baseline.read_text())
    candidate_payload = json.loads(candidate.read_text())
    metric_paths = [item.strip() for item in metrics.split(",") if item.strip()]
    lower_is_better = {
        "instrument_switch_error.rate",
        "boundary_errors.omission.rate",
        "boundary_errors.truncation.rate",
        "boundary_errors.duplication.rate",
    }
    result = paired_track_bootstrap(
        baseline_payload["tracks"],
        candidate_payload["tracks"],
        metric_paths=metric_paths,
        samples=samples,
        seed=seed,
        lower_is_better=lower_is_better,
    )
    if include_elapsed_strata:
        elapsed_paths = [
            f"elapsed_time_strata.{label}.{metric}"
            for label in ("0-40", "40-80", "80-160", "160+")
            for metric in (
                "multi.f1",
                "boundary_multi.f1",
                "long_gap_reidentification.10+.f1",
                "instrument_switch_error.rate",
            )
        ]
        if all(
            "elapsed_time_strata" in row
            for payload in (baseline_payload, candidate_payload)
            for row in payload.get("tracks", [])
        ):
            result["elapsed_time_strata"] = paired_track_bootstrap(
                baseline_payload["tracks"],
                candidate_payload["tracks"],
                metric_paths=elapsed_paths,
                samples=samples,
                seed=seed,
                lower_is_better={
                    path
                    for path in elapsed_paths
                    if path.endswith("instrument_switch_error.rate")
                },
            )
        else:
            result["elapsed_time_strata"] = {
                "status": "unavailable",
                "reason": "input evaluations predate elapsed-time strata",
            }
    result["baseline"] = str(baseline)
    result["candidate"] = str(candidate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _benchmark_once(
    model: TranscriptionModel,
    clip: torch.Tensor,
    *,
    seconds: float,
    long_context: str,
    prelude_forcing: bool,
    clear_cuda_cache: bool = False,
) -> dict:
    device = getattr(model, "_device", torch.device("cpu"))
    use_cuda = torch.cuda.is_available() and torch.device(device).type == "cuda"
    if use_cuda:
        if clear_cuda_cache:
            torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    list(
        model.transcribe(
            (clip, 16_000),
            use_sampling=False,
            batch_size=1,
            long_context=long_context,
            prelude_forcing=prelude_forcing,
            profile_generation=True,
        )
    )
    if use_cuda:
        torch.cuda.synchronize(device)
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
    else:
        allocated = reserved = 0
    elapsed = time.perf_counter() - start
    profile = getattr(model, "last_generation_profile", None)
    return {
        "elapsed_seconds": elapsed,
        "mean_chunk_latency_seconds": elapsed / max(1, math.ceil(seconds / 5.0)),
        "real_time_factor": elapsed / seconds,
        "audio_seconds_per_second": seconds / elapsed,
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_reserved_bytes": reserved,
        "streaming_state": model.last_streaming_state,
        "generation_profile": profile,
        "token_stream_sha256": getattr(model, "last_token_stream_sha256", None),
    }


def _stable_summary(values) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values),
    }


def _local_weights_sha256(model_path: str) -> str | None:
    path = Path(model_path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@app.command("benchmark")
def benchmark(
    model_path: Annotated[str, typer.Option("--model", "-m")],
    audio: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    lengths: str = "5,10,20,40,80,160,full",
    device: str = "cuda",
    dtype: str = "float32",
    long_context: str = "carry",
    prelude_forcing: Annotated[
        bool, typer.Option("--prelude-forcing/--no-prelude-forcing")
    ] = False,
    warmup_runs: int = 1,
    repeats: int = 3,
) -> None:
    """Measure cold and stable whole-track latency, memory, and state size."""

    if warmup_runs < 0 or repeats <= 0:
        raise typer.BadParameter(
            "warmup-runs must be non-negative and repeats positive"
        )
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise typer.BadParameter("dtype must be float32, float16, or bfloat16")
    model = TranscriptionModel.load_model(model_path, device=device, dtype=dtype)
    waveform = load_audio(audio, target_sr=16_000)
    full_seconds = waveform.shape[-1] / 16_000
    requested: list[tuple[str, float]] = []
    for raw in lengths.split(","):
        raw = raw.strip().lower()
        if raw == "full":
            requested.append(("full", full_seconds))
        else:
            requested.append((raw, float(raw)))

    rows = []
    for label, seconds in requested:
        samples = max(1, round(seconds * 16_000))
        if waveform.shape[-1] < samples:
            repeat = (samples + waveform.shape[-1] - 1) // waveform.shape[-1]
            clip = waveform.repeat(1, repeat)[:, :samples]
        else:
            clip = waveform[:, :samples]

        try:
            cold = _benchmark_once(
                model,
                clip,
                seconds=seconds,
                long_context=long_context,
                prelude_forcing=prelude_forcing,
                clear_cuda_cache=True,
            )
            for _ in range(warmup_runs):
                _benchmark_once(
                    model,
                    clip,
                    seconds=seconds,
                    long_context=long_context,
                    prelude_forcing=prelude_forcing,
                )
            stable = [
                _benchmark_once(
                    model,
                    clip,
                    seconds=seconds,
                    long_context=long_context,
                    prelude_forcing=prelude_forcing,
                )
                for _ in range(repeats)
            ]
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rows.append(
                {
                    "length_label": label,
                    "audio_seconds": seconds,
                    "source_mode": (
                        "full" if label == "full" else "repeat_or_trim_prefix"
                    ),
                    "status": "oom",
                    "error_type": type(error).__name__,
                    "error": str(error).splitlines()[0],
                }
            )
            continue
        stable_summary = {
            key: _stable_summary(row[key] for row in stable)
            for key in (
                "elapsed_seconds",
                "real_time_factor",
                "audio_seconds_per_second",
                "peak_memory_allocated_bytes",
                "peak_memory_reserved_bytes",
                "mean_chunk_latency_seconds",
            )
        }
        state_bytes = [
            float((row.get("streaming_state") or {}).get("bytes", 0)) for row in stable
        ]
        cache_lengths = [
            float(
                max(
                    (row.get("streaming_state") or {}).get("local_cache_lengths", [])
                    or [0]
                )
            )
            for row in stable
        ]
        stable_summary["streaming_state_bytes"] = _stable_summary(state_bytes)
        stable_summary["maximum_local_cache_length"] = _stable_summary(cache_lengths)
        for profile_key in (
            "prefix_prefill_seconds",
            "autoregressive_decode_seconds",
            "prefix_prefill_tokens",
            "autoregressive_decode_tokens",
            "generated_steps",
        ):
            values = [
                float((row.get("generation_profile") or {}).get(profile_key, 0))
                for row in stable
            ]
            stable_summary[profile_key] = _stable_summary(values)

        def profile_rate(row: dict, token_key: str, seconds_key: str) -> float:
            profile = row.get("generation_profile") or {}
            seconds_value = float(profile.get(seconds_key, 0))
            return (
                float(profile.get(token_key, 0)) / seconds_value
                if seconds_value
                else 0.0
            )

        stable_summary["prefix_prefill_tokens_per_second"] = _stable_summary(
            profile_rate(
                row, "prefix_prefill_tokens", "prefix_prefill_seconds"
            )
            for row in stable
        )
        stable_summary["autoregressive_decode_tokens_per_second"] = _stable_summary(
            profile_rate(
                row,
                "autoregressive_decode_tokens",
                "autoregressive_decode_seconds",
            )
            for row in stable
        )
        rows.append(
            {
                "length_label": label,
                "audio_seconds": seconds,
                "source_mode": ("full" if label == "full" else "repeat_or_trim_prefix"),
                "status": "ok",
                "cold_run": cold,
                "stable_runs": stable,
                "stable_summary": stable_summary,
            }
        )

    device_info = {"requested": device, "torch": torch.__version__}
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        device_info.update(
            {
                "name": torch.cuda.get_device_name(torch.device(device)),
                "cuda": torch.version.cuda,
            }
        )
    core_model = getattr(model, "_model", None)
    parameter_count = (
        sum(parameter.numel() for parameter in core_model.parameters())
        if core_model is not None
        else None
    )
    payload = {
        "protocol": {
            "model": model_path,
            "model_weights_sha256": _local_weights_sha256(model_path),
            "model_parameter_count": parameter_count,
            "audio": str(audio),
            "sample_rate": 16_000,
            "batch_size": 1,
            "weight_dtype": dtype,
            "cuda_autocast_dtype": (
                "float16"
                if torch.cuda.is_available() and torch.device(device).type == "cuda"
                else None
            ),
            "decoding": {"sampling": False, "beam_size": 1},
            "long_context": long_context,
            "prelude_forcing": prelude_forcing,
            "prefill_mode": getattr(
                getattr(core_model, "model_config", None), "prefill_mode", None
            ),
            "oracle_instruments": False,
            "warmup_runs": warmup_runs,
            "stable_repeats": repeats,
            "device": device_info,
        },
        "lengths": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _benchmark_rows(payload: dict) -> dict[float, dict]:
    return {
        float(row["audio_seconds"]): row
        for row in payload.get("lengths", [])
        if row.get("length_label") != "full"
    }


def _growth(first: dict, last: dict, metric: str) -> float | None:
    initial = float(first["stable_summary"][metric]["mean"])
    final = float(last["stable_summary"][metric]["mean"])
    return final / initial if initial > 0 else None


def _chunk_prefill_gate(
    baseline_payload: dict,
    candidate_payload: dict,
    baseline_rows: dict[float, dict],
    candidate_rows: dict[float, dict],
    *,
    minimum_speedup: float = 1.25,
) -> dict:
    """Evaluate carried-prefix speed at 5/10/20/40 seconds of prior state."""

    baseline_protocol = baseline_payload.get("protocol", {})
    candidate_protocol = candidate_payload.get("protocol", {})
    same_weights = (
        baseline_protocol.get("model_weights_sha256") is not None
        and baseline_protocol.get("model_weights_sha256")
        == candidate_protocol.get("model_weights_sha256")
    )
    modes_ok = baseline_protocol.get("prefill_mode") == "step" and (
        candidate_protocol.get("prefill_mode") in {"auto", "chunk"}
    )
    if not same_weights or not modes_ok:
        return {
            "status": "not_applicable",
            "reason": (
                "requires identical local weights, step baseline, and auto/chunk candidate"
            ),
            "passed": False,
        }

    # To time the prefix after N seconds of carried state, inspect chunk N/5
    # in a run long enough to contain the following chunk.
    probes = {5.0: (10.0, 1), 10.0: (20.0, 2), 20.0: (40.0, 4), 40.0: (80.0, 8)}
    rows = []
    for context_seconds, (audio_seconds, chunk_index) in probes.items():
        baseline_row = baseline_rows.get(audio_seconds)
        candidate_row = candidate_rows.get(audio_seconds)
        if (
            baseline_row is None
            or candidate_row is None
            or baseline_row.get("status", "ok") != "ok"
            or candidate_row.get("status", "ok") != "ok"
        ):
            rows.append(
                {
                    "context_seconds": context_seconds,
                    "status": "missing_or_oom",
                    "passed": False,
                }
            )
            continue

        def prefix_times(row: dict) -> list[float]:
            result = []
            for run in row.get("stable_runs", []):
                chunks = (run.get("generation_profile") or {}).get("chunks", [])
                if len(chunks) > chunk_index:
                    result.append(
                        float(chunks[chunk_index].get("prefix_prefill_seconds", 0))
                    )
            return result

        baseline_times = prefix_times(baseline_row)
        candidate_times = prefix_times(candidate_row)
        baseline_hashes = {
            run.get("token_stream_sha256")
            for run in baseline_row.get("stable_runs", [])
        }
        candidate_hashes = {
            run.get("token_stream_sha256")
            for run in candidate_row.get("stable_runs", [])
        }
        tokens_match = (
            None not in baseline_hashes
            and len(baseline_hashes) == 1
            and baseline_hashes == candidate_hashes
        )
        if not baseline_times or not candidate_times:
            speedup = None
        else:
            candidate_median = statistics.median(candidate_times)
            speedup = (
                statistics.median(baseline_times) / candidate_median
                if candidate_median > 0
                else None
            )
        passed = bool(
            speedup is not None and speedup >= minimum_speedup and tokens_match
        )
        rows.append(
            {
                "context_seconds": context_seconds,
                "status": "ok",
                "step_prefix_median_seconds": (
                    statistics.median(baseline_times) if baseline_times else None
                ),
                "chunk_prefix_median_seconds": (
                    statistics.median(candidate_times) if candidate_times else None
                ),
                "speedup": speedup,
                "decode_tokens_identical": tokens_match,
                "passed": passed,
            }
        )
    return {
        "status": "ok",
        "minimum_speedup": minimum_speedup,
        "required_context_seconds": sorted(probes),
        "probes": rows,
        "passed": all(row["passed"] for row in rows),
    }


@app.command("compare-efficiency")
def compare_efficiency(
    baseline: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidate: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
) -> None:
    """Compare matched length-scaling curves and preserve OOM observations."""

    baseline_payload = json.loads(baseline.read_text())
    candidate_payload = json.loads(candidate.read_text())
    baseline_protocol = baseline_payload.get("protocol", {})
    candidate_protocol = candidate_payload.get("protocol", {})
    matched_fields = (
        "audio",
        "sample_rate",
        "batch_size",
        "weight_dtype",
        "cuda_autocast_dtype",
        "decoding",
        "long_context",
        "prelude_forcing",
        "warmup_runs",
        "stable_repeats",
        "device",
    )
    mismatches = [
        field
        for field in matched_fields
        if baseline_protocol.get(field) != candidate_protocol.get(field)
    ]
    if mismatches:
        raise typer.BadParameter(
            f"efficiency protocols differ on matched fields: {mismatches}"
        )

    baseline_rows = _benchmark_rows(baseline_payload)
    candidate_rows = _benchmark_rows(candidate_payload)
    if baseline_rows.keys() != candidate_rows.keys():
        raise typer.BadParameter("efficiency benchmarks use different length grids")
    successful_seconds = sorted(
        seconds
        for seconds in baseline_rows
        if baseline_rows[seconds].get("status", "ok") == "ok"
        and candidate_rows[seconds].get("status", "ok") == "ok"
    )
    comparisons = []
    for seconds in successful_seconds:
        baseline_row = baseline_rows[seconds]
        candidate_row = candidate_rows[seconds]
        comparisons.append(
            {
                "audio_seconds": seconds,
                "baseline": baseline_row["stable_summary"],
                "candidate": candidate_row["stable_summary"],
                "candidate_over_baseline": {
                    metric: (
                        float(candidate_row["stable_summary"][metric]["mean"])
                        / float(baseline_row["stable_summary"][metric]["mean"])
                        if float(baseline_row["stable_summary"][metric]["mean"]) > 0
                        else None
                    )
                    for metric in (
                        "elapsed_seconds",
                        "real_time_factor",
                        "peak_memory_allocated_bytes",
                        "peak_memory_reserved_bytes",
                    )
                },
            }
        )

    scaling = None
    if len(successful_seconds) >= 2:
        first_seconds, last_seconds = successful_seconds[0], successful_seconds[-1]
        baseline_first, baseline_last = (
            baseline_rows[first_seconds],
            baseline_rows[last_seconds],
        )
        candidate_first, candidate_last = (
            candidate_rows[first_seconds],
            candidate_rows[last_seconds],
        )
        baseline_time_growth = _growth(baseline_first, baseline_last, "elapsed_seconds")
        candidate_time_growth = _growth(
            candidate_first, candidate_last, "elapsed_seconds"
        )
        baseline_memory_growth = _growth(
            baseline_first, baseline_last, "peak_memory_allocated_bytes"
        )
        candidate_memory_growth = _growth(
            candidate_first, candidate_last, "peak_memory_allocated_bytes"
        )
        time_better = (
            baseline_time_growth is not None
            and candidate_time_growth is not None
            and candidate_time_growth < baseline_time_growth
        )
        memory_better = (
            baseline_memory_growth is not None
            and candidate_memory_growth is not None
            and candidate_memory_growth < baseline_memory_growth
        )
        scaling = {
            "first_audio_seconds": first_seconds,
            "last_common_audio_seconds": last_seconds,
            "baseline_elapsed_growth": baseline_time_growth,
            "candidate_elapsed_growth": candidate_time_growth,
            "baseline_allocated_memory_growth": baseline_memory_growth,
            "candidate_allocated_memory_growth": candidate_memory_growth,
            "runtime_growth_better": time_better,
            "memory_growth_better": memory_better,
            "common_range_gate_passed": time_better and memory_better,
        }

    baseline_oom = sorted(
        seconds for seconds, row in baseline_rows.items() if row.get("status") == "oom"
    )
    candidate_oom = sorted(
        seconds for seconds, row in candidate_rows.items() if row.get("status") == "oom"
    )
    candidate_survives_baseline_oom = any(
        candidate_rows[seconds].get("status", "ok") == "ok" for seconds in baseline_oom
    )
    result = {
        "protocol": {
            "baseline": str(baseline),
            "candidate": str(candidate),
            "matched_fields": list(matched_fields),
            "claim_boundary": (
                "H3 requires measured runtime and allocated-memory growth; "
                "this descriptive gate is not a quality result."
            ),
        },
        "comparisons": comparisons,
        "scaling": scaling,
        "oom": {
            "baseline_seconds": baseline_oom,
            "candidate_seconds": candidate_oom,
            "candidate_survives_baseline_oom": candidate_survives_baseline_oom,
        },
        "chunk_prefill_gate": _chunk_prefill_gate(
            baseline_payload,
            candidate_payload,
            baseline_rows,
            candidate_rows,
        ),
        "h3_descriptive_gate_passed": bool(
            scaling and scaling["common_range_gate_passed"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
