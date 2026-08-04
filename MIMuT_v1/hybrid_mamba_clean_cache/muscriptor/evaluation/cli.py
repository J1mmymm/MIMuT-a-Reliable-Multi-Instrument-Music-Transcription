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
            "formal seed aggregation requires checkpoint-verified seed meë¯z¶‰žËkºwµç@€€€€€€€€€€€€€€€€€€€‰•ÉÉ½ÈˆèÍÑÈ¡•ÉÉ½È¤¹ÍÁ±¥Ñ±¥¹•Ì ¥lÁt°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€ÍÑ…‰±•}ÍÕµµ…Éä€ôì(€€€€€€€€€€€­•äè}ÍÑ…‰±•}ÍÕµµ…Éä¡É½Ým­•åt™½ÈÉ½Ü¥¸ÍÑ…‰±”¤(€€€€€€€€€€€™½È­•ä¥¸€ (€€€€€€€€€€€€€€€€‰•±…ÁÍ•‘}Í•½¹‘Ìˆ°(€€€€€€€€€€€€€€€€‰É•…±}Ñ¥µ•}™…Ñ½Èˆ°(€€€€€€€€€€€€€€€€‰…Õ‘¥½}Í•½¹‘Í}Á•É}Í•½¹ˆ°(€€€€€€€€€€€€€€€€‰Á•…­}µ•µ½Éå}…±±½…Ñ•‘}‰åÑ•Ìˆ°(€€€€€€€€€€€€€€€€‰Á•…­}µ•µ½Éå}É•Í•ÉÙ•‘}‰åÑ•Ìˆ°(€€€€€€€€€€€€€€€€‰µ•…¹}¡Õ¹­}±…Ñ•¹å}Í•½¹‘Ìˆ°(€€€€€€€€€€€€¤(€€€€€€€ô(€€€€€€€ÍÑ…Ñ•}‰åÑ•Ì€ôl(€€€€€€€€€€€™±½…Ð ¡É½Ü¹•Ð ‰ÍÑÉ•…µ¥¹}ÍÑ…Ñ”ˆ¤½Èíô¤¹•Ð ‰‰åÑ•Ìˆ°€À¤¤™½ÈÉ½Ü¥¸ÍÑ…‰±”(€€€€€€€t(€€€€€€€…¡•}±•¹Ñ¡Ì€ôl(€€€€€€€€€€€™±½…Ð (€€€€€€€€€€€€€€€µ…à (€€€€€€€€€€€€€€€€€€€€¡É½Ü¹•Ð ‰ÍÑÉ•…µ¥¹}ÍÑ…Ñ”ˆ¤½Èíô¤¹•Ð ‰±½…±}…¡•}±•¹Ñ¡Ìˆ°mt¤(€€€€€€€€€€€€€€€€€€€½ÈlÁt(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€¤(€€€€€€€€€€€™½ÈÉ½Ü¥¸ÍÑ…‰±”(€€€€€€€t(€€€€€€€ÍÑ…‰±•}ÍÕµµ…Éål‰ÍÑÉ•…µ¥¹}ÍÑ…Ñ•}‰åÑ•Ì‰t€ô}ÍÑ…‰±•}ÍÕµµ…Éä¡ÍÑ…Ñ•}‰åÑ•Ì¤(€€€€€€€ÍÑ…‰±•}ÍÕµµ…Éål‰µ…á¥µÕµ}±½…±}…¡•}±•¹Ñ ‰t€ô}ÍÑ…‰±•}ÍÕµµ…Éä¡…¡•}±•¹Ñ¡Ì¤(€€€€€€€™½ÈÁÉ½™¥±•}­•ä¥¸€ (€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Í•½¹‘Ìˆ°(€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Í•½¹‘Ìˆ°(€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Ñ½­•¹Ìˆ°(€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Ñ½­•¹Ìˆ°(€€€€€€€€€€€€‰•¹•É…Ñ•‘}ÍÑ•ÁÌˆ°(€€€€€€€€¤è(€€€€€€€€€€€Ù…±Õ•Ì€ôl(€€€€€€€€€€€€€€€™±½…Ð ¡É½Ü¹•Ð ‰•¹•É…Ñ¥½¹}ÁÉ½™¥±”ˆ¤½Èíô¤¹•Ð¡ÁÉ½™¥±•}­•ä°€À¤¤(€€€€€€€€€€€€€€€™½ÈÉ½Ü¥¸ÍÑ…‰±”(€€€€€€€€€€€t(€€€€€€€€€€€ÍÑ…‰±•}ÍÕµµ…ÉåmÁÉ½™¥±•}­•åt€ô}ÍÑ…‰±•}ÍÕµµ…Éä¡Ù…±Õ•Ì¤((€€€€€€€‘•˜ÁÉ½™¥±•}É…Ñ”¡É½Üè‘¥Ð°Ñ½­•¹}­•äèÍÑÈ°Í•½¹‘Í}­•äèÍÑÈ¤€´ø™±½…Ðè(€€€€€€€€€€€ÁÉ½™¥±”€ôÉ½Ü¹•Ð ‰•¹•É…Ñ¥½¹}ÁÉ½™¥±”ˆ¤½Èíô(€€€€€€€€€€€Í•½¹‘Í}Ù…±Õ”€ô™±½…Ð¡ÁÉ½™¥±”¹•Ð¡Í•½¹‘Í}­•ä°€À¤¤(€€€€€€€€€€€É•ÑÕÉ¸€ (€€€€€€€€€€€€€€€™±½…Ð¡ÁÉ½™¥±”¹•Ð¡Ñ½­•¹}­•ä°€À¤¤€¼Í•½¹‘Í}Ù…±Õ”(€€€€€€€€€€€€€€€¥˜Í•½¹‘Í}Ù…±Õ”(€€€€€€€€€€€€€€€•±Í”€À¸À(€€€€€€€€€€€€¤((€€€€€€€ÍÑ…‰±•}ÍÕµµ…Éål‰ÁÉ•™¥á}ÁÉ•™¥±±}Ñ½­•¹Í}Á•É}Í•½¹‰t€ô}ÍÑ…‰±•}ÍÕµµ…Éä (€€€€€€€€€€€ÁÉ½™¥±•}É…Ñ” (€€€€€€€€€€€€€€€É½Ü°€‰ÁÉ•™¥á}ÁÉ•™¥±±}Ñ½­•¹Ìˆ°€‰ÁÉ•™¥á}ÁÉ•™¥±±}Í•½¹‘Ìˆ(€€€€€€€€€€€€¤(€€€€€€€€€€€™½ÈÉ½Ü¥¸ÍÑ…‰±”(€€€€€€€€¤(€€€€€€€ÍÑ…‰±•}ÍÕµµ…Éål‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Ñ½­•¹Í}Á•É}Í•½¹‰t€ô}ÍÑ…‰±•}ÍÕµµ…Éä (€€€€€€€€€€€ÁÉ½™¥±•}É…Ñ” (€€€€€€€€€€€€€€€É½Ü°(€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Ñ½­•¹Ìˆ°(€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Í•½¹‘Ìˆ°(€€€€€€€€€€€€¤(€€€€€€€€€€€™½ÈÉ½Ü¥¸ÍÑ…‰±”(€€€€€€€€¤(€€€€€€€É½ÝÌ¹…ÁÁ•¹ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰±•¹Ñ¡}±…‰•°ˆè±…‰•°°(€€€€€€€€€€€€€€€€‰…Õ‘¥½}Í•½¹‘ÌˆèÍ•½¹‘Ì°(€€€€€€€€€€€€€€€€‰Í½ÕÉ•}µ½‘”ˆè€ ‰™Õ±°ˆ¥˜±…‰•°€ôô€‰™Õ±°ˆ•±Í”€‰É•Á•…Ñ}½É}ÑÉ¥µ}ÁÉ•™¥àˆ¤°(€€€€€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰½¬ˆ°(€€€€€€€€€€€€€€€€‰½±‘}ÉÕ¸ˆè½±°(€€€€€€€€€€€€€€€€‰ÍÑ…‰±•}ÉÕ¹ÌˆèÍÑ…‰±”°(€€€€€€€€€€€€€€€€‰ÍÑ…‰±•}ÍÕµµ…ÉäˆèÍÑ…‰±•}ÍÕµµ…Éä°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€‘•Ù¥•}¥¹™¼€ôì‰É•ÅÕ•ÍÑ•ˆè‘•Ù¥”°€‰Ñ½É ˆèÑ½É ¹}}Ù•ÉÍ¥½¹}}ô(€€€¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤…¹Ñ½É ¹‘•Ù¥”¡‘•Ù¥”¤¹ÑåÁ”€ôô€‰Õ‘„ˆè(€€€€€€€‘•Ù¥•}¥¹™¼¹ÕÁ‘…Ñ” (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰¹…µ”ˆèÑ½É ¹Õ‘„¹•Ñ}‘•Ù¥•}¹…µ”¡Ñ½É ¹‘•Ù¥”¡‘•Ù¥”¤¤°(€€€€€€€€€€€€€€€€‰Õ‘„ˆèÑ½É ¹Ù•ÉÍ¥½¸¹Õ‘„°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€½É•}µ½‘•°€ô•Ñ…ÑÑÈ¡µ½‘•°°€‰}µ½‘•°ˆ°9½¹”¤(€€€Á…É…µ•Ñ•É}½Õ¹Ð€ô€ (€€€€€€€ÍÕ´¡Á…É…µ•Ñ•È¹¹Õµ•° ¤™½ÈÁ…É…µ•Ñ•È¥¸½É•}µ½‘•°¹Á…É…µ•Ñ•ÉÌ ¤¤(€€€€€€€¥˜½É•}µ½‘•°¥Ì¹½Ð9½¹”(€€€€€€€•±Í”9½¹”(€€€€¤(€€€Á…å±½…€ôì(€€€€€€€€‰ÁÉ½Ñ½½°ˆèì(€€€€€€€€€€€€‰µ½‘•°ˆèµ½‘•±}Á…Ñ °(€€€€€€€€€€€€‰µ½‘•±}Ý•¥¡ÑÍ}Í¡„ÈÔØˆè}±½…±}Ý•¥¡ÑÍ}Í¡„ÈÔØ¡µ½‘•±}Á…Ñ ¤°(€€€€€€€€€€€€‰µ½‘•±}Á…É…µ•Ñ•É}½Õ¹ÐˆèÁ…É…µ•Ñ•É}½Õ¹Ð°(€€€€€€€€€€€€‰…Õ‘¥¼ˆèÍÑÈ¡…Õ‘¥¼¤°(€€€€€€€€€€€€‰Í…µÁ±•}É…Ñ”ˆè€ÄÙ|ÀÀÀ°(€€€€€€€€€€€€‰‰…Ñ¡}Í¥é”ˆè€Ä°(€€€€€€€€€€€€‰Ý•¥¡Ñ}‘ÑåÁ”ˆè‘ÑåÁ”°(€€€€€€€€€€€€‰Õ‘…}…ÕÑ½…ÍÑ}‘ÑåÁ”ˆè€ (€€€€€€€€€€€€€€€€‰™±½…ÐÄØˆ(€€€€€€€€€€€€€€€¥˜Ñ½É ¹Õ‘„¹¥Í}…Ù…¥±…‰±” ¤…¹Ñ½É ¹‘•Ù¥”¡‘•Ù¥”¤¹ÑåÁ”€ôô€‰Õ‘„ˆ(€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€¤°(€€€€€€€€€€€€‰‘•½‘¥¹œˆèì‰Í…µÁ±¥¹œˆè…±Í”°€‰‰•…µ}Í¥é”ˆè€Åô°(€€€€€€€€€€€€‰±½¹}½¹Ñ•áÐˆè±½¹}½¹Ñ•áÐ°(€€€€€€€€€€€€‰ÁÉ•±Õ‘•}™½É¥¹œˆèÁÉ•±Õ‘•}™½É¥¹œ°(€€€€€€€€€€€€‰ÁÉ•™¥±±}µ½‘”ˆè•Ñ…ÑÑÈ (€€€€€€€€€€€€€€€•Ñ…ÑÑÈ¡½É•}µ½‘•°°€‰µ½‘•±}½¹™¥œˆ°9½¹”¤°€‰ÁÉ•™¥±±}µ½‘”ˆ°9½¹”(€€€€€€€€€€€€¤°(€€€€€€€€€€€€‰½É…±•}¥¹ÍÑÉÕµ•¹ÑÌˆè…±Í”°(€€€€€€€€€€€€‰Ý…ÉµÕÁ}ÉÕ¹ÌˆèÝ…ÉµÕÁ}ÉÕ¹Ì°(€€€€€€€€€€€€‰ÍÑ…‰±•}É•Á•…ÑÌˆèÉ•Á•…ÑÌ°(€€€€€€€€€€€€‰‘•Ù¥”ˆè‘•Ù¥•}¥¹™¼°(€€€€€€€ô°(€€€€€€€€‰±•¹Ñ¡ÌˆèÉ½ÝÌ°(€€€ô(€€€½ÕÑÁÕÐ¹Á…É•¹Ð¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€½ÕÑÁÕÐ¹ÝÉ¥Ñ•}Ñ•áÐ¡©Í½¸¹‘ÕµÁÌ¡Á…å±½…°¥¹‘•¹ÐôÈ¤€¬€‰q¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(()‘•˜}‰•¹¡µ…É­}É½ÝÌ¡Á…å±½…è‘¥Ð¤€´ø‘¥Ñm™±½…Ð°‘¥Ñtè(€€€É•ÑÕÉ¸ì(€€€€€€€™±½…Ð¡É½Ýl‰…Õ‘¥½}Í•½¹‘Ì‰t¤èÉ½Ü(€€€€€€€™½ÈÉ½Ü¥¸Á…å±½…¹•Ð ‰±•¹Ñ¡Ìˆ°mt¤(€€€€€€€¥˜É½Ü¹•Ð ‰±•¹Ñ¡}±…‰•°ˆ¤€„ô€‰™Õ±°ˆ(€€€ô(()‘•˜}É½ÝÑ ¡™¥ÉÍÐè‘¥Ð°±…ÍÐè‘¥Ð°µ•ÑÉ¥ŒèÍÑÈ¤€´ø™±½…Ðð9½¹”è(€€€¥¹¥Ñ¥…°€ô™±½…Ð¡™¥ÉÍÑl‰ÍÑ…‰±•}ÍÕµµ…Éä‰umµ•ÑÉ¥ul‰µ•…¸‰t¤(€€€™¥¹…°€ô™±½…Ð¡±…ÍÑl‰ÍÑ…‰±•}ÍÕµµ…Éä‰umµ•ÑÉ¥ul‰µ•…¸‰t¤(€€€É•ÑÕÉ¸™¥¹…°€¼¥¹¥Ñ¥…°¥˜¥¹¥Ñ¥…°€ø€À•±Í”9½¹”(()‘•˜}¡Õ¹­}ÁÉ•™¥±±}…Ñ” (€€€‰…Í•±¥¹•}Á…å±½…è‘¥Ð°(€€€…¹‘¥‘…Ñ•}Á…å±½…è‘¥Ð°(€€€‰…Í•±¥¹•}É½ÝÌè‘¥Ñm™±½…Ð°‘¥Ñt°(€€€…¹‘¥‘…Ñ•}É½ÝÌè‘¥Ñm™±½…Ð°‘¥Ñt°(€€€€¨°(€€€µ¥¹¥µÕµ}ÍÁ••‘ÕÀè™±½…Ð€ô€Ä¸ÈÔ°(¤€´ø‘¥Ðè(€€€€ˆˆ‰Ù…±Õ…Ñ”…ÉÉ¥•µÁÉ•™¥àÍÁ••…Ð€Ô¼ÄÀ¼ÈÀ¼ÐÀÍ•½¹‘Ì½˜ÁÉ¥½ÈÍÑ…Ñ”¸ˆˆˆ((€€€‰…Í•±¥¹•}ÁÉ½Ñ½½°€ô‰…Í•±¥¹•}Á…å±½…¹•Ð ‰ÁÉ½Ñ½½°ˆ°íô¤(€€€…¹‘¥‘…Ñ•}ÁÉ½Ñ½½°€ô…¹‘¥‘…Ñ•}Á…å±½…¹•Ð ‰ÁÉ½Ñ½½°ˆ°íô¤(€€€Í…µ•}Ý•¥¡ÑÌ€ô€ (€€€€€€€‰…Í•±¥¹•}ÁÉ½Ñ½½°¹•Ð ‰µ½‘•±}Ý•¥¡ÑÍ}Í¡„ÈÔØˆ¤¥Ì¹½Ð9½¹”(€€€€€€€…¹‰…Í•±¥¹•}ÁÉ½Ñ½½°¹•Ð ‰µ½‘•±}Ý•¥¡ÑÍ}Í¡„ÈÔØˆ¤(€€€€€€€€ôô…¹‘¥‘…Ñ•}ÁÉ½Ñ½½°¹•Ð ‰µ½‘•±}Ý•¥¡ÑÍ}Í¡„ÈÔØˆ¤(€€€€¤(€€€µ½‘•Í}½¬€ô‰…Í•±¥¹•}ÁÉ½Ñ½½°¹•Ð ‰ÁÉ•™¥±±}µ½‘”ˆ¤€ôô€‰ÍÑ•Àˆ…¹€ (€€€€€€€…¹‘¥‘…Ñ•}ÁÉ½Ñ½½°¹•Ð ‰ÁÉ•™¥±±}µ½‘”ˆ¤¥¸ì‰…ÕÑ¼ˆ°€‰¡Õ¹¬‰ô(€€€€¤(€€€¥˜¹½ÐÍ…µ•}Ý•¥¡ÑÌ½È¹½Ðµ½‘•Í}½¬è(€€€€€€€É•ÑÕÉ¸ì(€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰¹½Ñ}…ÁÁ±¥…‰±”ˆ°(€€€€€€€€€€€€‰É•…Í½¸ˆè€ (€€€€€€€€€€€€€€€€‰É•ÅÕ¥É•Ì¥‘•¹Ñ¥…°±½…°Ý•¥¡ÑÌ°ÍÑ•À‰…Í•±¥¹”°…¹…ÕÑ¼½¡Õ¹¬…¹‘¥‘…Ñ”ˆ(€€€€€€€€€€€€¤°(€€€€€€€€€€€€‰Á…ÍÍ•ˆè…±Í”°(€€€€€€€ô((€€€€ŒQ¼Ñ¥µ”Ñ¡”ÁÉ•™¥à…™Ñ•È8Í•½¹‘Ì½˜…ÉÉ¥•ÍÑ…Ñ”°¥¹ÍÁ•Ð¡Õ¹¬8¼Ô(€€€€Œ¥¸„ÉÕ¸±½¹œ•¹½Õ Ñ¼½¹Ñ…¥¸Ñ¡”™½±±½Ý¥¹œ¡Õ¹¬¸(€€€ÁÉ½‰•Ì€ôìÔ¸Àè€ ÄÀ¸À°€Ä¤°€ÄÀ¸Àè€ ÈÀ¸À°€È¤°€ÈÀ¸Àè€ ÐÀ¸À°€Ð¤°€ÐÀ¸Àè€ àÀ¸À°€à¥ô(€€€É½ÝÌ€ômt(€€€™½È½¹Ñ•áÑ}Í•½¹‘Ì°€¡…Õ‘¥½}Í•½¹‘Ì°¡Õ¹­}¥¹‘•à¤¥¸ÁÉ½‰•Ì¹¥Ñ•µÌ ¤è(€€€€€€€‰…Í•±¥¹•}É½Ü€ô‰…Í•±¥¹•}É½ÝÌ¹•Ð¡…Õ‘¥½}Í•½¹‘Ì¤(€€€€€€€…¹‘¥‘…Ñ•}É½Ü€ô…¹‘¥‘…Ñ•}É½ÝÌ¹•Ð¡…Õ‘¥½}Í•½¹‘Ì¤(€€€€€€€¥˜€ (€€€€€€€€€€€‰…Í•±¥¹•}É½Ü¥Ì9½¹”(€€€€€€€€€€€½È…¹‘¥‘…Ñ•}É½Ü¥Ì9½¹”(€€€€€€€€€€€½È‰…Í•±¥¹•}É½Ü¹•Ð ‰ÍÑ…ÑÕÌˆ°€‰½¬ˆ¤€„ô€‰½¬ˆ(€€€€€€€€€€€½È…¹‘¥‘…Ñ•}É½Ü¹•Ð ‰ÍÑ…ÑÕÌˆ°€‰½¬ˆ¤€„ô€‰½¬ˆ(€€€€€€€€¤è(€€€€€€€€€€€É½ÝÌ¹…ÁÁ•¹ (€€€€€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€€€€€‰½¹Ñ•áÑ}Í•½¹‘Ìˆè½¹Ñ•áÑ}Í•½¹‘Ì°(€€€€€€€€€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰µ¥ÍÍ¥¹}½É}½½´ˆ°(€€€€€€€€€€€€€€€€€€€€‰Á…ÍÍ•ˆè…±Í”°(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€¤(€€€€€€€€€€€½¹Ñ¥¹Õ”((€€€€€€€‘•˜ÁÉ•™¥á}Ñ¥µ•Ì¡É½Üè‘¥Ð¤€´ø±¥ÍÑm™±½…Ñtè(€€€€€€€€€€€É•ÍÕ±Ð€ômt(€€€€€€€€€€€™½ÈÉÕ¸¥¸É½Ü¹•Ð ‰ÍÑ…‰±•}ÉÕ¹Ìˆ°mt¤è(€€€€€€€€€€€€€€€¡Õ¹­Ì€ô€¡ÉÕ¸¹•Ð ‰•¹•É…Ñ¥½¹}ÁÉ½™¥±”ˆ¤½Èíô¤¹•Ð ‰¡Õ¹­Ìˆ°mt¤(€€€€€€€€€€€€€€€¥˜±•¸¡¡Õ¹­Ì¤€ø¡Õ¹­}¥¹‘•àè(€€€€€€€€€€€€€€€€€€€É•ÍÕ±Ð¹…ÁÁ•¹ (€€€€€€€€€€€€€€€€€€€€€€€™±½…Ð¡¡Õ¹­Ím¡Õ¹­}¥¹‘•át¹•Ð ‰ÁÉ•™¥á}ÁÉ•™¥±±}Í•½¹‘Ìˆ°€À¤¤(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€É•ÑÕÉ¸É•ÍÕ±Ð((€€€€€€€‰…Í•±¥¹•}Ñ¥µ•Ì€ôÁÉ•™¥á}Ñ¥µ•Ì¡‰…Í•±¥¹•}É½Ü¤(€€€€€€€…¹‘¥‘…Ñ•}Ñ¥µ•Ì€ôÁÉ•™¥á}Ñ¥µ•Ì¡…¹‘¥‘…Ñ•}É½Ü¤(€€€€€€€‰…Í•±¥¹•}¡…Í¡•Ì€ôì(€€€€€€€€€€€ÉÕ¸¹•Ð ‰Ñ½­•¹}ÍÑÉ•…µ}Í¡„ÈÔØˆ¤(€€€€€€€€€€€™½ÈÉÕ¸¥¸‰…Í•±¥¹•}É½Ü¹•Ð ‰ÍÑ…‰±•}ÉÕ¹Ìˆ°mt¤(€€€€€€€ô(€€€€€€€…¹‘¥‘…Ñ•}¡…Í¡•Ì€ôì(€€€€€€€€€€€ÉÕ¸¹•Ð ‰Ñ½­•¹}ÍÑÉ•…µ}Í¡„ÈÔØˆ¤(€€€€€€€€€€€™½ÈÉÕ¸¥¸…¹‘¥‘…Ñ•}É½Ü¹•Ð ‰ÍÑ…‰±•}ÉÕ¹Ìˆ°mt¤(€€€€€€€ô(€€€€€€€Ñ½­•¹Í}µ…Ñ €ô€ (€€€€€€€€€€€9½¹”¹½Ð¥¸‰…Í•±¥¹•}¡…Í¡•Ì(€€€€€€€€€€€…¹±•¸¡‰…Í•±¥¹•}¡…Í¡•Ì¤€ôô€Ä(€€€€€€€€€€€…¹‰…Í•±¥¹•}¡…Í¡•Ì€ôô…¹‘¥‘…Ñ•}¡…Í¡•Ì(€€€€€€€€¤(€€€€€€€¥˜¹½Ð‰…Í•±¥¹•}Ñ¥µ•Ì½È¹½Ð…¹‘¥‘…Ñ•}Ñ¥µ•Ìè(€€€€€€€€€€€ÍÁ••‘ÕÀ€ô9½¹”(€€€€€€€•±Í”è(€€€€€€€€€€€…¹‘¥‘…Ñ•}µ•‘¥…¸€ôÍÑ…Ñ¥ÍÑ¥Ì¹µ•‘¥…¸¡…¹‘¥‘…Ñ•}Ñ¥µ•Ì¤(€€€€€€€€€€€ÍÁ••‘ÕÀ€ô€ (€€€€€€€€€€€€€€€ÍÑ…Ñ¥ÍÑ¥Ì¹µ•‘¥…¸¡‰…Í•±¥¹•}Ñ¥µ•Ì¤€¼…¹‘¥‘…Ñ•}µ•‘¥…¸(€€€€€€€€€€€€€€€¥˜…¹‘¥‘…Ñ•}µ•‘¥…¸€ø€À(€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€¤(€€€€€€€Á…ÍÍ•€ô‰½½° (€€€€€€€€€€€ÍÁ••‘ÕÀ¥Ì¹½Ð9½¹”…¹ÍÁ••‘ÕÀ€øôµ¥¹¥µÕµ}ÍÁ••‘ÕÀ…¹Ñ½­•¹Í}µ…Ñ (€€€€€€€€¤(€€€€€€€É½ÝÌ¹…ÁÁ•¹ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰½¹Ñ•áÑ}Í•½¹‘Ìˆè½¹Ñ•áÑ}Í•½¹‘Ì°(€€€€€€€€€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰½¬ˆ°(€€€€€€€€€€€€€€€€‰ÍÑ•Á}ÁÉ•™¥á}µ•‘¥…¹}Í•½¹‘Ìˆè€ (€€€€€€€€€€€€€€€€€€€ÍÑ…Ñ¥ÍÑ¥Ì¹µ•‘¥…¸¡‰…Í•±¥¹•}Ñ¥µ•Ì¤¥˜‰…Í•±¥¹•}Ñ¥µ•Ì•±Í”9½¹”(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€‰¡Õ¹­}ÁÉ•™¥á}µ•‘¥…¹}Í•½¹‘Ìˆè€ (€€€€€€€€€€€€€€€€€€€ÍÑ…Ñ¥ÍÑ¥Ì¹µ•‘¥…¸¡…¹‘¥‘…Ñ•}Ñ¥µ•Ì¤¥˜…¹‘¥‘…Ñ•}Ñ¥µ•Ì•±Í”9½¹”(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€€€€€€‰ÍÁ••‘ÕÀˆèÍÁ••‘ÕÀ°(€€€€€€€€€€€€€€€€‰‘•½‘•}Ñ½­•¹Í}¥‘•¹Ñ¥…°ˆèÑ½­•¹Í}µ…Ñ °(€€€€€€€€€€€€€€€€‰Á…ÍÍ•ˆèÁ…ÍÍ•°(€€€€€€€€€€€ô(€€€€€€€€¤(€€€É•ÑÕÉ¸ì(€€€€€€€€‰ÍÑ…ÑÕÌˆè€‰½¬ˆ°(€€€€€€€€‰µ¥¹¥µÕµ}ÍÁ••‘ÕÀˆèµ¥¹¥µÕµ}ÍÁ••‘ÕÀ°(€€€€€€€€‰É•ÅÕ¥É•‘}½¹Ñ•áÑ}Í•½¹‘ÌˆèÍ½ÉÑ•¡ÁÉ½‰•Ì¤°(€€€€€€€€‰ÁÉ½‰•ÌˆèÉ½ÝÌ°(€€€€€€€€‰Á…ÍÍ•ˆè…±°¡É½Ýl‰Á…ÍÍ•‰t™½ÈÉ½Ü¥¸É½ÝÌ¤°(€€€ô(()…ÁÀ¹½µµ…¹ ‰½µÁ…É”µ•™™¥¥•¹äˆ¤)‘•˜½µÁ…É•}•™™¥¥•¹ä (€€€‰…Í•±¥¹”è¹¹½Ñ…Ñ•‘mA…Ñ °ÑåÁ•È¹=ÁÑ¥½¸¡•á¥ÍÑÌõQÉÕ”°‘¥É}½­…äõ…±Í”¥t°(€€€…¹‘¥‘…Ñ”è¹¹½Ñ…Ñ•‘mA…Ñ °ÑåÁ•È¹=ÁÑ¥½¸¡•á¥ÍÑÌõQÉÕ”°‘¥É}½­…äõ…±Í”¥t°(€€€½ÕÑÁÕÐè¹¹½Ñ…Ñ•‘mA…Ñ °ÑåÁ•È¹=ÁÑ¥½¸ ˆ´µ½ÕÑÁÕÐˆ°€ˆµ¼ˆ¥t°(¤€´ø9½¹”è(€€€€ˆˆ‰½µÁ…É”µ…Ñ¡•±•¹Ñ µÍ…±¥¹œÕÉÙ•Ì…¹ÁÉ•Í•ÉÙ”==4½‰Í•ÉÙ…Ñ¥½¹Ì¸ˆˆˆ((€€€‰…Í•±¥¹•}Á…å±½…€ô©Í½¸¹±½…‘Ì¡‰…Í•±¥¹”¹É•…‘}Ñ•áÐ ¤¤(€€€…¹‘¥‘…Ñ•}Á…å±½…€ô©Í½¸¹±½…‘Ì¡…¹‘¥‘…Ñ”¹É•…‘}Ñ•áÐ ¤¤(€€€‰…Í•±¥¹•}ÁÉ½Ñ½½°€ô‰…Í•±¥¹•}Á…å±½…¹•Ð ‰ÁÉ½Ñ½½°ˆ°íô¤(€€€…¹‘¥‘…Ñ•}ÁÉ½Ñ½½°€ô…¹‘¥‘…Ñ•}Á…å±½…¹•Ð ‰ÁÉ½Ñ½½°ˆ°íô¤(€€€µ…Ñ¡•‘}™¥•±‘Ì€ô€ (€€€€€€€€‰…Õ‘¥¼ˆ°(€€€€€€€€‰Í…µÁ±•}É…Ñ”ˆ°(€€€€€€€€‰‰…Ñ¡}Í¥é”ˆ°(€€€€€€€€‰Ý•¥¡Ñ}‘ÑåÁ”ˆ°(€€€€€€€€‰Õ‘…}…ÕÑ½…ÍÑ}‘ÑåÁ”ˆ°(€€€€€€€€‰‘•½‘¥¹œˆ°(€€€€€€€€‰±½¹}½¹Ñ•áÐˆ°(€€€€€€€€‰ÁÉ•±Õ‘•}™½É¥¹œˆ°(€€€€€€€€‰Ý…ÉµÕÁ}ÉÕ¹Ìˆ°(€€€€€€€€‰ÍÑ…‰±•}É•Á•…ÑÌˆ°(€€€€€€€€‰‘•Ù¥”ˆ°(€€€€¤(€€€µ¥Íµ…Ñ¡•Ì€ôl(€€€€€€€™¥•±(€€€€€€€™½È™¥•±¥¸µ…Ñ¡•‘}™¥•±‘Ì(€€€€€€€¥˜‰…Í•±¥¹•}ÁÉ½Ñ½½°¹•Ð¡™¥•±¤€„ô…¹‘¥‘…Ñ•}ÁÉ½Ñ½½°¹•Ð¡™¥•±¤(€€€t(€€€¥˜µ¥Íµ…Ñ¡•Ìè(€€€€€€€É…¥Í”ÑåÁ•È¹	…‘A…É…µ•Ñ•È (€€€€€€€€€€€˜‰•™™¥¥•¹äÁÉ½Ñ½½±Ì‘¥™™•È½¸µ…Ñ¡•™¥•±‘Ìèíµ¥Íµ…Ñ¡•Íôˆ(€€€€€€€€¤((€€€‰…Í•±¥¹•}É½ÝÌ€ô}‰•¹¡µ…É­}É½ÝÌ¡‰…Í•±¥¹•}Á…å±½…¤(€€€…¹‘¥‘…Ñ•}É½ÝÌ€ô}‰•¹¡µ…É­}É½ÝÌ¡…¹‘¥‘…Ñ•}Á…å±½…¤(€€€¥˜‰…Í•±¥¹•}É½ÝÌ¹­•åÌ ¤€„ô…¹‘¥‘…Ñ•}É½ÝÌ¹­•åÌ ¤è(€€€€€€€É…¥Í”ÑåÁ•È¹	…‘A…É…µ•Ñ•È ‰•™™¥¥•¹ä‰•¹¡µ…É­ÌÕÍ”‘¥™™•É•¹Ð±•¹Ñ É¥‘Ìˆ¤(€€€ÍÕ•ÍÍ™Õ±}Í•½¹‘Ì€ôÍ½ÉÑ• (€€€€€€€Í•½¹‘Ì(€€€€€€€™½ÈÍ•½¹‘Ì¥¸‰…Í•±¥¹•}É½ÝÌ(€€€€€€€¥˜‰…Í•±¥¹•}É½ÝÍmÍ•½¹‘Ít¹•Ð ‰ÍÑ…ÑÕÌˆ°€‰½¬ˆ¤€ôô€‰½¬ˆ(€€€€€€€…¹…¹‘¥‘…Ñ•}É½ÝÍmÍ•½¹‘Ít¹•Ð ‰ÍÑ…ÑÕÌˆ°€‰½¬ˆ¤€ôô€‰½¬ˆ(€€€€¤(€€€½µÁ…É¥Í½¹Ì€ômt(€€€™½ÈÍ•½¹‘Ì¥¸ÍÕ•ÍÍ™Õ±}Í•½¹‘Ìè(€€€€€€€‰…Í•±¥¹•}É½Ü€ô‰…Í•±¥¹•}É½ÝÍmÍ•½¹‘Ít(€€€€€€€…¹‘¥‘…Ñ•}É½Ü€ô…¹‘¥‘…Ñ•}É½ÝÍmÍ•½¹‘Ít(€€€€€€€½µÁ…É¥Í½¹Ì¹…ÁÁ•¹ (€€€€€€€€€€€ì(€€€€€€€€€€€€€€€€‰…Õ‘¥½}Í•½¹‘ÌˆèÍ•½¹‘Ì°(€€€€€€€€€€€€€€€€‰‰…Í•±¥¹”ˆè‰…Í•±¥¹•}É½Ýl‰ÍÑ…‰±•}ÍÕµµ…Éä‰t°(€€€€€€€€€€€€€€€€‰…¹‘¥‘…Ñ”ˆè…¹‘¥‘…Ñ•}É½Ýl‰ÍÑ…‰±•}ÍÕµµ…Éä‰t°(€€€€€€€€€€€€€€€€‰…¹‘¥‘…Ñ•}½Ù•É}‰…Í•±¥¹”ˆèì(€€€€€€€€€€€€€€€€€€€µ•ÑÉ¥Œè€ (€€€€€€€€€€€€€€€€€€€€€€€™±½…Ð¡…¹‘¥‘…Ñ•}É½Ýl‰ÍÑ…‰±•}ÍÕµµ…Éä‰umµ•ÑÉ¥ul‰µ•…¸‰t¤(€€€€€€€€€€€€€€€€€€€€€€€€¼™±½…Ð¡‰…Í•±¥¹•}É½Ýl‰ÍÑ…‰±•}ÍÕµµ…Éä‰umµ•ÑÉ¥ul‰µ•…¸‰t¤(€€€€€€€€€€€€€€€€€€€€€€€¥˜™±½…Ð¡‰…Í•±¥¹•}É½Ýl‰ÍÑ…‰±•}ÍÕµµ…Éä‰umµ•ÑÉ¥ul‰µ•…¸‰t¤€ø€À(€€€€€€€€€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€™½Èµ•ÑÉ¥Œ¥¸€ (€€€€€€€€€€€€€€€€€€€€€€€€‰•±…ÁÍ•‘}Í•½¹‘Ìˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰É•…±}Ñ¥µ•}™…Ñ½Èˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰Á•…­}µ•µ½Éå}…±±½…Ñ•‘}‰åÑ•Ìˆ°(€€€€€€€€€€€€€€€€€€€€€€€€‰Á•…­}µ•µ½Éå}É•Í•ÉÙ•‘}‰åÑ•Ìˆ°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€ô(€€€€€€€€¤((€€€Í…±¥¹œ€ô9½¹”(€€€¥˜±•¸¡ÍÕ•ÍÍ™Õ±}Í•½¹‘Ì¤€øô€Èè(€€€€€€€™¥ÉÍÑ}Í•½¹‘Ì°±…ÍÑ}Í•½¹‘Ì€ôÍÕ•ÍÍ™Õ±}Í•½¹‘ÍlÁt°ÍÕ•ÍÍ™Õ±}Í•½¹‘Íl´Åt(€€€€€€€‰…Í•±¥¹•}™¥ÉÍÐ°‰…Í•±¥¹•}±…ÍÐ€ô€ (€€€€€€€€€€€‰…Í•±¥¹•}É½ÝÍm™¥ÉÍÑ}Í•½¹‘Ít°(€€€€€€€€€€€‰…Í•±¥¹•}É½ÝÍm±…ÍÑ}Í•½¹‘Ít°(€€€€€€€€¤(€€€€€€€…¹‘¥‘…Ñ•}™¥ÉÍÐ°…¹‘¥‘…Ñ•}±…ÍÐ€ô€ (€€€€€€€€€€€…¹‘¥‘…Ñ•}É½ÝÍm™¥ÉÍÑ}Í•½¹‘Ít°(€€€€€€€€€€€…¹‘¥‘…Ñ•}É½ÝÍm±…ÍÑ}Í•½¹‘Ít°(€€€€€€€€¤(€€€€€€€‰…Í•±¥¹•}Ñ¥µ•}É½ÝÑ €ô}É½ÝÑ ¡‰…Í•±¥¹•}™¥ÉÍÐ°‰…Í•±¥¹•}±…ÍÐ°€‰•±…ÁÍ•‘}Í•½¹‘Ìˆ¤(€€€€€€€…¹‘¥‘…Ñ•}Ñ¥µ•}É½ÝÑ €ô}É½ÝÑ  (€€€€€€€€€€€…¹‘¥‘…Ñ•}™¥ÉÍÐ°…¹‘¥‘…Ñ•}±…ÍÐ°€‰•±…ÁÍ•‘}Í•½¹‘Ìˆ(€€€€€€€€¤(€€€€€€€‰…Í•±¥¹•}µ•µ½Éå}É½ÝÑ €ô}É½ÝÑ  (€€€€€€€€€€€‰…Í•±¥¹•}™¥ÉÍÐ°‰…Í•±¥¹•}±…ÍÐ°€‰Á•…­}µ•µ½Éå}…±±½…Ñ•‘}‰åÑ•Ìˆ(€€€€€€€€¤(€€€€€€€…¹‘¥‘…Ñ•}µ•µ½Éå}É½ÝÑ €ô}É½ÝÑ  (€€€€€€€€€€€…¹‘¥‘…Ñ•}™¥ÉÍÐ°…¹‘¥‘…Ñ•}±…ÍÐ°€‰Á•…­}µ•µ½Éå}…±±½…Ñ•‘}‰åÑ•Ìˆ(€€€€€€€€¤(€€€€€€€Ñ¥µ•}‰•ÑÑ•È€ô€ (€€€€€€€€€€€‰…Í•±¥¹•}Ñ¥µ•}É½ÝÑ ¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹…¹‘¥‘…Ñ•}Ñ¥µ•}É½ÝÑ ¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹…¹‘¥‘…Ñ•}Ñ¥µ•}É½ÝÑ €ð‰…Í•±¥¹•}Ñ¥µ•}É½ÝÑ (€€€€€€€€¤(€€€€€€€µ•µ½Éå}‰•ÑÑ•È€ô€ (€€€€€€€€€€€‰…Í•±¥¹•}µ•µ½Éå}É½ÝÑ ¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹…¹‘¥‘…Ñ•}µ•µ½Éå}É½ÝÑ ¥Ì¹½Ð9½¹”(€€€€€€€€€€€…¹…¹‘¥‘…Ñ•}µ•µ½Éå}É½ÝÑ €ð‰…Í•±¥¹•}µ•µ½Éå}É½ÝÑ (€€€€€€€€¤(€€€€€€€Í…±¥¹œ€ôì(€€€€€€€€€€€€‰™¥ÉÍÑ}…Õ‘¥½}Í•½¹‘Ìˆè™¥ÉÍÑ}Í•½¹‘Ì°(€€€€€€€€€€€€‰±…ÍÑ}½µµ½¹}…Õ‘¥½}Í•½¹‘Ìˆè±…ÍÑ}Í•½¹‘Ì°(€€€€€€€€€€€€‰‰…Í•±¥¹•}•±…ÁÍ•‘}É½ÝÑ ˆè‰…Í•±¥¹•}Ñ¥µ•}É½ÝÑ °(€€€€€€€€€€€€‰…¹‘¥‘…Ñ•}•±…ÁÍ•‘}É½ÝÑ ˆè…¹‘¥‘…Ñ•}Ñ¥µ•}É½ÝÑ °(€€€€€€€€€€€€‰‰…Í•±¥¹•}…±±½…Ñ•‘}µ•µ½Éå}É½ÝÑ ˆè‰…Í•±¥¹•}µ•µ½Éå}É½ÝÑ °(€€€€€€€€€€€€‰…¹‘¥‘…Ñ•}…±±½…Ñ•‘}µ•µ½Éå}É½ÝÑ ˆè…¹‘¥‘…Ñ•}µ•µ½Éå}É½ÝÑ °(€€€€€€€€€€€€‰ÉÕ¹Ñ¥µ•}É½ÝÑ¡}‰•ÑÑ•ÈˆèÑ¥µ•}‰•ÑÑ•È°(€€€€€€€€€€€€‰µ•µ½Éå}É½ÝÑ¡}‰•ÑÑ•Èˆèµ•µ½Éå}‰•ÑÑ•È°(€€€€€€€€€€€€‰½µµ½¹}É…¹•}…Ñ•}Á…ÍÍ•ˆèÑ¥µ•}‰•ÑÑ•È…¹µ•µ½Éå}‰•ÑÑ•È°(€€€€€€€ô((€€€‰…Í•±¥¹•}½½´€ôÍ½ÉÑ• (€€€€€€€Í•½¹‘Ì™½ÈÍ•½¹‘Ì°É½Ü¥¸‰…Í•±¥¹•}É½ÝÌ¹¥Ñ•µÌ ¤¥˜É½Ü¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰½½´ˆ(€€€€¤(€€€…¹‘¥‘…Ñ•}½½´€ôÍ½ÉÑ• (€€€€€€€Í•½¹‘Ì™½ÈÍ•½¹‘Ì°É½Ü¥¸…¹‘¥‘…Ñ•}É½ÝÌ¹¥Ñ•µÌ ¤¥˜É½Ü¹•Ð ‰ÍÑ…ÑÕÌˆ¤€ôô€‰½½´ˆ(€€€€¤(€€€…¹‘¥‘…Ñ•}ÍÕÉÙ¥Ù•Í}‰…Í•±¥¹•}½½´€ô…¹ä (€€€€€€€…¹‘¥‘…Ñ•}É½ÝÍmÍ•½¹‘Ít¹•Ð ‰ÍÑ…ÑÕÌˆ°€‰½¬ˆ¤€ôô€‰½¬ˆ™½ÈÍ•½¹‘Ì¥¸‰…Í•±¥¹•}½½´(€€€€¤(€€€É•ÍÕ±Ð€ôì(€€€€€€€€‰ÁÉ½Ñ½½°ˆèì(€€€€€€€€€€€€‰‰…Í•±¥¹”ˆèÍÑÈ¡‰…Í•±¥¹”¤°(€€€€€€€€€€€€‰…¹‘¥‘…Ñ”ˆèÍÑÈ¡…¹‘¥‘…Ñ”¤°(€€€€€€€€€€€€‰µ…Ñ¡•‘}™¥•±‘Ìˆè±¥ÍÐ¡µ…Ñ¡•‘}™¥•±‘Ì¤°(€€€€€€€€€€€€‰±…¥µ}‰½Õ¹‘…Éäˆè€ (€€€€€€€€€€€€€€€€‰ ÌÉ•ÅÕ¥É•Ìµ•…ÍÕÉ•ÉÕ¹Ñ¥µ”…¹…±±½…Ñ•µµ•µ½ÉäÉ½ÝÑ ì€ˆ(€€€€€€€€€€€€€€€€‰Ñ¡¥Ì‘•ÍÉ¥ÁÑ¥Ù”…Ñ”¥Ì¹½Ð„ÅÕ…±¥ÑäÉ•ÍÕ±Ð¸ˆ(€€€€€€€€€€€€¤°(€€€€€€€ô°(€€€€€€€€‰½µÁ…É¥Í½¹Ìˆè½µÁ…É¥Í½¹Ì°(€€€€€€€€‰Í…±¥¹œˆèÍ…±¥¹œ°(€€€€€€€€‰½½´ˆèì(€€€€€€€€€€€€‰‰…Í•±¥¹•}Í•½¹‘Ìˆè‰…Í•±¥¹•}½½´°(€€€€€€€€€€€€‰…¹‘¥‘…Ñ•}Í•½¹‘Ìˆè…¹‘¥‘…Ñ•}½½´°(€€€€€€€€€€€€‰…¹‘¥‘…Ñ•}ÍÕÉÙ¥Ù•Í}‰…Í•±¥¹•}½½´ˆè…¹‘¥‘…Ñ•}ÍÕÉÙ¥Ù•Í}‰…Í•±¥¹•}½½´°(€€€€€€€ô°(€€€€€€€€‰¡Õ¹­}ÁÉ•™¥±±}…Ñ”ˆè}¡Õ¹­}ÁÉ•™¥±±}…Ñ” (€€€€€€€€€€€‰…Í•±¥¹•}Á…å±½…°(€€€€€€€€€€€…¹‘¥‘…Ñ•}Á…å±½…°(€€€€€€€€€€€‰…Í•±¥¹•}É½ÝÌ°(€€€€€€€€€€€…¹‘¥‘…Ñ•}É½ÝÌ°(€€€€€€€€¤°(€€€€€€€€‰ Í}‘•ÍÉ¥ÁÑ¥Ù•}…Ñ•}Á…ÍÍ•ˆè‰½½° (€€€€€€€€€€€Í…±¥¹œ…¹Í…±¥¹l‰½µµ½¹}É…¹•}…Ñ•}Á…ÍÍ•‰t(€€€€€€€€¤°(€€€ô(€€€½ÕÑÁÕÐ¹Á…É•¹Ð¹µ­‘¥È¡Á…É•¹ÑÌõQÉÕ”°•á¥ÍÑ}½¬õQÉÕ”¤(€€€½ÕÑÁÕÐ¹ÝÉ¥Ñ•}Ñ•áÐ¡©Í½¸¹‘ÕµÁÌ¡É•ÍÕ±Ð°¥¹‘•¹ÐôÈ¤€¬€‰q¸ˆ°•¹½‘¥¹œô‰ÕÑ˜´àˆ¤(()‘•˜µ…¥¸ ¤€´ø9½¹”è(€€€…ÁÀ ¤(()¥˜}}¹…µ•}|€ôô€‰}}µ…¥¹}|ˆè(€€€µ…¥¸ ¤(