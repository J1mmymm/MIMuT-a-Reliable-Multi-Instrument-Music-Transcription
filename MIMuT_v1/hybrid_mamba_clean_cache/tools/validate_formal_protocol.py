"""Validate the frozen MIMuT experiment matrix without starting training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from muscriptor.data.manifest import (
    manifest_fingerprint,
    read_manifest,
    validate_no_leakage,
)
from muscriptor.models.config import estimate_total_parameters
from muscriptor.training.config import load_experiment_config


def _config_paths(section: dict) -> list[str]:
    paths = list(section.get("backbones", {}).values())
    paths.extend(section.get("mechanism_ablations", {}).values())
    paths.extend(section.get("boundary_state_candidates", {}).values())
    return paths


def validate_matrix(matrix_path: Path, manifest: Path | None = None) -> dict:
    matrix = yaml.safe_load(matrix_path.read_text())
    config_root = matrix_path.parent
    errors: list[str] = []
    summaries = {}

    if matrix.get("seeds") != [3407, 3408, 3409]:
        errors.append("formal seeds must be exactly 3407, 3408, 3409")
    if (
        matrix.get("paper_title")
        != "MIMuT: Reliable Multi-Instrument Music Transcription"
    ):
        errors.append("paper title has drifted from the frozen MIMuT title")
    if matrix.get("evidence_status") != "no_formal_results":
        errors.append("matrix must not claim formal results before they exist")
    main_condition = matrix.get("main_condition", {})
    if (
        main_condition.get("audio_condition_dropout"),
        main_condition.get("instrument_condition_dropout"),
        main_condition.get("dataset_condition_dropout"),
    ) != (0.0, 1.0, 1.0):
        errors.append("matrix main condition must retain audio and null metadata")
    if main_condition.get("oracle_instruments") is not False:
        errors.append("matrix main condition must disable oracle instruments")

    data_protocol = matrix.get("data_protocol", {})
    training_datasets = set(data_protocol.get("training_datasets", []))
    test_only_datasets = set(data_protocol.get("test_only_datasets", []))
    if len(training_datasets) != 11:
        errors.append("data protocol must freeze exactly 11 training datasets")
    if test_only_datasets != {"rwc_pop", "urmp"}:
        errors.append("formal test-only datasets must be rwc_pop and urmp")
    if data_protocol.get("sampling") != "sqrt_dataset_duration":
        errors.append("formal sampling must use square-root dataset duration")
    if float(data_protocol.get("maximum_dataset_probability", -1)) != 0.35:
        errors.append("formal dataset probability cap must be 0.35")

    inference_factorial = matrix.get("inference_factorial", {})
    if inference_factorial.get("long_context") != [
        "reset",
        "carry",
    ] or inference_factorial.get("prelude_forcing") != [False, True]:
        errors.append("inference protocol must retain the reset/carry x prelude 2x2")
    statistics_protocol = matrix.get("statistics", {})
    if (
        statistics_protocol.get("unit") != "whole_track"
        or statistics_protocol.get("paired_bootstrap_samples") != 10_000
        or statistics_protocol.get("multiplicity") != "holm"
    ):
        errors.append("whole-track 10k bootstrap with Holm correction is required")

    project_root = matrix_path.parent.parent
    required_artifacts = (
        project_root / "paper" / "MIMUT_AAAI_DRAFT.md",
        project_root / "paper" / "TABLE_TEMPLATES.md",
        project_root / "paper" / "FIGURES.md",
        project_root / "paper" / "references.bib",
        project_root / "docs" / "LITERATURE_EVIDENCE.md",
    )
    for artifact in required_artifacts:
        if not artifact.is_file():
            errors.append(f"missing paper artifact: {artifact}")
    draft_path = project_root / "paper" / "MIMUT_AAAI_DRAFT.md"
    if draft_path.is_file():
        draft = draft_path.read_text(encoding="utf-8")
        if not draft.startswith(
            "# MIMuT: Reliable Multi-Instrument Music Transcription"
        ):
            errors.append("paper draft title has drifted")
        if "no formal results are available" not in draft:
            errors.append(
                "paper draft must disclose that formal results are unavailable"
            )
        if draft.count("<RESULT_TODO") < 5:
            errors.append("paper draft removed result placeholders before evaluation")
        forbidden_claims = ("state-of-the-art", "significantly outperforms")
        for phrase in forbidden_claims:
            if phrase in draft.lower():
                errors.append(f"paper draft contains premature result claim: {phrase}")

    expected_backbones = {
        "phase_b_106m": {
            "A0_global_transformer": "transformer",
            "A1_local_transformer": "local_transformer",
            "A2_pure_mamba": "pure_mamba",
            "A3_mimut_hybrid": "hybrid_mamba",
        },
        "phase_c_312m": {
            "mimut_hybrid": "hybrid_mamba",
            "global_transformer": "transformer",
            "local_transformer": "local_transformer",
            "pure_mamba": "pure_mamba",
        },
    }

    for phase_name in ("phase_b_106m", "phase_c_312m"):
        phase = matrix[phase_name]
        expected_steps = int(phase["expected_steps"])
        configs = {}
        for relative in sorted(set(_config_paths(phase))):
            path = config_root / relative
            if not path.exists():
                errors.append(f"missing config: {path}")
                continue
            config = load_experiment_config(path)
            total_steps = sum(stage.steps for stage in config.stages)
            if total_steps != expected_steps:
                errors.append(
                    f"{relative}: {total_steps} steps, expected {expected_steps}"
                )
            dropout = config.condition_dropout
            if (dropout.audio, dropout.instrument, dropout.dataset) != (0.0, 1.0, 1.0):
                errors.append(f"{relative}: formal condition is not audio-only")
            if config.seed != 3407:
                errors.append(f"{relative}: base seed must be 3407")
            if config.distillation.teacher_checkpoint is not None:
                errors.append(f"{relative}: distillation is not part of the protocol")
            for stage in config.stages:
                if stage.split != "train":
                    errors.append(f"{relative}/{stage.name}: formal stage is not train")
                if stage.dataset_weights:
                    errors.append(
                        f"{relative}/{stage.name}: manual dataset weights bypass "
                        "square-root-duration sampling"
                    )
                unknown = set(stage.datasets) - training_datasets
                if unknown:
                    errors.append(
                        f"{relative}/{stage.name}: non-frozen datasets {sorted(unknown)}"
                    )
                if set(stage.datasets) not in (
                    {"slakh2100_redux"},
                    training_datasets,
                ):
                    errors.append(
                        f"{relative}/{stage.name}: stage must use Slakh-only or "
                        "all 11 frozen training datasets"
                    )
            if "_v2" in relative:
                if config.optimizer.min_lr_ratio != 0.05:
                    errors.append(f"{relative}: v2 LR floor must be 0.05")
                if config.conditioning.mode != "audio_only":
                    errors.append(f"{relative}: v2 main must be explicit audio_only")
            configs[relative] = {
                "name": config.name,
                "backbone": config.model.backbone,
                "parameters": estimate_total_parameters(config.model),
                "steps": total_steps,
                "boundary_state_supervision": config.boundary_state_supervision.enabled,
            }

        for experiment_id, expected_backbone in expected_backbones[phase_name].items():
            relative = phase.get("backbones", {}).get(experiment_id)
            if (
                relative in configs
                and configs[relative]["backbone"] != expected_backbone
            ):
                errors.append(
                    f"{relative}: backbone is {configs[relative]['backbone']}, "
                    f"expected {expected_backbone}"
                )

        backbone_paths = list(phase.get("backbones", {}).values())
        counts = [
            configs[path]["parameters"] for path in backbone_paths if path in configs
        ]
        if counts:
            tolerance = float(phase["parameter_tolerance_fraction"])
            reference = sum(counts) / len(counts)
            for path in backbone_paths:
                if (
                    path in configs
                    and abs(configs[path]["parameters"] - reference) / reference
                    > tolerance
                ):
                    errors.append(f"{path}: parameter count exceeds +/-{tolerance:.0%}")
        summaries[phase_name] = configs

    phase_b = matrix["phase_b_106m"]
    expected_boundary_weights = {
        "lambda_005": 0.05,
        "lambda_010": 0.10,
        "lambda_020": 0.20,
    }
    for candidate, expected_weight in expected_boundary_weights.items():
        relative = phase_b.get("boundary_state_candidates", {}).get(candidate)
        if relative is None:
            errors.append(f"missing boundary-state candidate {candidate}")
            continue
        config = load_experiment_config(matrix_path.parent / relative)
        boundary = config.boundary_state_supervision
        if not boundary.enabled or (
            boundary.active_weight,
            boundary.reentry_weight,
        ) != (expected_weight, expected_weight):
            errors.append(
                f"{relative}: expected enabled active/reentry weights {expected_weight}"
            )

    expected_windows = {
        "local_window_1024": 1024,
        "local_window_2048": 2048,
        "local_window_4096": 4096,
    }
    for ablation, expected_window in expected_windows.items():
        relative = phase_b.get("mechanism_ablations", {}).get(ablation)
        if relative:
            config = load_experiment_config(matrix_path.parent / relative)
            if config.model.local_window != expected_window:
                errors.append(f"{relative}: local window must be {expected_window}")

    five_second = phase_b.get("mechanism_ablations", {}).get("five_second_only")
    if five_second:
        config = load_experiment_config(matrix_path.parent / five_second)
        if any(stage.context_chunks != 1 for stage in config.stages):
            errors.append(f"{five_second}: fixed-context control contains a long stage")

    manifest_summary = None
    if manifest is not None:
        records = read_manifest(manifest)
        validate_no_leakage(records)
        strict_test = test_only_datasets
        leaked = sorted(
            {
                record.dataset
                for record in records
                if record.dataset.lower() in strict_test and record.split != "test"
            }
        )
        if leaked:
            errors.append(f"strict test datasets outside test split: {leaked}")
        manifest_datasets = {record.dataset for record in records}
        missing_training = sorted(training_datasets - manifest_datasets)
        missing_test = sorted(test_only_datasets - manifest_datasets)
        unexpected_development = sorted(
            {
                record.dataset
                for record in records
                if record.split in {"train", "validation"}
                and record.dataset not in training_datasets
            }
        )
        if missing_training:
            errors.append(f"manifest is missing training datasets: {missing_training}")
        if missing_test:
            errors.append(f"manifest is missing test-only datasets: {missing_test}")
        if unexpected_development:
            errors.append(
                "manifest contains non-frozen development datasets: "
                f"{unexpected_development}"
            )
        manifest_summary = {
            "sha256": manifest_fingerprint(manifest),
            "records": len(records),
            "splits": {
                split: sum(record.split == split for record in records)
                for split in sorted({record.split for record in records})
            },
            "datasets": sorted({record.dataset for record in records}),
        }

    return {
        "status": "ok" if not errors else "error",
        "matrix": str(matrix_path),
        "errors": errors,
        "phases": summaries,
        "manifest": manifest_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(__file__).parents[1] / "configs" / "formal_experiment_matrix.yaml",
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = validate_matrix(args.matrix, args.manifest)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
