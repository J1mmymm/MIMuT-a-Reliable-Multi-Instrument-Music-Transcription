import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from muscriptor.data.manifest import (
    ManifestRecord,
    manifest_fingerprint,
    write_manifest,
)
from muscriptor.evaluation.cli import app
import muscriptor.evaluation.cli as evaluation_cli
from muscriptor.evaluation.statistics import (
    macro_average,
    micro_average,
    paired_track_bootstrap,
)


def _metric_row(track_id: str, value: float) -> dict:
    return {
        "track_id": track_id,
        "dataset": "d",
        "multi": {
            "precision": value,
            "recall": value,
            "f1": value,
            "true_positive": int(value * 10),
            "predicted": 10,
            "reference": 10,
        },
        "instrument_switch_error_rate": 1.0 - value,
    }


def test_micro_average_recomputes_prf_from_pooled_counts():
    rows = [
        {
            "m": {
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2 / 3,
                "true_positive": 1,
                "predicted": 1,
                "reference": 2,
            }
        },
        {
            "m": {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positive": 0,
                "predicted": 3,
                "reference": 1,
            }
        },
    ]
    pooled = micro_average(rows)["m"]
    assert pooled["true_positive"] == 1
    assert pooled["predicted"] == 4
    assert pooled["reference"] == 3


def test_macro_average_excludes_undefined_sparse_prf_tracks():
    empty = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "true_positive": 0,
        "predicted": 0,
        "reference": 0,
    }
    defined = {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "true_positive": 1,
        "predicted": 1,
        "reference": 1,
    }
    macro = macro_average([empty, defined])
    assert macro["f1"] == 1.0
    assert macro["defined_track_count"] == 1


def test_paired_bootstrap_requires_and_preserves_track_pairing():
    baseline = [_metric_row("a", 0.4), _metric_row("b", 0.5)]
    candidate = [_metric_row("a", 0.6), _metric_row("b", 0.7)]
    result = paired_track_bootstrap(
        baseline,
        candidate,
        metric_paths=["multi.f1", "instrument_switch_error_rate"],
        lower_is_better={"instrument_switch_error_rate"},
        samples=200,
        seed=3407,
    )
    assert result["paired_track_count"] == 2
    assert result["metrics"][0]["improvement"] > 0
    assert result["metrics"][1]["improvement"] > 0


def test_chunk_prefill_gate_uses_carried_prefix_medians_and_token_identity():
    def benchmark_row(audio_seconds: float, prefix_seconds: float, token_hash: str):
        chunks = [
            {"prefix_prefill_seconds": prefix_seconds}
            for _ in range(int(audio_seconds // 5))
        ]
        return {
            "status": "ok",
            "stable_runs": [
                {
                    "generation_profile": {"chunks": chunks},
                    "token_stream_sha256": token_hash,
                }
                for _ in range(3)
            ],
        }

    baseline_payload = {
        "protocol": {
            "model_weights_sha256": "a" * 64,
            "prefill_mode": "step",
        }
    }
    candidate_payload = {
        "protocol": {
            "model_weights_sha256": "a" * 64,
            "prefill_mode": "auto",
        }
    }
    baseline_rows = {
        seconds: benchmark_row(seconds, 2.0, f"tokens-{seconds}")
        for seconds in (10.0, 20.0, 40.0, 80.0)
    }
    candidate_rows = {
        seconds: benchmark_row(seconds, 1.0, f"tokens-{seconds}")
        for seconds in (10.0, 20.0, 40.0, 80.0)
    }
    result = evaluation_cli._chunk_prefill_gate(
        baseline_payload,
        candidate_payload,
        baseline_rows,
        candidate_rows,
    )
    assert result["passed"] is True
    assert [row["context_seconds"] for row in result["probes"]] == [
        5.0,
        10.0,
        20.0,
        40.0,
    ]
    assert all(row["speedup"] == pytest.approx(2.0) for row in result["probes"])


def test_compare_fails_closed_when_prediction_is_missing(tmp_path):
    notes = tmp_path / "ref.notes.json"
    notes.write_text(json.dumps({"notes": []}))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                track_id="x",
                dataset="d",
                split="test",
                audio_path=str(tmp_path / "x.wav"),
                notes_path=str(notes),
                duration=1.0,
                instrument_groups=[],
                group_id="x",
                is_multi_instrument=False,
            )
        ],
        manifest,
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": manifest_fingerprint(manifest),
                "split": "test",
                "expected_track_count": 1,
                "oracle_instruments": False,
                "condition_mode": "audio_only",
                "dataset_condition": None,
            }
        )
    )
    output = tmp_path / "evaluation.json"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "missing 1 of 1" in result.output


def test_predict_defaults_to_audio_only_and_records_protocol(tmp_path, monkeypatch):
    notes = tmp_path / "ref.notes.json"
    notes.write_text(json.dumps({"notes": []}))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                track_id="x",
                dataset="d",
                split="test",
                audio_path=str(tmp_path / "x.wav"),
                notes_path=str(notes),
                duration=1.0,
                instrument_groups=[0],
                group_id="x",
                is_multi_instrument=False,
            )
        ],
        manifest,
    )
    calls = []

    class FakeModel:
        def transcribe_to_midi(self, audio, **kwargs):
            calls.append((audio, kwargs))
            return b"MIDI"

    monkeypatch.setattr(
        evaluation_cli,
        "TranscriptionModel",
        SimpleNamespace(load_model=lambda *args, **kwargs: FakeModel()),
    )
    output = tmp_path / "predictions"
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model",
            "fake",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0][1]["instruments"] is None
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["oracle_instruments"] is False
    assert metadata["condition_mode"] == "audio_only"
    assert metadata["instrument_condition"] is None
    assert metadata["seed_verified"] is False


def test_predict_reads_and_enforces_checkpoint_seed_metadata(tmp_path, monkeypatch):
    notes = tmp_path / "ref.notes.json"
    notes.write_text(json.dumps({"notes": []}))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                track_id="x",
                dataset="d",
                split="test",
                audio_path=str(tmp_path / "x.wav"),
                notes_path=str(notes),
                duration=1.0,
                instrument_groups=[],
                group_id="x",
                is_multi_instrument=False,
            )
        ],
        manifest,
    )
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {"dummy": torch.zeros(1)},
        checkpoint,
        metadata={"experiment": json.dumps({"name": "mimut_seed3407", "seed": 3407})},
    )

    class FakeModel:
        def transcribe_to_midi(self, *args, **kwargs):
            return b"MIDI"

    monkeypatch.setattr(
        evaluation_cli,
        "TranscriptionModel",
        SimpleNamespace(load_model=lambda *args, **kwargs: FakeModel()),
    )
    mismatch = CliRunner().invoke(
        app,
        [
            "predict",
            "--model",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "mismatch"),
            "--seed",
            "3408",
            "--device",
            "cpu",
        ],
    )
    assert mismatch.exit_code != 0
    assert "disagrees with checkpoint seed" in mismatch.output

    output = tmp_path / "predictions"
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--model",
            str(checkpoint),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--experiment-id",
            "mimut",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 3407
    assert metadata["seed_verified"] is True


def test_compare_rejects_predictions_from_a_different_manifest(tmp_path):
    notes = tmp_path / "ref.notes.json"
    notes.write_text(json.dumps({"notes": []}))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                track_id="x",
                dataset="d",
                split="test",
                audio_path=str(tmp_path / "x.wav"),
                notes_path=str(notes),
                duration=1.0,
                instrument_groups=[],
                group_id="x",
                is_multi_instrument=False,
            )
        ],
        manifest,
    )
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    (predictions / "run.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "manifest_sha256": "wrong-sha",
                "split": "test",
                "expected_track_count": 1,
                "oracle_instruments": False,
                "condition_mode": "audio_only",
                "dataset_condition": None,
            }
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(tmp_path / "evaluation.json"),
        ],
    )
    assert result.exit_code != 0
    assert "different manifest SHA" in result.output


def test_external_predictions_require_audited_registration(tmp_path):
    notes = tmp_path / "ref.notes.json"
    notes.write_text(json.dumps({"notes": []}))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [
            ManifestRecord(
                track_id="x",
                dataset="d",
                split="test",
                audio_path=str(tmp_path / "x.wav"),
                notes_path=str(notes),
                duration=1.0,
                instrument_groups=[],
                group_id="x",
                is_multi_instrument=False,
            )
        ],
        manifest,
    )
    predictions = tmp_path / "external"
    (predictions / "d").mkdir(parents=True)
    (predictions / "d" / "x.notes.json").write_text(json.dumps({"notes": []}))
    result = CliRunner().invoke(
        app,
        [
            "register-external",
            "--predictions",
            str(predictions),
            "--manifest",
            str(manifest),
            "--model-name",
            "MT3 official",
            "--training-overlap",
            "unknown",
        ],
    )
    assert result.exit_code == 0, result.output
    metadata = json.loads((predictions / "run.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["producer"] == "external_checkpoint"
    assert metadata["training_overlap"] == "unknown"


def test_aggregate_seeds_averages_tracks_and_retains_seed_dispersion(tmp_path):
    paths = []
    for seed, value in ((3407, 0.4), (3408, 0.6)):
        row = _metric_row("a", value)
        payload = {
            "protocol": {
                "manifest_sha256": "frozen-manifest",
                "prediction_run": {
                    "seed": seed,
                    "seed_verified": True,
                    "experiment_id": "mimut-a3",
                    "condition_mode": "audio_only",
                    "checkpoint_audit": {
                        "model_config_sha256": "same-model",
                        "training_manifest_sha256": "same-training-manifest",
                    },
                },
            },
            "summary": {
                "complete": True,
                "macro": {"multi": row["multi"]},
                "micro": {"multi": row["multi"]},
            },
            "tracks": [row],
        }
        path = tmp_path / f"seed{seed}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    output = tmp_path / "aggregate.json"
    result = CliRunner().invoke(
        app,
        [
            "aggregate-seeds",
            "--inputs",
            ",".join(str(path) for path in paths),
            "--output",
            str(output),
            "--expected-seeds",
            "3407,3408",
        ],
    )
    assert result.exit_code == 0, result.output
    aggregate = json.loads(output.read_text(encoding="utf-8"))
    assert aggregate["protocol"]["seed_count"] == 2
    assert aggregate["protocol"]["seeds"] == [3407, 3408]
    assert aggregate["protocol"]["experiment_id"] == "mimut-a3"
    assert aggregate["tracks"][0]["multi"]["f1"] == pytest.approx(0.5)
    seed_stats = aggregate["summary"]["macro_seed_statistics"]["multi"]["f1"]
    assert seed_stats["mean"] == pytest.approx(0.5)
    assert seed_stats["std"] == pytest.approx(0.1)
    assert seed_stats["values"] == pytest.approx([0.4, 0.6])


def test_benchmark_fixes_batch_precision_and_records_scaling_protocol(
    tmp_path, monkeypatch
):
    audio = tmp_path / "track.wav"
    audio.write_bytes(b"placeholder")
    calls = []
    loads = []

    class FakeModel:
        _device = torch.device("cpu")
        last_streaming_state = {"bytes": 0, "local_cache_lengths": []}

        def transcribe(self, audio_input, **kwargs):
            clip, _sample_rate = audio_input
            calls.append((clip.shape[-1], kwargs))
            yield []

    monkeypatch.setattr(
        evaluation_cli,
        "TranscriptionModel",
        SimpleNamespace(
            load_model=lambda *args, **kwargs: (
                loads.append((args, kwargs)) or FakeModel()
            )
        ),
    )
    monkeypatch.setattr(
        evaluation_cli, "load_audio", lambda *args, **kwargs: torch.zeros(1, 160_000)
    )
    output = tmp_path / "benchmark.json"
    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--model",
            "fake",
            "--audio",
            str(audio),
            "--output",
            str(output),
            "--lengths",
            "5,10",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--warmup-runs",
            "0",
            "--repeats",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert loads[0][1]["dtype"] == "float32"
    assert payload["protocol"]["batch_size"] == 1
    assert payload["protocol"]["weight_dtype"] == "float32"
    assert all(row["status"] == "ok" for row in payload["lengths"])
    assert all(kwargs["batch_size"] == 1 for _, kwargs in calls)


def test_compare_efficiency_checks_growth_and_preserves_oom_frontier(tmp_path):
    protocol = {
        "audio": "same.wav",
        "sample_rate": 16_000,
        "batch_size": 1,
        "weight_dtype": "float32",
        "cuda_autocast_dtype": "float16",
        "decoding": {"sampling": False, "beam_size": 1},
        "long_context": "carry",
        "prelude_forcing": False,
        "warmup_runs": 1,
        "stable_repeats": 3,
        "device": {"name": "fixed-gpu"},
    }

    def successful(seconds, elapsed, memory):
        return {
            "length_label": str(seconds),
            "audio_seconds": float(seconds),
            "status": "ok",
            "stable_summary": {
                "elapsed_seconds": {"mean": elapsed, "std": 0.0},
                "real_time_factor": {"mean": elapsed / seconds, "std": 0.0},
                "audio_seconds_per_second": {
                    "mean": seconds / elapsed,
                    "std": 0.0,
                },
                "peak_memory_allocated_bytes": {"mean": memory, "std": 0.0},
                "peak_memory_reserved_bytes": {"mean": memory, "std": 0.0},
            },
        }

    baseline = {
        "protocol": protocol,
        "lengths": [
            successful(5, 1.0, 100.0),
            successful(10, 4.0, 400.0),
            {"length_label": "20", "audio_seconds": 20.0, "status": "oom"},
        ],
    }
    candidate = {
        "protocol": protocol,
        "lengths": [
            successful(5, 1.0, 100.0),
            successful(10, 2.0, 200.0),
            successful(20, 4.0, 300.0),
        ],
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    output = tmp_path / "comparison.json"
    result = CliRunner().invoke(
        app,
        [
            "compare-efficiency",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["scaling"]["runtime_growth_better"] is True
    assert comparison["scaling"]["memory_growth_better"] is True
    assert comparison["h3_descriptive_gate_passed"] is True
    assert comparison["oom"]["candidate_survives_baseline_oom"] is True
