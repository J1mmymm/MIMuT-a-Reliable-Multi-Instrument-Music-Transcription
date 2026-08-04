"""Fail-closed validation for v2 smoke/pilot logs and resume replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def _load_run(run: Path) -> dict:
    experiment_path = run / "experiment.json"
    latest_path = run / "latest.json"
    if not experiment_path.is_file() or not latest_path.is_file():
        raise RuntimeError(f"incomplete run metadata under {run}")
    experiment = json.loads(experiment_path.read_text())
    latest = json.loads(latest_path.read_text())
    trainer_path = run / latest["trainer"]
    if not trainer_path.is_file():
        raise RuntimeError(f"missing trainer state: {trainer_path}")
    trainer = torch.load(trainer_path, map_location="cpu", weights_only=False)

    metric_paths = sorted(run.glob("metrics.rank*.jsonl"))
    if not metric_paths:
        raise RuntimeError(f"no rank metrics under {run}")
    rows_by_rank = {}
    final_metric_hash_by_rank = {}
    teacher_enabled = bool(
        experiment.get("distillation", {}).get("teacher_checkpoint")
    )
    for path in metric_paths:
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise RuntimeError(f"empty metrics file: {path}")
        previous = 0
        for row in rows:
            for key in ("loss", "ce", "distill", "grad_norm"):
                if not math.isfinite(float(row[key])):
                    raise RuntimeError(f"non-finite {key} in {path}")
            if teacher_enabled and not (
                float(row["distill"]) > 0 and row.get("distill_positive") is True
            ):
                raise RuntimeError(f"teacher KL is not positive in {path}")
            if row.get("teacher_frozen") is not True:
                raise RuntimeError(f"teacher freeze check failed in {path}")
            if row.get("health_checked") is not True:
                raise RuntimeError(f"health check was skipped in {path}")
            if int(row["step"]) != int(row.get("optimizer_step", -1)):
                raise RuntimeError(f"optimizer-step mismatch in {path}")
            if int(row["step"]) <= previous:
                raise RuntimeError(f"non-monotonic optimizer steps in {path}")
            if len(str(row.get("source_sequence_sha256", ""))) != 64:
                raise RuntimeError(f"missing source-sequence audit in {path}")
            previous = int(row["step"])
        rows_by_rank[path.name] = rows
        rank_id = int(path.stem.rsplit("rank", 1)[1])
        final_metric_hash_by_rank[rank_id] = rows[-1]["source_sequence_sha256"]

    hashes = trainer.get("source_sequence_sha256_by_rank")
    if not hashes or any(len(str(value)) != 64 for value in hashes):
        raise RuntimeError("trainer state lacks per-rank source-sequence hashes")
    for rank_id, metric_hash in final_metric_hash_by_rank.items():
        if rank_id >= len(hashes) or metric_hash != hashes[rank_id]:
            raise RuntimeError(
                f"final metrics and trainer source hash disagree on rank {rank_id}"
            )
    if int(trainer["global_step"]) != int(latest["global_step"]):
        raise RuntimeError("latest.json and trainer optimizer steps disagree")
    return {
        "path": str(run),
        "global_step": int(trainer["global_step"]),
        "training_signature": trainer.get("training_signature"),
        "context_draw_counts": trainer.get("context_draw_counts"),
        "source_sequence_sha256_by_rank": list(hashes),
        "rank_metric_files": sorted(rows_by_rank),
        "teacher_enabled": teacher_enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--continuous-reference", type=Path)
    args = parser.parse_args()

    result = {"status": "ok", "run": _load_run(args.run.resolve())}
    if args.continuous_reference is not None:
        reference = _load_run(args.continuous_reference.resolve())
        result["continuous_reference"] = reference
        for key in (
            "global_step",
            "training_signature",
            "context_draw_counts",
            "source_sequence_sha256_by_rank",
        ):
            if result["run"][key] != reference[key]:
                raise RuntimeError(f"resume replay mismatch: {key}")
        result["resume_replay_identical"] = True
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
