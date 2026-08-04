"""Aggregation and paired whole-track bootstrap utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


_PRF_KEYS = {"precision", "recall", "f1", "true_positive", "predicted", "reference"}
_RATE_KEYS = {"rate", "count", "reference"}


def macro_average(rows: list[dict]) -> dict:
    """Recursively macro-average numeric leaves shared by every row."""

    if not rows:
        return {}
    result = {}
    keys = set.intersection(*(set(row) for row in rows))
    if _PRF_KEYS <= keys:
        # A track with neither reference nor predicted events has an
        # undefined per-track PRF, not a score of zero. Excluding it prevents
        # sparse boundary/re-entry bins from being dominated by empty tracks.
        valid = [
            row
            for row in rows
            if int(row["reference"]) > 0 or int(row["predicted"]) > 0
        ]
        if not valid:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positive": 0.0,
                "predicted": 0.0,
                "reference": 0.0,
                "defined_track_count": 0,
            }
        return {
            key: sum(float(row[key]) for row in valid) / len(valid)
            for key in sorted(_PRF_KEYS)
        } | {"defined_track_count": len(valid)}
    if _RATE_KEYS <= keys:
        valid = [row for row in rows if int(row["reference"]) > 0]
        if not valid:
            return {
                "rate": 0.0,
                "count": 0.0,
                "reference": 0.0,
                "defined_track_count": 0,
            }
        return {
            key: sum(float(row[key]) for row in valid) / len(valid)
            for key in sorted(_RATE_KEYS)
        } | {"defined_track_count": len(valid)}
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = macro_average(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            result[key] = sum(values) / len(values)
    return result


def micro_average(rows: list[dict]) -> dict:
    """Pool metric counts while macro-averaging count-free diagnostics."""

    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    if _PRF_KEYS <= keys:
        true_positive = sum(int(row["true_positive"]) for row in rows)
        predicted = sum(int(row["predicted"]) for row in rows)
        reference = sum(int(row["reference"]) for row in rows)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / reference if reference else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": true_positive,
            "predicted": predicted,
            "reference": reference,
        }
    if _RATE_KEYS <= keys:
        count = sum(int(row["count"]) for row in rows)
        reference = sum(int(row["reference"]) for row in rows)
        return {
            "rate": count / reference if reference else 0.0,
            "count": count,
            "reference": reference,
        }

    result = {}
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = micro_average(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            result[key] = sum(values) / len(values)
    return result


def nested_value(row: dict, path: str) -> float:
    value = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"metric path {path!r} is absent")
        value = value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"metric path {path!r} is not numeric")
    return float(value)


def _nested_object(row: dict, path: str) -> dict | None:
    value = row
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, dict) else None


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sample")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return Holm step-down family-wise adjusted p-values."""

    values = list(p_values)
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_track_bootstrap(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    metric_paths: list[str],
    samples: int = 10_000,
    seed: int = 3407,
    lower_is_better: set[str] | None = None,
) -> dict:
    """Bootstrap candidate-minus-baseline deltas over paired whole tracks."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    lower_is_better = lower_is_better or set()

    def keyed(rows: list[dict]) -> dict[tuple[str, str], dict]:
        result = {}
        for row in rows:
            key = (str(row["dataset"]), str(row["track_id"]))
            if key in result:
                raise ValueError(f"duplicate track row: {key}")
            result[key] = row
        return result

    baseline = keyed(baseline_rows)
    candidate = keyed(candidate_rows)
    if baseline.keys() != candidate.keys():
        missing_candidate = sorted(baseline.keys() - candidate.keys())
        missing_baseline = sorted(candidate.keys() - baseline.keys())
        raise ValueError(
            "paired bootstrap requires identical track sets; "
            f"candidate missing={missing_candidate[:5]}, "
            f"baseline missing={missing_baseline[:5]}"
        )
    keys = sorted(baseline)
    if not keys:
        raise ValueError("paired bootstrap requires at least one track")

    results = []
    raw_p_values = []
    tested_result_indices = []
    for metric_path in metric_paths:
        # Reuse the same track-resampling sequence for every endpoint.
        rng = random.Random(seed)
        parent_path = metric_path.rsplit(".", 1)[0] if "." in metric_path else ""
        metric_keys = []
        for key in keys:
            baseline_parent = _nested_object(baseline[key], parent_path)
            candidate_parent = _nested_object(candidate[key], parent_path)
            if (
                baseline_parent is not None
                and candidate_parent is not None
                and (
                    (
                        _PRF_KEYS <= baseline_parent.keys()
                        and _PRF_KEYS <= candidate_parent.keys()
                        and int(baseline_parent["reference"]) == 0
                        and int(baseline_parent["predicted"]) == 0
                        and int(candidate_parent["reference"]) == 0
                        and int(candidate_parent["predicted"]) == 0
                    )
                    or (
                        _RATE_KEYS <= baseline_parent.keys()
                        and _RATE_KEYS <= candidate_parent.keys()
                        and int(baseline_parent["reference"]) == 0
                        and int(candidate_parent["reference"]) == 0
                    )
                )
            ):
                continue
            metric_keys.append(key)
        if not metric_keys:
            results.append(
                {
                    "metric": metric_path,
                    "status": "undefined",
                    "paired_track_count": 0,
                    "reason": "metric is undefined for every paired track",
                }
            )
            continue
        deltas = [
            nested_value(candidate[key], metric_path)
            - nested_value(baseline[key], metric_path)
            for key in metric_keys
        ]
        observed = sum(deltas) / len(deltas)
        replicates = []
        for _ in range(samples):
            replicates.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
            )
        replicates.sort()
        lower = _percentile(replicates, 0.025)
        upper = _percentile(replicates, 0.975)
        non_positive = (sum(value <= 0 for value in replicates) + 1) / (samples + 1)
        non_negative = (sum(value >= 0 for value in replicates) + 1) / (samples + 1)
        p_value = min(1.0, 2 * min(non_positive, non_negative))
        raw_p_values.append(p_value)
        sign = -1.0 if metric_path in lower_is_better else 1.0
        results.append(
            {
                "metric": metric_path,
                "status": "ok",
                "paired_track_count": len(metric_keys),
                "direction": ("lower_is_better" if sign < 0 else "higher_is_better"),
                "candidate_minus_baseline": observed,
                "improvement": sign * observed,
                "ci95_candidate_minus_baseline": [lower, upper],
                "p_value_two_sided": p_value,
            }
        )
        tested_result_indices.append(len(results) - 1)

    adjusted = holm_adjust(raw_p_values)
    for result_index, value in zip(tested_result_indices, adjusted):
        results[result_index]["p_value_holm"] = value
    return {
        "unit": "whole_track",
        "paired_track_count": len(keys),
        "bootstrap_samples": samples,
        "seed": seed,
        "tested_metric_count": len(tested_result_indices),
        "metrics": results,
    }
