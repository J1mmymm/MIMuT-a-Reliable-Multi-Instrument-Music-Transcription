"""Learning-rate schedule and resume-signature compatibility helpers.

Torch-free on purpose: pure functions of the experiment configuration, unit
tested without CUDA.

The historical schedule was a single global warmup + cosine over the total
step budget.  With the curriculum's long-context stages at the very end, that
meant the model's core novelty — cross-chunk state use — trained at ~3% of
the base LR and below.  Stages may now override this with ``lr_scale`` (peak
LR multiplier) and ``rewarm_steps`` (linear re-warm at the stage start); an
overridden stage runs its own cosine from ``lr_scale`` down to
``lr_scale * MIN_STAGE_LR_RATIO``.  Stages without overrides keep the exact
legacy global schedule, so old configs reproduce old behavior bit for bit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any


# An overridden stage's cosine floors at this fraction of its peak so a
# middle stage never parks the optimizer at LR exactly 0.
MIN_STAGE_LR_RATIO = 0.05


def legacy_lambda(step: int, *, warmup: int, total_steps: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _stage_lambda(
    stage_step: int, *, steps: int, lr_scale: float, rewarm_steps: int
) -> float:
    ramp = 1.0
    if rewarm_steps > 0:
        ramp = min(1.0, stage_step / rewarm_steps)
    cosine_span = max(1, steps - rewarm_steps)
    progress = min(1.0, max(0, stage_step - rewarm_steps) / cosine_span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    shaped = MIN_STAGE_LR_RATIO + (1.0 - MIN_STAGE_LR_RATIO) * cosine
    return lr_scale * ramp * shaped


def make_scheduler_lambda(
    stages: Sequence[Any], *, warmup: int
) -> Callable[[int], float]:
    """Build the LambdaLR multiplier for a stage list.

    ``stages`` are objects with ``steps`` and optional ``lr_scale`` /
    ``rewarm_steps`` attributes (StageConfig).  Deterministic in the global
    step alone, so checkpoint resume replays the exact schedule.
    """
    boundaries: list[tuple[int, int, float, int]] = []  # start, steps, scale, rewarm
    start = 0
    for stage in stages:
        boundaries.append(
            (
                start,
                int(stage.steps),
                float(getattr(stage, "lr_scale", 1.0)),
                int(getattr(stage, "rewarm_steps", 0)),
            )
        )
        start += int(stage.steps)
    total_steps = start

    def scheduler_lambda(step: int) -> float:
        selected = boundaries[-1]
        for stage in boundaries:
            if step < stage[0] + stage[1]:
                selected = stage
                break
        stage_start, steps, lr_scale, rewarm = selected
        if lr_scale == 1.0 and rewarm == 0:
            return legacy_lambda(step, warmup=warmup, total_steps=total_steps)
        return _stage_lambda(
            step - stage_start,
            steps=steps,
            lr_scale=lr_scale,
            rewarm_steps=rewarm,
        )

    return scheduler_lambda


# Fields added to the experiment configuration after checkpoints already
# existed in the wild.  A stored training signature that predates a field is
# still compatible when the current run keeps that field at its safe default;
# any other difference is a real configuration change and must fail resume.
SIGNATURE_FIELD_DEFAULTS: dict[str, Any] = {
    "lr_scale": 1.0,
    "rewarm_steps": 0,
    "tie_corruption": 0.0,
    "position_encoding": "sinusoidal",
}


def signatures_compatible(stored: Any, current: Any) -> bool:
    """Order-insensitive structural comparison with new-field tolerance."""
    if isinstance(stored, dict) and isinstance(current, dict):
        for key in stored:
            if key not in current:
                return False
            if not signatures_compatible(stored[key], current[key]):
                return False
        for key in current:
            if key in stored:
                continue
            if key not in SIGNATURE_FIELD_DEFAULTS:
                return False
            if current[key] != SIGNATURE_FIELD_DEFAULTS[key]:
                return False
        return True
    if isinstance(stored, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(stored) != len(current):
            return False
        return all(
            signatures_compatible(s, c) for s, c in zip(stored, current)
        )
    return stored == current
