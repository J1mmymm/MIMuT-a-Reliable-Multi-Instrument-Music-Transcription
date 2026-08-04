"""Validated YAML experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from muscriptor.models.config import MODEL_PRESETS, ModelConfig


@dataclass(frozen=True)
class StageConfig:
    name: str
    steps: int
    context_chunks: int
    datasets: list[str] = field(default_factory=list)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    split: str = "train"
    # LR overrides (see muscriptor/training/schedule.py).  Defaults reproduce
    # the historical single global warmup+cosine.  A late curriculum stage
    # (e.g. the long-context stages) should set lr_scale (peak multiplier on
    # optimizer.learning_rate) and rewarm_steps so it does not inherit the
    # near-zero tail of the global cosine.
    lr_scale: float = 1.0
    rewarm_steps: int = 0
    # Optional per-stage override.  Long-context stages use a weaker teacher
    # constraint so supervised labels can reward information beyond 5 seconds.
    distillation_weight: float | None = None


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class DistillationConfig:
    teacher_checkpoint: str | None = None
    weight: float = 0.25
    temperature: float = 2.0


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    every_steps: int = 2000
    num_batches: int = 4
    batch_size: int = 1
    split: str = "validation"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    manifest: str
    output_dir: str
    model: ModelConfig
    stages: list[StageConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    distillation: DistillationConfig = DistillationConfig()
    validation: ValidationConfig = ValidationConfig()
    # Used for a supervised warm start even when no KL teacher is retained.
    initialization_checkpoint: str | None = None
    seed: int = 1337
    precision: str = "bfloat16"
    global_batch_audio_seconds: int = 320
    gradient_accumulation_steps: int = 1
    condition_dropout: float = 0.2
    # Probability that a training chunk's tie prologue is fed to the model
    # with substituted (wrong) pitch/program tokens while its labels are
    # masked — simulating an imperfect forced prelude so carried state and
    # continuation are robust to earlier mistakes at inference.  0 disables
    # (historical behavior).
    tie_corruption: float = 0.0
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 2000
    distributed: str = "ddp"
    gradient_checkpointing: bool = True
    resume: str | None = None
    # Exact resume restores the saved RNG stream and therefore normally
    # requires the original DDP world size.  Set this only for an intentional
    # elastic restart (for example 2 -> 4 GPUs); model/optimizer/scheduler are
    # restored exactly, while each rank receives a deterministic fresh seed.
    allow_world_size_change_on_resume: bool = False


def _model_from_raw(raw: dict[str, Any] | str) -> ModelConfig:
    if isinstance(raw, str):
        if raw not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset: {raw}")
        return MODEL_PRESETS[raw]
    raw = dict(raw)
    preset = raw.pop("preset", None)
    base = MODEL_PRESETS[preset].to_dict() if preset else {}
    base.update(raw)
    return ModelConfig.from_dict(base)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"cyclic experiment config inheritance at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("experiment YAML must contain an object")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = (path.parent / parent).resolve()
    return _deep_merge(_load_raw(parent_path, seen), raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = _load_raw(path)
    model = _model_from_raw(raw.pop("model"))
    stages = [StageConfig(**stage) for stage in raw.pop("stages")]
    optimizer = OptimizerConfig(**raw.pop("optimizer", {}))
    distillation = DistillationConfig(**raw.pop("distillation", {}))
    validation = ValidationConfig(**raw.pop("validation", {}))
    config = ExperimentConfig(
        model=model,
        stages=stages,
        optimizer=optimizer,
        distillation=distillation,
        validation=validation,
        **raw,
    )
    if config.precision not in {"bfloat16", "float16", "float32"}:
        raise ValueError("precision must be bfloat16, float16, or float32")
    if config.distributed not in {"ddp", "fsdp", "none"}:
        raise ValueError("distributed must be ddp, fsdp, or none")
    if not 0 <= config.condition_dropout < 1:
        raise ValueError("condition_dropout must be in [0, 1)")
    if not 0 <= config.tie_corruption < 1:
        raise ValueError("tie_corruption must be in [0, 1)")
    if not config.stages:
        raise ValueError("at least one training stage is required")
    for stage in config.stages:
        if stage.lr_scale <= 0:
            raise ValueError(f"stage {stage.name!r}: lr_scale must be positive")
        if stage.rewarm_steps < 0 or stage.rewarm_steps >= stage.steps:
            raise ValueError(
                f"stage {stage.name!r}: rewarm_steps must be in [0, steps)"
            )
        if stage.distillation_weight is not None and stage.distillation_weight < 0:
            raise ValueError(
                f"stage {stage.name!r}: distillation_weight must be non-negative"
            )
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.validation.every_steps <= 0:
        raise ValueError("validation.every_steps must be positive")
    if config.validation.num_batches <= 0 or config.validation.batch_size <= 0:
        raise ValueError("validation batch counts and sizes must be positive")
    return config
