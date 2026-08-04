"""Validated YAML experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from muscriptor.models.config import MODEL_PRESETS, ModelConfig


@dataclass(frozen=True)
class StageConfig:
    name: str
    steps: int
    context_chunks: int | None = None
    context_distribution: dict[int, float] = field(default_factory=dict)
    datasets: list[str] = field(default_factory=list)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    split: str = "train"
    augmentation: bool | None = None

    def context_probabilities(self) -> dict[int, float]:
        """Return the normalized, backward-compatible context curriculum."""

        if self.context_distribution:
            total = sum(self.context_distribution.values())
            if total <= 0:
                raise ValueError(
                    f"stage {self.name!r} context distribution has no positive mass"
                )
            return {
                int(chunks): float(weight) / total
                for chunks, weight in sorted(self.context_distribution.items())
            }
        if self.context_chunks is None:
            raise ValueError(f"stage {self.name!r} has no context configuration")
        return {int(self.context_chunks): 1.0}


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.0
    gradient_clip: float = 1.0


@dataclass(frozen=True)
class DistillationConfig:
    teacher_checkpoint: str | None = None
    teacher_sha256: str | None = None
    teacher_revision: str | None = None
    teacher_microbatch_size: int = 4
    weight: float = 0.25
    temperature: float = 2.0


@dataclass(frozen=True)
class ConditioningConfig:
    """Condition semantics shared by training and formal inference."""

    mode: str = "audio_only"
    dropout_scope: str = "sequence"


@dataclass(frozen=True)
class AugmentationConfig:
    """Deterministic, leakage-safe isolated-source augmentation policy."""

    enabled: bool = False
    catalog: str | None = None
    remix_probability: float = 0.25
    pitch_shift_probability: float = 0.20
    source_count_probabilities: dict[int, float] = field(
        default_factory=lambda: {2: 0.50, 3: 0.30, 4: 0.20}
    )
    gain_db_min: float = -12.0
    gain_db_max: float = 0.0
    semitone_choices: tuple[int, ...] = (-3, -2, -1, 1, 2, 3)
    allowed_datasets: tuple[str, ...] = (
        "slakh2100_redux",
        "choralebricks",
        "maestro",
        "gaps",
        "guitarset",
        "idmt_smt_bass",
    )


@dataclass(frozen=True)
class ConditionDropoutConfig:
    """Independent condition masking probabilities.

    Formal MIMuT runs are audio-only: acoustic frames are always retained,
    while instrument and dataset metadata are always mapped to the null class.
    The three probabilities remain configurable for explicit oracle/CFG
    diagnostics, but those runs must not be mixed into the main table.
    """

    audio: float = 0.0
    instrument: float = 1.0
    dataset: float = 1.0


@dataclass(frozen=True)
class BoundaryStateSupervisionConfig:
    """Training-only probes for musically structured recurrent state."""

    enabled: bool = False
    active_weight: float = 0.1
    reentry_weight: float = 0.1


@dataclass(frozen=True)
class CleanCacheTrainingConfig:
    """Chunk-transaction training matching Clean Acoustic Cache inference."""

    enabled: bool = False
    symbolic_state_curriculum: bool = False
    prediction_threshold: float = 0.5
    max_predicted_active_notes: int = 64


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    manifest: str
    output_dir: str
    model: ModelConfig
    stages: list[StageConfig]
    optimizer: OptimizerConfig = OptimizerConfig()
    distillation: DistillationConfig = DistillationConfig()
    seed: int = 1337
    precision: str = "bfloat16"
    global_batch_audio_seconds: int = 320
    conditioning: ConditioningConfig = ConditioningConfig()
    condition_dropout: ConditionDropoutConfig = ConditionDropoutConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    boundary_state_supervision: BoundaryStateSupervisionConfig = (
        BoundaryStateSupervisionConfig()
    )
    clean_cache_training: CleanCacheTrainingConfig = CleanCacheTrainingConfig()
    true_eos_loss_weight: float = 1.0
    num_workers: int = 4
    log_every: int = 20
    save_every: int = 2000
    distributed: str = "ddp"
    gradient_checkpointing: bool = True
    resume: str | None = None


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
    conditioning_was_explicit = "conditioning" in raw
    model = _model_from_raw(raw.pop("model"))
    stages = []
    for stage_raw in raw.pop("stages"):
        stage_raw = dict(stage_raw)
        distribution = stage_raw.get("context_distribution", {})
        if distribution:
            stage_raw["context_distribution"] = {
                int(chunks): float(weight)
                for chunks, weight in distribution.items()
            }
        stages.append(StageConfig(**stage_raw))
    optimizer = OptimizerConfig(**raw.pop("optimizer", {}))
    distillation = DistillationConfig(**raw.pop("distillation", {}))
    conditioning = ConditioningConfig(**raw.pop("conditioning", {}))
    augmentation_raw = dict(raw.pop("augmentation", {}))
    if "source_count_probabilities" in augmentation_raw:
        augmentation_raw["source_count_probabilities"] = {
            int(count): float(weight)
            for count, weight in augmentation_raw[
                "source_count_probabilities"
            ].items()
        }
    if "semitone_choices" in augmentation_raw:
        augmentation_raw["semitone_choices"] = tuple(
            int(value) for value in augmentation_raw["semitone_choices"]
        )
    if "allowed_datasets" in augmentation_raw:
        augmentation_raw["allowed_datasets"] = tuple(
            str(value) for value in augmentation_raw["allowed_datasets"]
        )
    augmentation = AugmentationConfig(**augmentation_raw)
    boundary_state_supervision = BoundaryStateSupervisionConfig(
        **raw.pop("boundary_state_supervision", {})
    )
    clean_cache_training = CleanCacheTrainingConfig(
        **raw.pop("clean_cache_training", {})
    )
    dropout_raw = raw.pop("condition_dropout", {})
    if isinstance(dropout_raw, (int, float)):
        # Backward-compatible parsing for pre-MIMuT YAML files.  Reusing the
        # single probability reproduces their historical behavior exactly.
        condition_dropout = ConditionDropoutConfig(
            audio=float(dropout_raw),
            instrument=float(dropout_raw),
            dataset=float(dropout_raw),
        )
    elif isinstance(dropout_raw, dict):
        condition_dropout = ConditionDropoutConfig(**dropout_raw)
    else:
        raise ValueError("condition_dropout must be a number or an object")
    # Old configurations predate the explicit mode contract.  Preserve their
    # historical non-audio-only dropout behavior instead of interpreting it as
    # a malformed new audio-only experiment.
    if (
        not conditioning_was_explicit
        and condition_dropout
        != ConditionDropoutConfig(audio=0.0, instrument=1.0, dataset=1.0)
    ):
        conditioning = ConditioningConfig(mode="conditioned")
    config = ExperimentConfig(
        model=model,
        stages=stages,
        optimizer=optimizer,
        distillation=distillation,
        conditioning=conditioning,
        condition_dropout=condition_dropout,
        augmentation=augmentation,
        boundary_state_supervision=boundary_state_supervision,
        clean_cache_training=clean_cache_training,
        **raw,
    )
    if config.precision not in {"bfloat16", "float16", "float32"}:
        raise ValueError("precision must be bfloat16, float16, or float32")
    if config.distributed not in {"ddp", "fsdp", "none"}:
        raise ValueError("distributed must be ddp, fsdp, or none")
    if not 0 <= config.optimizer.min_lr_ratio <= 1:
        raise ValueError("optimizer.min_lr_ratio must be in [0, 1]")
    if config.conditioning.mode not in {"audio_only", "conditioned"}:
        raise ValueError("conditioning.mode must be audio_only or conditioned")
    if config.conditioning.dropout_scope != "sequence":
        raise ValueError("conditioning.dropout_scope must be sequence")
    for name, probability in asdict(config.condition_dropout).items():
        if not 0 <= probability <= 1:
            raise ValueError(f"condition_dropout.{name} must be in [0, 1]")
    if config.conditioning.mode == "audio_only" and config.condition_dropout != (
        ConditionDropoutConfig(audio=0.0, instrument=1.0, dataset=1.0)
    ):
        raise ValueError(
            "audio_only requires condition_dropout audio=0, instrument=1, dataset=1"
        )
    if (
        config.conditioning.mode == "conditioned"
        and config.condition_dropout.dataset <= 0
    ):
        raise ValueError(
            "conditioned mode requires positive dataset dropout so NULL is trained"
        )
    if (
        conditioning_was_explicit
        and
        config.conditioning.mode == "conditioned"
        and abs(config.condition_dropout.dataset - 0.10) > 1e-12
    ):
        raise ValueError("conditioned mode fixes dataset dropout at 0.10")
    if config.distillation.weight < 0 or config.distillation.temperature <= 0:
        raise ValueError("distillation weight/temperature are invalid")
    if config.distillation.teacher_microbatch_size <= 0:
        raise ValueError("teacher_microbatch_size must be positive")
    if config.distillation.teacher_checkpoint:
        digest = (config.distillation.teacher_sha256 or "").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("enabled distillation requires a full teacher_sha256")
        if not config.distillation.teacher_revision:
            raise ValueError("enabled distillation requires teacher_revision")
        if config.distillation.teacher_microbatch_size != 4:
            raise ValueError("teacher_microbatch_size is fixed at 4")
        if abs(config.distillation.weight - 0.25) > 1e-12:
            raise ValueError("distillation weight is fixed at 0.25")
        if abs(config.distillation.temperature - 2.0) > 1e-12:
            raise ValueError("distillation temperature is fixed at 2.0")
    for name in ("remix_probability", "pitch_shift_probability"):
        probability = getattr(config.augmentation, name)
        if not 0 <= probability <= 1:
            raise ValueError(f"augmentation.{name} must be in [0, 1]")
    if config.augmentation.enabled and not config.augmentation.catalog:
        raise ValueError("enabled augmentation requires a catalog path")
    source_mass = sum(config.augmentation.source_count_probabilities.values())
    if source_mass <= 0 or any(
        count not in {2, 3, 4} or weight < 0
        for count, weight in config.augmentation.source_count_probabilities.items()
    ):
        raise ValueError("augmentation source-count probabilities are invalid")
    if config.augmentation.gain_db_min > config.augmentation.gain_db_max:
        raise ValueError("augmentation gain_db_min exceeds gain_db_max")
    if not config.augmentation.semitone_choices or any(
        value == 0 or abs(value) > 3 for value in config.augmentation.semitone_choices
    ):
        raise ValueError("augmentation semitone choices must be nonzero and within +/-3")
    if config.augmentation.enabled:
        expected = AugmentationConfig()
        if abs(config.augmentation.remix_probability - 0.25) > 1e-12:
            raise ValueError("augmentation remix_probability is fixed at 0.25")
        if abs(config.augmentation.pitch_shift_probability - 0.20) > 1e-12:
            raise ValueError("augmentation pitch_shift_probability is fixed at 0.20")
        if config.augmentation.source_count_probabilities != {
            2: 0.50,
            3: 0.30,
            4: 0.20,
        }:
            raise ValueError("augmentation source-count policy must be 2/3/4=.50/.30/.20")
        if (
            abs(config.augmentation.gain_db_min + 12.0) > 1e-12
            or abs(config.augmentation.gain_db_max) > 1e-12
        ):
            raise ValueError("augmentation gain range is fixed at -12..0 dB")
        if tuple(config.augmentation.semitone_choices) != expected.semitone_choices:
            raise ValueError("augmentation semitone choices are fixed at +/-1..3")
        if set(config.augmentation.allowed_datasets) != set(expected.allowed_datasets):
            raise ValueError("augmentation dataset whitelist is fixed")
    if config.boundary_state_supervision.active_weight < 0:
        raise ValueError(
            "boundary_state_supervision.active_weight must be non-negative"
        )
    if config.boundary_state_supervision.reentry_weight < 0:
        raise ValueError(
            "boundary_state_supervision.reentry_weight must be non-negative"
        )
    if config.boundary_state_supervision.enabled and not (
        config.boundary_state_supervision.active_weight
        or config.boundary_state_supervision.reentry_weight
    ):
        raise ValueError("enabled boundary-state supervision needs a positive weight")
    if config.true_eos_loss_weight <= 0:
        raise ValueError("true_eos_loss_weight must be positive")
    if not 0 < config.clean_cache_training.prediction_threshold < 1:
        raise ValueError("clean-cache prediction threshold must be in (0, 1)")
    if config.clean_cache_training.max_predicted_active_notes <= 0:
        raise ValueError("max_predicted_active_notes must be positive")
    if config.clean_cache_training.enabled:
        if not config.model.clean_acoustic_cache:
            raise ValueError(
                "clean_cache_training requires model.clean_acoustic_cache=true"
            )
        if config.conditioning.mode != "audio_only":
            raise ValueError("clean-cache training is strictly audio-only")
        if not config.boundary_state_supervision.enabled:
            raise ValueError("clean-cache training requires boundary supervision")
        if config.distillation.teacher_checkpoint:
            raise ValueError("clean-cache pilot does not support distillation")
        if config.augmentation.enabled:
            raise ValueError("clean-cache pilot disables augmentation")
    if not config.stages:
        raise ValueError("at least one training stage is required")
    for stage in config.stages:
        if stage.steps <= 0:
            raise ValueError(f"stage {stage.name!r} steps must be positive")
        if stage.context_chunks is not None and stage.context_distribution:
            raise ValueError(
                f"stage {stage.name!r} must use context_chunks or "
                "context_distribution, not both"
            )
        probabilities = stage.context_probabilities()
        if any(chunks not in {1, 2, 4, 8} for chunks in probabilities):
            raise ValueError(
                f"stage {stage.name!r} contexts must be one of 1, 2, 4, 8"
            )
        if any(weight <= 0 for weight in probabilities.values()):
            raise ValueError(
                f"stage {stage.name!r} context weights must be positive"
            )
    return config
