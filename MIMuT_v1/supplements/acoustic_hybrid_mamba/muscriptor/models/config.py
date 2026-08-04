"""Serializable model/backbone configuration.

The published MuScriptor checkpoints predate this module and are represented
by ``backbone="transformer"`` with ``use_type_embeddings=False``.  New
Hybrid-Mamba checkpoints persist every field below in ``config.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


BackboneKind = Literal["transformer", "hybrid_mamba", "local_transformer", "pure_mamba"]
AcousticEncoderKind = Literal["none", "mamba2"]
PositionEncodingKind = Literal["sinusoidal", "rope"]


@dataclass(frozen=True)
class ModelConfig:
    dim: int
    num_heads: int
    num_layers: int
    card: int = 1395
    backbone: BackboneKind = "transformer"
    hidden_scale: int = 4
    max_period: float = 10_000.0
    local_window: int = 2048
    local_query_chunk: int = 256
    # "sinusoidal" adds absolute sin/cos embeddings to the input (historical
    # behavior; extrapolates poorly past the trained sequence length).
    # "rope" applies rotary embeddings inside the bounded local-attention
    # layers instead: attention scores then depend only on relative offsets,
    # which the sliding window caps at ``local_window``, so whole-song
    # streaming never sees an out-of-distribution position.  Only meaningful
    # for the mixed/local backbones; the full-attention "transformer"
    # backbone keeps its historical sinusoidal path regardless.
    position_encoding: PositionEncodingKind = "sinusoidal"
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_backend: str = "official"
    # The autoregressive backbone and acoustic encoder are deliberately
    # separate.  ``acoustic_encoder=mamba2`` applies a causal Mamba stack only
    # to log-mel frames.  Its state may cross audio chunks; the decoder state
    # is always freshly allocated for each five-second chunk, irrespective of
    # whether that decoder is Transformer or Hybrid-Mamba.
    acoustic_encoder: AcousticEncoderKind = "none"
    acoustic_num_layers: int = 0
    acoustic_identity_init: bool = True
    use_type_embeddings: bool = False
    segment_duration: float = 5.0
    tokenizer_name: str = "MT3_FULL_PLUS"
    tokenizer_max_shift_steps: int = 1001
    num_instrument_condition_classes: int = 1000
    num_dataset_condition_classes: int = 4
    correct_class_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.dim <= 0 or self.num_layers <= 0:
            raise ValueError("dim and num_layers must be positive")
        if self.dim % self.num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if self.local_window <= 0 or self.local_query_chunk <= 0:
            raise ValueError("local attention sizes must be positive")
        if self.backbone not in {
            "transformer",
            "hybrid_mamba",
            "local_transformer",
            "pure_mamba",
        }:
            raise ValueError(f"unsupported backbone: {self.backbone!r}")
        if self.position_encoding not in {"sinusoidal", "rope"}:
            raise ValueError(
                f"unsupported position_encoding: {self.position_encoding!r}"
            )
        if self.position_encoding == "rope" and (self.dim // self.num_heads) % 2:
            raise ValueError("rope requires an even attention head dimension")
        if self.backbone in {"hybrid_mamba", "pure_mamba"}:
            inner = self.dim * self.mamba_expand
            if inner % self.mamba_headdim:
                raise ValueError(
                    "dim * mamba_expand must be divisible by mamba_headdim"
                )
            heads = inner // self.mamba_headdim
            fused_projection = 2 * inner + 2 * self.mamba_d_state + heads
            if self.mamba_backend == "official" and fused_projection % 8:
                raise ValueError(
                    "official causal-conv1d requires the fused Mamba projection "
                    "stride to be divisible by 8; adjust dim, d_state, or headdim"
                )
        if self.acoustic_encoder not in {"none", "mamba2"}:
            raise ValueError(
                f"unsupported acoustic_encoder: {self.acoustic_encoder!r}"
            )
        if self.acoustic_encoder == "none" and self.acoustic_num_layers != 0:
            raise ValueError(
                "acoustic_num_layers must be zero when acoustic_encoder='none'"
            )
        if self.acoustic_encoder == "mamba2":
            if self.acoustic_num_layers <= 0:
                raise ValueError(
                    "acoustic_num_layers must be positive for the Mamba2 encoder"
                )
            inner = self.dim * self.mamba_expand
            if inner % self.mamba_headdim:
                raise ValueError(
                    "dim * mamba_expand must be divisible by mamba_headdim"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})


LEGACY_PRESETS: dict[str, ModelConfig] = {
    "small": ModelConfig(dim=768, num_heads=12, num_layers=14, card=1393),
    "medium": ModelConfig(dim=1024, num_heads=16, num_layers=24, card=1395),
    "large": ModelConfig(dim=1536, num_heads=24, num_layers=48, card=1395),
}


HYBRID_PRESETS: dict[str, ModelConfig] = {
    "hybrid-small": ModelConfig(
        dim=768,
        num_heads=12,
        num_layers=13,
        backbone="hybrid_mamba",
        use_type_embeddings=True,
        num_dataset_condition_classes=64,
        correct_class_conditioning=True,
    ),
    "hybrid-medium": ModelConfig(
        dim=1024,
        num_heads=16,
        num_layers=22,
        backbone="hybrid_mamba",
        use_type_embeddings=True,
        num_dataset_condition_classes=64,
        correct_class_conditioning=True,
    ),
    "hybrid-large": ModelConfig(
        dim=1536,
        num_heads=24,
        num_layers=44,
        backbone="hybrid_mamba",
        use_type_embeddings=True,
        num_dataset_condition_classes=64,
        correct_class_conditioning=True,
    ),
}


ACOUSTIC_MAMBA_PRESETS: dict[str, ModelConfig] = {
    # Compatibility preset for the first Acoustic-Mamba implementation.
    "acoustic-mamba-medium": ModelConfig(
        dim=1024,
        num_heads=16,
        num_layers=24,
        card=1395,
        backbone="transformer",
        acoustic_encoder="mamba2",
        acoustic_num_layers=6,
        acoustic_identity_init=True,
        use_type_embeddings=False,
        num_dataset_condition_classes=64,
        # Keep the released checkpoint's class-token indexing.  This makes
        # the null inference condition and copied instrument embeddings exact.
        correct_class_conditioning=False,
    ),
    # Main dual-Mamba architecture: whole-song acoustic state plus the
    # validated 2xMamba2/1xLocal-Attention token decoder.  Decoder state is
    # intentionally reset at every audio chunk so generated-token errors do
    # not enter the next chunk's recurrent state.
    "acoustic-hybrid-mamba-medium": ModelConfig(
        dim=1024,
        num_heads=16,
        num_layers=22,
        card=1395,
        backbone="hybrid_mamba",
        position_encoding="rope",
        local_window=4096,
        acoustic_encoder="mamba2",
        acoustic_num_layers=6,
        acoustic_identity_init=True,
        use_type_embeddings=True,
        num_dataset_condition_classes=64,
        correct_class_conditioning=True,
    ),
}


MODEL_PRESETS = {**LEGACY_PRESETS, **HYBRID_PRESETS, **ACOUSTIC_MAMBA_PRESETS}


# Parameter-matched full-attention controls used by the three experiment
# scales.  These are deliberately not the released MuScriptor depths: the
# Mamba2 mixer has slightly more parameters than self-attention at the same
# width, so 15/25/49 layers are the fair controls for 13/22/44 Hybrid layers.
MATCHED_TRANSFORMER_LAYERS = {
    "hybrid-small": 15,
    "hybrid-medium": 25,
    "hybrid-large": 49,
}


def estimate_backbone_parameters(config: ModelConfig) -> int:
    """Return the exact architectural parameter count without loading CUDA.

    The Mamba2 expression follows the official implementation with
    ``d_ssm=d_inner``, one group, disabled linear biases, enabled depthwise
    convolution bias, and gated RMSNorm.  A CUDA acceptance test compares
    this estimate against the instantiated official module.
    """

    dim = config.dim
    ffn = 2 * dim * (config.hidden_scale * dim)
    layer_norms = 4 * dim  # two LayerNorm weights and biases
    attention = 4 * dim * dim

    def local_layer() -> int:
        return attention + ffn + layer_norms

    def mamba_layer() -> int:
        d_inner = config.mamba_expand * dim
        heads = d_inner // config.mamba_headdim
        conv_dim = d_inner + 2 * config.mamba_d_state
        in_projection = dim * (2 * d_inner + 2 * config.mamba_d_state + heads)
        convolution = conv_dim * (config.mamba_d_conv + 1)
        dt_a_d = 3 * heads
        gated_rms_norm = d_inner
        out_projection = d_inner * dim
        mixer = in_projection + convolution + dt_a_d + gated_rms_norm + out_projection
        return mixer + ffn + layer_norms

    if config.backbone == "transformer":
        # Historical Transformer has two affine LayerNorms in every layer.
        return config.num_layers * (attention + ffn + layer_norms)
    if config.backbone == "local_transformer":
        return config.num_layers * local_layer()
    if config.backbone == "pure_mamba":
        return config.num_layers * mamba_layer()
    # Mamba2 -> Mamba2 -> local attention.
    return sum(
        local_layer() if (index + 1) % 3 == 0 else mamba_layer()
        for index in range(config.num_layers)
    )


def estimate_total_parameters(config: ModelConfig) -> int:
    """Estimate LM plus all three MuScriptor conditioning modules."""

    dim = config.dim
    total = estimate_backbone_parameters(config)
    total += (config.card + 1) * dim  # input tokens
    total += config.card * dim  # logits head
    total += 2 * dim  # output LayerNorm
    total += 512 * dim + dim  # mel projection weight and bias
    total += (config.num_instrument_condition_classes + 1) * dim
    total += (config.num_dataset_condition_classes + 1) * dim
    if config.use_type_embeddings:
        total += 4 * dim + dim  # type table and chunk marker
    if config.acoustic_encoder == "mamba2":
        total += config.acoustic_num_layers * _estimate_mamba_ffn_layer(config)
    return total


def _estimate_mamba_ffn_layer(config: ModelConfig) -> int:
    """Parameters in one pre-norm Mamba2 + 4D FFN acoustic block."""

    dim = config.dim
    d_inner = config.mamba_expand * dim
    heads = d_inner // config.mamba_headdim
    conv_dim = d_inner + 2 * config.mamba_d_state
    in_projection = dim * (2 * d_inner + 2 * config.mamba_d_state + heads)
    convolution = conv_dim * (config.mamba_d_conv + 1)
    dt_a_d = 3 * heads
    gated_rms_norm = d_inner
    out_projection = d_inner * dim
    mixer = in_projection + convolution + dt_a_d + gated_rms_norm + out_projection
    ffn = 2 * dim * (config.hidden_scale * dim)
    layer_norms = 4 * dim
    return mixer + ffn + layer_norms


def estimate_acoustic_encoder_parameters(config: ModelConfig) -> int:
    if config.acoustic_encoder == "none":
        return 0
    return config.acoustic_num_layers * _estimate_mamba_ffn_layer(config)


def matched_transformer_config(hybrid_name: str) -> ModelConfig:
    """Build the parameter-matched Transformer control for a Hybrid preset."""

    hybrid = HYBRID_PRESETS[hybrid_name]
    return ModelConfig(
        **{
            **hybrid.to_dict(),
            "backbone": "transformer",
            "num_layers": MATCHED_TRANSFORMER_LAYERS[hybrid_name],
        }
    )
