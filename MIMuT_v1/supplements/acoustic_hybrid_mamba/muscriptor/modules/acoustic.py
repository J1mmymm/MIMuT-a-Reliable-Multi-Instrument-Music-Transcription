"""Causal Mamba2 encoder for long-context log-mel conditioning.

This module is independent of the autoregressive MIDI decoder backbone.  Only
acoustic frames enter it, so carrying its state between five-second chunks
cannot feed generated-token errors back into later chunks.  Transformer and
Hybrid-Mamba decoders both allocate fresh token-generation state per chunk.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from muscriptor.models.config import ModelConfig
from muscriptor.modules.hybrid import StreamingMamba2
from muscriptor.modules.streaming import ModelState


class AcousticMambaBlock(nn.Module):
    """Pre-norm Mamba2 residual followed by a pre-norm 4D GELU FFN."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_index: int,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm1 = nn.LayerNorm(config.dim, eps=1e-5, **factory)
        self.mamba = StreamingMamba2(
            config,
            layer_idx=layer_index,
            **factory,
        )
        self.norm2 = nn.LayerNorm(config.dim, eps=1e-5, **factory)
        self.linear1 = nn.Linear(
            config.dim,
            config.hidden_scale * config.dim,
            bias=False,
            **factory,
        )
        self.linear2 = nn.Linear(
            config.hidden_scale * config.dim,
            config.dim,
            bias=False,
            **factory,
        )

        if config.acoustic_identity_init:
            # Both branches initially contribute exactly zero.  Together with
            # the released mel projection this makes a warm-started student
            # reproduce the teacher before the first optimiser update.
            nn.init.zeros_(self.mamba.mixer.out_proj.weight)
            nn.init.zeros_(self.linear2.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:
        hidden = inputs + self.mamba(self.norm1(inputs), model_state=model_state)
        return hidden + self.linear2(F.gelu(self.linear1(self.norm2(hidden))))


class AcousticMambaEncoder(nn.Module):
    """Stack of causal Mamba2 blocks operating only on acoustic frames."""

    def __init__(self, config: ModelConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        if config.acoustic_encoder != "mamba2":
            raise ValueError("AcousticMambaEncoder requires acoustic_encoder='mamba2'")
        self.layers = nn.ModuleList(
            AcousticMambaBlock(
                config,
                layer_index=index,
                device=device,
                dtype=dtype,
            )
            for index in range(config.acoustic_num_layers)
        )
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and model_state is None:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden, model_state=model_state)
        return hidden
