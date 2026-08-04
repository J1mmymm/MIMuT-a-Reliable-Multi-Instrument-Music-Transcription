"""Hybrid Mamba/local-attention backbones.

The production Mamba path deliberately delegates the selective scan and
streaming step implementation to the official ``mamba-ssm`` package.  This
module only adapts its cache to MuScriptor's explicit ``ModelState`` API.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


logger = logging.getLogger(__name__)

from muscriptor.models.config import ModelConfig
from muscriptor.modules.streaming import ModelState, State, StatefulModule
from muscriptor.modules.transformer import (
    StreamingTransformer,
    create_sin_embedding,
)


class MambaDependencyError(RuntimeError):
    pass


def _load_mamba() -> tuple[type[nn.Module], type[Any]]:
    try:
        from mamba_ssm import Mamba2
        from mamba_ssm.utils.generation import InferenceParams
    except Exception as exc:  # pragma: no cover - exercised on CUDA install
        raise MambaDependencyError(
            "Hybrid-Mamba requires the optional CUDA dependencies. Install "
            "the project with `pip install -e '.[train,mamba]'` on Linux/CUDA."
        ) from exc
    return Mamba2, InferenceParams


def _load_flash_attention() -> Callable | None:
    try:  # pragma: no cover - availability depends on the CUDA environment
        from flash_attn import flash_attn_func
    except Exception:
        return None
    return flash_attn_func


def _rope_rotate(
    x: torch.Tensor,  # [B, H, T, Dh]
    positions: torch.Tensor,  # [T] absolute positions
    inv_freq: torch.Tensor,  # [Dh / 2]
) -> torch.Tensor:
    """Apply rotary position embeddings (half-split convention).

    Because rotary phases cancel inside the q·k dot product, attention scores
    depend only on relative offsets — bounded by the sliding window — so any
    absolute stream position is in-distribution.  Angles are computed in
    float64 where supported: at whole-song offsets (10^5+ tokens) float32
    ``position * inv_freq`` alone loses ~centirad phase accuracy.
    """
    angle_dtype = (
        torch.float32 if positions.device.type == "mps" else torch.float64
    )
    angles = (
        positions.to(angle_dtype)[:, None] * inv_freq.to(angle_dtype)[None, :]
    )  # [T, Dh/2]
    cos = torch.cos(angles)[None, None].to(torch.float32)  # [1, 1, T, Dh/2]
    sin = torch.sin(angles)[None, None].to(torch.float32)
    half = x.shape[-1] // 2
    x32 = x.to(torch.float32)
    x1, x2 = x32[..., :half], x32[..., half:]
    rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return rotated.to(x.dtype)


class StreamingLocalAttention(StatefulModule):
    """Causal self-attention with a bounded sliding KV cache.

    CUDA uses FlashAttention's native window when available.  The fallback
    processes short query blocks with SDPA and never materializes a T x T
    attention matrix.  With ``rope=True`` rotary embeddings are applied to
    q/k using absolute stream offsets (relative inside the dot product), and
    the backbone skips its additive absolute sinusoidal embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        query_chunk_size: int = 256,
        rope: bool = False,
        rope_max_period: float = 10_000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        factory = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.query_chunk_size = query_chunk_size
        self.rope = rope
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory)
        self._flash_attention = _load_flash_attention()
        if rope:
            if self.head_dim % 2:
                raise ValueError("rope requires an even head dimension")
            inv_freq = 1.0 / (
                rope_max_period
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.float32)
                    / self.head_dim
                )
            )
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        del sequence_length
        weight = self.in_proj.weight
        shape = (batch_size, self.num_heads, 0, self.head_dim)
        return {
            "key": torch.empty(shape, device=weight.device, dtype=weight.dtype),
            "value": torch.empty(shape, device=weight.device, dtype=weight.dtype),
            "offset": 0,
        }

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offset"] += increment

    def reorder_state(self, state: State, indices: torch.Tensor) -> None:
        for name in ("key", "value"):
            value = state[name]
            state[name] = value.index_select(0, indices.to(value.device))

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, t, _ = x.shape
        qkv = self.in_proj(x).view(b, t, 3, self.num_heads, self.head_dim)
        # SDPA layout: [B, H, T, Dh].
        return tuple(qkv[:, :, i].transpose(1, 2) for i in range(3))

    def _sdpa_window(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_offset: int,
        k_offset: int,
    ) -> torch.Tensor:
        outputs = []
        total_q = q.shape[2]
        for start in range(0, total_q, self.query_chunk_size):
            end = min(total_q, start + self.query_chunk_size)
            abs_q_start = q_offset + start
            abs_q_end = q_offset + end
            key_start_abs = max(k_offset, abs_q_start - self.window_size + 1)
            key_end_abs = min(k_offset + k.shape[2], abs_q_end)
            ks = key_start_abs - k_offset
            ke = key_end_abs - k_offset
            q_part = q[:, :, start:end]
            k_part = k[:, :, ks:ke]
            v_part = v[:, :, ks:ke]

            q_pos = torch.arange(abs_q_start, abs_q_end, device=q.device).view(-1, 1)
            k_pos = torch.arange(key_start_abs, key_end_abs, device=q.device).view(
                1, -1
            )
            allowed = (k_pos <= q_pos) & (k_pos >= q_pos - self.window_size + 1)
            out = F.scaled_dot_product_attention(
                q_part,
                k_part,
                v_part,
                attn_mask=allowed.view(1, 1, end - start, ke - ks),
                dropout_p=0.0,
            )
            outputs.append(out)
        return torch.cat(outputs, dim=2)

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        q, new_k, new_v = self._project(x)
        state = self.get_state(model_state)

        if state is None:
            k, v = new_k, new_v
            q_offset = k_offset = 0
        else:
            old_k, old_v = state["key"], state["value"]
            k = torch.cat([old_k, new_k], dim=2)
            v = torch.cat([old_v, new_v], dim=2)
            q_offset = int(state["offset"])
            k_offset = q_offset - old_k.shape[2]

        if self.rope:
            # The cached keys are stored *unrotated*; rotate the full k window
            # on every call so cached and fresh keys share one phase basis.
            q_pos = torch.arange(
                q_offset, q_offset + q.shape[2], device=x.device
            )
            k_pos = torch.arange(
                k_offset, k_offset + k.shape[2], device=x.device
            )
            if state is not None:
                state["key"] = k[:, :, -self.window_size :].detach()
                state["value"] = v[:, :, -self.window_size :].detach()
            q = _rope_rotate(q, q_pos, self.rope_inv_freq)
            k = _rope_rotate(k, k_pos, self.rope_inv_freq)

        use_flash = (
            self._flash_attention is not None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:  # pragma: no cover - CUDA-only optimized path
            out = self._flash_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0,
                causal=True,
                window_size=(self.window_size - 1, 0),
            ).transpose(1, 2)
        else:
            out = self._sdpa_window(q, k, v, q_offset, k_offset)

        if state is not None and not self.rope:
            # (With rope the unrotated window was already stored above.)
            state["key"] = k[:, :, -self.window_size :].detach()
            state["value"] = v[:, :, -self.window_size :].detach()

        b, _, t, _ = out.shape
        out = out.transpose(1, 2).reshape(b, t, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm = nn.LayerNorm(dim, eps=1e-5, **factory)
        self.linear1 = nn.Linear(dim, hidden_dim, bias=False, **factory)
        self.linear2 = nn.Linear(hidden_dim, dim, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear2(F.gelu(self.linear1(self.norm(x))))


class LocalTransformerLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-5, **factory)
        self.mixer = StreamingLocalAttention(
            cfg.dim,
            cfg.num_heads,
            cfg.local_window,
            cfg.local_query_chunk,
            rope=cfg.position_encoding == "rope",
            rope_max_period=cfg.max_period,
            **factory,
        )
        self.ffn = FeedForward(cfg.dim, cfg.hidden_scale * cfg.dim, **factory)

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), model_state=model_state)
        return self.ffn(x)


def _fast_prefill_enabled() -> bool:
    return os.environ.get("MUSCRIPTOR_MAMBA_FAST_PREFILL", "0").lower() not in {
        "",
        "0",
        "false",
    }


class StreamingMamba2(StatefulModule):
    def __init__(self, cfg: ModelConfig, layer_idx: int, device=None, dtype=None):
        super().__init__()
        Mamba2, InferenceParams = _load_mamba()
        self._inference_params_cls = InferenceParams
        self._fast_prefill_failed = False
        self.layer_idx = layer_idx
        self.mixer = Mamba2(
            d_model=cfg.dim,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        cache = self.mixer.allocate_inference_cache(
            batch_size=batch_size,
            max_seqlen=sequence_length,
        )
        params = self._inference_params_cls(
            max_seqlen=sequence_length,
            max_batch_size=batch_size,
            key_value_memory_dict={self.layer_idx: cache},
        )
        return {"inference_params": params}

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["inference_params"].seqlen_offset += increment

    def reorder_state(self, state: State, indices: torch.Tensor) -> None:
        params = state["inference_params"]
        cache = params.key_value_memory_dict[self.layer_idx]

        def reorder(value):
            if isinstance(value, torch.Tensor):
                return value.index_select(0, indices.to(value.device))
            if isinstance(value, tuple):
                return tuple(reorder(v) for v in value)
            if isinstance(value, list):
                return [reorder(v) for v in value]
            return value

        params.key_value_memory_dict[self.layer_idx] = reorder(cache)
        lengths = getattr(params, "lengths_per_sample", None)
        if isinstance(lengths, torch.Tensor):
            params.lengths_per_sample = lengths.index_select(
                0, indices.to(lengths.device)
            )

    def _fast_prefill(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Multi-token continuation through the chunked-scan kernel.

        Replicates the official ``Mamba2.forward`` computation but threads the
        cached conv/SSM state in as initial state, so a ~500-token chunk
        prefix costs one kernel launch instead of ~500 sequential ``step``
        calls per layer.  Only used when MUSCRIPTOR_MAMBA_FAST_PREFILL=1; the
        CUDA test in tests/test_fast_prefill_cuda.py verifies equivalence
        with the step path before you enable it.
        """
        from einops import rearrange
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

        m = self.mixer
        if not getattr(m, "rmsnorm", True):
            raise RuntimeError("fast prefill supports the default rmsnorm=True only")
        d_state = m.d_state
        d_conv = m.d_conv
        ngroups = getattr(m, "ngroups", 1)
        d_ssm = getattr(m, "d_ssm", None) or m.d_inner
        nheads = m.nheads
        conv_state, ssm_state = params.key_value_memory_dict[self.layer_idx]

        zxbcdt = m.in_proj(x)  # [B, L, d_in_proj]
        d_mlp = (
            zxbcdt.shape[-1] - 2 * d_ssm - 2 * ngroups * d_state - nheads
        ) // 2
        if d_mlp != 0:
            raise RuntimeError("fast prefill supports the default head layout only")
        z, xBC, dt = torch.split(
            zxbcdt, [d_ssm, d_ssm + 2 * ngroups * d_state, nheads], dim=-1
        )

        # Depthwise causal conv, seeded with the cached last (d_conv - 1)
        # inputs; then refresh the cache with the stream's last d_conv inputs.
        xBC_t = xBC.transpose(1, 2)  # [B, conv_dim, L]
        prev = conv_state[:, :, 1:].to(xBC_t.dtype)
        padded = torch.cat([prev, xBC_t], dim=-1)
        conv = F.conv1d(
            padded, m.conv1d.weight, m.conv1d.bias, groups=xBC_t.shape[1]
        )
        conv_state.copy_(
            torch.cat([conv_state.to(xBC_t.dtype), xBC_t], dim=-1)[:, :, -d_conv:]
        )
        xBC = m.act(conv).transpose(1, 2)  # [B, L, conv_dim]

        x_part, B_part, C_part = torch.split(
            xBC, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1
        )
        A = -torch.exp(m.A_log.float())
        D = (
            rearrange(m.D, "(h p) -> h p", p=m.headdim)
            if getattr(m, "D_has_hdim", False)
            else m.D
        )
        y, last_state = mamba_chunk_scan_combined(
            rearrange(x_part, "b l (h p) -> b l h p", p=m.headdim),
            dt,
            A,
            rearrange(B_part, "b l (g n) -> b l g n", g=ngroups),
            rearrange(C_part, "b l (g n) -> b l g n", g=ngroups),
            chunk_size=m.chunk_size,
            D=D,
            z=None,
            dt_bias=m.dt_bias,
            initial_states=ssm_state.to(A.dtype),
            dt_softplus=True,
            return_final_states=True,
        )
        ssm_state.copy_(last_state)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = m.norm(y, z)
        return m.out_proj(y)

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        state = self.get_state(model_state)
        params = None if state is None else state["inference_params"]
        # Mamba2's cached ``step`` path accepts exactly one token.  At a new
        # five-second chunk we intentionally keep the recurrent cache but feed
        # a multi-token prefix (chunk marker, conditioning, and BOS).  Advance
        # that prefix one token at a time while temporarily presenting the
        # correct offset to Mamba2.  ``LMModel.generate`` performs the single
        # authoritative state increment after this call, so restore the base
        # offset to avoid counting these tokens twice.
        if params is not None and params.seqlen_offset > 0 and x.shape[1] > 1:
            if _fast_prefill_enabled() and not self._fast_prefill_failed:
                try:
                    return self._fast_prefill(x, params)
                except Exception:  # pragma: no cover - depends on mamba version
                    self._fast_prefill_failed = True
                    logger.warning(
                        "fast Mamba prefill failed; falling back to the "
                        "token-by-token step path",
                        exc_info=True,
                    )
            base_offset = params.seqlen_offset
            outputs = []
            try:
                for token_index in range(x.shape[1]):
                    params.seqlen_offset = base_offset + token_index
                    outputs.append(
                        self.mixer(
                            x[:, token_index : token_index + 1],
                            inference_params=params,
                        )
                    )
            finally:
                params.seqlen_offset = base_offset
            return torch.cat(outputs, dim=1)
        return self.mixer(x, inference_params=params)


class MambaLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm = nn.LayerNorm(cfg.dim, eps=1e-5, **factory)
        self.mixer = StreamingMamba2(cfg, layer_idx, **factory)
        self.ffn = FeedForward(cfg.dim, cfg.hidden_scale * cfg.dim, **factory)

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), model_state=model_state)
        return self.ffn(x)


class StreamingMixedBackbone(StatefulModule):
    """Stack of Mamba2 and bounded local-attention layers."""

    def __init__(self, cfg: ModelConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg
        self.max_period = cfg.max_period
        self.gradient_checkpointing = False
        layers: list[nn.Module] = []
        mamba_idx = 0
        for index in range(cfg.num_layers):
            if cfg.backbone == "local_transformer":
                kind = "local"
            elif cfg.backbone == "pure_mamba":
                kind = "mamba"
            else:
                # Mamba2 -> Mamba2 -> local attention.
                kind = "local" if index % 3 == 2 else "mamba"
            if kind == "local":
                layers.append(LocalTransformerLayer(cfg, device=device, dtype=dtype))
            else:
                layers.append(
                    MambaLayer(cfg, layer_idx=mamba_idx, device=device, dtype=dtype)
                )
                mamba_idx += 1
        self.layers = nn.ModuleList(layers)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def init_state(self, batch_size: int, sequence_length: int) -> State:
        device = next(self.parameters()).device
        return {"offsets": torch.zeros(batch_size, dtype=torch.long, device=device)}

    def increment_step(self, state: State, increment: int = 1) -> None:
        state["offsets"] += increment

    def reorder_state(self, state: State, indices: torch.Tensor) -> None:
        offsets = state["offsets"]
        state["offsets"] = offsets.index_select(0, indices.to(offsets.device))

    def forward(
        self,
        x: torch.Tensor,
        prepend_length: int = 0,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:
        del prepend_length
        b, t, dim = x.shape
        state = self.get_state(model_state)
        if self.cfg.position_encoding != "rope":
            offsets = (
                state["offsets"]
                if state is not None
                else torch.zeros(b, dtype=torch.long, device=x.device)
            )
            positions = torch.arange(t, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos = create_sin_embedding(
                positions, dim, max_period=self.max_period, dtype=torch.float32
            ).to(x.dtype)
            x = x + pos

        for layer in self.layers:
            if self.gradient_checkpointing and self.training and model_state is None:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x, model_state=model_state)
        return x


def build_backbone(cfg: ModelConfig, device=None, dtype=None) -> nn.Module:
    if cfg.backbone == "transformer":
        return StreamingTransformer(
            d_model=cfg.dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.hidden_scale * cfg.dim,
            max_period=cfg.max_period,
            device=device,
            dtype=dtype,
        )
    return StreamingMixedBackbone(cfg, device=device, dtype=dtype)
