"""Hybrid Mamba/local-attention backbones.

The production Mamba path deliberately delegates the selective scan and
streaming step implementation to the official ``mamba-ssm`` package.  This
module only adapts its cache to MuScriptor's explicit ``ModelState`` API.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

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


def _load_mamba_prefill_kernels() -> tuple[Callable, Callable, Callable] | None:
    # The vectorized state update mirrors these two audited upstream releases.
    # ``auto`` must fall back rather than guessing against a different cache or
    # kernel ABI; explicit ``chunk`` mode will then fail with a clear error.
    try:
        if version("mamba-ssm") != "2.3.2.post1" or version(
            "causal-conv1d"
        ) != "1.6.2.post1":
            return None
    except PackageNotFoundError:
        return None
    try:  # pragma: no cover - CUDA-only optional dependency
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except Exception:
        return None
    return causal_conv1d_fn, causal_conv1d_update, mamba_chunk_scan_combined


class StreamingLocalAttention(StatefulModule):
    """Causal self-attention with a bounded sliding KV cache.

    CUDA uses FlashAttention's native window when available.  The fallback
    processes short query blocks with SDPA and never materializes a T x T
    attention matrix.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        query_chunk_size: int = 256,
        position_encoding: str = "sinusoidal",
        max_period: float = 10_000.0,
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
        self.position_encoding = position_encoding
        self.max_period = max_period
        if position_encoding not in {"sinusoidal", "rope"}:
            raise ValueError("position_encoding must be sinusoidal or rope")
        if position_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False, **factory)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, **factory)
        self._flash_attention = _load_flash_attention()

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

    def _rope(self, value: torch.Tensor, offset: int) -> torch.Tensor:
        """Apply interleaved RoPE using FP32 phases at absolute stream offsets."""

        positions = torch.arange(
            offset,
            offset + value.shape[2],
            device=value.device,
            dtype=torch.float32,
        )
        frequencies = torch.arange(
            0, self.head_dim, 2, device=value.device, dtype=torch.float32
        )
        frequencies = self.max_period ** (-frequencies / self.head_dim)
        angles = positions[:, None] * frequencies[None, :]
        cos = angles.cos().view(1, 1, value.shape[2], -1)
        sin = angles.sin().view(1, 1, value.shape[2], -1)
        fp32 = value.float()
        even = fp32[..., 0::2]
        odd = fp32[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)
        return rotated.to(value.dtype)

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
            if old_k.dtype != new_k.dtype:
                old_k = old_k.to(new_k.dtype)
                old_v = old_v.to(new_v.dtype)
                state["key"], state["value"] = old_k, old_v
            k = torch.cat([old_k, new_k], dim=2)
            v = torch.cat([old_v, new_v], dim=2)
            q_offset = int(state["offset"])
            k_offset = q_offset - old_k.shape[2]

        if self.position_encoding == "rope":
            q = self._rope(q, q_offset)
            new_k = self._rope(new_k, q_offset)
            if state is None:
                k = new_k
            else:
                k = torch.cat([old_k, new_k], dim=2)

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

        if state is not None:
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
            cfg.position_encoding,
            cfg.max_period,
            **factory,
        )
        self.ffn = FeedForward(cfg.dim, cfg.hidden_scale * cfg.dim, **factory)

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm(x), model_state=model_state)
        return self.ffn(x)


class StreamingMamba2(StatefulModule):
    def __init__(self, cfg: ModelConfig, layer_idx: int, device=None, dtype=None):
        super().__init__()
        Mamba2, InferenceParams = _load_mamba()
        self._inference_params_cls = InferenceParams
        self.layer_idx = layer_idx
        self.prefill_mode = cfg.prefill_mode
        self._prefill_kernels = _load_mamba_prefill_kernels()
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

    def _chunk_prefill(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Vectorized cached prefill equivalent to official repeated ``step``."""

        if self._prefill_kernels is None:
            raise MambaDependencyError(
                "chunk prefill needs causal-conv1d and Mamba SSD kernels"
            )
        (
            causal_conv1d_fn,
            causal_conv1d_update,
            mamba_chunk_scan_combined,
        ) = self._prefill_kernels
        mixer = self.mixer
        conv_state, ssm_state = params.key_value_memory_dict[self.layer_idx]

        zxbcdt = mixer.in_proj(x)
        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * mixer.d_ssm
            - 2 * mixer.ngroups * mixer.d_state
            - mixer.nheads
        ) // 2
        z0, x0, z, xbc, dt = torch.split(
            zxbcdt,
            [
                d_mlp,
                d_mlp,
                mixer.d_ssm,
                mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state,
                mixer.nheads,
            ],
            dim=-1,
        )
        # v1.6.2's update kernel accepts a whole sequence and updates the
        # d_conv-sized cache in place, retaining exactly the history used by
        # the single-token path.
        # Preserve the channel-last storage layout (channel stride == 1):
        # causal-conv's differentiable initial-state path requires it.
        # ``xbc`` is a split view into a wider projection whose parent stride
        # is not necessarily 8-aligned.  Materialize B,L,D first, then
        # transpose so the kernel sees channel stride 1 and aligned B/L strides.
        raw_xbc = xbc.contiguous().transpose(1, 2)
        if conv_state.dtype != raw_xbc.dtype:
            conv_state = conv_state.to(raw_xbc.dtype)
            ssm_state = ssm_state.to(raw_xbc.dtype)
            params.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        differentiable = torch.is_grad_enabled() and zxbcdt.requires_grad
        if differentiable:
            initial_conv = conv_state[..., -(mixer.d_conv - 1) :]
            xbc = causal_conv1d_fn(
                raw_xbc,
                rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
                bias=mixer.conv1d.bias,
                initial_states=initial_conv,
                activation=mixer.activation,
            ).transpose(1, 2)
        else:
            xbc = causal_conv1d_update(
                raw_xbc.contiguous(),
                conv_state,
                rearrange(mixer.conv1d.weight, "d 1 w -> d w"),
                mixer.conv1d.bias,
                mixer.activation,
            ).transpose(1, 2)
        x_branch, b_branch, c_branch = torch.split(
            xbc,
            [
                mixer.d_ssm,
                mixer.ngroups * mixer.d_state,
                mixer.ngroups * mixer.d_state,
            ],
            dim=-1,
        )
        a = -torch.exp(mixer.A_log.float())
        dt_limit = (
            {}
            if mixer.dt_limit == (0.0, float("inf"))
            else {"dt_limit": mixer.dt_limit}
        )
        d_skip = (
            rearrange(mixer.D, "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else mixer.D
        )
        scanned = mamba_chunk_scan_combined(
            rearrange(x_branch, "b l (h p) -> b l h p", p=mixer.headdim),
            dt,
            a,
            rearrange(
                b_branch,
                "b l (g n) -> b l g n",
                g=mixer.ngroups,
            ),
            rearrange(
                c_branch,
                "b l (g n) -> b l g n",
                g=mixer.ngroups,
            ),
            chunk_size=mixer.chunk_size,
            D=d_skip,
            z=(
                rearrange(z, "b l (h p) -> b l h p", p=mixer.headdim)
                if not mixer.rmsnorm
                else None
            ),
            dt_bias=mixer.dt_bias,
            dt_softplus=True,
            initial_states=ssm_state,
            return_final_states=True,
            **dt_limit,
        )
        y, final_state = scanned[:2]
        if not differentiable:
            with torch.no_grad():
                ssm_state.copy_(final_state)
        y = rearrange(y, "b l h p -> b l (h p)")
        if mixer.rmsnorm:
            y = mixer.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        return mixer.out_proj(y)

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

    def forward(
        self, x: torch.Tensor, model_state: ModelState | None = None
    ) -> torch.Tensor:
        state = self.get_state(model_state)
        params = None if state is None else state["inference_params"]
        if params is not None:
            cache = params.key_value_memory_dict[self.layer_idx]

            def compute_dtype(value):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    return value.to(dtype=x.dtype)
                if isinstance(value, tuple):
                    return tuple(compute_dtype(item) for item in value)
                if isinstance(value, list):
                    return [compute_dtype(item) for item in value]
                return value

            if any(
                isinstance(item, torch.Tensor) and item.dtype != x.dtype
                for item in cache
            ):
                params.key_value_memory_dict[self.layer_idx] = compute_dtype(cache)
        # Mamba2's cached ``step`` path accepts exactly one token.  At a new
        # five-second chunk we intentionally keep the recurrent cache but feed
        # a multi-token prefix (chunk marker, conditioning, and BOS).  Advance
        # that prefix one token at a time while temporarily presenting the
        # correct offset to Mamba2.  ``LMModel.generate`` performs the single
        # authoritative state increment after this call, so restore the base
        # offset to avoid counting these tokens twice.
        if params is not None and x.shape[1] > 1:
            if self.prefill_mode != "step" and self._prefill_kernels is not None:
                return self._chunk_prefill(x, params)
            if self.prefill_mode == "chunk":
                raise MambaDependencyError(
                    "prefill_mode=chunk requested but optimized kernels are unavailable"
                )
            base_offset = params.seqlen_offset
            outputs = []
            try:
                for token_index in range(x.shape[1]):
                    params.seqlen_offset = base_offset + token_index
                    # mamba-ssm 2.3.2's full-sequence causal-conv kernel has
                    # a batch-one/channel-last stride restriction even for a
                    # single token.  A zero cache is a valid first recurrent
                    # step, so enter the official ``step`` branch directly.
                    if params.seqlen_offset == 0:
                        params.seqlen_offset = 1
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
        if self.cfg.position_encoding == "sinusoidal":
            state = self.get_state(model_state)
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
