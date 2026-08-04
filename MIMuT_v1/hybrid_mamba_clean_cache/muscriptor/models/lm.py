"""Language model for MIDI token generation.

Adapted from audiocraft/models/lm.py.
"""

import logging
import time
from collections.abc import Iterator

import torch
from torch import nn

import muscriptor.accelerator
from muscriptor.models.config import ModelConfig
from muscriptor.modules.conditioners import (
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    nullify_all_conditions,
)
from muscriptor.modules.streaming import (
    ModelState,
    increment_steps,
    init_states,
    reorder_states,
)
from muscriptor.modules.hybrid import build_backbone
import muscriptor.utils.sampling as utils


logger = logging.getLogger(__name__)
ConditionTensors = dict[str, ConditionType]


# ---------------------------------------------------------------------------
# ScaledEmbedding  (used for token embeddings, keeps weight compatible with ckpt)
# ---------------------------------------------------------------------------


class ScaledEmbedding(nn.Embedding):
    """Embedding that maps zero_idx (a negative index) to a zero vector."""

    def __init__(self, *args, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        assert zero_idx < 0
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        return torch.where(is_zero[..., None], torch.zeros_like(y), y)


# ---------------------------------------------------------------------------
# TorchAutocast
# ---------------------------------------------------------------------------


class TorchAutocast:
    """Minimal autocast context manager (matches the audiocraft interface)."""

    def __init__(
        self,
        enabled: bool = False,
        device_type: str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        self.enabled = enabled
        self.device_type = device_type
        self.dtype = dtype
        self._ctx = None

    def __enter__(self):
        if self.enabled:
            self._ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled and self._ctx is not None:
            self._ctx.__exit__(*args)


# ---------------------------------------------------------------------------
# LMModel
# ---------------------------------------------------------------------------


class LMModel(nn.Module):
    """Causal transformer LM for MIDI token generation.

    Single-stream
    Supports classifier-free guidance at inference time.
    """

    def __init__(
        self,
        condition_provider: ConditioningProvider,
        card: int = 1024,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        cfg_coef: float = 1.0,
        autocast: TorchAutocast | None = None,
        model_config: ModelConfig | None = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_provider = condition_provider
        self.card = card
        self.dim = dim
        self.cfg_coef = cfg_coef
        if model_config is None:
            model_config = ModelConfig(
                dim=dim,
                num_heads=num_heads,
                num_layers=kwargs.pop("num_layers"),
                card=card,
                hidden_scale=hidden_scale,
                max_period=kwargs.pop("max_period", 10_000),
            )
        elif kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected model kwargs with model_config: {unknown}")
        self.model_config = model_config
        self.last_generation_profile: dict[str, float | int] | None = None
        self.autocast = (
            autocast if autocast is not None else TorchAutocast(enabled=False)
        )

        self.emb = ScaledEmbedding(
            self.card + 1,
            dim,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )

        # Keep the historical attribute name so published Transformer
        # checkpoints retain their exact ``transformer.*`` state-dict keys.
        self.transformer = build_backbone(model_config, device=device, dtype=dtype)
        self.type_embedding: nn.Embedding | None = None
        self.chunk_start: nn.Parameter | None = None
        if model_config.use_type_embeddings:
            self.type_embedding = nn.Embedding(4, dim, device=device, dtype=dtype)
            self.chunk_start = nn.Parameter(
                torch.zeros(1, 1, dim, device=device, dtype=dtype)
            )
            nn.init.normal_(self.chunk_start, std=dim**-0.5)
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.linear = nn.Linear(dim, card, bias=False)
        self.symbolic_state_marker: nn.Parameter | None = None
        self.symbolic_program_embedding: nn.Embedding | None = None
        self.symbolic_pitch_embedding: nn.Embedding | None = None
        self.symbolic_reentry_embedding: nn.Embedding | None = None
        self.boundary_projection: nn.Linear | None = None
        self.active_note_head: nn.Linear | None = None
        self.reentry_head: nn.Linear | None = None
        if model_config.clean_acoustic_cache:
            groups = model_config.num_symbolic_instrument_groups
            pitches = model_config.num_symbolic_pitches
            reentry = model_config.num_reentry_classes
            bottleneck = model_config.boundary_state_dim
            self.symbolic_state_marker = nn.Parameter(
                torch.zeros(1, 1, dim, device=device, dtype=dtype)
            )
            nn.init.normal_(self.symbolic_state_marker, std=dim**-0.5)
            self.symbolic_program_embedding = nn.Embedding(
                groups, dim, device=device, dtype=dtype
            )
            self.symbolic_pitch_embedding = nn.Embedding(
                pitches, dim, device=device, dtype=dtype
            )
            self.symbolic_reentry_embedding = nn.Embedding(
                reentry, dim, device=device, dtype=dtype
            )
            self.boundary_projection = nn.Linear(
                dim, bottleneck, device=device, dtype=dtype
            )
            self.active_note_head = nn.Linear(
                bottleneck, groups * pitches, device=device, dtype=dtype
            )
            self.reentry_head = nn.Linear(
                bottleneck, groups * reentry, device=device, dtype=dtype
            )

    # ------------------------------------------------------------------
    # Token ID properties
    # ------------------------------------------------------------------

    @property
    def initial_token_id(self) -> int:
        return self.card

    @property
    def zero_token_id(self) -> int:
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        return -2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:
        """Return normalized hidden states for an already packed sequence."""
        hidden = self.transformer(embeddings, prepend_length=0, model_state=model_state)
        return self.out_norm(hidden)

    @staticmethod
    def _ordered_condition_keys(condition_tensors: ConditionTensors) -> list[str]:
        preferred = ["self_wav", "instrument_group", "dataset_name"]
        ordered = [key for key in preferred if key in condition_tensors]
        ordered.extend(key for key in condition_tensors if key not in ordered)
        return ordered

    def condition_prefix_embeddings(
        self,
        condition_tensors: ConditionTensors,
        *,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Build the clean chunk-marker/audio/NULL-metadata prefix.

        MIDI embeddings, BOS, prompts, and generated events are deliberately
        excluded.  This is the only sequence allowed to update the persistent
        Clean Acoustic Cache.
        """

        if self.type_embedding is None or self.chunk_start is None:
            raise ValueError("clean audio state requires use_type_embeddings=True")
        if condition_tensors:
            inferred = next(iter(condition_tensors.values()))[0].shape[0]
            if batch_size is not None and batch_size != inferred:
                raise ValueError("condition batch size mismatch")
            batch_size = inferred
        if batch_size is None:
            raise ValueError("batch_size is required when conditions are empty")

        dtype = self.emb.weight.dtype
        device = self.emb.weight.device
        chunk = self.chunk_start.expand(batch_size, -1, -1)
        chunk = chunk + self.type_embedding(
            torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        )
        pieces = [chunk]
        for key in self._ordered_condition_keys(condition_tensors):
            cond, mask = condition_tensors[key]
            cond = cond.to(device=device, dtype=dtype)
            kind = 1 if key == "self_wav" else 2
            types = torch.full(
                cond.shape[:2], kind, dtype=torch.long, device=cond.device
            )
            cond = cond + self.type_embedding(types)
            cond = cond * mask.to(device=device).unsqueeze(-1).to(cond.dtype)
            pieces.append(cond)
        return torch.cat(pieces, dim=1)

    def symbolic_state_embeddings(
        self,
        active_notes: torch.Tensor,
        reentry: torch.Tensor | None = None,
        reentry_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode structured committed state as one event-side token.

        The representation is factorized over instrument group and pitch, so
        it precisely preserves active pairs without a multi-million-parameter
        dense projection.  It belongs only to the temporary decode branch.
        """

        if (
            self.symbolic_state_marker is None
            or self.symbolic_program_embedding is None
            or self.symbolic_pitch_embedding is None
            or self.symbolic_reentry_embedding is None
            or self.type_embedding is None
        ):
            raise ValueError("model has no committed-symbolic-state encoder")
        cfg = self.model_config
        expected = (
            cfg.num_symbolic_instrument_groups,
            cfg.num_symbolic_pitches,
        )
        if tuple(active_notes.shape[-2:]) != expected:
            raise ValueError(
                f"active_notes must end in {expected}, got {active_notes.shape}"
            )
        active = active_notes.to(self.emb.weight.dtype)
        program_counts = active.sum(dim=-1)
        pitch_counts = active.sum(dim=-2)
        pooled = program_counts @ self.symbolic_program_embedding.weight
        pooled = pooled + pitch_counts @ self.symbolic_pitch_embedding.weight
        pooled = pooled / active.sum(dim=(-2, -1), keepdim=False).clamp_min(1)[
            ..., None
        ]

        if reentry is not None:
            if tuple(reentry.shape[-1:]) != (
                cfg.num_symbolic_instrument_groups,
            ):
                raise ValueError("reentry has the wrong instrument-group axis")
            embedded = self.symbolic_reentry_embedding(reentry.long())
            valid = (
                torch.ones_like(reentry, dtype=torch.bool)
                if reentry_valid is None
                else reentry_valid.bool()
            )
            weights = valid.to(embedded.dtype).unsqueeze(-1)
            pooled = pooled + (embedded * weights).sum(dim=-2) / weights.sum(
                dim=-2
            ).clamp_min(1)

        batch = active_notes.shape[0]
        token = self.symbolic_state_marker.expand(batch, -1, -1)
        token = token + pooled.unsqueeze(1)
        event_type = torch.full(
            (batch, 1), 3, dtype=torch.long, device=token.device
        )
        return token + self.type_embedding(event_type)

    def boundary_state_logits(
        self, audio_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict end-of-chunk active notes and instrument re-entry class."""

        if (
            self.boundary_projection is None
            or self.active_note_head is None
            or self.reentry_head is None
        ):
            raise ValueError("model has no audio-only boundary heads")
        cfg = self.model_config
        bottleneck = torch.nn.functional.silu(
            self.boundary_projection(audio_hidden)
        )
        active = self.active_note_head(bottleneck).view(
            *audio_hidden.shape[:-1],
            cfg.num_symbolic_instrument_groups,
            cfg.num_symbolic_pitches,
        )
        reentry = self.reentry_head(bottleneck).view(
            *audio_hidden.shape[:-1],
            cfg.num_symbolic_instrument_groups,
            cfg.num_reentry_classes,
        )
        return active, reentry

    @torch.no_grad()
    def prefill_symbolic_state(
        self,
        model_state: ModelState,
        active_notes: torch.Tensor,
        reentry: torch.Tensor | None = None,
        reentry_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write one structured state token into a temporary decode state."""

        token = self.symbolic_state_embeddings(
            active_notes, reentry=reentry, reentry_valid=reentry_valid
        )
        with self.autocast:
            hidden = self.encode_embeddings(token, model_state=model_state)
        increment_steps(self.transformer, model_state, increment=1)
        return hidden

    def prepare_condition_tensors(
        self,
        conditions: list[ConditioningAttributes],
        *,
        cfg_coef: float = 1.0,
        log_timing: bool = False,
    ) -> ConditionTensors:
        """Tokenize and encode conditions using the formal CFG ordering."""

        if not conditions:
            return {}
        source = conditions
        if cfg_coef != 1.0:
            source = conditions + nullify_all_conditions(conditions)
        prepared = self.condition_provider.tokenize(source)
        if log_timing:
            muscriptor.accelerator.synchronize()
            started = time.perf_counter()
        result = self.condition_provider(prepared)
        if log_timing:
            muscriptor.accelerator.synchronize()
            print(
                f"[muscriptor] encode conditions (total): "
                f"{time.perf_counter() - started:.3f}s"
            )
        return result

    @torch.no_grad()
    def prefill_condition_prefix(
        self,
        conditions: list[ConditioningAttributes],
        *,
        model_state: ModelState | None = None,
        state_sequence_length: int | ×¯=¶‰žËkºwµçeá}…±É•…‘å}ÁÉ•™¥±±•õÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•°(€€€€€€€€¤(€€€€€€€¥˜ÕÍ•}Í…µÁ±¥¹œ…¹Ñ•µÀ€ø€À¸Àè(€€€€€€€€€€€ÁÉ½‰Ì€ôÑ½É ¹Í½™Ñµ…à¡±½¥ÑÌ€¼Ñ•µÀ°‘¥´ô´Ä¤(€€€€€€€€€€€¹•áÑ}Ñ½­•¹Ì€ôÕÑ¥±Ì¹Í…µÁ±•}™É½µ}ÁÉ½‰Ì¡ÁÉ½‰Ì°Ñ½Á}ÀõÑ½Á}À°Ñ½Á}¬õÑ½Á}¬¥lè°€Át(€€€€€€€•±Í”è(€€€€€€€€€€€¹•áÑ}Ñ½­•¹Ì€ôÑ½É ¹…Éµ…à¡±½¥ÑÌ°‘¥´ô´Ä¤€€Œm	t(€€€€€€€É•ÑÕÉ¸¹•áÑ}Ñ½­•¹Ì€€Œm	t((€€€€Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(€€€€Œ•¹•É…Ñ¥½¸(€€€€Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´((€€€Ñ½É ¹¹½}É… ¤(€€€‘•˜•¹•É…Ñ” (€€€€€€€Í•±˜°(€€€€€€€ÁÉ½µÁÐèÑ½É ¹Q•¹Í½Èð9½¹”€ô9½¹”°(€€€€€€€½¹‘¥Ñ¥½¹Ìè±¥ÍÑm½¹‘¥Ñ¥½¹¥¹ÑÑÉ¥‰ÕÑ•Ít€ômt°(€€€€€€€¹Õµ}Í…µÁ±•Ìè¥¹Ðð9½¹”€ô9½¹”°(€€€€€€€µ…á}•¹}±•¸è¥¹Ð€ô€ÈÔØ°(€€€€€€€ÕÍ•}Í…µÁ±¥¹œè‰½½°€ôQÉÕ”°(€€€€€€€Ñ•µÀè™±½…Ð€ô€Ä¸À°(€€€€€€€Ñ½Á}¬è¥¹Ð€ô€À°(€€€€€€€Ñ½Á}Àè™±½…Ð€ô€À¸À°(€€€€€€€™}½•˜è™±½…Ðð9½¹”€ô9½¹”°(€€€€€€€•…É±å}ÍÑ½Á}½¹}Ñ½­•¸è¥¹Ðð9½¹”€ô9½¹”°(€€€€€€€‰•…µ}Í¥é”è¥¹Ð€ô€Ä°(€€€€€€€‰•…µ}±•¹Ñ¡}Í½É•}…±Á¡„è™±½…Ð€ô€À¸ÜÔ°(€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹ÌèÑ½É ¹Q•¹Í½Èð±¥ÍÑm¥¹Ñtð9½¹”€ô9½¹”°(€€€€€€€µ½‘•±}ÍÑ…Ñ”è5½‘•±MÑ…Ñ”ð9½¹”€ô9½¹”°(€€€€€€€ÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ è¥¹Ðð9½¹”€ô9½¹”°(€€€€€€€ÁÉ½™¥±•}•¹•É…Ñ¥½¸è‰½½°€ô…±Í”°(€€€€€€€ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•è‰½½°€ô…±Í”°(€€€€¤€´ø%Ñ•É…Ñ½ÉmÑ½É ¹Q•¹Í½Étè(€€€€€€€€ˆˆ‰ÕÑ½É•É•ÍÍ¥Ù•±ä•¹•É…Ñ”Ñ½­•¹Ì°å¥•±‘¥¹œ½¹”Ñ¥µ•ÍÑ•À…Ð„Ñ¥µ”¸((€€€€€€€… å¥•±¥Ì„m¹Õµ}Í…µÁ±•Íu€Ñ•¹Í½È¸½È‰•…µ}Í¥é”€ôô€Ä€¡‘•™…Õ±Ð¤°(€€€€€€€Ñ½­•¹Ì…É”å¥•±‘•…ÌÑ¡•ä…É”•¹•É…Ñ•¸½È‰•…µ}Í¥é”€ø€Ä°‰•…´Í•…É (€€€€€€€¥ÌÉÕ¸¹½¸µÍÑÉ•…µ¥¹±ä…¹…±°Ñ½­•¹Ì…É”å¥•±‘•…ÐÑ¡”•¹¸((€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Í€…É”Ñ½­•¸¥‘ÌÝ¡½Í”±½¥ÑÌ…É”™½É•Ñ¼€µ¥¹˜…Ð(€€€€€€€•Ù•ÉäÍÑ•À°Í¼Ñ¡•ä…¸¹•Ù•È‰”Í…µÁ±•€¡É••‘ä°Í…µÁ±¥¹œ½È‰•…´¤¸(€€€€€€€€ˆˆˆ(€€€€€€€…ÍÍ•ÉÐ¹½ÐÍ•±˜¹ÑÉ…¥¹¥¹œ(€€€€€€€¥˜‰•…µ}Í¥é”€ø€Äè(€€€€€€€€€€€…ÍÍ•ÉÐ•…É±å}ÍÑ½Á}½¹}Ñ½­•¸¥Ì¹½Ð9½¹”°€ (€€€€€€€€€€€€€€€€‰‰•…´Í•…É É•ÅÕ¥É•Ì•…É±å}ÍÑ½Á}½¹}Ñ½­•¸ˆ(€€€€€€€€€€€€¤(€€€€€€€‘•Ù¥”€ôÍ•±˜¹•µˆ¹Ý•¥¡Ð¹‘•Ù¥”(€€€€€€€Í•±˜¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±”€ô9½¹”((€€€€€€€¥˜™½É‰¥‘‘•¹}Ñ½­•¹Ì¥Ì¹½Ð9½¹”…¹¹½Ð¥Í¥¹ÍÑ…¹” (€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ì°Ñ½É ¹Q•¹Í½È(€€€€€€€€¤è(€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ì€ôÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ì°‘•Ù¥”õ‘•Ù¥”°‘ÑåÁ”õÑ½É ¹±½¹œ(€€€€€€€€€€€€¤((€€€€€€€¥˜¹Õµ}Í…µÁ±•Ì¥Ì9½¹”è(€€€€€€€€€€€¹Õµ}Í…µÁ±•Ì€ô€ (€€€€€€€€€€€€€€€±•¸¡½¹‘¥Ñ¥½¹Ì¤(€€€€€€€€€€€€€€€¥˜½¹‘¥Ñ¥½¹Ì(€€€€€€€€€€€€€€€•±Í”€¡ÁÉ½µÁÐ¹Í¡…Á•lÁt¥˜ÁÉ½µÁÐ¥Ì¹½Ð9½¹”•±Í”€Ä¤(€€€€€€€€€€€€¤((€€€€€€€™}½•˜€ôÍ•±˜¹™}½•˜¥˜™}½•˜¥Ì9½¹”•±Í”™}½•˜((€€€€€€€€Œ	Õ¥±½¹‘¥Ñ¥½¸Ñ•¹Í½ÉÌ€¡Ý¥Ñ ¹Õ±°½¹‘¥Ñ¥½¹Ì…ÁÁ•¹‘•™½È¤¸(€€€€€€€™}½¹‘¥Ñ¥½¹Ì€ôÍ•±˜¹ÁÉ•Á…É•}½¹‘¥Ñ¥½¹}Ñ•¹Í½ÉÌ (€€€€€€€€€€€½¹‘¥Ñ¥½¹Ì°™}½•˜õ™}½•˜°±½}Ñ¥µ¥¹œõ‰½½°¡½¹‘¥Ñ¥½¹Ì¤(€€€€€€€€¤((€€€€€€€•™™}‰…Ñ €ô¹Õµ}Í…µÁ±•Ì€¨‰•…µ}Í¥é”((€€€€€€€€ŒáÁ…¹½¹‘¥Ñ¥½¹ÌÍ¼•… ‰•…´•ÑÌ¥ÑÌ½Ý¸½Áä€¡¥¹Ñ•É±•…Ù•™½È¤¸(€€€€€€€¥˜‰•…µ}Í¥é”€ø€Ä…¹™}½¹‘¥Ñ¥½¹Ìè(€€€€€€€€€€€™}½¹‘¥Ñ¥½¹Ì€ôì(€€€€€€€€€€€€€€€¬è€ (€€€€€€€€€€€€€€€€€€€Ñ½É ¹É•Á•…Ñ}¥¹Ñ•É±•…Ù”¡½¹°‰•…µ}Í¥é”°‘¥´ôÀ¤°(€€€€€€€€€€€€€€€€€€€Ñ½É ¹É•Á•…Ñ}¥¹Ñ•É±•…Ù”¡µ…Í¬°‰•…µ}Í¥é”°‘¥´ôÀ¤°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€™½È¬°€¡½¹°µ…Í¬¤¥¸™}½¹‘¥Ñ¥½¹Ì¹¥Ñ•µÌ ¤(€€€€€€€€€€€ô((€€€€€€€€Œ%¹¥Ñ¥…±¥Í”•¹•É…Ñ¥½¸‰Õ™™•È€¡•™™}‰…Ñ É½ÝÌ€ô¹Õµ}Í…µÁ±•Ìƒ\‰•…µ}Í¥é”¤(€€€€€€€Õ¹•¹•É…Ñ•€ôÍ•±˜¹Õ¹•¹•É…Ñ•‘}Ñ½­•¹}¥(€€€€€€€•¹}Í•ÅÕ•¹”€ôÑ½É ¹™Õ±° (€€€€€€€€€€€€¡•™™}‰…Ñ °µ…á}•¹}±•¸€¬€Ä¤°(€€€€€€€€€€€Õ¹•¹•É…Ñ•°(€€€€€€€€€€€‘•Ù¥”õ‘•Ù¥”°(€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹±½¹œ°(€€€€€€€€¤(€€€€€€€•¹}Í•ÅÕ•¹•lè°€Át€ôÍ•±˜¹¥¹¥Ñ¥…±}Ñ½­•¹}¥((€€€€€€€ÍÑ…ÉÑ}½™™Í•Ð€ô€À(€€€€€€€¥˜ÁÉ½µÁÐ¥Ì¹½Ð9½¹”è(€€€€€€€€€€€AP€ôÁÉ½µÁÐ¹Í¡…Á•l´Åt(€€€€€€€€€€€¥˜‰•…µ}Í¥é”€ø€Äè(€€€€€€€€€€€€€€€ÁÉ½µÁÐ€ôÑ½É ¹É•Á•…Ñ}¥¹Ñ•É±•…Ù”¡ÁÉ½µÁÐ°‰•…µ}Í¥é”°‘¥´ôÀ¤(€€€€€€€€€€€•¹}Í•ÅÕ•¹•lè°€Ä€è€Ä€¬AQt€ôÁÉ½µÁÐ(€€€€€€€€€€€Õ¹•¹•É…Ñ•‘}ÍÑ•ÁÌ€ô€¡•¹}Í•ÅÕ•¹”€ôôÕ¹•¹•É…Ñ•¤¹¹½¹é•É¼ ¥lè°€Åt(€€€€€€€€€€€ÍÑ…ÉÑ}½™™Í•Ð€ôµ…à À°¥¹Ð¡Õ¹•¹•É…Ñ•‘}ÍÑ•ÁÌ¹…µ¥¸ ¤¤€´€Ä¤((€€€€€€€ÁÉ•Á•¹‘}±•¹Ñ €ôÍÕ´¡½¹¹Í¡…Á•lÅt™½È½¹°|¥¸™}½¹‘¥Ñ¥½¹Ì¹Ù…±Õ•Ì ¤¤(€€€€€€€€Œ9•Ü¡•­Á½¥¹ÑÌÁÉ•Á•¹…¸•áÁ±¥¥Ð¡Õ¹¬µÍÑ…ÉÐ•µ‰•‘‘¥¹œ¥¸(€€€€€€€€Œ…‘‘¥Ñ¥½¸Ñ¼Ñ¡”½¹‘¥Ñ¥½¸Ñ•¹Í½ÉÌ¸€%ÐÁ…ÉÑ¥¥Á…Ñ•Ì¥¸•Ù•Éä(€€€€€€€€ŒÍÑÉ•…µ¥¹œ±…å•È…¹µÕÍÐÑ¡•É•™½É”…‘Ù…¹”•Ù•Éä…¡”½™™Í•Ð¸(€€€€€€€¥˜Í•±˜¹ÑåÁ•}•µ‰•‘‘¥¹œ¥Ì¹½Ð9½¹”…¹¹½ÐÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•è(€€€€€€€€€€€ÁÉ•Á•¹‘}±•¹Ñ €¬ô€Ä(€€€€€€€¥˜ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•è(€€€€€€€€€€€¥˜µ½‘•±}ÍÑ…Ñ”¥Ì9½¹”è(€€€€€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È (€€€€€€€€€€€€€€€€€€€€‰ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•É•ÅÕ¥É•Ì…¸•áÑ•É¹…°µ½‘•±}ÍÑ…Ñ”ˆ(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜½¹‘¥Ñ¥½¹Ìè(€€€€€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È (€€€€€€€€€€€€€€€€€€€€‰ÁÉ•™¥±±••¹•É…Ñ¥½¸µÕÍÐ¹½Ð•¹½‘”½¹‘¥Ñ¥½¹Ì„Í•½¹Ñ¥µ”ˆ(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€ÁÉ•Á•¹‘}±•¹Ñ €ô€À(€€€€€€€…¡•}‰…Ñ¡}Í¥é”€ô•™™}‰…Ñ €¨€ Ä¥˜™}½•˜€ôô€Ä¸À•±Í”€È¤(€€€€€€€…¡•}Í•Å}±•¸€ôÁÉ•Á•¹‘}±•¹Ñ €¬µ…á}•¹}±•¸(€€€€€€€¥˜µ½‘•±}ÍÑ…Ñ”¥Ì9½¹”è(€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”€ô¥¹¥Ñ}ÍÑ…Ñ•Ì (€€€€€€€€€€€€€€€Í•±˜°(€€€€€€€€€€€€€€€‰…Ñ¡}Í¥é”õ…¡•}‰…Ñ¡}Í¥é”°(€€€€€€€€€€€€€€€Í•ÅÕ•¹•}±•¹Ñ õÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ ½È…¡•}Í•Å}±•¸°(€€€€€€€€€€€€¤((€€€€€€€€ŒÕµÕ±…Ñ•±½œµÁÉ½ˆÍ½É•Ì°½¹”Á•È‰•…´É½Ü¸(€€€€€€€‰•…µ}Í½É•Ì€ôÑ½É ¹é•É½Ì¡•™™}‰…Ñ °‘•Ù¥”õ‘•Ù¥”°‘ÑåÁ”õÑ½É ¹™±½…Ð¤((€€€€€€€Õ‘…}ÁÉ½™¥±”€ôÁÉ½™¥±•}•¹•É…Ñ¥½¸…¹‘•Ù¥”¹ÑåÁ”€ôô€‰Õ‘„ˆ(€€€€€€€ÁÉ½™¥±•}•Ù•¹ÑÌè±¥ÍÑmÑÕÁ±•mÍÑÈ°½‰©•Ð°½‰©•Ð°¥¹Ñut€ômt(€€€€€€€ÁÉ½™¥±•}ÁÕ}Í•½¹‘Ì€ôì‰ÁÉ•™¥±°ˆè€À¸À°€‰‘•½‘”ˆè€À¸Áô(€€€€€€€ÁÉ½™¥±•}Ñ½­•¹Ì€ôì‰ÁÉ•™¥±°ˆè€À°€‰‘•½‘”ˆè€Áô((€€€€€€€‘•˜ÁÉ½™¥±•}ÍÑ…ÉÐ ¤è(€€€€€€€€€€€¥˜¹½ÐÁÉ½™¥±•}•¹•É…Ñ¥½¸è(€€€€€€€€€€€€€€€É•ÑÕÉ¸9½¹”(€€€€€€€€€€€¥˜Õ‘…}ÁÉ½™¥±”è(€€€€€€€€€€€€€€€•Ù•¹Ð€ôÑ½É ¹Õ‘„¹Ù•¹Ð¡•¹…‰±•}Ñ¥µ¥¹œõQÉÕ”¤(€€€€€€€€€€€€€€€•Ù•¹Ð¹É•½É ¤(€€€€€€€€€€€€€€€É•ÑÕÉ¸•Ù•¹Ð(€€€€€€€€€€€É•ÑÕÉ¸Ñ¥µ”¹Á•É™}½Õ¹Ñ•È ¤((€€€€€€€‘•˜ÁÉ½™¥±•}•¹¡ÍÑ…ÉÐ°€¨°­¥¹èÍÑÈ°Ñ½­•¹Ìè¥¹Ð¤€´ø9½¹”è(€€€€€€€€€€€¥˜ÍÑ…ÉÐ¥Ì9½¹”è(€€€€€€€€€€€€€€€É•ÑÕÉ¸(€€€€€€€€€€€ÁÉ½™¥±•}Ñ½­•¹Ím­¥¹‘t€¬ôÑ½­•¹Ì(€€€€€€€€€€€¥˜Õ‘…}ÁÉ½™¥±”è(€€€€€€€€€€€€€€€•Ù•¹Ð€ôÑ½É ¹Õ‘„¹Ù•¹Ð¡•¹…‰±•}Ñ¥µ¥¹œõQÉÕ”¤(€€€€€€€€€€€€€€€•Ù•¹Ð¹É•½É ¤(€€€€€€€€€€€€€€€ÁÉ½™¥±•}•Ù•¹ÑÌ¹…ÁÁ•¹ ¡­¥¹°ÍÑ…ÉÐ°•Ù•¹Ð°Ñ½­•¹Ì¤¤(€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€ÁÉ½™¥±•}ÁÕ}Í•½¹‘Ím­¥¹‘t€¬ôÑ¥µ”¹Á•É™}½Õ¹Ñ•È ¤€´™±½…Ð¡ÍÑ…ÉÐ¤((€€€€€€€€Œ½ÈÉ••‘ä½Í…µÁ±¥¹œ•µ¥ÐÁÉ½µÁÐÍÑ•ÁÌ¹½Üì‰•…´Í•…É •µ¥ÑÌ…ÐÑ¡”•¹¸(€€€€€€€¥˜‰•…µ}Í¥é”€ôô€Äè(€€€€€€€€€€€™½ÈÐ¥¸É…¹”¡ÍÑ…ÉÑ}½™™Í•Ð¤è(€€€€€€€€€€€€€€€å¥•±•¹}Í•ÅÕ•¹•lè°Ð€¬€Åt((€€€€€€€±…ÍÑ}½™™Í•Ð€ôÍÑ…ÉÑ}½™™Í•Ð€´€Ä(€€€€€€€Ý¥Ñ Í•±˜¹…ÕÑ½…ÍÐè(€€€€€€€€€€€™½È½™™Í•Ð¥¸É…¹”¡ÍÑ…ÉÑ}½™™Í•Ð°µ…á}•¹}±•¸¤è(€€€€€€€€€€€€€€€±…ÍÑ}½™™Í•Ð€ô½™™Í•Ð(€€€€€€€€€€€€€€€™¥ÉÍÑ}¥Ñ•È€ô½™™Í•Ð€ôôÍÑ…ÉÑ}½™™Í•Ð(€€€€€€€€€€€€€€€¥¹ÁÕÑ|€ô€ (€€€€€€€€€€€€€€€€€€€•¹}Í•ÅÕ•¹•lè°€è½™™Í•Ð€¬€Åt(€€€€€€€€€€€€€€€€€€€¥˜™¥ÉÍÑ}¥Ñ•È(€€€€€€€€€€€€€€€€€€€•±Í”•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€è½™™Í•Ð€¬€Åt(€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€¥˜‰•…µ}Í¥é”€ôô€Äè(€€€€€€€€€€€€€€€€€€€€ŒƒŠRŠR MÑ…¹‘…ÉÉ••‘ä€¼Í…µÁ±¥¹œÁ…Ñ ƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (€€€€€€€€€€€€€€€€€€€¥˜•…É±å}ÍÑ½Á}½¹}Ñ½­•¸¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€€€€€€€€€‘½¹”€ô€¡•¹}Í•ÅÕ•¹”€ôô•…É±å}ÍÑ½Á}½¹}Ñ½­•¸¤¹…¹ä¡‘¥´ôÄ¤¹…±° ¤(€€€€€€€€€€€€€€€€€€€€€€€¥˜‘½¹”è(€€€€€€€€€€€€€€€€€€€€€€€€€€€‰É•…¬((€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±•}­¥¹€ô€‰ÁÉ•™¥±°ˆ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€‰‘•½‘”ˆ(€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ•€ôÁÉ½™¥±•}ÍÑ…ÉÐ ¤(€€€€€€€€€€€€€€€€€€€¹•áÑ}Ñ½­•¸€ôÍ•±˜¹}Í…µÁ±•}¹•áÑ}Ñ½­•¸ (€€€€€€€€€€€€€€€€€€€€€€€¥¹ÁÕÑ|°(€€€€€€€€€€€€€€€€€€€€€€€™}½¹‘¥Ñ¥½¹Ì°(€€€€€€€€€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€€€€€€€€€™¥ÉÍÑ}ÍÑ•Àõ™¥ÉÍÑ}¥Ñ•È°(€€€€€€€€€€€€€€€€€€€€€€€ÕÍ•}Í…µÁ±¥¹œõÕÍ•}Í…µÁ±¥¹œ°(€€€€€€€€€€€€€€€€€€€€€€€Ñ•µÀõÑ•µÀ°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á}¬õÑ½Á}¬°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á}ÀõÑ½Á}À°(€€€€€€€€€€€€€€€€€€€€€€€™}½•˜õ™}½•˜°(€€€€€€€€€€€€€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ìõ™½É‰¥‘‘•¹}Ñ½­•¹Ì°(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•õÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•°(€€€€€€€€€€€€€€€€€€€€¤€€Œm	t(€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±•}•¹ (€€€€€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ•°(€€€€€€€€€€€€€€€€€€€€€€€­¥¹õÁÉ½™¥±•}­¥¹°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½­•¹Ìõ¥¹ÁÕÑ|¹Í¡…Á•l´Åt(€€€€€€€€€€€€€€€€€€€€€€€€¬€¡ÁÉ•Á•¹‘}±•¹Ñ ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€À¤°(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€¥¹ÁÕÑ}P€ô¥¹ÁÕÑ|¹Í¡…Á•l´Åt(€€€€€€€€€€€€€€€€€€€¥¹É•µ•¹Ñ}ÍÑ•ÁÌ (€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹ÑÉ…¹Í™½Éµ•È°(€€€€€€€€€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€€€€€€€€€¥¹É•µ•¹Ðõ¥¹ÁÕÑ}P€¬€¡ÁÉ•Á•¹‘}±•¹Ñ ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€À¤°(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€Ñ¡¥Í}•¹}ÍÑ•À€ô•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€¬€Åt(€€€€€€€€€€€€€€€€€€€¹•áÑ}Ñ½­•¸€ôÑ½É ¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€€€€€Ñ¡¥Í}•¹}ÍÑ•À€ôôÕ¹•¹•É…Ñ•°¹•áÑ}Ñ½­•¸°Ñ¡¥Í}•¹}ÍÑ•À(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€¬€Åt€ô¹•áÑ}Ñ½­•¸((€€€€€€€€€€€€€€€€€€€å¥•±•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€¬€Åt€€Œm¹Õµ}Í…µÁ±•Ít((€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€€ŒƒŠRŠR 	•…´Í•…É ÍÑ•ÀƒŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠRŠR (€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±•}­¥¹€ô€‰ÁÉ•™¥±°ˆ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€‰‘•½‘”ˆ(€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ•€ôÁÉ½™¥±•}ÍÑ…ÉÐ ¤(€€€€€€€€€€€€€€€€€€€±½¥ÑÌ€ôÍ•±˜¹}½µÁÕÑ•}±½¥ÑÌ (€€€€€€€€€€€€€€€€€€€€€€€¥¹ÁÕÑ|°(€€€€€€€€€€€€€€€€€€€€€€€™}½¹‘¥Ñ¥½¹Ì°(€€€€€€€€€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€€€€€€€€€™¥ÉÍÑ}ÍÑ•Àõ™¥ÉÍÑ}¥Ñ•È°(€€€€€€€€€€€€€€€€€€€€€€€™}½•˜õ™}½•˜°(€€€€€€€€€€€€€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ìõ™½É‰¥‘‘•¹}Ñ½­•¹Ì°(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•õÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•°(€€€€€€€€€€€€€€€€€€€€¤€€Œm•™™}‰…Ñ °…É‘t(€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±•}•¹ (€€€€€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ•°(€€€€€€€€€€€€€€€€€€€€€€€­¥¹õÁÉ½™¥±•}­¥¹°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½­•¹Ìõ¥¹ÁÕÑ|¹Í¡…Á•l´Åt(€€€€€€€€€€€€€€€€€€€€€€€€¬€¡ÁÉ•Á•¹‘}±•¹Ñ ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€À¤°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€¥¹ÁÕÑ}P€ô¥¹ÁÕÑ|¹Í¡…Á•l´Åt(€€€€€€€€€€€€€€€€€€€¥¹É•µ•¹Ñ}ÍÑ•ÁÌ (€€€€€€€€€€€€€€€€€€€€€€€Í•±˜¹ÑÉ…¹Í™½Éµ•È°(€€€€€€€€€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€€€€€€€€€¥¹É•µ•¹Ðõ¥¹ÁÕÑ}P€¬€¡ÁÉ•Á•¹‘}±•¹Ñ ¥˜™¥ÉÍÑ}¥Ñ•È•±Í”€À¤°(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€±½}ÁÉ½‰Ì€ôÑ½É ¹±½}Í½™Ñµ…à¡±½¥ÑÌ¹™±½…Ð ¤°‘¥´ô´Ä¤((€€€€€€€€€€€€€€€€€€€€ŒQ½À‰•…µ}Í¥é”…¹‘¥‘…Ñ”Ñ½­•¹ÌÁ•ÈÕÉÉ•¹Ð‰•…´(€€€€€€€€€€€€€€€€€€€Ñ½Á­}Í½É•Ì°Ñ½Á­}Ñ½­•¹Ì€ôÑ½É ¹Ñ½Á¬ (€€€€€€€€€€€€€€€€€€€€€€€±½}ÁÉ½‰Ì°¬õ‰•…µ}Í¥é”°‘¥´ô´Ä(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€€ŒQÉ…¬Ý¡¥ ‰•…µÌ¡…Ù”…±É•…‘ä•µ¥ÑÑ•=L(€€€€€€€€€€€€€€€€€€€•½Í}µ…Í¬€ô•¹}Í•ÅÕ•¹”€ôô•…É±å}ÍÑ½Á}½¹}Ñ½­•¸(€€€€€€€€€€€€€€€€€€€‰•…µ}¡…Í}•¹‘•€ô•½Í}µ…Í¬¹…¹ä¡‘¥´ô´Ä¤(€€€€€€€€€€€€€€€€€€€•½Í}Á½Ì€ô•½Í}µ…Í¬¹¥¹Ð ¤¹…Éµ…à¡‘¥´ô´Ä¤¹±…µÀ¡µ¥¸ôÄ¤(€€€€€€€€€€€€€€€€€€€‰•…µ}±•¹Ñ¡Ì€ôÑ½É ¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€€€€€‰•…µ}¡…Í}•¹‘•°(€€€€€€€€€€€€€€€€€€€€€€€•½Í}Á½Ì°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹™Õ±±}±¥­”¡•½Í}Á½Ì°½™™Í•Ð€¬€Ä¤°(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€€Œ¥¹¥Í¡•‰•…µÌè‘½¸Ð•áÁ…¹™ÕÉÑ¡•È(€€€€€€€€€€€€€€€€€€€Ñ½Á­}Í½É•Ì€ôÑ½É ¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€€€€€‰•…µ}¡…Í}•¹‘•¹Õ¹ÍÅÕ••é” ´Ä¤°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹é•É½Í}±¥­”¡Ñ½Á­}Í½É•Ì¤°(€€€€€€€€€€€€€€€€€€€€€€€Ñ½Á­}Í½É•Ì°(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€€€€€€€€€€Œ1•¹Ñ µ¹½Éµ…±¥é•…¹‘¥‘…Ñ”Í½É•Ìèm•™™}‰…Ñ °‰•…µ}Í¥é•t(€€€€€€€€€€€€€€€€€€€±À€ô€Ä¸À€¼€¡‰•…µ}±•¹Ñ¡Ì¹™±½…Ð ¤€¨¨‰•…µ}±•¹Ñ¡}Í½É•}…±Á¡„¤(€€€€€€€€€€€€€€€€€€€…¹€ô€¡‰•…µ}Í½É•Ì¹Õ¹ÍÅÕ••é” ´Ä¤€¬Ñ½Á­}Í½É•Ì¤€¨±À¹Õ¹ÍÅÕ••é” ´Ä¤((€€€€€€€€€€€€€€€€€€€€ŒI•Í¡…Á”Ñ¼m¹Õµ}Í…µÁ±•Ì°‰•…µ}Í¥é—
Ét™½ÈÉ½ÍÌµ‰•…´Í•±•Ñ¥½¸(€€€€€€€€€€€€€€€€€€€…¹‘|É€ô…¹¹É•Í¡…Á”¡¹Õµ}Í…µÁ±•Ì°‰•…µ}Í¥é”€¨‰•…µ}Í¥é”¤((€€€€€€€€€€€€€€€€€€€¥˜½™™Í•Ð€ôôÍÑ…ÉÑ}½™™Í•Ðè(€€€€€€€€€€€€€€€€€€€€€€€€Œ±°‰•…µÌ¥‘•¹Ñ¥…°…ÐÍÑ…ÉÐƒŠPÑ…­”™¥ÉÍÐ‰•…µ}Í¥é”Ñ½­•¹Ì(€€€€€€€€€€€€€€€€€€€€€€€¹•Ý}Í½É•Ì€ô…¹‘|É‘lè°€é‰•…µ}Í¥é•t(€€€€€€€€€€€€€€€€€€€€€€€‰•ÍÑ}¥‘à€ô€ (€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹…É…¹”¡‰•…µ}Í¥é”°‘•Ù¥”õ‘•Ù¥”¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¹Õ¹ÍÅÕ••é” À¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€€¹•áÁ…¹¡¹Õµ}Í…µÁ±•Ì°€´Ä¤(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€€€€€¹•Ý}Í½É•Ì°‰•ÍÑ}¥‘à€ôÑ½É ¹Ñ½Á¬¡…¹‘|É°¬õ‰•…µ}Í¥é”°‘¥´ô´Ä¤((€€€€€€€€€€€€€€€€€€€€Œ•½‘”™±…Ð¥¹‘•àƒŠH€¡ÁÉ•Ù}‰•…µ}Ý¥Ñ¡¥¹}Í…µÁ±”°Ñ½­•¹}É…¹¬¤(€€€€€€€€€€€€€€€€€€€ÁÉ•Ù}±½…°€ô€¡‰•ÍÑ}¥‘à€¼¼‰•…µ}Í¥é”¤¹É•Í¡…Á” ´Ä¤(€€€€€€€€€€€€€€€€€€€Ñ½­}É…¹¬€ô€¡‰•ÍÑ}¥‘à€”‰•…µ}Í¥é”¤¹É•Í¡…Á” ´Ä¤((€€€€€€€€€€€€€€€€€€€€Œ5…ÀÑ¼±½‰…°É½Ü¥¹‘¥•Ì¥¸m•™™}‰…Ñ °ƒŠ™tÑ•¹Í½ÉÌ(€€€€€€€€€€€€€€€€€€€Í…µÁ±•}‰…Í”€ô€ (€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹…É…¹”¡¹Õµ}Í…µÁ±•Ì°‘•Ù¥”õ‘•Ù¥”¤¹É•Á•…Ñ}¥¹Ñ•É±•…Ù” (€€€€€€€€€€€€€€€€€€€€€€€€€€€‰•…µ}Í¥é”(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€€¨‰•…µ}Í¥é”(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€ÁÉ•Ù}±½‰…°€ôÍ…µÁ±•}‰…Í”€¬ÁÉ•Ù}±½…°((€€€€€€€€€€€€€€€€€€€€ŒQ½­•¸™½È•… ¹•Ü‰•…´(€€€€€€€€€€€€€€€€€€€¹•áÑ}Ñ½­•¸€ôÑ½Á­}Ñ½­•¹ÍmÁÉ•Ù}±½‰…°°Ñ½­}É…¹­t((€€€€€€€€€€€€€€€€€€€€ŒUÁ‘…Ñ”‰•…´Í½É•Ì€¡ÍÑ½É”Õ¸µ¹½Éµ…±¥é•™½ÈÑ¡”¹•áÐÍÑ•À¤(€€€€€€€€€€€€€€€€€€€‰•…µ}Í½É•Ì€ô¹•Ý}Í½É•Ì¹É•Í¡…Á” ´Ä¤€¼±ÁmÁÉ•Ù}±½‰…±t((€€€€€€€€€€€€€€€€€€€€ŒI•½É‘•È•¹•É…Ñ¥½¸Í•ÅÕ•¹•ÌÑ¼µ…Ñ Ý¥¹¹¥¹œ‰•…µÌ(€€€€€€€€€€€€€€€€€€€•¹}Í•ÅÕ•¹”€ô•¹}Í•ÅÕ•¹•mÁÉ•Ù}±½‰…±t((€€€€€€€€€€€€€€€€€€€É•½É‘•È€ôÁÉ•Ù}±½‰…°(€€€€€€€€€€€€€€€€€€€¥˜™}½•˜€„ô€Ä¸Àè(€€€€€€€€€€€€€€€€€€€€€€€É•½É‘•È€ôÑ½É ¹…Ð¡mÁÉ•Ù}±½‰…°°ÁÉ•Ù}±½‰…°€¬•™™}‰…Ñ¡t¤(€€€€€€€€€€€€€€€€€€€É•½É‘•É}ÍÑ…Ñ•Ì¡Í•±˜¹ÑÉ…¹Í™½Éµ•È°µ½‘•±}ÍÑ…Ñ”°É•½É‘•È¤((€€€€€€€€€€€€€€€€€€€€Œ]É¥Ñ”¹•áÐÑ½­•¸€¡É•ÍÁ•Ñ¥¹œÁÉ”µ™¥±±•ÁÉ½µÁÐÁ½Í¥Ñ¥½¹Ì¤(€€€€€€€€€€€€€€€€€€€Ñ¡¥Í}ÍÑ•À€ô•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€¬€Åt(€€€€€€€€€€€€€€€€€€€¹•áÑ}Ñ½­•¸€ôÑ½É ¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€€€€€Ñ¡¥Í}ÍÑ•À€ôôÕ¹•¹•É…Ñ•°¹•áÑ}Ñ½­•¸°Ñ¡¥Í}ÍÑ•À(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€•¹}Í•ÅÕ•¹•lè°½™™Í•Ð€¬€Åt€ô¹•áÑ}Ñ½­•¸((€€€€€€€€€€€€€€€€€€€€Œ…É±äÍÑ½ÀÝ¡•¸•Ù•Éä‰•…´¥¸•Ù•ÉäÍ…µÁ±”¡…Ì•µ¥ÑÑ•=L(€€€€€€€€€€€€€€€€€€€¥˜€¡•¹}Í•ÅÕ•¹”€ôô•…É±å}ÍÑ½Á}½¹}Ñ½­•¸¤¹…¹ä¡‘¥´ô´Ä¤¹…±° ¤è(€€€€€€€€€€€€€€€€€€€€€€€‰É•…¬((€€€€€€€¥˜ÁÉ½™¥±•}•¹•É…Ñ¥½¸è(€€€€€€€€€€€¥˜Õ‘…}ÁÉ½™¥±”è(€€€€€€€€€€€€€€€Ñ½É ¹Õ‘„¹Íå¹¡É½¹¥é”¡‘•Ù¥”¤(€€€€€€€€€€€€€€€™½È­¥¹°ÍÑ…ÉÑ}•Ù•¹Ð°•¹‘}•Ù•¹Ð°|¥¸ÁÉ½™¥±•}•Ù•¹ÑÌè(€€€€€€€€€€€€€€€€€€€ÁÉ½™¥±•}ÁÕ}Í•½¹‘Ím­¥¹‘t€¬ô€ (€€€€€€€€€€€€€€€€€€€€€€€ÍÑ…ÉÑ}•Ù•¹Ð¹•±…ÁÍ•‘}Ñ¥µ”¡•¹‘}•Ù•¹Ð¤€¼€ÄÀÀÀ¸À(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€Í•±˜¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±”€ôì(€€€€€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Í•½¹‘ÌˆèÁÉ½™¥±•}ÁÕ}Í•½¹‘Íl‰ÁÉ•™¥±°‰t°(€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Í•½¹‘ÌˆèÁÉ½™¥±•}ÁÕ}Í•½¹‘Íl‰‘•½‘”‰t°(€€€€€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Ñ½­•¹ÌˆèÁÉ½™¥±•}Ñ½­•¹Íl‰ÁÉ•™¥±°‰t°(€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Ñ½­•¹ÌˆèÁÉ½™¥±•}Ñ½­•¹Íl‰‘•½‘”‰t°(€€€€€€€€€€€€€€€€‰•¹•É…Ñ•‘}ÍÑ•ÁÌˆèµ…à À°±…ÍÑ}½™™Í•Ð€´ÍÑ…ÉÑ}½™™Í•Ð€¬€Ä¤°(€€€€€€€€€€€ô((€€€€€€€€Œ	•…´Í•…É èÍ•±•Ð‰•ÍÐ‰•…´Á•ÈÍ…µÁ±”…¹å¥•±…±°Ñ½­•¹Ì…Ð½¹”(€€€€€€€¥˜‰•…µ}Í¥é”€ø€Äè(€€€€€€€€€€€‰•ÍÑ}‰•…´€ô‰•…µ}Í½É•Ì¹É•Í¡…Á”¡¹Õµ}Í…µÁ±•Ì°‰•…µ}Í¥é”¤¹…Éµ…à¡‘¥´ô´Ä¤(€€€€€€€€€€€‰•ÍÑ}±½‰…°€ô€ (€€€€€€€€€€€€€€€Ñ½É ¹…É…¹”¡¹Õµ}Í…µÁ±•Ì°‘•Ù¥”õ‘•Ù¥”¤€¨‰•…µ}Í¥é”€¬‰•ÍÑ}‰•…´(€€€€€€€€€€€€¤(€€€€€€€€€€€‰•ÍÑ}Í•ÅÕ•¹”€ô•¹}Í•ÅÕ•¹•m‰•ÍÑ}±½‰…±t€€Œm¹Õµ}Í…µÁ±•Ì°Qt(€€€€€€€€€€€™½ÈÐ¥¸É…¹”¡±…ÍÑ}½™™Í•Ð€¬€Ä¤è(€€€€€€€€€€€€€€€å¥•±‰•ÍÑ}Í•ÅÕ•¹•lè°Ð€¬€Åt