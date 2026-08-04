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
        state_sequence_length: int | None = None,
        cfg_coef: float = 1.0,
        log_timing: bool = True,
    ) -> tuple[ModelState, torch.Tensor]:
        """Advance a persistent state with only clean acoustic conditions."""

        tensors = self.prepare_condition_tensors(
            conditions, cfg_coef=cfg_coef, log_timing=log_timing
        )
        prefix = self.condition_prefix_embeddings(tensors)
        if model_state is None:
            model_state = init_states(
                self,
                batch_size=prefix.shape[0],
                sequence_length=state_sequence_length or prefix.shape[1],
            )
        with self.autocast:
            hidden = self.encode_embeddings(prefix, model_state=model_state)
        increment_steps(self.transformer, model_state, increment=prefix.shape[1])
        return model_state, hidden

    def forward_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        model_state: ModelState | None = None,
    ) -> torch.Tensor:
        """Project an already packed multimodal sequence to token logits."""
        return self.linear(
            self.encode_embeddings(embeddings, model_state=model_state)
        )

    def forward(
        self,
        sequence: torch.Tensor,  # [B, S]
        condition_tensors: ConditionTensors,
        first_step: bool = False,
        model_state: ModelState | None = None,
        prepend_condition_prefix: bool | None = None,
    ) -> torch.Tensor:  # [B, S, card]
        B, S = sequence.shape

        input_ = self.emb(sequence)  # [B, S, D]
        if self.type_embedding is not None:
            token_type = torch.full_like(sequence, 3)
            input_ = input_ + self.type_embedding(token_type)

        prepend_length = 0
        if prepend_condition_prefix is None:
            prepend_condition_prefix = first_step
        if first_step and prepend_condition_prefix:
            if self.type_embedding is not None:
                prefix = self.condition_prefix_embeddings(
                    condition_tensors, batch_size=B
                )
                input_ = torch.cat([prefix, input_], dim=1)
            else:
                # Preserve the exact historical prepend order for released
                # Transformer checkpoints.
                for cond, _ in condition_tensors.values():
                    input_ = torch.cat([cond.to(input_.dtype), input_], dim=1)
            prepend_length = input_.shape[1] - S

        logits = self.forward_embeddings(input_, model_state=model_state)
        # Remove prepended conditioning tokens.
        if prepend_length > 0:
            logits = logits[:, -S:]
        return logits  # [B, S, card]

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _compute_logits(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
        prefix_already_prefilled: bool = False,
    ) -> torch.Tensor:  # [B, card]
        """Run the forward pass and return masked logits at the last timestep."""
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        if cfg_coef == 1.0:
            logits = self(
                sequence,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
                prepend_condition_prefix=not prefix_already_prefilled,
            )
        else:
            doubled = torch.cat([sequence, sequence], dim=0)
            all_logits = self(
                doubled,
                cfg_conditions,
                first_step=first_step,
                model_state=model_state,
                prepend_condition_prefix=not prefix_already_prefilled,
            )
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef

        logits = logits[:, -1, :].float()  # [B, card] — last timestep
        logits[:, 1393:] = -torch.inf  # mask reserved / OOV tokens
        if forbidden_tokens is not None:
            logits[:, forbidden_tokens] = -torch.inf
        return logits

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        cfg_conditions: ConditionTensors,
        model_state: ModelState,
        first_step: bool,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        forbidden_tokens: torch.Tensor | None = None,
        prefix_already_prefilled: bool = False,
    ) -> torch.Tensor:  # [B]
        logits = self._compute_logits(
            sequence,
            cfg_conditions,
            model_state,
            first_step,
            cfg_coef,
            forbidden_tokens=forbidden_tokens,
            prefix_already_prefilled=prefix_already_prefilled,
        )
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            next_tokens = utils.sample_from_probs(probs, top_p=top_p, top_k=top_k)[:, 0]
        else:
            next_tokens = torch.argmax(logits, dim=-1)  # [B]
        return next_tokens  # [B]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor | None = None,
        conditions: list[ConditioningAttributes] = [],
        num_samples: int | None = None,
        max_gen_len: int = 256,
        use_sampling: bool = True,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        early_stop_on_token: int | None = None,
        beam_size: int = 1,
        beam_length_score_alpha: float = 0.75,
        forbidden_tokens: torch.Tensor | list[int] | None = None,
        model_state: ModelState | None = None,
        state_sequence_length: int | None = None,
        profile_generation: bool = False,
        prefix_already_prefilled: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Autoregressively generate tokens, yielding one timestep at a time.

        Each yield is a ``[num_samples]`` tensor. For beam_size == 1 (default),
        tokens are yielded as they are generated. For beam_size > 1, beam search
        is run non-streamingly and all tokens are yielded at the end.

        ``forbidden_tokens`` are token ids whose logits are forced to -inf at
        every step, so they can never be sampled (greedy, sampling or beam).
        """
        assert not self.training
        if beam_size > 1:
            assert early_stop_on_token is not None, (
                "beam search requires early_stop_on_token"
            )
        device = self.emb.weight.device
        self.last_generation_profile = None

        if forbidden_tokens is not None and not isinstance(
            forbidden_tokens, torch.Tensor
        ):
            forbidden_tokens = torch.tensor(
                forbidden_tokens, device=device, dtype=torch.long
            )

        if num_samples is None:
            num_samples = (
                len(conditions)
                if conditions
                else (prompt.shape[0] if prompt is not None else 1)
            )

        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef

        # Build condition tensors (with null conditions appended for CFG).
        cfg_conditions = self.prepare_condition_tensors(
            conditions, cfg_coef=cfg_coef, log_timing=bool(conditions)
        )

        eff_batch = num_samples * beam_size

        # Expand conditions so each beam gets its own copy (interleaved for CFG).
        if beam_size > 1 and cfg_conditions:
            cfg_conditions = {
                k: (
                    torch.repeat_interleave(cond, beam_size, dim=0),
                    torch.repeat_interleave(mask, beam_size, dim=0),
                )
                for k, (cond, mask) in cfg_conditions.items()
            }

        # Initialise generation buffer (eff_batch rows = num_samples × beam_size)
        ungenerated = self.ungenerated_token_id
        gen_sequence = torch.full(
            (eff_batch, max_gen_len + 1),
            ungenerated,
            device=device,
            dtype=torch.long,
        )
        gen_sequence[:, 0] = self.initial_token_id

        start_offset = 0
        if prompt is not None:
            PT = prompt.shape[-1]
            if beam_size > 1:
                prompt = torch.repeat_interleave(prompt, beam_size, dim=0)
            gen_sequence[:, 1 : 1 + PT] = prompt
            ungenerated_steps = (gen_sequence == ungenerated).nonzero()[:, 1]
            start_offset = max(0, int(ungenerated_steps.amin()) - 1)

        prepend_length = sum(cond.shape[1] for cond, _ in cfg_conditions.values())
        # New checkpoints prepend an explicit chunk-start embedding in
        # addition to the condition tensors.  It participates in every
        # streaming layer and must therefore advance every cache offset.
        if self.type_embedding is not None and not prefix_already_prefilled:
            prepend_length += 1
        if prefix_already_prefilled:
            if model_state is None:
                raise ValueError(
                    "prefix_already_prefilled requires an external model_state"
                )
            if conditions:
                raise ValueError(
                    "prefilled generation must not encode conditions a second time"
                )
            prepend_length = 0
        cache_batch_size = eff_batch * (1 if cfg_coef == 1.0 else 2)
        cache_seq_len = prepend_length + max_gen_len
        if model_state is None:
            model_state = init_states(
                self,
                batch_size=cache_batch_size,
                sequence_length=state_sequence_length or cache_seq_len,
            )

        # Accumulated log-prob scores, one per beam row.
        beam_scores = torch.zeros(eff_batch, device=device, dtype=torch.float)

        cuda_profile = profile_generation and device.type == "cuda"
        profile_events: list[tuple[str, object, object, int]] = []
        profile_cpu_seconds = {"prefill": 0.0, "decode": 0.0}
        profile_tokens = {"prefill": 0, "decode": 0}

        def profile_start():
            if not profile_generation:
                return None
            if cuda_profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event
            return time.perf_counter()

        def profile_end(start, *, kind: str, tokens: int) -> None:
            if start is None:
                return
            profile_tokens[kind] += tokens
            if cuda_profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                profile_events.append((kind, start, event, tokens))
            else:
                profile_cpu_seconds[kind] += time.perf_counter() - float(start)

        # For greedy/sampling emit prompt steps now; beam search emits at the end.
        if beam_size == 1:
            for t in range(start_offset):
                yield gen_sequence[:, t + 1]

        last_offset = start_offset - 1
        with self.autocast:
            for offset in range(start_offset, max_gen_len):
                last_offset = offset
                first_iter = offset == start_offset
                input_ = (
                    gen_sequence[:, : offset + 1]
                    if first_iter
                    else gen_sequence[:, offset : offset + 1]
                )

                if beam_size == 1:
                    # ── Standard greedy / sampling path ──────────────────
                    if early_stop_on_token is not None:
                        done = (gen_sequence == early_stop_on_token).any(dim=1).all()
                        if done:
                            break

                    profile_kind = "prefill" if first_iter else "decode"
                    started = profile_start()
                    next_token = self._sample_next_token(
                        input_,
                        cfg_conditions,
                        model_state,
                        first_step=first_iter,
                        use_sampling=use_sampling,
                        temp=temp,
                        top_k=top_k,
                        top_p=top_p,
                        cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                        prefix_already_prefilled=prefix_already_prefilled,
                    )  # [B]
                    profile_end(
                        started,
                        kind=profile_kind,
                        tokens=input_.shape[-1]
                        + (prepend_length if first_iter else 0),
                    )

                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    this_gen_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(
                        this_gen_step == ungenerated, next_token, this_gen_step
                    )
                    gen_sequence[:, offset + 1] = next_token

                    yield gen_sequence[:, offset + 1]  # [num_samples]

                else:
                    # ── Beam search step ──────────────────────────────────
                    profile_kind = "prefill" if first_iter else "decode"
                    started = profile_start()
                    logits = self._compute_logits(
                        input_,
                        cfg_conditions,
                        model_state,
                        first_step=first_iter,
                        cfg_coef=cfg_coef,
                        forbidden_tokens=forbidden_tokens,
                        prefix_already_prefilled=prefix_already_prefilled,
                    )  # [eff_batch, card]
                    profile_end(
                        started,
                        kind=profile_kind,
                        tokens=input_.shape[-1]
                        + (prepend_length if first_iter else 0),
                    )
                    input_T = input_.shape[-1]
                    increment_steps(
                        self.transformer,
                        model_state,
                        increment=input_T + (prepend_length if first_iter else 0),
                    )

                    log_probs = torch.log_softmax(logits.float(), dim=-1)

                    # Top beam_size candidate tokens per current beam
                    topk_scores, topk_tokens = torch.topk(
                        log_probs, k=beam_size, dim=-1
                    )

                    # Track which beams have already emitted EOS
                    eos_mask = gen_sequence == early_stop_on_token
                    beam_has_ended = eos_mask.any(dim=-1)
                    eos_pos = eos_mask.int().argmax(dim=-1).clamp(min=1)
                    beam_lengths = torch.where(
                        beam_has_ended,
                        eos_pos,
                        torch.full_like(eos_pos, offset + 1),
                    )

                    # Finished beams: don't expand further
                    topk_scores = torch.where(
                        beam_has_ended.unsqueeze(-1),
                        torch.zeros_like(topk_scores),
                        topk_scores,
                    )

                    # Length-normalized candidate scores: [eff_batch, beam_size]
                    lp = 1.0 / (beam_lengths.float() ** beam_length_score_alpha)
                    cand = (beam_scores.unsqueeze(-1) + topk_scores) * lp.unsqueeze(-1)

                    # Reshape to [num_samples, beam_size²] for cross-beam selection
                    cand_2d = cand.reshape(num_samples, beam_size * beam_size)

                    if offset == start_offset:
                        # All beams identical at start — take first beam_size tokens
                        new_scores = cand_2d[:, :beam_size]
                        best_idx = (
                            torch.arange(beam_size, device=device)
                            .unsqueeze(0)
                            .expand(num_samples, -1)
                        )
                    else:
                        new_scores, best_idx = torch.topk(cand_2d, k=beam_size, dim=-1)

                    # Decode flat index → (prev_beam_within_sample, token_rank)
                    prev_local = (best_idx // beam_size).reshape(-1)
                    tok_rank = (best_idx % beam_size).reshape(-1)

                    # Map to global row indices in [eff_batch, …] tensors
                    sample_base = (
                        torch.arange(num_samples, device=device).repeat_interleave(
                            beam_size
                        )
                        * beam_size
                    )
                    prev_global = sample_base + prev_local

                    # Token for each new beam
                    next_token = topk_tokens[prev_global, tok_rank]

                    # Update beam scores (store un-normalized for the next step)
                    beam_scores = new_scores.reshape(-1) / lp[prev_global]

                    # Reorder generation sequences to match winning beams
                    gen_sequence = gen_sequence[prev_global]

                    reorder = prev_global
                    if cfg_coef != 1.0:
                        reorder = torch.cat([prev_global, prev_global + eff_batch])
                    reorder_states(self.transformer, model_state, reorder)

                    # Write next token (respecting pre-filled prompt positions)
                    this_step = gen_sequence[:, offset + 1]
                    next_token = torch.where(
                        this_step == ungenerated, next_token, this_step
                    )
                    gen_sequence[:, offset + 1] = next_token

                    # Early stop when every beam in every sample has emitted EOS
                    if (gen_sequence == early_stop_on_token).any(dim=-1).all():
                        break

        if profile_generation:
            if cuda_profile:
                torch.cuda.synchronize(device)
                for kind, start_event, end_event, _ in profile_events:
                    profile_cpu_seconds[kind] += (
                        start_event.elapsed_time(end_event) / 1000.0
                    )
            self.last_generation_profile = {
                "prefix_prefill_seconds": profile_cpu_seconds["prefill"],
                "autoregressive_decode_seconds": profile_cpu_seconds["decode"],
                "prefix_prefill_tokens": profile_tokens["prefill"],
                "autoregressive_decode_tokens": profile_tokens["decode"],
                "generated_steps": max(0, last_offset - start_offset + 1),
            }

        # Beam search: select best beam per sample and yield all tokens at once
        if beam_size > 1:
            best_beam = beam_scores.reshape(num_samples, beam_size).argmax(dim=-1)
            best_global = (
                torch.arange(num_samples, device=device) * beam_size + best_beam
            )
            best_sequence = gen_sequence[best_global]  # [num_samples, T]
            for t in range(last_offset + 1):
                yield best_sequence[:, t + 1]
