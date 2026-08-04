"""TranscriptionModel: main user-facing entry point."""

import contextlib
import hashlib
import io
import json
import math
import re
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F

from safetensors.torch import load_file

import muscriptor.accelerator
from muscriptor.events import (
    ChunkBoundary,
    NoteEndEvent,
    NoteStartEvent,
    OpenNoteTracker,
    ProgressEvent,
    decode_model_tokens,
)
from muscriptor.models.lm import LMModel, TorchAutocast
from muscriptor.models.config import LEGACY_PRESETS, ModelConfig
from muscriptor.modules.streaming import (
    clone_model_state,
    init_states,
    state_size_bytes,
)
from muscriptor.utils.download import download_companion, download_if_necessary
from muscriptor.modules.conditioners import (
    MelSpectrogramConditioner,
    ClassConditioner,
    ConditioningProvider,
    ConditioningAttributes,
    WavCondition,
)
from muscriptor.tokenizer.mt3 import (
    MT3_FULL_PLUS_GROUP_NAMES,
    MT3Tokenizer,
    instrument_group_from_names,
)
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Note,
    trim_overlapping_notes,
    validate_notes,
)
from muscriptor.utils.audio import load_audio, resample
from muscriptor.utils.midi import notes_to_midi


@contextlib.contextmanager
def _timed(label: str, store: list[tuple[str, float]] | None = None):
    """Print and (optionally) record how long a block of work takes."""
    muscriptor.accelerator.synchronize()
    t0 = time.perf_counter()
    yield
    muscriptor.accelerator.synchronize()
    dt = time.perf_counter() - t0
    print(f"[muscriptor] {label}: {dt:.2f}s", file=sys.stderr)
    if store is not None:
        store.append((label, dt))


# Published model variants live at hf://MuScriptor/muscriptor-<size>. A bare
# size keyword ("small"/"medium"/"large") resolves to the matching repo; the
# architecture is then read from that repo's config.json (see _resolve_config).
_HF_REPO_TEMPLATE = "hf://MuScriptor/muscriptor-{size}/model.safetensors"
_MODEL_SIZES = ("small", "medium", "large")
_DEFAULT_SIZE = "medium"


def _resolve_source(weights_path: str | Path | None) -> str | Path:
    """Map a --model value to a weights location.

    A size keyword ("small"/"medium"/"large") â€” or None, which defaults to
    ``medium`` â€” becomes the corresponding HuggingFace repo URL. Anything else
    (a local path, an ``hf://`` or ``http(s)://`` URL) is passed through as-is.
    """
    if weights_path is None:
        weights_path = _DEFAULT_SIZE
    if isinstance(weights_path, str) and weights_path in _MODEL_SIZES:
        return _HF_REPO_TEMPLATE.format(size=weights_path)
    return weights_path


_SAMPLE_RATE = 16000
# Must match the segment duration used during training / evaluation.
_SEGMENT_DURATION = 5.0


# Per-variant configs, keyed by the size that appears in the HF repo name
# (muscriptor-<size>). Each published repo also ships these values in its
# config.json; this table is the fallback when no config.json is present.
_ModelConfig = ModelConfig
_CONFIGS: dict[str, ModelConfig] = LEGACY_PRESETS

_DEFAULT_CONFIG = _CONFIGS["large"]

# Legacy local checkpoints identified by the 8-hex tag in their filename,
# mapped to the equivalent variant config.
_LEGACY_CONFIGS: dict[str, ModelConfig] = {
    "01684fbb": _CONFIGS["large"],
    "0ac4ce03": _CONFIGS["small"],
    "8f59580c": _CONFIGS["medium"],
    "e84904c4": _CONFIGS["large"],
}

_CONFIG_FILENAME = "config.json"


def _config_from_json(path: Path) -> ModelConfig:
    """Read a _ModelConfig from a HuggingFace-style config.json."""
    data = json.loads(path.read_text())
    return ModelConfig.from_dict(data)


def _resolve_config(source: str | Path, weights_path: Path) -> ModelConfig:
    """Determine the model architecture for a set of weights.

    Resolution order, most to least authoritative:
      1. ``config.json`` sitting next to the weights â€” the self-describing,
         HuggingFace-idiomatic source of truth (local dir or hf:// repo).
      2. the ``muscriptor-<size>`` segment of an ``hf://`` repo name.
      3. the legacy 8-hex tag embedded in a local checkpoint filename.
    """
    config_path = weights_path.parent / _CONFIG_FILENAME
    if not config_path.exists():
        fetched = download_companion(source, _CONFIG_FILENAME)
        if fetched is not None:
            config_path = fetched
    if config_path.exists():
        return _config_from_json(config_path)

    m = re.search(r"muscriptor-(large|medium|small)", str(source))
    if m:
        return _CONFIGS[m.group(1)]

    m = re.search(r"_([0-9a-f]{8})_", weights_path.name)
    if m and m.group(1) in _LEGACY_CONFIGS:
        return _LEGACY_CONFIGS[m.group(1)]
    return _DEFAULT_CONFIG


def _remap_single_codebook_keys(state_dict: dict) -> dict:
    """Adapt legacy multi-codebook checkpoints to the single-stream LMModel.

    Older checkpoints store the token embedding and output head as the first
    entry of an ``nn.ModuleList`` (``emb.0.*`` / ``linears.0.*``). LMModel is
    single-stream, so those map to ``emb.*`` / ``linear.*``. Checkpoints with a
    second codebook (``emb.1.*`` etc.) are unsupported and rejected.
    """
    if any(k.startswith(("emb.1.", "linears.1.")) for k in state_dict):
        raise ValueError(
            "Checkpoint has more than one codebook (n_q > 1); "
            "only single-stream models are supported."
        )
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("emb.0."):
            key = "emb." + key[len("emb.0.") :]
        elif key.startswith("linears.0."):
            key = "linear." + key[len("linears.0.") :]
        remapped[key] = value
    return remapped


def _build_model(device: torch.device, cfg: ModelConfig = _DEFAULT_CONFIG) -> LMModel:
    mel_cond = MelSpectrogramConditioner(
        output_dim=cfg.dim,
        device=device,
        sample_rate=_SAMPLE_RATE,
        n_fft=2048,
        frame_rate=100,
        n_mel_bins=512,
        log_scale=True,
        eps=1e-6,
        normalize_audio=False,
    )
    inst_cond = ClassConditioner(
        num_classes=cfg.num_instrument_condition_classes,
        output_dim=cfg.dim,
        device=device,
        correct_padding=cfg.correct_class_conditioning,
        learned_null=cfg.learned_null_conditioning,
    )
    ds_cond = ClassConditioner(
        num_classes=cfg.num_dataset_condition_classes,
        output_dim=cfg.dim,
        device=device,
        correct_padding=cfg.correct_class_conditioning,
        learned_null=cfg.learned_null_conditioning,
    )

    condition_provider = ConditioningProvider(
        conditioners={
            "self_wav": mel_cond,
            "instrument_group": inst_cond,
            "dataset_name": ds_cond,
        },
        device=device,
    )

    # Disabled off-CUDA: on MPS half precision comes from native fp16 weights
    # (see load_model) â€” autocast there is measurably slower than fp32.
    autocast = TorchAutocast(enabled=False)
    if device.type == "cuda":
        autocast = TorchAutocast(enabled=True, device_type="cuda", dtype=torch.float16)

    model = LMModel(
        condition_provider=condition_provider,
        card=cfg.card,
        dim=cfg.dim,
        num_heads=cfg.num_heads,
        hidden_scale=4,
        cfg_coef=1.0,
        autocast=autocast,
        model_config=cfg,
        device=device,
    )
    return model


def _build_instrument_for_program(tokenizer: MT3Tokenizer) -> Callable[[int], str]:
    """Map a decoded program int â†’ human-readable instrument name.

    MT3_FULL_PLUS groups multiple GM programs together; the decoded program
    is always the first program of the group. We map that representative
    back to the readable group name.
    """
    group_map = tokenizer.group_program_map
    program_to_name: dict[int, str] = {}
    for name, gid in MT3_FULL_PLUS_GROUP_NAMES.items():
        if gid in group_map and group_map[gid]:
            program_to_name[group_map[gid][0]] = name

    def lookup(program: int) -> str:
        if program == DRUM_PROGRAM:
            return "drums"
        return program_to_name.get(program, f"program_{program}")

    return lookup


class TranscriptionModel:
    """Transcribes audio to MIDI using the muscriptor model.

    Example::

        from pathlib import Path

        model = TranscriptionModel.load_model()
        for event in model.transcribe("audio.wav"):
            print(event)

        Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))
    """

    def __init__(self, model: LMModel, tokenizer: MT3Tokenizer, device: torch.device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._instrument_for_program = _build_instrument_for_program(tokenizer)
        self.last_streaming_state: dict[str, int | list[int]] | None = None
        self.last_generation_profile: dict[str, object] | None = None
        self.last_token_stream_sha256: str | None = None

    @classmethod
    def load_model(
        cls,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> "TranscriptionModel":
        """Load model weights and return a ready-to-use TranscriptionModel.

        Args:
            weights_path: A size keyword (``"small"``/``"medium"``/``"large"``)
                selecting a published HuggingFace variant, a local safetensors
                path, an ``hf://`` or ``https://`` URL, or None.  If None, the
                default ``medium`` variant is downloaded from HuggingFace.
                Remote URLs are cached under ~/.cache/muscriptor/.
            device: Torch device to use.  Defaults to the current accelerator
                (CUDA, MPS, ...) if one is available, else CPU.
            dtype: Transformer weight/compute dtype: ``"float32"``,
                ``"float16"``, ``"bfloat16"`` (or the torch dtypes). ``None``
                picks per device: float16 on MPS (halves memory traffic â€”
                decode is bandwidth-bound), float32 elsewhere (CUDA gets fp16
                compute via autocast instead). The conditioning pipeline
                (mel-spectrogram/class embeddings) always stays in fp32; its
                outputs are cast at the transformer boundary.
        """
        if device is None:
            device = (
                muscriptor.accelerator.current_accelerator()
                if muscriptor.accelerator.is_available()
                else torch.device("cpu")
            )
        elif isinstance(device, str):
            device = torch.device(device)

        if dtype is None:
            dtype = torch.float16 if device.type == "mps" else torch.float32
        elif isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        source = _resolve_source(weights_path)
        weights_path = download_if_necessary(source)
        model = _build_model(device, _resolve_config(source, weights_path))
        model.eval()

        state_dict = load_file(weights_path, device=str(device))
        state_dict = _remap_single_codebook_keys(state_dict)
        model.load_state_dict(state_dict)
        model.to(device)
        if dtype != torch.float32:
            model.to(dtype)
            # Conditioners keep fp32 numerics (log-mel of quiet passages
            # underflows in fp16); LMModel.forward casts their outputs.
            model.condition_provider.float()

        tokenizer = MT3Tokenizer(
            instrument_vocabulary="MT3_FULL_PLUS",
            max_shift_steps=1001,
        )

        return cls(model=model, tokenizer=tokenizer, device=device)

    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio: str | Path | tuple[torch.Tensor, int],
        use_sampling: bool = False,
        temperature: float = 1.0,
        cfg_coef: float = 1.0,
        instruments: list[str] | None = None,
        batch_size: int | None = None,
        no_eos_is_ok: bool = True,
        beam_size: int = 1,
        prelude_forcing: bool = True,
        long_context: str = "auto",
        profile_generation: bool = False,
        dataset_name: str | None = None,
        committed_symbolic_state: bool = False,
    ) -> Iterator[NoteStartEvent | NoteEndEvent | ProgressEvent]:
        """Transcribe audio into a stream of note events.

        See the README for full argument documentation and the streaming /
        chunk-ordering guarantees. The audio is split into 5-second chunks;
        within each chunk events arrive in temporal order, and all events
        from chunk N are yielded before any event from chunk N+1.

        ``instruments``, when given, is a hard constraint: every program/drum
        token outside the listed groups is masked out during generation, so
        no other instrument can appear in the output. Leave it unset to let
        the model decode whatever instruments it detects.

        ``prelude_forcing`` (default True) teacher-forces each chunk's tie
        prologue â€” the tokens declaring which notes are sustained from the
        previous chunk â€” from the previous chunk's actually-unfinished notes,
        instead of letting the model guess (and occasionally re-enter with
        the wrong instruments). It requires chunks to be generated strictly
        in order, so while it is on the batch size defaults to (and must be)
        1; combining it with ``batch_size > 1`` raises ValueError â€” pass
        ``prelude_forcing=False`` explicitly to trade chunk-boundary quality
        for batched throughput.

        Interleaved with the note events are coarse :class:`ProgressEvent`
        anchors (``completed`` of ``total`` chunks): one up front with
        ``completed == 0``, then one as each chunk finishes. Consumers that
        only care about notes can ignore them.
        """
        if long_context not in {"auto", "carry", "reset", "clean_audio"}:
            raise ValueError(
                "long_context must be 'auto', 'carry', 'reset', or 'clean_audio'"
            )
        if long_context == "auto":
            model_config = getattr(getattr(self, "_model", None), "model_config", None)
            long_context = (
                "reset"
                if getattr(model_config, "backbone", "transformer") == "transformer"
                else "carry"
            )
        if long_context in {"carry", "clean_audio"} and beam_size != 1:
            raise ValueError(
                f"long-context {long_context} currently requires beam_size=1"
            )
        batch_size = self._resolve_batch_size(batch_size, prelude_forcing)
        if long_context in {"carry", "clean_audio"} and batch_size != 1:
            raise ValueError(f"long-context {long_context} requires batch_size=1")
        if committed_symbolic_state ç^ú¶‰žËkºwµçEÕ‘¥¼ÕÉÉ•¹Ñ±äÉ•ÅÕ¥É•Ì‰•…µ}Í¥é”ôÄˆ¤(€€€€€€€¥˜±•¸¡…±±}½¹‘¥Ñ¥½¹Ì¤€„ô±•¸¡Í••­}Ñ¥µ•Ì¤è(€€€€€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È ‰½¹‘¥Ñ¥½¸½Í••¬µÑ¥µ”±•¹Ñ µ¥Íµ…Ñ ˆ¤((€€€€€€€•½Í}¥€ôÍ•±˜¹}Ñ½­•¹¥é•È¹•½Í}¥(€€€€€€€¹Õµ}¡Õ¹­Ì€ô±•¸¡Í••­}Ñ¥µ•Ì¤(€€€€€€€±•…¹}ÍÑ…Ñ”€ô9½¹”(€€€€€€€ÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ €ô¹Õµ}¡Õ¹­Ì€¨€¡µ…á}•¹}±•¸€¬€ØÐÀ¤(€€€€€€€½µµ¥ÑÑ•€ô€ (€€€€€€€€€€€=Á•¹9½Ñ•QÉ…­•È¡Í•±˜¹}Ñ½­•¹¥é•È¹}Ù½…ˆ°Í•±˜¹}Ñ½­•¹¥é•È¹™É…µ•}É…Ñ”¤(€€€€€€€€€€€¥˜ÁÉ•±Õ‘•}™½É¥¹œ½È½µµ¥ÑÑ•‘}Íåµ‰½±¥}ÍÑ…Ñ”(€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€¤(€€€€€€€½µµ¥ÑÑ•‘}¡Õ¹­Ì€ô€À(€€€€€€€É•©•Ñ•‘}¡Õ¹­Ì€ô€À(€€€€€€€±•…¹}ÁÉ•™¥á}Ñ½­•¹Ì€ô€À(€€€€€€€‰½Õ¹‘…Éå}…É••µ•¹Ñ}ÍÕ´€ô€À¸À(€€€€€€€‰½Õ¹‘…Éå}…É••µ•¹Ñ}½Õ¹Ð€ô€À(€€€€€€€ÁÉ•Ù¥½ÕÍ}É••¹ÑÉä€ô9½¹”(€€€€€€€ÁÉ•Ù¥½ÕÍ}É••¹ÑÉå}Ù…±¥€ô9½¹”(€€€€€€€•¹•É…Ñ¥½¹}ÁÉ½™¥±•Ìè±¥ÍÑm‘¥ÑmÍÑÈ°™±½…Ðð¥¹Ñut€ômt((€€€€€€€™½È¡Õ¹­}¥¹‘•à°€¡½¹‘¥Ñ¥½¸°Í••­}Ñ¥µ”¤¥¸•¹Õµ•É…Ñ” (€€€€€€€€€€€é¥À¡…±±}½¹‘¥Ñ¥½¹Ì°Í••­}Ñ¥µ•Ì¤(€€€€€€€€¤è(€€€€€€€€€€€¹•áÑ}Í••¬€ô€ (€€€€€€€€€€€€€€€Í••­}Ñ¥µ•Ím¡Õ¹­}¥¹‘•à€¬€Åt(€€€€€€€€€€€€€€€¥˜¡Õ¹­}¥¹‘•à€¬€Ä€ð¹Õµ}¡Õ¹­Ì(€€€€€€€€€€€€€€€•±Í”9½¹”(€€€€€€€€€€€€¤(€€€€€€€€€€€‰½Õ¹‘…Éä€ô¡Õ¹­	½Õ¹‘…Éä¡Í••­}Ñ¥µ”°¹•áÑ}Í••¬¤(€€€€€€€€€€€…¹‘¥‘…Ñ”€ô½µµ¥ÑÑ•¹™½É¬ ¤¥˜½µµ¥ÑÑ•¥Ì¹½Ð9½¹”•±Í”9½¹”(€€€€€€€€€€€ÁÉ½µÁÐ€ô9½¹”(€€€€€€€€€€€¥˜…¹‘¥‘…Ñ”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”¹™••¡‰½Õ¹‘…Éä¤(€€€€€€€€€€€€€€€¥˜ÁÉ•±Õ‘•}™½É¥¹œ…¹¡Õ¹­}¥¹‘•à€ø€Àè(€€€€€€€€€€€€€€€€€€€ÁÉ½µÁÐ€ôÑ½É ¹Ñ•¹Í½È (€€€€€€€€€€€€€€€€€€€€€€€mÍ•±˜¹}Ñ½­•¹¥é•È¹Ñ¥•}Í•Ñ¥½¹}Ñ½­•¹}¥‘Ì¡…¹‘¥‘…Ñ”¹½Á•¹}­•åÌ ¤¥t°(€€€€€€€€€€€€€€€€€€€€€€€‘•Ù¥”õÍ•±˜¹}‘•Ù¥”°(€€€€€€€€€€€€€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹±½¹œ°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€å¥•±‰½Õ¹‘…Éä((€€€€€€€€€€€±•…¹}ÍÑ…Ñ”°±•…¹}¡¥‘‘•¸€ôÍ•±˜¹}µ½‘•°¹ÁÉ•™¥±±}½¹‘¥Ñ¥½¹}ÁÉ•™¥à (€€€€€€€€€€€€€€€m½¹‘¥Ñ¥½¹t°(€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”õ±•…¹}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€ÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ õÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ °(€€€€€€€€€€€€€€€™}½•˜õ™}½•˜°(€€€€€€€€€€€€€€€±½}Ñ¥µ¥¹œõQÉÕ”°(€€€€€€€€€€€€¤(€€€€€€€€€€€±•…¹}ÁÉ•™¥á}Ñ½­•¹Ì€¬ô¥¹Ð¡±•…¹}¡¥‘‘•¸¹Í¡…Á•lÅt¤(€€€€€€€€€€€‘•½‘•}ÍÑ…Ñ”€ô±½¹•}µ½‘•±}ÍÑ…Ñ”¡±•…¹}ÍÑ…Ñ”°‘•Ñ… õQÉÕ”¤(€€€€€€€€€€€ÕÉÉ•¹Ñ}…Ñ¥Ù•}±½¥ÑÌ€ôÕÉÉ•¹Ñ}É••¹ÑÉå}±½¥ÑÌ€ô9½¹”(€€€€€€€€€€€¥˜Í•±˜¹}µ½‘•°¹µ½‘•±}½¹™¥œ¹±•…¹}…½ÕÍÑ¥}…¡”è(€€€€€€€€€€€€€€€ÕÉÉ•¹Ñ}…Ñ¥Ù•}±½¥ÑÌ°ÕÉÉ•¹Ñ}É••¹ÑÉå}±½¥ÑÌ€ô€ (€€€€€€€€€€€€€€€€€€€Í•±˜¹}µ½‘•°¹‰½Õ¹‘…Éå}ÍÑ…Ñ•}±½¥ÑÌ¡±•…¹}¡¥‘‘•¹lè°€´Åt¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€¥˜½µµ¥ÑÑ•‘}Íåµ‰½±¥}ÍÑ…Ñ”è(€€€€€€€€€€€€€€€¥˜…¹‘¥‘…Ñ”¥Ì9½¹”è(€€€€€€€€€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È ‰½µµ¥ÑÑ•ÑÉ…­•ÈÝ…Ì¹½Ð¥¹¥Ñ¥…±¥é•ˆ¤(€€€€€€€€€€€€€€€™œ€ôÍ•±˜¹}µ½‘•°¹µ½‘•±}½¹™¥œ(€€€€€€€€€€€€€€€ÍÑ…Ñ•}‰…Ñ €ô€Ä¥˜™}½•˜€ôô€Ä¸À•±Í”€È(€€€€€€€€€€€€€€€…Ñ¥Ù”€ôÑ½É ¹é•É½Ì (€€€€€€€€€€€€€€€€€€€ÍÑ…Ñ•}‰…Ñ °(€€€€€€€€€€€€€€€€€€€™œ¹¹Õµ}Íåµ‰½±¥}¥¹ÍÑÉÕµ•¹Ñ}É½ÕÁÌ°(€€€€€€€€€€€€€€€€€€€™œ¹¹Õµ}Íåµ‰½±¥}Á¥Ñ¡•Ì°(€€€€€€€€€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹‰½½°°(€€€€€€€€€€€€€€€€€€€‘•Ù¥”õÍ•±˜¹}‘•Ù¥”°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€É•Ù•ÉÍ”€ôì(€€€€€€€€€€€€€€€€€€€ÁÉ½É…´èÉ½ÕÀ(€€€€€€€€€€€€€€€€€€€™½ÈÉ½ÕÀ°ÁÉ½É…µÌ¥¸Í•±˜¹}Ñ½­•¹¥é•È¹É½ÕÁ}ÁÉ½É…µ}µ…À¹¥Ñ•µÌ ¤(€€€€€€€€€€€€€€€€€€€™½ÈÁÉ½É…´¥¸ÁÉ½É…µÌ(€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€™½ÈÁÉ½É…´°Á¥Ñ ¥¸…¹‘¥‘…Ñ”¹½Á•¹}­•åÌ ¤è(€€€€€€€€€€€€€€€€€€€É½ÕÀ€ôÉ•Ù•ÉÍ”¹•Ð¡ÁÉ½É…´¤(€€€€€€€€€€€€€€€€€€€¥˜É½ÕÀ¥Ì¹½Ð9½¹”…¹€À€ðôÁ¥Ñ €ð…Ñ¥Ù”¹Í¡…Á•l´Åtè(€€€€€€€€€€€€€€€€€€€€€€€…Ñ¥Ù•lÀ°É½ÕÀ°Á¥Ñ¡t€ôQÉÕ”(€€€€€€€€€€€€€€€É••¹ÑÉä€ôÉ••¹ÑÉå}Ù…±¥€ô9½¹”(€€€€€€€€€€€€€€€¥˜ÁÉ•Ù¥½ÕÍ}É••¹ÑÉä¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€€€€€É••¹ÑÉä€ôÑ½É ¹™Õ±° (€€€€€€€€€€€€€€€€€€€€€€€€¡ÍÑ…Ñ•}‰…Ñ °™œ¹¹Õµ}Íåµ‰½±¥}¥¹ÍÑÉÕµ•¹Ñ}É½ÕÁÌ¤°(€€€€€€€€€€€€€€€€€€€€€€€™œ¹¹Õµ}É••¹ÑÉå}±…ÍÍ•Ì€´€Ä°(€€€€€€€€€€€€€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹±½¹œ°(€€€€€€€€€€€€€€€€€€€€€€€‘•Ù¥”õÍ•±˜¹}‘•Ù¥”°(€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€É••¹ÑÉå}Ù…±¥€ôÑ½É ¹é•É½Í}±¥­”¡É••¹ÑÉä°‘ÑåÁ”õÑ½É ¹‰½½°¤(€€€€€€€€€€€€€€€€€€€É••¹ÑÉålÁt€ôÁÉ•Ù¥½ÕÍ}É••¹ÑÉä(€€€€€€€€€€€€€€€€€€€É••¹ÑÉå}Ù…±¥‘lÁt€ôÁÉ•Ù¥½ÕÍ}É••¹ÑÉå}Ù…±¥(€€€€€€€€€€€€€€€Í•±˜¹}µ½‘•°¹ÁÉ•™¥±±}Íåµ‰½±¥}ÍÑ…Ñ” (€€€€€€€€€€€€€€€€€€€‘•½‘•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€€€€€…Ñ¥Ù”°(€€€€€€€€€€€€€€€€€€€É••¹ÑÉäõÉ••¹ÑÉä°(€€€€€€€€€€€€€€€€€€€É••¹ÑÉå}Ù…±¥õÉ••¹ÑÉå}Ù…±¥°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€•µ¥ÑÑ•‘}•½Ì€ô…±Í”(€€€€€€€€€€€™½ÈÍÑ•À¥¸Í•±˜¹}µ½‘•°¹•¹•É…Ñ” (€€€€€€€€€€€€€€€ÁÉ½µÁÐõÁÉ½µÁÐ°(€€€€€€€€€€€€€€€½¹‘¥Ñ¥½¹Ìõmt°(€€€€€€€€€€€€€€€¹Õµ}Í…µÁ±•ÌôÄ°(€€€€€€€€€€€€€€€µ…á}•¹}±•¸õµ…á}•¹}±•¸°(€€€€€€€€€€€€€€€ÕÍ•}Í…µÁ±¥¹œõÕÍ•}Í…µÁ±¥¹œ°(€€€€€€€€€€€€€€€Ñ•µÀõÑ•µÁ•É…ÑÕÉ”°(€€€€€€€€€€€€€€€Ñ½Á}¬ôÀ°(€€€€€€€€€€€€€€€Ñ½Á}ÀôÀ¸À°(€€€€€€€€€€€€€€€™}½•˜õ™}½•˜°(€€€€€€€€€€€€€€€•…É±å}ÍÑ½Á}½¹}Ñ½­•¸õ•½Í}¥°(€€€€€€€€€€€€€€€‰•…µ}Í¥é”õ‰•…µ}Í¥é”°(€€€€€€€€€€€€€€€™½É‰¥‘‘•¹}Ñ½­•¹Ìõ™½É‰¥‘‘•¹}Ñ½­•¹Ì°(€€€€€€€€€€€€€€€µ½‘•±}ÍÑ…Ñ”õ‘•½‘•}ÍÑ…Ñ”°(€€€€€€€€€€€€€€€ÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ õÍÑ…Ñ•}Í•ÅÕ•¹•}±•¹Ñ °(€€€€€€€€€€€€€€€ÁÉ½™¥±•}•¹•É…Ñ¥½¸õÁÉ½™¥±•}•¹•É…Ñ¥½¸°(€€€€€€€€€€€€€€€ÁÉ•™¥á}…±É•…‘å}ÁÉ•™¥±±•õQÉÕ”°(€€€€€€€€€€€€¤è(€€€€€€€€€€€€€€€Ñ½­•¸€ô¥¹Ð¡ÍÑ•ÁlÁt¤(€€€€€€€€€€€€€€€¥˜Ñ½­•¸€ôô•½Í}¥è(€€€€€€€€€€€€€€€€€€€•µ¥ÑÑ•‘}•½Ì€ôQÉÕ”(€€€€€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€€€€€¥˜…¹‘¥‘…Ñ”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€€€€€…¹‘¥‘…Ñ”¹™••¡Ñ½­•¸¤(€€€€€€€€€€€€€€€å¥•±Ñ½­•¸((€€€€€€€€€€€€Œ‘•½‘•}ÍÑ…Ñ•€¥Ì¥¹Ñ•¹Ñ¥½¹…±±ä¹½Ð…ÍÍ¥¹•‰…¬Ñ¼±•…¹}ÍÑ…Ñ”¸(€€€€€€€€€€€‘•°‘•½‘•}ÍÑ…Ñ”(€€€€€€€€€€€¥˜…¹‘¥‘…Ñ”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€¥˜•µ¥ÑÑ•‘}•½Ì…¹…¹‘¥‘…Ñ”¹½µµ¥Ñ}É•…‘äè(€€€€€€€€€€€€€€€€€€€½µµ¥ÑÑ•€ô…¹‘¥‘…Ñ”(€€€€€€€€€€€€€€€€€€€½µµ¥ÑÑ•‘}¡Õ¹­Ì€¬ô€Ä(€€€€€€€€€€€€€€€€€€€¥˜ÕÉÉ•¹Ñ}…Ñ¥Ù•}±½¥ÑÌ¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€€€€€€€€€™œ€ôÍ•±˜¹}µ½‘•°¹µ½‘•±}½¹™¥œ(€€€€€€€€€€€€€€€€€€€€€€€ÑÉÕÑ €ôÑ½É ¹é•É½Ì (€€€€€€€€€€€€€€€€€€€€€€€€€€€™œ¹¹Õµ}Íåµ‰½±¥}¥¹ÍÑÉÕµ•¹Ñ}É½ÕÁÌ°(€€€€€€€€€€€€€€€€€€€€€€€€€€€™œ¹¹Õµ}Íåµ‰½±¥}Á¥Ñ¡•Ì°(€€€€€€€€€€€€€€€€€€€€€€€€€€€‘ÑåÁ”õÑ½É ¹‰½½°°(€€€€€€€€€€€€€€€€€€€€€€€€€€€‘•Ù¥”õÍ•±˜¹}‘•Ù¥”°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€É•Ù•ÉÍ”€ôì(€€€€€€€€€€€€€€€€€€€€€€€€€€€ÁÉ½É…´èÉ½ÕÀ(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½ÈÉ½ÕÀ°ÁÉ½É…µÌ¥¸Í•±˜¹}Ñ½­•¹¥é•È¹É½ÕÁ}ÁÉ½É…µ}µ…À¹¥Ñ•µÌ ¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€™½ÈÁÉ½É…´¥¸ÁÉ½É…µÌ(€€€€€€€€€€€€€€€€€€€€€€€ô(€€€€€€€€€€€€€€€€€€€€€€€™½ÈÁÉ½É…´°Á¥Ñ ¥¸…¹‘¥‘…Ñ”¹½Á•¹}­•åÌ ¤è(€€€€€€€€€€€€€€€€€€€€€€€€€€€É½ÕÀ€ôÉ•Ù•ÉÍ”¹•Ð¡ÁÉ½É…´¤(€€€€€€€€€€€€€€€€€€€€€€€€€€€¥˜É½ÕÀ¥Ì¹½Ð9½¹”…¹€À€ðôÁ¥Ñ €ðÑÉÕÑ ¹Í¡…Á•l´Åtè(€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ÑÉÕÑ¡mÉ½ÕÀ°Á¥Ñ¡t€ôQÉÕ”(€€€€€€€€€€€€€€€€€€€€€€€ÁÉ•‘¥Ñ•€ôÑ½É ¹Í¥µ½¥¡ÕÉÉ•¹Ñ}…Ñ¥Ù•}±½¥ÑÍlÁt¤€øô€À¸Ô(€€€€€€€€€€€€€€€€€€€€€€€¥¹Ñ•ÉÍ•Ñ¥½¸€ô€¡ÁÉ•‘¥Ñ•€˜ÑÉÕÑ ¤¹ÍÕ´ ¤¹™±½…Ð ¤(€€€€€€€€€€€€€€€€€€€€€€€‘•¹½µ¥¹…Ñ½È€ôÁÉ•‘¥Ñ•¹ÍÕ´ ¤€¬ÑÉÕÑ ¹ÍÕ´ ¤(€€€€€€€€€€€€€€€€€€€€€€€…É••µ•¹Ð€ôÑ½É ¹Ý¡•É” (€€€€€€€€€€€€€€€€€€€€€€€€€€€‘•¹½µ¥¹…Ñ½È€ø€À°(€€€€€€€€€€€€€€€€€€€€€€€€€€€€È€¨¥¹Ñ•ÉÍ•Ñ¥½¸€¼‘•¹½µ¥¹…Ñ½È°(€€€€€€€€€€€€€€€€€€€€€€€€€€€Ñ½É ¹½¹•Í}±¥­”¡¥¹Ñ•ÉÍ•Ñ¥½¸¤°(€€€€€€€€€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€€€€€€€€€‰½Õ¹‘…Éå}…É••µ•¹Ñ}ÍÕ´€¬ô™±½…Ð¡…É••µ•¹Ð¤(€€€€€€€€€€€€€€€€€€€€€€€‰½Õ¹‘…Éå}…É••µ•¹Ñ}½Õ¹Ð€¬ô€Ä(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€É•©•Ñ•‘}¡Õ¹­Ì€¬ô€Ä((€€€€€€€€€€€¥˜ÕÉÉ•¹Ñ}É••¹ÑÉå}±½¥ÑÌ¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€ÁÉ½‰…‰¥±¥Ñ¥•Ì€ôÑ½É ¹Í½™Ñµ…à¡ÕÉÉ•¹Ñ}É••¹ÑÉå}±½¥ÑÍlÁt¹™±½…Ð ¤°‘¥´ô´Ä¤(€€€€€€€€€€€€€€€½¹™¥‘•¹”°ÁÉ•Ù¥½ÕÍ}É••¹ÑÉä€ôÁÉ½‰…‰¥±¥Ñ¥•Ì¹µ…à¡‘¥´ô´Ä¤(€€€€€€€€€€€€€€€ÁÉ•Ù¥½ÕÍ}É••¹ÑÉå}Ù…±¥€ô½¹™¥‘•¹”€øô€À¸ÐÀ((€€€€€€€€€€€¥˜¹½Ð•µ¥ÑÑ•‘}•½Ìè(€€€€€€€€€€€€€€€µ•ÍÍ…”€ô€ (€€€€€€€€€€€€€€€€€€€˜‰¡Õ¹¬í¡Õ¹­}¥¹‘•áô€¡Í••¬õíÍ••­}Ñ¥µ”è¸Å™õÌ¤‘¥¹½Ð•µ¥Ð=L€ˆ(€€€€€€€€€€€€€€€€€€€˜‰Ý¥Ñ¡¥¸íµ…á}•¹}±•¹ôÑ½­•¹Ìˆ(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€¥˜¹½}•½Í}¥Í}½¬è(€€€€€€€€€€€€€€€€€€€Ý…É¹¥¹Ì¹Ý…É¸¡µ•ÍÍ…”°IÕ¹Ñ¥µ•]…É¹¥¹œ°ÍÑ…­±•Ù•°ôÈ¤(€€€€€€€€€€€€€€€•±Í”è(€€€€€€€€€€€€€€€€€€€É…¥Í”IÕ¹Ñ¥µ•ÉÉ½È (€€€€€€€€€€€€€€€€€€€€€€€µ•ÍÍ…”€¬€ˆ€¡Ñ¡¥Ì¥Ì½¹±äÉ…¥Í•Õ¹‘•È€´µÍÑÉ¥Ðµ•½Ì¤ˆ(€€€€€€€€€€€€€€€€€€€€¤((€€€€€€€€€€€å¥•±AÉ½É•ÍÍÙ•¹Ð¡½µÁ±•Ñ•õ¡Õ¹­}¥¹‘•à€¬€Ä°Ñ½Ñ…°õ¹Õµ}¡Õ¹­Ì¤(€€€€€€€€€€€¥˜ÁÉ½™¥±•}•¹•É…Ñ¥½¸…¹Í•±˜¹}µ½‘•°¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€€€€€•¹•É…Ñ¥½¹}ÁÉ½™¥±•Ì¹…ÁÁ•¹¡‘¥Ð¡Í•±˜¹}µ½‘•°¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±”¤¤((€€€€€€€¥˜±•…¹}ÍÑ…Ñ”¥Ì¹½Ð9½¹”è(€€€€€€€€€€€±½…±}…¡•}±•¹Ñ¡Ì€ôl(€€€€€€€€€€€€€€€¥¹Ð¡Ù…±Õ•l‰­•ä‰t¹Í¡…Á•lÉt¤(€€€€€€€€€€€€€€€™½ÈÙ…±Õ”¥¸±•…¹}ÍÑ…Ñ”¹Ù…±Õ•Ì ¤(€€€€€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡Ù…±Õ”°‘¥Ð¤…¹€‰­•äˆ¥¸Ù…±Õ”(€€€€€€€€€€€t(€€€€€€€€€€€Í•±˜¹±…ÍÑ}ÍÑÉ•…µ¥¹}ÍÑ…Ñ”€ôì(€€€€€€€€€€€€€€€€‰‰åÑ•ÌˆèÍÑ…Ñ•}Í¥é•}‰åÑ•Ì¡±•…¹}ÍÑ…Ñ”¤°(€€€€€€€€€€€€€€€€‰±½…±}…¡•}±•¹Ñ¡Ìˆè±½…±}…¡•}±•¹Ñ¡Ì°(€€€€€€€€€€€€€€€€‰±•…¹}ÁÉ•™¥á}Ñ½­•¹Ìˆè±•…¹}ÁÉ•™¥á}Ñ½­•¹Ì°(€€€€€€€€€€€€€€€€‰½µµ¥ÑÑ•‘}Íåµ‰½±¥}¡Õ¹­Ìˆè½µµ¥ÑÑ•‘}¡Õ¹­Ì°(€€€€€€€€€€€€€€€€‰É•©•Ñ•‘}Íåµ‰½±¥}¡Õ¹­ÌˆèÉ•©•Ñ•‘}¡Õ¹­Ì°(€€€€€€€€€€€€€€€€‰‰½Õ¹‘…Éå}…Ñ¥Ù•}˜Äˆè€ (€€€€€€€€€€€€€€€€€€€‰½Õ¹‘…Éå}…É••µ•¹Ñ}ÍÕ´€¼‰½Õ¹‘…Éå}…É••µ•¹Ñ}½Õ¹Ð(€€€€€€€€€€€€€€€€€€€¥˜‰½Õ¹‘…Éå}…É••µ•¹Ñ}½Õ¹Ð(€€€€€€€€€€€€€€€€€€€•±Í”€À¸À(€€€€€€€€€€€€€€€€¤°(€€€€€€€€€€€ô(€€€€€€€¥˜ÁÉ½™¥±•}•¹•É…Ñ¥½¸è(€€€€€€€€€€€Í•±˜¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±”€ôì(€€€€€€€€€€€€€€€­•äèÍÕ´¡™±½…Ð¡ÁÉ½™¥±”¹•Ð¡­•ä°€À¤¤™½ÈÁÉ½™¥±”¥¸•¹•É…Ñ¥½¹}ÁÉ½™¥±•Ì¤(€€€€€€€€€€€€€€€™½È­•ä¥¸€ (€€€€€€€€€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Í•½¹‘Ìˆ°(€€€€€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Í•½¹‘Ìˆ°(€€€€€€€€€€€€€€€€€€€€‰ÁÉ•™¥á}ÁÉ•™¥±±}Ñ½­•¹Ìˆ°(€€€€€€€€€€€€€€€€€€€€‰…ÕÑ½É•É•ÍÍ¥Ù•}‘•½‘•}Ñ½­•¹Ìˆ°(€€€€€€€€€€€€€€€€€€€€‰•¹•É…Ñ•‘}ÍÑ•ÁÌˆ°(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€ô(€€€€€€€€€€€Í•±˜¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±•l‰±•…¹}…Õ‘¥½}ÁÉ•™¥±±}Ñ½­•¹Ì‰t€ô€ (€€€€€€€€€€€€€€€±•…¹}ÁÉ•™¥á}Ñ½­•¹Ì(€€€€€€€€€€€€¤(€€€€€€€€€€€Í•±˜¹±…ÍÑ}•¹•É…Ñ¥½¹}ÁÉ½™¥±•l‰¡Õ¹­Ì‰t€ô•¹•É…Ñ¥½¹}ÁÉ½™¥±•Ì((€€€€Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(€€€‘•˜ÑÉ…¹ÍÉ¥‰•}Ñ½}µ¥‘¤ (€€€€€€€Í•±˜°(€€€€€€€…Õ‘¥¼èÍÑÈðA…Ñ ðÑÕÁ±•mÑ½É ¹Q•¹Í½È°¥¹Ñt°(€€€€€€€ÕÍ•}Í…µÁ±¥¹œè‰½½°€ô…±Í”°(€€€€€€€Ñ•µÁ•É…ÑÕÉ”è™±½…Ð€ô€Ä¸À°(€€€€€€€™}½•˜è™±½…Ð€ô€Ä¸À°(€€€€€€€¥¹ÍÑÉÕµ•¹ÑÌè±¥ÍÑmÍÑÉtð9½¹”€ô9½¹”°(€€€€€€€‰…Ñ¡}Í¥é”è¥¹Ðð9½¹”€ô9½¹”°(€€€€€€€¹½}•½Í}¥Í}½¬è‰½½°€ôQÉÕ”°(€€€€€€€‰•…µ}Í¥é”è¥¹Ð€ô€Ä°(€€€€€€€ÁÉ•±Õ‘•}™½É¥¹œè‰½½°€ôQÉÕ”°(€€€€€€€±½¹}½¹Ñ•áÐèÍÑÈ€ô€‰…ÕÑ¼ˆ°(€€€€€€€‘…Ñ…Í•Ñ}¹…µ”èÍÑÈð9½¹”€ô9½¹”°(€€€€€€€½µµ¥ÑÑ•‘}Íåµ‰½±¥}ÍÑ…Ñ”è‰½½°€ô…±Í”°(€€€€¤€´ø‰åÑ•Ìè(€€€€€€€€ˆˆ‰M…µ”…Ì€éµ•Ñ éÑÉ…¹ÍÉ¥‰•€‰ÕÐÉ•ÑÕÉ¹Ì„5%$™¥±”…Ì‰åÑ•Ì¸ˆˆˆ(€€€€€€€•Ù•¹ÑÌ€ôÍ•±˜¹ÑÉ…¹ÍÉ¥‰” (€€€€€€€€€€€…Õ‘¥¼°(€€€€€€€€€€€ÕÍ•}Í…µÁ±¥¹œõÕÍ•}Í…µÁ±¥¹œ°(€€€€€€€€€€€Ñ•µÁ•É…ÑÕÉ”õÑ•µÁ•É…ÑÕÉ”°(€€€€€€€€€€€™}½•˜õ™}½•˜°(€€€€€€€€€€€¥¹ÍÑÉÕµ•¹ÑÌõ¥¹ÍÑÉÕµ•¹ÑÌ°(€€€€€€€€€€€‰…Ñ¡}Í¥é”õ‰…Ñ¡}Í¥é”°(€€€€€€€€€€€¹½}•½Í}¥Í}½¬õ¹½}•½Í}¥Í}½¬°(€€€€€€€€€€€‰•…µ}Í¥é”õ‰•…µ}Í¥é”°(€€€€€€€€€€€ÁÉ•±Õ‘•}™½É¥¹œõÁÉ•±Õ‘•}™½É¥¹œ°(€€€€€€€€€€€±½¹}½¹Ñ•áÐõ±½¹}½¹Ñ•áÐ°(€€€€€€€€€€€‘…Ñ…Í•Ñ}¹…µ”õ‘…Ñ…Í•Ñ}¹…µ”°(€€€€€€€€€€€½µµ¥ÑÑ•‘}Íåµ‰½±¥}ÍÑ…Ñ”õ½µµ¥ÑÑ•‘}Íåµ‰½±¥}ÍÑ…Ñ”°(€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸Í•±˜¹•Ù•¹ÑÍ}Ñ½}µ¥‘¥}‰åÑ•Ì¡•Ù•¹ÑÌ¤((€€€‘•˜•Ù•¹ÑÍ}Ñ½}µ¥‘¥}‰åÑ•Ì (€€€€€€€Í•±˜°•Ù•¹ÑÌè%Ñ•É…Ñ½Ém9½Ñ•MÑ…ÉÑÙ•¹Ðð9½Ñ•¹‘Ù•¹ÐðAÉ½É•ÍÍÙ•¹Ñt(€€€€¤€´ø‰åÑ•Ìè(€€€€€€€€ˆˆ‰I•…ÍÍ•µ‰±”9½Ñ•Ì™É½´„9½Ñ•MÑ…ÉÐ½9½Ñ•¹ÍÑÉ•…´…¹Í•É¥…±¥é”5%$¸((€€€€€€€M¡…É•‰ä€éµ•Ñ éÑÉ…¹ÍÉ¥‰•}Ñ½}µ¥‘¥€…¹Ñ¡”!QQ@Í•ÉÙ•È°Í¼Ñ¡”5%$(€€€€€€€‰åÑ•Ì…É”¥‘•¹Ñ¥…°É•…É‘±•ÍÌ½˜¡½ÜÑ¡”•Ù•¹ÑÌÝ•É”½‰Ñ…¥¹•¸(€€€€€€€€ˆˆˆ(€€€€€€€¹½Ñ•Ìè±¥ÍÑm9½Ñ•t€ômt(€€€€€€€½Á•¹}¹½Ñ•Ìè‘¥Ñm¥¹Ð°9½Ñ•t€ôíô(€€€€€€€ÁÉ½É…µ}¹…µ•Ìè‘¥Ñm¥¹Ð°ÍÑÉt€ôíô(€€€€€€€™½È•Ø¥¸•Ù•¹ÑÌè(€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡•Ø°AÉ½É•ÍÍÙ•¹Ð¤è(€€€€€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡•Ø°9½Ñ•MÑ…ÉÑÙ•¹Ð¤è(€€€€€€€€€€€€€€€¥Í}‘ÉÕ´€ô•Ø¹¥¹ÍÑÉÕµ•¹Ð€ôô€‰‘ÉÕµÌˆ(€€€€€€€€€€€€€€€ÁÉ½É…´€ô€ (€€€€€€€€€€€€€€€€€€€IU5}AI=I4(€€€€€€€€€€€€€€€€€€€¥˜¥Í}‘ÉÕ´(€€€€€€€€€€€€€€€€€€€•±Í”Í•±˜¹}ÁÉ½É…µ}™½É}¥¹ÍÑÉÕµ•¹Ð¡•Ø¹¥¹ÍÑÉÕµ•¹Ð¤(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€ÁÉ½É…µ}¹…µ•ÍmÁÉ½É…µt€ô•Ø¹¥¹ÍÑÉÕµ•¹Ð¹É•Á±…” ‰|ˆ°€ˆ€ˆ¤(€€€€€€€€€€€€€€€¹½Ñ”€ô9½Ñ” (€€€€€€€€€€€€€€€€€€€¥Í}‘ÉÕ´õ¥Í}‘ÉÕ´°(€€€€€€€€€€€€€€€€€€€ÁÉ½É…´õÁÉ½É…´°(€€€€€€€€€€€€€€€€€€€½¹Í•Ðõ•Ø¹ÍÑ…ÉÑ}Ñ¥µ”°(€€€€€€€€€€€€€€€€€€€½™™Í•Ðõ•Ø¹ÍÑ…ÉÑ}Ñ¥µ”°€€ŒÁ…Ñ¡•½¸9½Ñ•¹‘Ù•¹Ð(€€€€€€€€€€€€€€€€€€€Á¥Ñ õ•Ø¹Á¥Ñ °(€€€€€€€€€€€€€€€€¤(€€€€€€€€€€€€€€€½Á•¹}¹½Ñ•Ím•Ø¹¥¹‘•át€ô¹½Ñ”(€€€€€€€€€€€•±Í”è€€Œ9½Ñ•¹‘Ù•¹Ð(€€€€€€€€€€€€€€€¹½Ñ”€ô½Á•¹}¹½Ñ•Ì¹Á½À¡•Ø¹ÍÑ…ÉÑ}•Ù•¹Ñ}¥¹‘•à¤(€€€€€€€€€€€€€€€¹½Ñ”¹½™™Í•Ð€ô•Ø¹•¹‘}Ñ¥µ”(€€€€€€€€€€€€€€€¹½Ñ•Ì¹…ÁÁ•¹¡¹½Ñ”¤((€€€€€€€€Œ5…Ñ Ñ¡”±•…ä‘•½‘•ÈÌ¹½Ñ”µ±•…¹ÕÀÁ…ÍÌÍ¼Ñ¡”5%$‰åÑ•Ì(€€€€€€€€Œ‘½¸Ð‘É¥™Ð™É½´•…É±¥•ÈÉ•™•É•¹”½ÕÑÁÕÑÌ¸(€€€€€€€¹½Ñ•Ì€ôÙ…±¥‘…Ñ•}¹½Ñ•Ì¡¹½Ñ•Ì°™¥àõQÉÕ”¤(€€€€€€€¹½Ñ•Ì€ôÑÉ¥µ}½Ù•É±…ÁÁ¥¹}¹½Ñ•Ì¡¹½Ñ•Ì°Í½ÉÐõQÉÕ”¤(€€€€€€€µ¥‘¤€ô¹½Ñ•Í}Ñ½}µ¥‘¤¡¹½Ñ•Ì°ÁÉ½É…µ}¹…µ•ÌõÁÉ½É…µ}¹…µ•Ì¤(€€€€€€€‰Õ˜€ô¥¼¹	åÑ•Í%< ¤(€€€€€€€µ¥‘¤¹Í…Ù”¡™¥±”õ‰Õ˜¤(€€€€€€€É•ÑÕÉ¸‰Õ˜¹•ÑÙ…±Õ” ¤((€€€‘•˜}ÁÉ½É…µ}™½É}¥¹ÍÑÉÕµ•¹Ð¡Í•±˜°¥¹ÍÑÉÕµ•¹ÐèÍÑÈ¤€´ø¥¹Ðè(€€€€€€€€ˆˆ‰%¹Ù•ÉÍ”½˜}¥¹ÍÑÉÕµ•¹Ñ}™½É}ÁÉ½É…µ€™½È¹½¸µ‘ÉÕ´¥¹ÍÑÉÕµ•¹ÑÌ¸ˆˆˆ(€€€€€€€¥˜¹½Ð¡…Í…ÑÑÈ¡Í•±˜°€‰}¥¹ÍÑ}Ñ½}ÁÉ½É…´ˆ¤è(€€€€€€€€€€€É½ÕÁ}µ…À€ôÍ•±˜¹}Ñ½­•¹¥é•È¹É½ÕÁ}ÁÉ½É…µ}µ…À(€€€€€€€€€€€Í•±˜¹}¥¹ÍÑ}Ñ½}ÁÉ½É…´€ôì(€€€€€€€€€€€€€€€¹…µ”èÉ½ÕÁ}µ…Ám¥‘ulÁt(€€€€€€€€€€€€€€€™½È¹…µ”°¥¥¸5PÍ}U11}A1UM}I=UA}95L¹¥Ñ•µÌ ¤(€€€€€€€€€€€€€€€¥˜¥¥¸É½ÕÁ}µ…À…¹É½ÕÁ}µ…Ám¥‘t(€€€€€€€€€€€ô(€€€€€€€¥˜¥¹ÍÑÉÕµ•¹Ð¥¸Í•±˜¹}¥¹ÍÑ}Ñ½}ÁÉ½É…´è(€€€€€€€€€€€É•ÑÕÉ¸Í•±˜¹}¥¹ÍÑ}Ñ½}ÁÉ½É…µm¥¹ÍÑÉÕµ•¹Ñt(€€€€€€€€Œ™…±±‰…¬™½ÈÕ¹­¹½Ý¸¹…µ•Ì±¥­”€‰ÁÉ½É…µ|ÐÈˆ(€€€€€€€¥˜¥¹ÍÑÉÕµ•¹Ð¹ÍÑ…ÉÑÍÝ¥Ñ  ‰ÁÉ½É…µ|ˆ¤è(€€€€€€€€€€€É•ÑÕÉ¸¥¹Ð¡¥¹ÍÑÉÕµ•¹Ð¹É•µ½Ù•ÁÉ•™¥à ‰ÁÉ½É…µ|ˆ¤¤(€€€€€€€É…¥Í”Y…±Õ•ÉÉ½È¡˜‰U¹­¹½Ý¸¥¹ÍÑÉÕµ•¹Ð¹…µ”èí¥¹ÍÑÉÕµ•¹Ð…Éôˆ¤((€€€€Œ€´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´(€€€‘•˜}±½…‘}Ý…Ø (€€€€€€€Í•±˜°…Õ‘¥¼èÍÑÈðA…Ñ ðÑ½É ¹Q•¹Í½È°Í…µÁ±•}É…Ñ”è¥¹Ðð9½¹”(€€€€¤€´øÑ½É ¹Q•¹Í½Èè(€€€€€€€€ˆˆ‰I•ÑÕÉ¸µ½¹¼™±½…ÐÌÈÝ…Ù•™½É´…Ð€ÄØ­!è°Í¡…Á”lÄ°Qt¸ˆˆˆ(€€€€€€€¥˜¥Í¥¹ÍÑ…¹”¡…Õ‘¥¼°€¡ÍÑÈ°A…Ñ ¤¤è(€€€€€€€€€€€Ý…Ø€ô±½…‘}…Õ‘¥¼¡…Õ‘¥¼°Ñ…É•Ñ}ÍÈõ}M5A1}IQ¤(€€€€€€€•±Í”è(€€€€€€€€€€€Ý…Ø€ô…Õ‘¥¼¹™±½…Ð ¤(€€€€€€€€€€€¥˜Ý…Ø¹‘¥´ ¤€ôô€Äè(€€€€€€€€€€€€€€€Ý…Ø€ôÝ…Ø¹Õ¹ÍÅÕ••é” À¤(€€€€€€€€€€€¥˜Ý…Ø¹‘¥´ ¤€ôô€Ìè(€€€€€€€€€€€€€€€Ý…Ø€ôÝ…Ø¹ÍÅÕ••é” À¤(€€€€€€€€€€€¥˜Ý…Ø¹Í¡…Á•lÁt€ø€Äè(€€€€€€€€€€€€€€€Ý…Ø€ôÝ…Ø¹µ•…¸ À°­••Á‘¥´õQÉÕ”¤(€€€€€€€€€€€¥˜Í…µÁ±•}É…Ñ”¥Ì¹½Ð9½¹”…¹Í…µÁ±•}É…Ñ”€„ô}M5A1}IQè(€€€€€€€€€€€€€€€Ý…Ø€ôÉ•Í…µÁ±”¡Ý…Ø°Í…µÁ±•}É…Ñ”°}M5A1}IQ¤(€€€€€€€É•ÑÕÉ¸Ý…Ø¹Ñ¼¡Í•±˜¹}‘•Ù¥”¤((€€€‘•˜}‰Õ¥±‘}½¹‘¥Ñ¥½¹Ì (€€€€€€€Í•±˜°(€€€€€€€Ý…ØèÑ½É ¹Q•¹Í½È°(€€€€€€€¥¹ÍÑÉÕµ•¹Ñ}É½ÕÀèÍÑÈð9½¹”€ô9½¹”°(€€€€€€€‘…Ñ…Í•Ñ}¹…µ”èÍÑÈð9½¹”€ô9½¹”°(€€€€¤€´ø±¥ÍÑm½¹‘¥Ñ¥½¹¥¹ÑÑÉ¥‰ÕÑ•Ítè(€€€€€€€€ˆˆ‰	Õ¥±„Í¥¹±”µ•±•µ•¹Ð±¥ÍÐ½˜½¹‘¥Ñ¥½¹¥¹ÑÑÉ¥‰ÕÑ•Ì™½È½¹”€ÔµÍ•½¹¡Õ¹¬¸ˆˆˆ(€€€€€€€P€ôÝ…Ø¹Í¡…Á•l´Åt(€€€€€€€Ý…Ù|Í€ôÝ…Ø¹Õ¹ÍÅÕ••é” À¤€€ŒlÄ°€Ä°Qt(€€€€€€€±•¹Ñ €ôÑ½É ¹Ñ•¹Í½È¡mQt°‘•Ù¥”õÍ•±˜¹}‘•Ù¥”¤(€€€€€€€Ý…Ù}½¹€ô]…Ù½¹‘¥Ñ¥½¸ (€€€€€€€€€€€Ý…ØõÝ…Ù|Í°(€€€€€€€€€€€±•¹Ñ õ±•¹Ñ °(€€€€€€€€€€€Í…µÁ±•}É…Ñ”õm}M5A1}IQt°(€€€€€€€€€€€Á…Ñ õm9½¹•t°(€€€€€€€€€€€Í••­}Ñ¥µ”õlÀ¸Át°(€€€€€€€€¤(€€€€€€€É•ÑÕÉ¸l(€€€€€€€€€€€½¹‘¥Ñ¥½¹¥¹ÑÑÉ¥‰ÕÑ•Ì (€€€€€€€€€€€€€€€Ý…Øõì‰Í•±™}Ý…ØˆèÝ…Ù}½¹‘ô°(€€€€€€€€€€€€€€€Ñ•áÐõì(€€€€€€€€€€€€€€€€€€€€‰¥¹ÍÑÉÕµ•¹Ñ}É½ÕÀˆè¥¹ÍÑÉÕµ•¹Ñ}É½ÕÀ°(€€€€€€€€€€€€€€€€€€€€Œ±Ý…åÌÕ¹½¹‘¥Ñ¥½¹…°½¸‘…Ñ…Í•ÐèÑ¡”¹Õ±°½Á…±…ÍÌ¸(€€€€€€€€€€€€€€€€€€€€‰‘…Ñ…Í•Ñ}¹…µ”ˆè‘…Ñ…Í•Ñ}¹…µ”°(€€€€€€€€€€€€€€€ô°(€€€€€€€€€€€€¤(€€€€€€€t