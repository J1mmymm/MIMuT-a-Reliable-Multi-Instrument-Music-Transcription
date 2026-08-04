"""Distributed supervised training entry point."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from muscriptor.data.dataset import (
    BalancedWindowBatchSampler,
    ContinuousChunkDataset,
    TrainingBatch,
    collate_training_examples,
)
from muscriptor.data.manifest import (
    manifest_fingerprint,
    read_manifest,
)
from muscriptor.models.lm import LMModel
from muscriptor.models.training import (
    IGNORE_INDEX,
    _condition_attributes,
    prepare_decoder_training_batch,
)
from muscriptor.modules.conditioners import MelSpectrogramConditioner
from muscriptor.training.config import ExperimentConfig, StageConfig
from muscriptor.training.schedule import (
    make_scheduler_lambda,
    signatures_compatible,
)
from muscriptor.transcription_model import TranscriptionModel, _build_model


class LongContextTrainingModel(nn.Module):
    def __init__(self, model: LMModel):
        super().__init__()
        self.model = model

    def forward(
        self, batch: TrainingBatch, condition_dropout: float
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        prepared = prepare_decoder_training_batch(
            self.model, batch, condition_dropout=condition_dropout
        )
        logits = self.model(
            prepared.inputs,
            prepared.conditions,
            first_step=True,
        )
        return logits, prepared.labels, prepared.token_count


def _training_signature(config: ExperimentConfig) -> dict[str, Any]:
    signature = asdict(config)
    signature.pop("resume", None)
    signature.pop("output_dir", None)
    # Operational resume policy does not change model or optimizer semantics.
    signature.pop("allow_world_size_change_on_resume", None)
    return signature


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(device: torch.device, precision: str):
    if precision == "float32" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=getattr(torch, precision),
    )


def _unwrap_lm(wrapped: nn.Module, raw_wrapper: LongContextTrainingModel) -> LMModel:
    if isinstance(wrapped, DistributedDataParallel):
        return wrapped.module.model
    return raw_wrapper.model


def _teacher_logits(
    teacher: LMModel,
    batch: TrainingBatch,
) -> torch.Tensor:
    bsz, chunks, max_length = batch.target_ids.shape
    flat = bsz * chunks
    attributes = _condition_attributes(batch, sample_rate=16_000, condition_dropout=0.0)
    # Released teachers use their original small dataset taxonomy.  Distill
    # them in the same dataset-unconditional mode used at inference rather
    # than indexing that table with this run's expanded taxonomy.
    for attribute in attributes:
        attribute.text["dataset_name"] = None
    conditions = teacher.condition_provider(
        teacher.condition_provider.tokenize(attributes)
    )
    targets = batch.target_ids.reshape(flat, max_length)
    # Match the student's teacher-forcing context exactly (including any
    # tie-corrupted inputs) so the distillation targets align per position.
    sources = (
        batch.input_ids if batch.input_ids is not None else batch.target_ids
    ).reshape(flat, max_length)
    inputs = torch.zeros_like(targets)
    inputs[:, 0] = teacher.initial_token_id
    inputs[:, 1:] = sources[:, :-1]
    logits = teacher(inputs, conditions, first_step=True)
    return logits


def _distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    supervised = labels.ne(IGNORE_INDEX)
    selected = student_logits[supervised]
    teacher_selected = teacher_logits[supervised]
    if selected.shape[0] != teacher_selected.shape[0]:
        raise RuntimeError("teacher/student token alignment mismatch")
    vocab = min(selected.shape[-1], teacher_selected.shape[-1], 1393)
    student = F.log_softmax(selected[:, :vocab].float() / temperature, dim=-1)
    teacher = F.softmax(teacher_selected[:, :vocab].float() / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature * temperature


def _initialize_from_teacher(student: LMModel, teacher: LMModel) -> dict[str, int]:
    """Warm-start every shape-compatible decoder/conditioner parameter.

    A released Transformer teacher contributes embeddings, heads and
    conditioners.  An earlier Hybrid-Mamba checkpoint additionally has the
    same decoder backbone, chunk marker and type embeddings as the dual-Mamba
    student, so those parameters are copied exactly.
    """

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    exact_prefixes = (
        "emb.",
        "transformer.",
        "type_embedding.",
        "chunk_start",
        "out_norm.",
        "linear.",
        "condition_provider.conditioners.self_wav.output_proj.",
        "condition_provider.conditioners.instrument_group.embed.",
    )
    copied = 0
    copied_parameters = 0
    decoder_tensors = 0
    decoder_parameters = 0
    with torch.no_grad():
        for key, source in teacher_state.items():
            if not key.startswith(exact_prefixes):
                continue
            destination = student_state.get(key)
            if destination is None or destination.shape != source.shape:
                continue
            destination.copy_(source.to(destination))
            copied += 1
            copied_parameters += destination.numel()
            if key.startswith(("transformer.", "type_embedding.", "chunk_start")):
                decoder_tensors += 1
                decoder_parameters += destination.numel()

        dataset_key = "condition_provider.conditioners.dataset_name.embed.weight"
        if dataset_key in student_state and dataset_key in teacher_state:
            destination = student_state[dataset_key]
            source = teacher_state[dataset_key]
            rows = min(destination.shape[0], source.shape[0])
            destination[:rows].copy_(source[:rows].to(destination))
            copied += 1
            copied_parameters += rows * destination.shape[1]

    return {
        "tensors": copied,
        "parameters": copied_parameters,
        "decoder_tensors": decoder_tensors,
        "decoder_parameters": decoder_parameters,
    }


def _resolve_training_checkpoint(path: str | None) -> str | None:
    """Resolve a run directory through its atomic ``latest.json`` pointer."""

    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_dir():
        return path
    latest_path = candidate / "latest.json"
    if not latest_path.is_file():
        raise FileNotFoundError(
            f"checkpoint directory has no latest.json: {candidate}"
        )
    latest = json.loads(latest_path.read_text())
    weights = candidate / latest["weights"]
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint weights not found: {weights}")
    return str(weights)


def _optimizer_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _save_checkpoint(
    *,
    wrapped: nn.Module,
    raw_wrapper: LongContextTrainingModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentConfig,
    output_dir: Path,
    global_step: int,
    stage_index: int,
    stage_step: int,
    manifest_hash: str,
    taxonomy: dict[str, Any],
    rank: int,
) -> None:
    if config.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

        with FSDP.state_dict_type(
            wrapped,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            wrapper_state = wrapped.state_dict()
            optimizer_state = FSDP.optim_state_dict(wrapped, optimizer)
        state = {
            key.removeprefix("model."): value.detach().cpu().contiguous()
            for key, value in wrapper_state.items()
            if key.startswith("model.")
        }
    else:
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in _unwrap_lm(wrapped, raw_wrapper).state_dict().items()
        }
        optimizer_state = optimizer.state_dict()

    local_rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
    }
    if dist.is_initialized():
        rng_by_rank: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(rng_by_rank, local_rng)
    else:
        rng_by_rank = [local_rng]
    if rank != 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"checkpoint_{global_step:08d}"
    tokenizer_metadata = {
        "name": config.model.tokenizer_name,
        "max_shift_steps": config.model.tokenizer_max_shift_steps,
        "segment_duration": config.model.segment_duration,
    }
    save_file(
        state,
        output_dir / f"{stem}.safetensors",
        metadata={
            "model_config": json.dumps(config.model.to_dict(), sort_keys=True),
            "tokenizer": json.dumps(tokenizer_metadata, sort_keys=True),
            "taxonomy": json.dumps(taxonomy, sort_keys=True),
            "manifest_sha256": manifest_hash,
        },
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.model.to_dict(), indent=2) + "\n"
    )
    (output_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer_metadata, indent=2) + "\n"
    )
    (output_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy, indent=2, sort_keys=True) + "\n"
    )
    trainer_state = {
        "global_step": global_step,
        "stage_index": stage_index,
        "stage_step": stage_step,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "manifest_sha256": manifest_hash,
        "taxonomy": taxonomy,
        "world_size": len(rng_by_rank),
        "rng_by_rank": rng_by_rank,
        "training_signature": _training_signature(config),
    }
    torch.save(trainer_state, output_dir / f"{stem}.trainer.pt")
    latest_payload = (
        json.dumps(
            {
                "weights": f"{stem}.safetensors",
                "trainer": f"{stem}.trainer.pt",
                "global_step": global_step,
            },
            indent=2,
        )
        + "\n"
    )
    latest_tmp = output_dir / f".latest.{os.getpid()}.json.tmp"
    latest_tmp.write_text(latest_payload)
    os.replace(latest_tmp, output_dir / "latest.json")


def _load_resume_files(
    config: ExperimentConfig,
    model: LMModel,
    manifest_hash: str,
    taxonomy: dict[str, Any],
    device: torch.device,
    rank: int,
) -> dict[str, Any] | None:
    if not config.resume:
        return None
    resume = Path(config.resume)
    if resume.is_dir():
        latest = json.loads((resume / "latest.json").read_text())
        weights = resume / latest["weights"]
        trainer_path = resume / latest["trainer"]
    else:
        trainer_path = resume
        weights = Path(str(resume).replace(".trainer.pt", ".safetensors"))
    from safetensors.torch import load_file

    model.load_state_dict(load_file(weights, device=str(device)))
    state = torch.load(trainer_path, map_location=device, weights_only=False)
    if state["manifest_sha256"] != manifest_hash:
        raise RuntimeError("manifest changed since the resumed checkpoint")
    if state.get("taxonomy") != taxonomy:
        raise RuntimeError("dataset taxonomy changed since the resumed checkpoint")
    if state.get("training_signature") is not None and not signatures_compatible(
        state["training_signature"], _training_signature(config)
    ):
        raise RuntimeError(
            "experiment configuration changed since the resumed checkpoint"
        )
    if "rng_by_rank" in state:
        current_world_size = dist.get_world_size() if dist.is_initialized() else 1
        if state.get("world_size") != current_world_size:
            if not config.allow_world_size_change_on_resume:
                raise RuntimeError(
                    "deterministic resume requires the original world size "
                    f"({state.get('world_size')} != {current_world_size}); set "
                    "allow_world_size_change_on_resume=true for an intentional "
                    "elastic restart"
                )
            # The batch sampler is already indexed by stage_step, rank and a
            # fixed seed.  Re-seed the remaining stochastic transforms for the
            # new rank topology; this is reproducible but intentionally not
            # bit-identical to the old world-size trajectory.
            elastic_seed = config.seed + 1_009 * int(state["global_step"])
            _seed_everything(elastic_seed, rank)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "elastic_resume": {
                                "checkpoint_world_size": state.get("world_size"),
                                "current_world_size": current_world_size,
                                "global_step": int(state["global_step"]),
                                "seed": elastic_seed,
                            }
                        }
                    ),
                    flush=True,
                )
        else:
            if rank >= len(state["rng_by_rank"]):
                raise RuntimeError(
                    "checkpoint world size is smaller than the resumed rank"
                )
            rank_rng = state["rng_by_rank"][rank]
            random.setstate(rank_rng["python"])
            np.random.set_state(rank_rng["numpy"])
            torch.set_rng_state(rank_rng["torch"].cpu())
            if torch.cuda.is_available() and rank_rng["cuda"] is not None:
                torch.cuda.set_rng_state(rank_rng["cuda"].cpu(), device=device)
    else:
        # Backward-compatible resume for checkpoints created before per-rank
        # RNG capture was introduced.
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state


def _stage_loader(
    config: ExperimentConfig,
    stage: StageConfig,
    records,
    *,
    world_size: int,
    rank: int,
    stage_step: int,
    dataset_ids: dict[str, int],
) -> DataLoader:
    selected = [
        record
        for record in records
        if not stage.datasets or record.dataset in stage.datasets
    ]
    dataset = ContinuousChunkDataset(
        selected,
        split=stage.split,
        context_chunks=stage.context_chunks,
        segment_duration=config.model.segment_duration,
        dataset_ids=dataset_ids,
        tie_corruption=config.tie_corruption,
    )
    if not len(dataset):
        raise RuntimeError(
            f"stage {stage.name!r} has no "
            f"{stage.context_chunks * config.model.segment_duration:g}s "
            "continuous training windows"
        )
    effective_global_examples = max(
        1,
        round(
            config.global_batch_audio_seconds
            / (config.model.segment_duration * stage.context_chunks)
        ),
    )
    micro_global_examples = max(
        1,
        effective_global_examples // config.gradient_accumulation_steps,
    )
    per_device = max(1, micro_global_examples // world_size)
    sampler = BalancedWindowBatchSampler(
        dataset,
        batch_size=per_device,
        num_batches=(stage.steps - stage_step)
        * config.gradient_accumulation_steps,
        seed=config.seed + 10_007 * stage.context_chunks,
        rank=rank,
        start_batch=stage_step * config.gradient_accumulation_steps,
        dataset_probabilities=stage.dataset_weights or None,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        collate_fn=collate_training_examples,
    )


def _validation_loader(
    config: ExperimentConfig,
    stage: StageConfig,
    records,
    *,
    dataset_ids: dict[str, int],
) -> DataLoader | None:
    if not config.validation.enabled:
        return None
    selected = [
        record
        for record in records
        if not stage.datasets or record.dataset in stage.datasets
    ]
    dataset = ContinuousChunkDataset(
        selected,
        split=config.validation.split,
        context_chunks=stage.context_chunks,
        segment_duration=config.model.segment_duration,
        dataset_ids=dataset_ids,
        tie_corruption=0.0,
    )
    if not len(dataset):
        return None
    return DataLoader(
        dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_training_examples,
    )


@torch.no_grad()
def _run_validation(
    raw_wrapper: LongContextTrainingModel,
    teacher: LMModel | None,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    num_batches: int,
    temperature: float,
    eos_id: int,
) -> dict[str, float | int]:
    raw_wrapper.eval()
    ce_sum = 0.0
    kd_sum = 0.0
    supervised_tokens = 0
    correct_tokens = 0
    eos_tokens = 0
    correct_eos = 0
    batches = 0
    try:
        for batch_index, batch in enumerate(loader):
            if batch_index >= num_batches:
                break
            batch = batch.to(device)
            with _autocast(device, precision):
                logits, labels, token_count = raw_wrapper(batch, 0.0)
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                supervised = labels.ne(IGNORE_INDEX)
                predictions = logits.argmax(dim=-1)
                ce_sum += float(ce)
                supervised_tokens += int(token_count)
                correct_tokens += int((predictions[supervised] == labels[supervised]).sum())
                eos_mask = supervised & labels.eq(eos_id)
                eos_tokens += int(eos_mask.sum())
                correct_eos += int((predictions[eos_mask] == eos_id).sum())
                if teacher is not None:
                    teacher_logits = _teacher_logits(teacher, batch)
                    kd = _distillation_loss(
                        logits,
                        labels,
                        teacher_logits,
                        temperature=temperature,
                    )
                    kd_sum += float(kd) * token_count
            batches += 1
    finally:
        raw_wrapper.train()

    denominator = max(1, supervised_tokens)
    return {
        "val_ce": ce_sum / denominator,
        "val_token_accuracy": correct_tokens / denominator,
        "val_eos_accuracy": correct_eos / max(1, eos_tokens),
        "val_kd": kd_sum / denominator if teacher is not None else 0.0,
        "val_tokens": supervised_tokens,
        "val_eos_tokens": eos_tokens,
        "val_batches": batches,
    }


def train(config: ExperimentConfig) -> None:
    rank, local_rank, world_size = _distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid-Mamba training requires a CUDA machine")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    _seed_everything(config.seed, rank)

    records = read_manifest(config.manifest)
    manifest_hash = manifest_fingerprint(config.manifest)
    dataset_ids = {
        name: index
        for index, name in enumerate(sorted({record.dataset for record in records}))
    }
    if len(dataset_ids) > config.model.num_dataset_condition_classes:
        raise ValueError(
            f"manifest has {len(dataset_ids)} datasets but the model allows "
            f"{config.model.num_dataset_condition_classes}"
        )
    taxonomy: dict[str, Any] = {
        "datasets": dataset_ids,
        "instrument_vocabulary": config.model.tokenizer_name,
        "instrument_condition_classes": (config.model.num_instrument_condition_classes),
    }
    raw_model = _build_model(device, config.model)
    for conditioner in raw_model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
            conditioner.set_gradient_checkpointing(config.gradient_checkpointing)
    if hasattr(raw_model.transformer, "set_gradient_checkpointing"):
        raw_model.transformer.set_gradient_checkpointing(config.gradient_checkpointing)

    teacher = None
    initializer = None
    teacher_path = _resolve_training_checkpoint(
        config.distillation.teacher_checkpoint
    )
    initialization_path = _resolve_training_checkpoint(
        config.initialization_checkpoint
        or teacher_path
    )
    if initialization_path and not config.resume:
        initializer = TranscriptionModel.load_model(
            initialization_path,
            device=device,
            dtype=config.precision,
        )._model
        initializer.eval()
        warm_start = _initialize_from_teacher(raw_model, initializer)
        if rank == 0:
            print(json.dumps({"warm_start": warm_start}), flush=True)

    if teacher_path:
        if (
            initializer is not None
            and initialization_path == teacher_path
        ):
            teacher = initializer
        else:
            teacher = TranscriptionModel.load_model(
                teacher_path,
                device=device,
                dtype=config.precision,
            )._model
        assert teacher is not None
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    elif initializer is not None:
        del initializer

    raw_wrapper = LongContextTrainingModel(raw_model).to(device)
    resume_state = _load_resume_files(
        config, raw_model, manifest_hash, taxonomy, device, rank
    )

    if config.distributed == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
        )

        wrapped: nn.Module = FSDP(
            raw_wrapper,
            device_id=device,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=getattr(torch, config.precision),
                reduce_dtype=torch.float32,
                buffer_dtype=getattr(torch, config.precision),
            ),
        )
    elif config.distributed == "ddp" and world_size > 1:
        wrapped = DistributedDataParallel(
            raw_wrapper,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        wrapped = raw_wrapper

    optimizer = torch.optim.AdamW(
        _optimizer_groups(wrapped, config.optimizer.weight_decay),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        make_scheduler_lambda(
            config.stages,
            warmup=config.optimizer.warmup_steps,
        ),
    )
    if resume_state is not None:
        if config.distributed == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            optimizer_state = FSDP.optim_state_dict_to_load(
                wrapped, optimizer, resume_state["optimizer"]
            )
            optimizer.load_state_dict(optimizer_state)
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        global_step = int(resume_state["global_step"])
        resume_stage = int(resume_state["stage_index"])
        resume_stage_step = int(resume_state["stage_step"])
    else:
        global_step = resume_stage = resume_stage_step = 0

    output_dir = Path(config.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "experiment.json").write_text(
            json.dumps(asdict(config), indent=2) + "\n"
        )
    log_path = output_dir / f"metrics.rank{rank}.jsonl"
    validation_log_path = output_dir / "validation.jsonl"
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(output_dir / "tensorboard")
        except ImportError:
            writer = None
    last_time = time.perf_counter()

    for stage_index, stage in enumerate(config.stages):
        if stage_index < resume_stage:
            continue
        stage_start = resume_stage_step if stage_index == resume_stage else 0
        if stage_start >= stage.steps:
            resume_stage_step = 0
            continue
        loader = _stage_loader(
            config,
            stage,
            records,
            world_size=world_size,
            rank=rank,
            stage_step=stage_start,
            dataset_ids=dataset_ids,
        )
        validation_loader = (
            _validation_loader(
                config,
                stage,
                records,
                dataset_ids=dataset_ids,
            )
            if rank == 0
            else None
        )
        wrapped.train()
        accumulation = config.gradient_accumulation_steps
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_ce = 0.0
        accumulated_kd = 0.0
        accumulated_tokens = 0
        optimiser_steps = stage_start
        kd_weight = (
            stage.distillation_weight
            if stage.distillation_weight is not None
            else config.distillation.weight
        )

        for micro_index, batch in enumerate(loader):
            batch = batch.to(device)
            final_micro_step = (micro_index + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if not final_micro_step and hasattr(wrapped, "no_sync"):
                sync_context = wrapped.no_sync()

            with sync_context:
                with _autocast(device, config.precision):
                    logits, labels, token_count = wrapped(
                        batch, config.condition_dropout
                    )
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                    loss = ce
                    kd = torch.zeros((), device=device)
                    if teacher is not None:
                        with torch.no_grad():
                            teacher_logits = _teacher_logits(teacher, batch)
                        kd = _distillation_loss(
                            logits,
                            labels,
                            teacher_logits,
                            temperature=config.distillation.temperature,
                        )
                        loss = loss + kd_weight * kd
                (loss / accumulation).backward()

            accumulated_loss += float(loss.detach())
            accumulated_ce += float(ce.detach())
            accumulated_kd += float(kd.detach())
            accumulated_tokens += token_count
            if not final_micro_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                wrapped.parameters(), config.optimizer.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            optimiser_steps += 1

            step_loss = accumulated_loss / accumulation
            step_ce = accumulated_ce / accumulation
            step_kd = accumulated_kd / accumulation
            step_tokens = accumulated_tokens
            accumulated_loss = accumulated_ce = accumulated_kd = 0.0
            accumulated_tokens = 0

            if global_step % config.log_every == 0:
                now = time.perf_counter()
                metrics: dict[str, Any] = {
                    "step": global_step,
                    "stage": stage.name,
                    "stage_step": optimiser_steps,
                    "context_chunks": stage.context_chunks,
                    "loss": step_loss,
                    "ce": step_ce,
                    "distill": step_kd,
                    "distill_weight": kd_weight if teacher is not None else 0.0,
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "tokens": step_tokens,
                    "gradient_accumulation_steps": accumulation,
                    "steps_per_second": config.log_every / (now - last_time),
                }
                last_time = now
                with log_path.open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
                if rank == 0:
                    print(json.dumps(metrics), flush=True)
                    if writer is not None:
                        for key in ("loss", "ce", "distill", "grad_norm", "lr"):
                            writer.add_scalar(key, metrics[key], global_step)

            if (
                config.validation.enabled
                and global_step % config.validation.every_steps == 0
            ):
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0 and validation_loader is not None:
                    validation_metrics: dict[str, Any] = {
                        "step": global_step,
                        "stage": stage.name,
                        "context_chunks": stage.context_chunks,
                    }
                    validation_metrics.update(
                        _run_validation(
                            raw_wrapper,
                            teacher,
                            validation_loader,
                            device=device,
                            precision=config.precision,
                            num_batches=config.validation.num_batches,
                            temperature=config.distillation.temperature,
                            eos_id=validation_loader.dataset.tokenizer.eos_id,
                        )
                    )
                    with validation_log_path.open("a") as stream:
                        stream.write(json.dumps(validation_metrics) + "\n")
                    print(json.dumps(validation_metrics), flush=True)
                    if writer is not None:
                        for key, value in validation_metrics.items():
                            if key.startswith("val_") and isinstance(value, float):
                                writer.add_scalar(key, value, global_step)
                if dist.is_initialized():
                    dist.barrier()
                wrapped.train()
                if teacher is not None:
                    teacher.eval()

            if global_step % config.save_every == 0:
                _save_checkpoint(
                    wrapped=wrapped,
                    raw_wrapper=raw_wrapper,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    output_dir=output_dir,
                    global_step=global_step,
                    stage_index=stage_index,
                    stage_step=optimiser_steps,
                    manifest_hash=manifest_hash,
                    taxonomy=taxonomy,
                    rank=rank,
                )
        resume_stage_step = 0

    _save_checkpoint(
        wrapped=wrapped,
        raw_wrapper=raw_wrapper,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=output_dir,
        global_step=global_step,
        stage_index=len(config.stages),
        stage_step=0,
        manifest_hash=manifest_hash,
        taxonomy=taxonomy,
        rank=rank,
    )
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    from muscriptor.training.config import load_experiment_config

    train(load_experiment_config(args.config))


if __name__ == "__main__":
    main()
