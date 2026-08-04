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
     