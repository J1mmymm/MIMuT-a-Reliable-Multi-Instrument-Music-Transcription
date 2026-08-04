"""One-GPU acceptance test for the production Clean Acoustic Cache model."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F

from muscriptor.data.dataset import (
    NUM_INSTRUMENT_GROUPS,
    NUM_MIDI_PITCHES,
    REENTRY_NONE_CLASS,
    TrainingBatch,
)
from muscriptor.models.config import estimate_total_parameters
from muscriptor.modules.conditioners import MelSpectrogramConditioner
from muscriptor.modules.streaming import (
    clone_model_state,
    increment_steps,
    init_states,
    iter_state_tensors,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.training.config import load_experiment_config
from muscriptor.training.train import LongContextTrainingModel
from muscriptor.transcription_model import _build_model


def snapshot(state):
    return {path: tensor.detach().clone() for path, tensor in iter_state_tensors(state)}


def assert_unchanged(state, before):
    after = dict(iter_state_tensors(state))
    if after.keys() != before.keys():
        raise AssertionError("state tensor paths changed")
    for path, tensor in after.items():
        torch.testing.assert_close(tensor, before[path], rtol=0, atol=0)


def synthetic_batch(tokenizer: MT3Tokenizer, device: torch.device) -> TrainingBatch:
    tie = next(
        index
        for index, event in enumerate(tokenizer._vocab)
        if event.type == "tie"
    )
    target = torch.tensor([tie, tokenizer.eos_id], device=device)
    targets = target.view(1, 1, -1).repeat(1, 2, 1)
    active = torch.zeros(
        1,
        2,
        NUM_INSTRUMENT_GROUPS,
        NUM_MIDI_PITCHES,
        dtype=torch.bool,
        device=device,
    )
    reentry = torch.full(
        (1, 2, NUM_INSTRUMENT_GROUPS),
        REENTRY_NONE_CLASS,
        dtype=torch.long,
        device=device,
    )
    valid = torch.ones_like(reentry, dtype=torch.bool)
    return TrainingBatch(
        waveform=torch.zeros(1, 2, 1, 80_000, device=device),
        target_ids=targets,
        target_lengths=torch.full((1, 2), 2, dtype=torch.long, device=device),
        instrument_groups=[None],
        dataset_ids=torch.zeros(1, dtype=torch.long, device=device),
        track_ids=["cuda-smoke"],
        start_times=torch.zeros(1, device=device),
        has_long_gap=torch.zeros(1, dtype=torch.bool, device=device),
        target_loss_mask=torch.ones_like(targets, dtype=torch.bool),
        active_note_targets=active,
        reentry_targets=reentry,
        reentry_valid=valid,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--isolation-tokens", type=int, default=2000)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer = MT3Tokenizer(
        instrument_vocabulary=config.model.tokenizer_name,
        max_shift_steps=config.model.tokenizer_max_shift_steps,
    )
    model = _build_model(device, config.model)
    for conditioner in model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    estimated_parameters = estimate_total_parameters(config.model)
    if actual_parameters != estimated_parameters:
        raise AssertionError(
            f"parameter estimate mismatch: {actual_parameters} != {estimated_parameters}"
        )

    wrapper = LongContextTrainingModel(
        model,
        config.boundary_state_supervision,
        tokenizer=tokenizer,
        clean_cache_config=config.clean_cache_training,
    ).to(device).train()
    batch = synthetic_batch(tokenizer, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, labels, _, active, reentry, _ = wrapper(
            batch, 0.0, 1.0, 1.0, global_step=20_000
        )
        loss = F.cross_entropy(logits.float(), labels)
        loss = loss + 0.1 * active.float().square().mean()
        loss = loss + 0.1 * reentry.float().square().mean()
    loss.backward()
    if model.boundary_projection.weight.grad is None:
        raise AssertionError("boundary head did not receive a gradient")
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise AssertionError("non-finite gradient in CUDA smoke step")

    wrapper.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefix = torch.randn(1, 32, model.dim, device=device)
        events = torch.randn(1, args.isolation_tokens, model.dim, device=device)
        clean = init_states(
            model,
            batch_size=1,
            sequence_length=args.isolation_tokens + 128,
        )
        model.encode_embeddings(prefix, model_state=clean)
        increment_steps(model.transformer, clean, prefix.shape[1])
        before = snapshot(clean)
        decode = clone_model_state(clean)
        model.encode_embeddings(events, model_state=decode)
        increment_steps(model.transformer, decode, events.shape[1])
        assert_unchanged(clean, before)

        clean_lengths = [
            int(value["key"].shape[2])
            for value in clean.values()
            if isinstance(value, dict) and "key" in value
        ]
        if not clean_lengths or set(clean_lengths) != {prefix.shape[1]}:
            raise AssertionError(f"unexpected clean Local KV lengths: {clean_lengths}")

    result = {
        "status": "ok",
        "parameters": actual_parameters,
        "training_loss": float(loss.detach()),
        "isolation_tokens": args.isolation_tokens,
        "clean_local_kv_lengths": clean_lengths,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
