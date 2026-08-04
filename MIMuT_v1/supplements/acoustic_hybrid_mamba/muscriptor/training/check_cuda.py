"""CUDA/Mamba2 acceptance check for the acoustic encoder."""

from __future__ import annotations

import argparse
import importlib.metadata
import json

import torch

from muscriptor.models.config import (
    ModelConfig,
    estimate_acoustic_encoder_parameters,
)
from muscriptor.modules.acoustic import AcousticMambaEncoder
from muscriptor.modules.streaming import (
    increment_steps,
    init_states,
    state_size_bytes,
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_check() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 12:
        cuda_parts = tuple(
            int(part) for part in (torch.version.cuda or "0.0").split(".")[:2]
        )
        if cuda_parts < (12, 8):
            raise RuntimeError(
                "Blackwell (SM120) requires a PyTorch build with CUDA >= 12.8"
            )

    config = ModelConfig(
        dim=256,
        num_heads=4,
        num_layers=1,
        card=128,
        backbone="transformer",
        acoustic_encoder="mamba2",
        acoustic_num_layers=2,
        acoustic_identity_init=False,
        mamba_d_state=32,
        mamba_headdim=64,
    )
    model = AcousticMambaEncoder(config, device=device, dtype=torch.bfloat16)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    estimated_parameters = estimate_acoustic_encoder_parameters(config)
    if actual_parameters != estimated_parameters:
        raise RuntimeError(
            "official Mamba2 parameter layout changed: "
            f"estimated={estimated_parameters}, actual={actual_parameters}"
        )

    # Kernel compilation plus a genuine backward pass.
    model.train()
    training_input = torch.randn(
        2,
        96,
        config.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = model(training_input)
    output.float().square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(training_input.grad).all():
        raise RuntimeError("non-finite gradient from Acoustic-Mamba")

    # Exercise the retained-state path with multi-token chunk prefixes.  With
    # the causal encoder must match the parallel execution.
    model.eval()
    stream_input = torch.randn(1, 48, config.dim, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        parallel = model(stream_input)
        state = init_states(model, batch_size=1, sequence_length=48)
        pieces = []
        start = 0
        for size in (13, 11, 24):
            pieces.append(
                model(
                    stream_input[:, start : start + size],
                    model_state=state,
                )
            )
            increment_steps(model, state, increment=size)
            start += size
        streaming = torch.cat(pieces, dim=1)
    max_error = float((parallel - streaming).abs().max())
    if not torch.allclose(parallel, streaming, atol=3e-2, rtol=3e-2):
        raise RuntimeError(f"parallel/streaming mismatch; maximum error is {max_error}")

    first_state_bytes = state_size_bytes(state)
    with torch.no_grad():
        model(stream_input[:, :16], model_state=state)
        increment_steps(model, state, increment=16)
    second_state_bytes = state_size_bytes(state)
    if second_state_bytes != first_state_bytes:
        raise RuntimeError("acoustic Mamba state grew with stream length")

    return {
        "status": "ok",
        "device": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gib": round(properties.total_memory / (1024**3), 2),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mamba_ssm": _version("mamba-ssm"),
        "causal_conv1d": _version("causal-conv1d"),
        "acoustic_encoder_parameters": actual_parameters,
        "streaming_max_abs_error": max_error,
        "state_bytes_after_48_tokens": state_size_bytes(state),
        "state_size_constant": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and validate Acoustic-Mamba2 CUDA kernels"
    )
    parser.add_argument(
        "--output",
        help="optional path for the JSON acceptance report",
    )
    args = parser.parse_args()
    report = run_check()
    rendered = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
