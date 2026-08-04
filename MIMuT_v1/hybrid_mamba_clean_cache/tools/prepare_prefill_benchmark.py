"""Create same-weight step/auto inference snapshots for the prefill gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_root = args.output_root.resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".safetensors":
        parser.error("--checkpoint must be a local safetensors file")
    source_config = checkpoint.with_name("config.json")
    if not source_config.is_file():
        parser.error("config.json must sit next to the checkpoint")
    if output_root.exists() and any(output_root.iterdir()):
        parser.error("refusing to reuse a non-empty output root")

    config = json.loads(source_config.read_text())
    if config.get("backbone") not in {"hybrid_mamba", "pure_mamba"}:
        parser.error("prefill comparison requires a Mamba backbone")

    created = {}
    for mode in ("step", "auto"):
        directory = output_root / mode
        directory.mkdir(parents=True, exist_ok=False)
        weights = directory / checkpoint.name
        try:
            os.link(checkpoint, weights)
        except OSError as exc:
            parser.error(
                "same-filesystem hardlink failed; choose an output root on the "
                f"checkpoint filesystem ({exc})"
            )
        local_config = dict(config)
        local_config["prefill_mode"] = mode
        (directory / "config.json").write_text(
            json.dumps(local_config, indent=2, sort_keys=True) + "\n"
        )
        created[mode] = str(weights)
    print(json.dumps(created, indent=2))


if __name__ == "__main__":
    main()
