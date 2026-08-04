"""Materialize a new-run v2 config after local teacher/catalog verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-revision")
    parser.add_argument("--augmentation-catalog", type=Path)
    args = parser.parse_args()

    base = args.base.resolve()
    output = args.output.resolve()
    if not base.is_file():
        parser.error(f"base config does not exist: {base}")
    if output.exists():
        parser.error(f"refusing to overwrite existing config: {output}")
    if Path(args.output_dir).exists():
        parser.error(
            "refusing to target an existing run directory; choose a new output-dir"
        )

    payload: dict = {
        "extends": Path(os.path.relpath(base, output.parent)).as_posix(),
        "name": args.name,
        "output_dir": args.output_dir,
    }
    audit = {"base": str(base), "output": str(output)}
    if args.teacher is not None:
        if not args.teacher_revision:
            parser.error("--teacher-revision is required with --teacher")
        teacher = args.teacher.resolve()
        if not teacher.is_file() or teacher.suffix.lower() != ".safetensors":
            parser.error("teacher must be a local .safetensors file")
        config_json = teacher.with_name("config.json")
        if not config_json.is_file():
            parser.error("teacher config.json must sit next to the safetensors file")
        digest = sha256_file(teacher)
        payload["distillation"] = {
            "teacher_checkpoint": str(teacher),
            "teacher_sha256": digest,
            "teacher_revision": args.teacher_revision,
            "teacher_microbatch_size": 4,
            "weight": 0.25,
            "temperature": 2.0,
        }
        audit["teacher_sha256"] = digest
        audit["teacher_revision"] = args.teacher_revision
    if args.augmentation_catalog is not None:
        catalog = args.augmentation_catalog.resolve()
        if not catalog.is_file():
            parser.error(f"augmentation catalog does not exist: {catalog}")
        payload["augmentation"] = {
            "enabled": True,
            "catalog": str(catalog),
        }
        audit["augmentation_catalog_sha256"] = hashlib.sha256(
            catalog.read_bytes()
        ).hexdigest()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
