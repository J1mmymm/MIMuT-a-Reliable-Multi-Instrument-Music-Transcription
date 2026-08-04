"""Command line interface for manifest construction and validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from muscriptor.data.manifest import (
    build_manifest,
    manifest_fingerprint,
    read_manifest,
)
from muscriptor.data.augmentation import (
    augmentation_catalog_fingerprint,
    build_augmentation_catalog,
    read_augmentation_catalog,
    validate_augmentation_catalog,
)
from muscriptor.data.schema import load_standardized_track


app = typer.Typer(add_completion=False, help="MuScriptor dataset tools")


@app.command("build")
def build(
    data_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    strict: Annotated[
        bool,
        typer.Option(help="Fail when any notes file cannot be matched to audio"),
    ] = False,
) -> None:
    records, issues = build_manifest(data_root, output, strict=strict)
    typer.echo(
        json.dumps(
            {
                "records": len(records),
                "issues": len(issues),
                "manifest": str(output.resolve()),
                "sha256": manifest_fingerprint(output),
            },
            indent=2,
        )
    )


@app.command("check")
def check(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    check_files: Annotated[bool, typer.Option("--check-files/--no-check-files")] = True,
) -> None:
    records = read_manifest(manifest)
    splits = Counter(record.split for record in records)
    datasets = Counter(record.dataset for record in records)
    hours = sum(record.duration for record in records) / 3600
    errors = []
    if check_files:
        for record in records:
            audio = Path(record.audio_path)
            notes = Path(record.notes_path)
            if not audio.is_file():
                errors.append(f"{record.track_id}: missing audio {audio}")
            if not notes.is_file():
                errors.append(f"{record.track_id}: missing notes {notes}")
            elif len(errors) < 100:
                try:
                    load_standardized_track(notes)
                except Exception as exc:
                    errors.append(f"{record.track_id}: {exc}")
    payload = {
        "records": len(records),
        "hours": round(hours, 3),
        "multi_instrument_fraction": (
            sum(record.is_multi_instrument for record in records) / len(records)
            if records
            else 0.0
        ),
        "splits": dict(sorted(splits.items())),
        "datasets": dict(sorted(datasets.items())),
        "file_or_schema_errors": errors,
        "sha256": manifest_fingerprint(manifest),
    }
    typer.echo(json.dumps(payload, indent=2))
    if errors:
        raise typer.Exit(code=1)


@app.command("augmentation-build")
def augmentation_build(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    slakh_root: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    cache_root: Annotated[Path | None, typer.Option(file_okay=False)] = None,
) -> None:
    """Build the strict train-only isolated-source augmentation catalog."""

    records = read_manifest(manifest)
    entries = build_augmentation_catalog(
        records,
        output,
        slakh_root=slakh_root,
        cache_root=cache_root,
    )
    typer.echo(
        json.dumps(
            {
                "entries": len(entries),
                "datasets": sorted({entry.dataset for entry in entries}),
                "catalog": str(output.resolve()),
                "sha256": augmentation_catalog_fingerprint(output),
            },
            indent=2,
        )
    )


@app.command("augmentation-check")
def augmentation_check(
    catalog: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    required_duration: float = 40.0,
    check_files: Annotated[bool, typer.Option("--check-files/--no-check-files")] = True,
) -> None:
    entries = read_augmentation_catalog(catalog)
    result = validate_augmentation_catalog(
        entries,
        read_manifest(manifest),
        required_duration=required_duration,
        check_files=check_files,
    )
    result["sha256"] = augmentation_catalog_fingerprint(catalog)
    typer.echo(json.dumps(result, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
