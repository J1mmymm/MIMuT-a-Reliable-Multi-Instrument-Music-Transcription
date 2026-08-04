"""Evaluation and efficiency benchmark CLI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import torch
import typer

from muscriptor.data.manifest import read_manifest
from muscriptor.evaluation.io import load_notes
from muscriptor.evaluation.metrics import evaluate_notes, mir_eval_note_metrics
from muscriptor.transcription_model import TranscriptionModel
from muscriptor.tokenizer.mt3 import MT3_FULL_PLUS_GROUP_NAMES
from muscriptor.utils.audio import load_audio


app = typer.Typer(add_completion=False, help="MuScriptor evaluation tools")


def _oracle_names(group_ids: list[int]) -> list[str]:
    inverse = {group_id: name for name, group_id in MT3_FULL_PLUS_GROUP_NAMES.items()}
    return [inverse[group_id] for group_id in group_ids if group_id in inverse]


def _prediction_path(directory: Path, dataset: str, track_id: str) -> Path | None:
    for suffix in (".notes.json", ".mid", ".midi"):
        for path in (
            directory / dataset / f"{track_id}{suffix}",
            directory / f"{track_id}{suffix}",
        ):
            if path.exists():
                return path
    return None


def _macro(rows: list[dict]) -> dict:
    """Recursively macro-average numeric metric leaves."""
    if not rows:
        return {}
    result = {}
    keys = set.intersection(*(set(row) for row in rows))
    for key in sorted(keys):
        values = [row[key] for row in rows]
        if all(isinstance(value, dict) for value in values):
            result[key] = _macro(values)
        elif all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            result[key] = sum(values) / len(values)
    return result


@app.command("compare")
def compare(
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    predictions: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: str = "test",
    with_mir_eval: Annotated[
        bool,
        typer.Option(
            "--mir-eval/--no-mir-eval",
            help=(
                "Additionally report the official mir_eval note metrics "
                "(onset F1 and onset+offset F1) so results are directly "
                "comparable to the literature. Requires mir_eval."
            ),
        ),
    ] = False,
) -> None:
    if with_mir_eval and mir_eval_note_metrics([], []) is None:
        raise typer.BadParameter(
            "--mir-eval requires the mir_eval package (pip install mir_eval)"
        )
    rows = []
    for record in read_manifest(manifest):
        if record.split != split:
            continue
        prediction_path = _prediction_path(predictions, record.dataset, record.track_id)
        if prediction_path is None:
            continue
        reference_notes = load_notes(record.notes_path)
        prediction_notes = load_notes(prediction_path)
        result = evaluate_notes(reference_notes, prediction_notes)
        row = {
            "track_id": record.track_id,
            "dataset": record.dataset,
            **result.to_dict(),
        }
        if with_mir_eval:
            row["mir_eval"] = mir_eval_note_metrics(
                reference_notes, prediction_notes
            )
        rows.append(row)
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_dataset[dataset] = _macro(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"track_id", "dataset"}
                }
                for row in rows
                if row["dataset"] == dataset
            ]
        )
    payload = {
        "summary": {
            "track_count": len(rows),
            "macro": _macro(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"track_id", "dataset"}
                    }
                    for row in rows
                ]
            ),
            "by_dataset": by_dataset,
        },
        "tracks": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    typer.echo(f"evaluated {len(rows)} tracks")


@app.command("predict")
def predict(
    model_path: Annotated[str, typer.Option("--model", "-m")],
    manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    split: str = "test",
    device: str = "cuda",
    long_context: str = "carry",
    oracle_instruments: Annotated[
        bool,
        typer.Option("--oracle-instruments/--no-oracle-instruments"),
    ] = True,
) -> None:
    """Transcribe a manifest into dataset-scoped MIDI prediction files."""
    model = TranscriptionModel.load_model(model_path, device=device)
    selected = [record for record in read_manifest(manifest) if record.split == split]
    for index, record in enumerate(selected, 1):
        instruments = (
            _oracle_names(record.instrument_groups) if oracle_instruments else None
        )
        midi = model.transcribe_to_midi(
            record.audio_path,
            instruments=instruments,
            long_context=long_context,
        )
        path = output / record.dataset / f"{record.track_id}.mid"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(midi)
        typer.echo(f"[{index}/{len(selected)}] {record.dataset}/{record.track_id}")


@app.command("benchmark")
def benchmark(
    model_path: Annotated[str, typer.Option("--model", "-m")],
    audio: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    lengths: str = "5,10,20,40,80,160",
    device: str = "cuda",
    long_context: str = "carry",
) -> None:
    model = TranscriptionModel.load_model(model_path, device=device)
    waveform = load_audio(audio, target_sr=16_000)
    rows = []
    for seconds in map(int, lengths.split(",")):
        samples = seconds * 16_000
        if waveform.shape[-1] < samples:
            repeat = (samples + waveform.shape[-1] - 1) // waveform.shape[-1]
            clip = waveform.repeat(1, repeat)[:, :samples]
        else:
            clip = waveform[:, :samples]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        list(
            model.transcribe(
                (clip, 16_000),
                use_sampling=False,
                long_context=long_context,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
        else:
            peak = 0
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "seconds": seconds,
                "elapsed": elapsed,
                "real_time_factor": elapsed / seconds,
                "audio_seconds_per_second": seconds / elapsed,
                "peak_memory_bytes": peak,
                "streaming_state": model.last_streaming_state,
                "long_context": long_context,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
