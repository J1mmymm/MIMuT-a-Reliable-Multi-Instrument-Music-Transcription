"""Whole-track transcription quality and long-horizon reliability metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.encode import encode_contiguous_chunks
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note


_TOKENIZER = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
_PROGRAM_TO_REPRESENTATIVE = {
    program: programs[0]
    for programs in _TOKENIZER.group_program_map.values()
    for program in programs
}
_GAP_BINS = ("0-5", "5-10", "10-20", "20+")


def _program(note: Note) -> int:
    if note.is_drum or note.program == DRUM_PROGRAM:
        return DRUM_PROGRAM
    return _PROGRAM_TO_REPRESENTATIVE.get(note.program, note.program)


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    true_positive: int
    predicted: int
    reference: int


@dataclass
class CountRate:
    rate: float
    count: int
    reference: int


@dataclass
class EvaluationResult:
    onset: PRF
    frame: PRF
    program_frame: PRF
    offset: PRF
    drums: PRF
    multi: PRF
    boundary_multi: PRF
    boundary_multi_by_radius: dict[str, PRF]
    boundary_errors: dict[str, CountRate]
    instrument_attribution_accuracy: float
    instrument_attribution_coverage: float
    instrument_switch_error: CountRate
    instrument_switch_error_rate: float
    instrument_switch_comparisons: int
    long_gap_reidentification: dict[str, PRF]
    long_gap_conditional_accuracy: dict[str, float]
    matched_note_count: int
    official_mir_eval: dict
    elapsed_time_strata: dict[str, dict]
    dense_chunk_multi: PRF
    dense_chunk_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _prf(tp: int, predicted: int, reference: int) -> PRF:
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, predicted, reference)


def _count_rate(count: int, reference: int) -> CountRate:
    return CountRate(count / reference if reference else 0.0, count, reference)


def _official_mir_eval(reference: list[Note], prediction: list[Note]) -> dict:
    """Return the package's canonical pitched-note onset/offset measures."""

    try:
        from mir_eval.transcription import precision_recall_f1_overlap
    except ImportError:
        return {
            "available": False,
            "reason": "install the eval extra to report official mir_eval metrics",
        }

    reference = [note for note in reference if not note.is_drum]
    prediction = [note for note in prediction if not note.is_drum]

    def arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
        intervals = np.asarray(
            [[note.onset, note.offset] for note in notes], dtype=np.float64
        ).reshape(-1, 2)
        midi = np.asarray([note.pitch for note in notes], dtype=np.float64)
        frequencies = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
        return intervals, frequencies

    reference_intervals, reference_pitches = arrays(reference)
    prediction_intervals, prediction_pitches = arrays(prediction)

    def run(offset_ratio: float | None) -> dict:
        precision, recall, f1, overlap = precision_recall_f1_overlap(
            reference_intervals,
            reference_pitches,
            prediction_intervals,
            prediction_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=offset_ratio,
            offset_min_tolerance=0.05,
            strict=False,
        )
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "average_overlap_ratio": float(overlap),
            "predicted": len(prediction),
            "reference": len(reference),
        }

    return {
        "available": True,
        "onset": run(None),
        "onset_offset": run(0.2),
    }


def _elapsed_time_strata(
    reference: list[Note],
    prediction: list[Note],
    *,
    segment_duration: float,
    boundary_radius: float,
) -> dict[str, dict]:
    boundaries = (
        ("0-40", 0.0, 40.0),
        ("40-80", 40.0, 80.0),
        ("80-160", 80.0, 160.0),
        ("160+", 160.0, float("inf")),
    )
    result = {}
    for label, start, end in boundaries:
        refs = [note for note in reference if start <= note.onset < end]
        preds = [note for note in prediction if start <= note.onset < end]
        boundary_refs = [
            note
            for note in refs
            if _near_boundary(note, segment_duration, boundary_radius)
        ]
        boundary_preds = [
            note
            for note in preds
            if _near_boundary(note, segment_duration, boundary_radius)
        ]
        result[label] = {
            "reference_notes": len(refs),
            "predicted_notes": len(preds),
            "multi": _metric(
                refs, preds, require_offset=True, require_program=True
            ),
            "boundary_multi": _metric(
                boundary_refs,
                boundary_preds,
                require_offset=True,
                require_program=True,
            ),
            "long_gap_reidentification": _reentry_metrics(refs, preds),
            "instrument_switch_error": _instrument_switch_metric(refs, preds),
        }
    return result


def _dense_chunk_metric(
    reference: list[Note],
    prediction: list[Note],
    *,
    segment_duration: float,
    threshold: int = 1500,
) -> tuple[PRF, int]:
    maximum = max([note.offset for note in reference] or [0.0])
    chunks = max(1, int(np.ceil(maximum / segment_duration)))
    encoded = encode_contiguous_chunks(
        _TOKENIZER,
        reference,
        start_time=0.0,
        num_chunks=chunks,
        duration=segment_duration,
    )
    dense = {
        index for index, chunk in enumerate(encoded) if len(chunk.target_ids) > threshold
    }
    refs = [
        note for note in reference if int(note.onset // segment_duration) in dense
    ]
    preds = [
        note for note in prediction if int(note.onset // segment_duration) in dense
    ]
    return _metric(refs, preds, require_offset=True, require_program=True), len(dense)


def _compatible(
    reference: Note,
    prediction: Note,
    *,
    require_offset: bool,
    require_program: bool,
) -> bool:
    if reference.pitch != prediction.pitch:
        return False
    if abs(reference.onset - prediction.onset) > 0.05:
        return False
    if require_program and _program(reference) != _program(prediction):
        return False
    if require_offset and not reference.is_drum:
        tolerance = max(0.05, 0.2 * (reference.offset - reference.onset))
        if abs(reference.offset - prediction.offset) > tolerance:
            return False
    return True


def _maximum_matching(
    adjacency: list[list[tuple[float, int]]]
) -> list[tuple[int, int]]:
    for row in adjacency:
        row.sort()
    prediction_to_reference: dict[int, int] = {}

    def augment(reference_index: int, seen: set[int]) -> bool:
        for _, prediction_index in adjacency[reference_index]:
            if prediction_index in seen:
                continue
            seen.add(prediction_index)
            previous = prediction_to_reference.get(prediction_index)
            if previous is None or augment(previous, seen):
                prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for reference_index in sorted(
        range(len(adjacency)), key=lambda index: len(adjacency[index])
    ):
        augment(reference_index, set())
    return sorted(
        (
            (reference_index, prediction_index)
            for prediction_index, reference_index in prediction_to_reference.items()
        ),
        key=lambda pair: pair[0],
    )


def _match(
    reference: list[Note],
    prediction: list[Note],
    *,
    require_offset: bool = False,
    require_program: bool = False,
) -> list[tuple[int, int]]:
    adjacency: list[list[tuple[float, int]]] = [[] for _ in reference]
    for reference_index, ref in enumerate(reference):
        for prediction_index, pred in enumerate(prediction):
            if not _compatible(
                ref,
                pred,
                require_offset=require_offset,
                require_program=require_program,
            ):
                continue
            cost = abs(ref.onset - pred.onset)
            if require_offset and not ref.is_drum:
                cost += abs(ref.offset - pred.offset)
            if not require_program:
                cost += 1e-6 * int(_program(ref) != _program(pred))
            adjacency[reference_index].append((cost, prediction_index))
    return _maximum_matching(adjacency)


def _metric(
    reference: list[Note],
    prediction: list[Note],
    *,
    require_offset: bool = False,
    require_program: bool = False,
) -> PRF:
    matches = _match(
        reference,
        prediction,
        require_offset=require_offset,
        require_program=require_program,
    )
    return _prf(len(matches), len(prediction), len(reference))


def _frame_metric(
    reference: list[Note],
    prediction: list[Note],
    *,
    require_program: bool = False,
    resolution: float = 0.0625,
) -> PRF:
    maximum = max([note.offset for note in reference + prediction] or [0.0])
    frames = max(1, int(maximum / resolution) + 1)
    active_sets: list[set[tuple[int, ...]]] = []
    for collection in (reference, prediction):
        active: set[tuple[int, ...]] = set()
        for note in collection:
            start = max(0, int(note.onset / resolution))
            end = max(start + 1, int(note.offset / resolution) + 1)
            for frame in range(start, min(end, frames)):
                key = (
                    (frame, note.pitch, _program(note))
                    if require_program
                    else (frame, note.pitch)
                )
                active.add(key)
        active_sets.append(active)
    ref_active, pred_active = active_sets
    return _prf(len(ref_active & pred_active), len(pred_active), len(ref_active))


def _near_boundary(note: Note, segment_duration: float, radius: float) -> bool:
    event_times = (note.onset,) if note.is_drum else (note.onset, note.offset)
    for event_time in event_times:
        boundary = round(event_time / segment_duration) * segment_duration
        if boundary > 0 and abs(event_time - boundary) <= radius:
            return True
    return False


def _gap_bin(gap: float) -> str:
    if gap < 5:
        return "0-5"
    if gap < 10:
        return "5-10"
    if gap < 20:
        return "10-20"
    return "20+"


@dataclass(frozen=True)
class _ReentryEvent:
    program: int
    onset: float
    gap: float


def _reentry_events(notes: list[Note]) -> list[_ReentryEvent]:
    """Return instrument activity-episode starts after the first episode."""

    grouped: dict[int, list[Note]] = {}
    for note in notes:
        grouped.setdefault(_program(note), []).append(note)
    events: list[_ReentryEvent] = []
    for program, program_notes in grouped.items():
        ordered = sorted(program_notes, key=lambda note: (note.onset, note.offset))
        if not ordered:
            continue
        active_until = ordered[0].offset
        for note in ordered[1:]:
            if note.onset > active_until:
                events.append(
                    _ReentryEvent(program, note.onset, note.onset - active_until)
                )
            active_until = max(active_until, note.offset)
    return sorted(events, key=lambda event: (event.onset, event.program))


def _reentry_metrics(reference: list[Note], prediction: list[Note]) -> dict[str, PRF]:
    reference_events = _reentry_events(reference)
    prediction_events = _reentry_events(prediction)
    result: dict[str, PRF] = {}
    for gap_bin in (*_GAP_BINS, "10+"):
        if gap_bin == "10+":
            refs = [event for event in reference_events if event.gap >= 10.0]
            preds = [event for event in prediction_events if event.gap >= 10.0]
        else:
            refs = [
                event for event in reference_events if _gap_bin(event.gap) == gap_bin
            ]
            preds = [
                event for event in prediction_events if _gap_bin(event.gap) == gap_bin
            ]
        adjacency: list[list[tuple[float, int]]] = [[] for _ in refs]
        for reference_index, ref in enumerate(refs):
            for prediction_index, pred in enumerate(preds):
                if ref.program == pred.program and abs(ref.onset - pred.onset) <= 0.05:
                    adjacency[reference_index].append(
                        (abs(ref.onset - pred.onset), prediction_index)
                    )
        matches = _maximum_matching(adjacency)
        result[gap_bin] = _prf(len(matches), len(preds), len(refs))
    return result


def _instrument_switch_metric(
    reference: list[Note], prediction: list[Note]
) -> CountRate:
    matched = dict(_match(reference, prediction))
    by_program: dict[int, list[int]] = {}
    for reference_index, note in enumerate(reference):
        by_program.setdefault(_program(note), []).append(reference_index)
    switches = comparisons = 0
    for indices in by_program.values():
        indices.sort(key=lambda index: reference[index].onset)
        for previous_index, current_index in zip(indices, indices[1:]):
            if previous_index not in matched or current_index not in matched:
                continue
            previous_program = _program(prediction[matched[previous_index]])
            current_program = _program(prediction[matched[current_index]])
            switches += int(previous_program != current_program)
            comparisons += 1
    return _count_rate(switches, comparisons)


def _boundary_errors(
    reference: list[Note],
    prediction: list[Note],
    *,
    segment_duration: float,
    radius: float,
) -> dict[str, CountRate]:
    maximum = max([note.offset for note in reference] or [0.0])
    boundary = segment_duration
    crossings = omitted = truncated = duplicated = 0
    while boundary < maximum:
        crossing = [
            note
            for note in reference
            if not note.is_drum and note.onset < boundary < note.offset
        ]
        crossings += len(crossing)
        matches = _match(crossing, prediction, require_program=True)
        matched_reference = {reference_index for reference_index, _ in matches}
        matched_prediction = {prediction_index for _, prediction_index in matches}
        omitted += len(crossing) - len(matched_reference)
        for reference_index, prediction_index in matches:
            ref = crossing[reference_index]
            pred = prediction[prediction_index]
            offset_tolerance = max(0.05, 0.2 * (ref.offset - ref.onset))
            if pred.offset < ref.offset - offset_tolerance:
                truncated += 1
        for prediction_index, pred in enumerate(prediction):
            if prediction_index in matched_prediction:
                continue
            if not boundary <= pred.onset <= boundary + radius:
                continue
            if any(
                pred.pitch == ref.pitch and _program(pred) == _program(ref)
                for ref in crossing
            ):
                duplicated += 1
        boundary += segment_duration
    return {
        "omission": _count_rate(omitted, crossings),
        "truncation": _count_rate(truncated, crossings),
        "duplication": _count_rate(duplicated, crossings),
    }


def evaluate_notes(
    reference: list[Note],
    prediction: list[Note],
    *,
    segment_duration: float = 5.0,
    boundary_radius: float = 0.25,
    boundary_radii: tuple[float, ...] = (0.25, 0.5),
) -> EvaluationResult:
    pitched_ref = [note for note in reference if not note.is_drum]
    pitched_pred = [note for note in prediction if not note.is_drum]
    drum_ref = [note for note in reference if note.is_drum]
    drum_pred = [note for note in prediction if note.is_drum]

    onset = _metric(reference, prediction)
    offset = _metric(pitched_ref, pitched_pred, require_offset=True)
    drums = _metric(drum_ref, drum_pred)
    multi = _metric(reference, prediction, require_offset=True, require_program=True)

    radii = tuple(sorted(set((*boundary_radii, boundary_radius))))
    boundary_multi_by_radius: dict[str, PRF] = {}
    for radius in radii:
        boundary_ref = [
            note for note in reference if _near_boundary(note, segment_duration, radius)
        ]
        boundary_pred = [
            note
            for note in prediction
            if _near_boundary(note, segment_duration, radius)
        ]
        boundary_multi_by_radius[f"{radius:.2f}"] = _metric(
            boundary_ref,
            boundary_pred,
            require_offset=True,
            require_program=True,
        )
    boundary_multi = boundary_multi_by_radius[f"{boundary_radius:.2f}"]

    attribution_matches = _match(reference, prediction)
    matched_prediction = dict(attribution_matches)
    correct = sum(
        int(_program(reference[ref_index]) == _program(prediction[pred_index]))
        for ref_index, pred_index in attribution_matches
    )
    attribution = correct / len(attribution_matches) if attribution_matches else 0.0
    coverage = len(attribution_matches) / len(reference) if reference else 0.0

    reference_by_program: dict[int, list[int]] = {}
    for reference_index, ref in enumerate(reference):
        reference_by_program.setdefault(_program(ref), []).append(reference_index)
    switches = comparisons = 0
    gap_correct = {key: [0, 0] for key in _GAP_BINS}
    for ref_program, indices in reference_by_program.items():
        indices.sort(key=lambda index: reference[index].onset)
        for previous_index, current_index in zip(indices, indices[1:]):
            if (
                previous_index not in matched_prediction
                or current_index not in matched_prediction
            ):
                continue
            previous_ref = reference[previous_index]
            current_ref = reference[current_index]
            previous_pred = prediction[matched_prediction[previous_index]]
            current_pred = prediction[matched_prediction[current_index]]
            previous_program = _program(previous_pred)
            current_program = _program(current_pred)
            switches += int(previous_program != current_program)
            comparisons += 1
            key = _gap_bin(max(0.0, current_ref.onset - previous_ref.offset))
            gap_correct[key][0] += int(
                previous_program == ref_program and current_program == ref_program
            )
            gap_correct[key][1] += 1

    dense_chunk_multi, dense_chunk_count = _dense_chunk_metric(
        reference,
        prediction,
        segment_duration=segment_duration,
    )

    return EvaluationResult(
        onset=onset,
        frame=_frame_metric(reference, prediction),
        program_frame=_frame_metric(reference, prediction, require_program=True),
        offset=offset,
        drums=drums,
        multi=multi,
        boundary_multi=boundary_multi,
        boundary_multi_by_radius=boundary_multi_by_radius,
        boundary_errors=_boundary_errors(
            reference,
            prediction,
            segment_duration=segment_duration,
            radius=boundary_radius,
        ),
        instrument_attribution_accuracy=attribution,
        instrument_attribution_coverage=coverage,
        instrument_switch_error=_count_rate(switches, comparisons),
        instrument_switch_error_rate=(switches / comparisons if comparisons else 0.0),
        instrument_switch_comparisons=comparisons,
        long_gap_reidentification=_reentry_metrics(reference, prediction),
        long_gap_conditional_accuracy={
            key: value[0] / value[1] if value[1] else 0.0
            for key, value in gap_correct.items()
        },
        matched_note_count=len(attribution_matches),
        official_mir_eval=_official_mir_eval(reference, prediction),
        elapsed_time_strata=_elapsed_time_strata(
            reference,
            prediction,
            segment_duration=segment_duration,
            boundary_radius=boundary_radius,
        ),
        dense_chunk_multi=dense_chunk_multi,
        dense_chunk_count=dense_chunk_count,
    )
