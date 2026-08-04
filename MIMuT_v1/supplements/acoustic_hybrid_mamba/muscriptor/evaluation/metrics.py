"""MuScriptor-compatible and long-horizon transcription metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note


_TOKENIZER = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
_PROGRAM_TO_REPRESENTATIVE = {
    program: programs[0]
    for programs in _TOKENIZER.group_program_map.values()
    for program in programs
}


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
class EvaluationResult:
    onset: PRF
    frame: PRF
    offset: PRF
    drums: PRF
    multi: PRF
    boundary_multi: PRF
    instrument_attribution_accuracy: float
    instrument_switch_error_rate: float
    long_gap_reidentification: dict[str, float]
    matched_note_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _prf(tp: int, predicted: int, reference: int) -> PRF:
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp, predicted, reference)


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
    if require_program:
        ref_program = _program(reference)
        pred_program = _program(prediction)
        if ref_program != pred_program:
            return False
    if require_offset and not reference.is_drum:
        tolerance = max(0.05, 0.2 * (reference.offset - reference.onset))
        if abs(reference.offset - prediction.offset) > tolerance:
            return False
    return True


def _match(
    reference: list[Note],
    prediction: list[Note],
    *,
    require_offset: bool = False,
    require_program: bool = False,
) -> list[tuple[int, int]]:
    adjacency: list[list[tuple[float, int]]] = [[] for _ in reference]
    for ref_index, ref in enumerate(reference):
        for pred_index, pred in enumerate(prediction):
            if _compatible(
                ref,
                pred,
                require_offset=require_offset,
                require_program=require_program,
            ):
                cost = abs(ref.onset - pred.onset)
                if require_offset and not ref.is_drum:
                    cost += abs(ref.offset - pred.offset)
                # Prefer a same-program pairing when otherwise-equivalent
                # onset/pitch matches exist. Program is still not required
                # unless requested by the caller.
                if not require_program:
                    ref_program = _program(ref)
                    pred_program = _program(pred)
                    cost += 1e-6 * int(ref_program != pred_program)
                adjacency[ref_index].append((cost, pred_index))
        adjacency[ref_index].sort()

    # Maximum-cardinality bipartite matching.  A global greedy pass can
    # under-count true positives when one reference has several candidates
    # and another has only one.
    prediction_to_reference: dict[int, int] = {}

    def augment(ref_index: int, seen: set[int]) -> bool:
        for _, pred_index in adjacency[ref_index]:
            if pred_index in seen:
                continue
            seen.add(pred_index)
            previous = prediction_to_reference.get(pred_index)
            if previous is None or augment(previous, seen):
                prediction_to_reference[pred_index] = ref_index
                return True
        return False

    for ref_index in sorted(
        range(len(reference)),
        key=lambda index: (len(adjacency[index]), reference[index].onset),
    ):
        augment(ref_index, set())
    return sorted(
        (
            (ref_index, pred_index)
            for pred_index, ref_index in prediction_to_reference.items()
        ),
        key=lambda pair: pair[0],
    )


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
    resolution: float = 0.0625,
) -> PRF:
    maximum = max([note.offset for note in reference + prediction] or [0.0])
    frames = max(1, int(maximum / resolution) + 1)
    ref_active = set()
    pred_active = set()
    for collection, output in (
        (reference, ref_active),
        (prediction, pred_active),
    ):
        for note in collection:
            start = max(0, int(note.onset / resolution))
            end = max(start + 1, int(note.offset / resolution) + 1)
            for frame in range(start, min(end, frames)):
                output.add((frame, note.pitch))
    return _prf(len(ref_active & pred_active), len(pred_active), len(ref_active))


def _near_boundary(note: Note, segment_duration: float, radius: float) -> bool:
    boundary = round(note.onset / segment_duration) * segment_duration
    return boundary > 0 and abs(note.onset - boundary) <= radius


def _gap_bin(gap: float) -> str:
    if gap < 5:
        return "0-5"
    if gap < 10:
        return "5-10"
    if gap < 20:
        return "10-20"
    return "20+"


def evaluate_notes(
    reference: list[Note],
    prediction: list[Note],
    *,
    segment_duration: float = 5.0,
    boundary_radius: float = 0.25,
) -> EvaluationResult:
    pitched_ref = [note for note in reference if not note.is_drum]
    pitched_pred = [note for note in prediction if not note.is_drum]
    drum_ref = [note for note in reference if note.is_drum]
    drum_pred = [note for note in prediction if note.is_drum]
    onset = _metric(reference, prediction)
    offset = _metric(pitched_ref, pitched_pred, require_offset=True)
    drums = _metric(drum_ref, drum_pred)
    multi_matches = _match(
        reference,
        prediction,
        require_offset=True,
        require_program=True,
    )
    multi = _prf(len(multi_matches), len(prediction), len(reference))

    boundary_ref = [
        note
        for note in reference
        if _near_boundary(note, segment_duration, boundary_radius)
    ]
    boundary_pred = [
        note
        for note in prediction
        if _near_boundary(note, segment_duration, boundary_radius)
    ]
    boundary_multi = _metric(
        boundary_ref,
        boundary_pred,
        require_offset=True,
        require_program=True,
    )

    attribution_matches = _match(reference, prediction)
    correct = 0
    reference_by_program: dict[int, list[int]] = {}
    matched_prediction = dict(attribution_matches)
    for ref_index, pred_index in attribution_matches:
        ref, pred = reference[ref_index], prediction[pred_index]
        ref_program = _program(ref)
        pred_program = _program(pred)
        correct += int(ref_program == pred_program)
    for ref_index, ref in enumerate(reference):
        ref_program = _program(ref)
        reference_by_program.setdefault(ref_program, []).append(ref_index)
    attribution = correct / len(attribution_matches) if attribution_matches else 0.0

    switches = comparisons = 0
    gap_correct = {key: [0, 0] for key in ("0-5", "5-10", "10-20", "20+")}
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

    return EvaluationResult(
        onset=onset,
        frame=_frame_metric(reference, prediction),
        offset=offset,
        drums=drums,
        multi=multi,
        boundary_multi=boundary_multi,
        instrument_attribution_accuracy=attribution,
        instrument_switch_error_rate=(switches / comparisons if comparisons else 0.0),
        long_gap_reidentification={
            key: value[0] / value[1] if value[1] else 0.0
            for key, value in gap_correct.items()
        },
        matched_note_count=len(attribution_matches),
    )


def mir_eval_note_metrics(
    reference: list[Note],
    prediction: list[Note],
) -> dict | None:
    """Official mir_eval note metrics for literature-comparable numbers.

    Returns onset-only F1 (50 ms tolerance) and onset+offset F1 (offset
    ratio 0.2, 50 ms floor) over pitched notes, computed by
    ``mir_eval.transcription`` itself rather than this module's re-
    implementation.  Returns None when mir_eval is not installed.
    """
    try:
        import mir_eval.transcription
    except ImportError:
        return None
    import numpy as np

    def arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
        pitched = [note for note in notes if not note.is_drum]
        intervals = np.array(
            [
                [note.onset, max(note.offset, note.onset + 1e-4)]
                for note in pitched
            ]
        ).reshape(-1, 2)
        pitches = np.array(
            [440.0 * 2.0 ** ((note.pitch - 69) / 12.0) for note in pitched]
        )
        return intervals, pitches

    ref_intervals, ref_pitches = arrays(reference)
    est_intervals, est_pitches = arrays(prediction)
    if not len(ref_intervals) or not len(est_intervals):
        empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return {"onset": dict(empty), "onset_offset": dict(empty)}

    onset_p, onset_r, onset_f, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            offset_ratio=None,
        )
    )
    full_p, full_r, full_f, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            offset_ratio=0.2,
            offset_min_tolerance=0.05,
        )
    )
    return {
        "onset": {"precision": onset_p, "recall": onset_r, "f1": onset_f},
        "onset_offset": {"precision": full_p, "recall": full_r, "f1": full_f},
    }
