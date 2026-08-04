"""Hermetic tests for the streaming token decoder.

Each test builds a synthetic per-chunk (note_events, tie_note_events) scenario,
encodes it to model tokens, and feeds the ChunkBoundary-delimited token stream
to `decode_model_tokens` — exercising the streaming decoder end to end without
a model or audio.
"""

from muscriptor.events import (
    ChunkBoundary,
    NoteEndEvent,
    NoteStartEvent,
    decode_model_tokens,
)
from muscriptor.tokenizer.notes import (
    MINIMUM_NOTE_DURATION_SEC,
    NoteEvent,
    TieNoteEvent,
    build_event_vocab,
)
from tests.encode_helpers import encode_note_events

# Match the model's shift range so within-chunk shifts (up to ~500 ticks) encode.
_MAX_SHIFT_STEPS = 1001
_VOCAB = build_event_vocab(_MAX_SHIFT_STEPS)


def _instr(program: int) -> str:
    return f"prog{program}"


def _on(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=1, pitch=pitch)


def _off(time: float, program: int, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=False, program=program, time=time, velocity=0, pitch=pitch)


def _drum(time: float, pitch: int) -> NoteEvent:
    return NoteEvent(is_drum=True, program=128, time=time, velocity=1, pitch=pitch)


def _decode(*chunks) -> list[NoteStartEvent | NoteEndEvent]:
    """Decode a sequence of chunks through the streaming decoder.

    Each chunk is ``(note_events, tie_note_events, seek_time, next_seek_time)``;
    it is encoded to tokens (tie prologue included) and prefixed with a
    ChunkBoundary, then the whole stream is decoded in one pass.
    """

    def stream():
        for note_events, tie_notes, seek, next_seek in chunks:
            yield ChunkBoundary(seek, next_seek)
            yield from encode_note_events(
                note_events,
                max_shift_steps=_MAX_SHIFT_STEPS,
                tie_note_events=tie_notes,
                start_time=seek,
            )

    return list(decode_model_tokens(stream(), _VOCAB, _instr))


# ---------------------------------------------------------------------------
# Basic note matching
# ---------------------------------------------------------------------------


def test_single_note_in_one_chunk():
    events = _decode(([_on(0.1, 0, 60), _off(0.5, 0, 60)], [], 0.0, 5.0))
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent)
    assert isinstance(end, NoteEndEvent)
    assert start.pitch == 60
    assert start.start_time == 0.1
    assert start.instrument == "prog0"
    assert end.end_time == 0.5
    assert end.start_event is start


def test_indices_are_unique_and_monotonic():
    events = _decode(
        (
            [_on(0.1, 0, 60), _on(0.2, 0, 64), _off(0.3, 0, 60), _off(0.4, 0, 64)],
            [],
            0.0,
            5.0,
        )
    )
    starts = [e for e in events if isinstance(e, NoteStartEvent)]
    assert [s.index for s in starts] == [0, 1]


def test_orphan_note_off_is_dropped():
    events = _decode(([_off(0.2, 0, 60)], [], 0.0, 5.0))
    assert events == []


def test_retrigger_closes_previous():
    events = _decode(
        ([_on(0.1, 0, 60), _on(0.3, 0, 60), _off(0.5, 0, 60)], [], 0.0, 5.0)
    )
    # expect: NoteStart#0, NoteEnd(prev), NoteStart#1, NoteEnd(curr)
    assert isinstance(events[0], NoteStartEvent) and events[0].index == 0
    assert isinstance(events[1], NoteEndEvent) and events[1].start_event_index == 0
    assert events[1].end_time == 0.3
    assert isinstance(events[2], NoteStartEvent) and events[2].index == 1
    assert isinstance(events[3], NoteEndEvent) and events[3].start_event_index == 1
    assert events[3].end_time == 0.5


# ---------------------------------------------------------------------------
# Drums
# ---------------------------------------------------------------------------


def test_drum_emits_start_and_end_pair():
    events = _decode(([_drum(1.0, 38)], [], 0.0, 5.0))
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent)
    assert start.pitch == 38
    assert start.instrument == "drums"
    assert isinstance(end, NoteEndEvent)
    assert end.end_time == 1.0 + MINIMUM_NOTE_DURATION_SEC
    assert end.start_event is start


# ---------------------------------------------------------------------------
# Chunk-boundary stitching
# ---------------------------------------------------------------------------


def test_note_sustains_across_chunk_via_tie():
    events = _decode(
        # chunk 0: note-on at 4.0s, no off
        ([_on(4.0, 0, 60)], [], 0.0, 5.0),
        # chunk 1: tie says (0, 60) continues; close at 6.0s
        ([_off(6.0, 0, 60)], [TieNoteEvent(program=0, pitch=60)], 5.0, None),
    )
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent) and start.pitch == 60
    assert isinstance(end, NoteEndEvent)
    assert end.start_event is start
    assert end.end_time == 6.0


def test_unsustained_note_closes_at_chunk_boundary():
    events = _decode(
        # chunk 0: note-on at 4.0s, no off
        ([_on(4.0, 0, 60)], [], 0.0, 5.0),
        # chunk 1: no tie → must close at 5.0s
        ([], [], 5.0, None),
    )
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent)
    assert isinstance(end, NoteEndEvent)
    assert end.start_event is start
    assert end.end_time == 5.0


def test_tie_for_unknown_note_is_ignored():
    # no prior open notes; tie says (0, 60) should continue — ignore it
    events = _decode(
        (
            [_on(0.5, 0, 64), _off(1.0, 0, 64)],
            [TieNoteEvent(program=0, pitch=60)],
            0.0,
            5.0,
        )
    )
    starts = [e for e in events if isinstance(e, NoteStartEvent)]
    assert len(starts) == 1
    assert starts[0].pitch == 64


# ---------------------------------------------------------------------------
# Multi-chunk filtering and end-of-stream flush
# ---------------------------------------------------------------------------


def test_events_past_next_seek_time_are_filtered():
    # Event at 5.2s should not be processed when next_seek_time=5.0
    events = _decode(
        ([_on(4.5, 0, 60), _off(4.9, 0, 60), _on(5.2, 0, 64)], [], 0.0, 5.0)
    )
    pitches = [e.pitch for e in events if isinstance(e, NoteStartEvent)]
    assert pitches == [60]


def test_end_of_stream_closes_remaining_open_notes():
    events = _decode(([_on(4.0, 0, 60)], [], 0.0, None))
    assert len(events) == 2
    start, end = events
    assert isinstance(start, NoteStartEvent)
    assert isinstance(end, NoteEndEvent)
    assert end.start_event is start
    assert end.end_time == 4.0 + MINIMUM_NOTE_DURATION_SEC


def test_no_extra_events_when_all_closed():
    events = _decode(([_on(0.1, 0, 60), _off(0.5, 0, 60)], [], 0.0, None))
    assert len(events) == 2
    assert isinstance(events[0], NoteStartEvent)
    assert isinstance(events[1], NoteEndEvent)
    assert events[1].end_time == 0.5


# ---------------------------------------------------------------------------
# Global invariants
# ---------------------------------------------------------------------------


def test_every_start_has_exactly_one_end():
    """Drive a synthetic 3-chunk session and check the contract."""
    all_events = _decode(
        ([_on(0.1, 0, 60), _on(0.2, 0, 64), _off(0.3, 0, 60)], [], 0.0, 5.0),
        # 64 is still open → tie keeps it
        (
            [_on(5.5, 0, 67), _off(6.0, 0, 64), _off(6.5, 0, 67)],
            [TieNoteEvent(program=0, pitch=64)],
            5.0,
            10.0,
        ),
        ([_drum(10.5, 36)], [], 10.0, None),
    )

    starts = [e for e in all_events if isinstance(e, NoteStartEvent)]
    ends = [e for e in all_events if isinstance(e, NoteEndEvent)]
    assert len(starts) == len(ends)
    start_ids = {s.index for s in starts}
    end_ids = {e.start_event_index for e in ends}
    assert start_ids == end_ids
    assert len(start_ids) == len(starts), "indices must be unique"
