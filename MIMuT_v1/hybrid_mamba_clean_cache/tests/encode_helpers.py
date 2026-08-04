"""Tokenizer helpers used only by the unit tests.

The shipped package decodes straight from token indices to streamed events
(:func:`muscriptor.events.decode_model_tokens`), so the note-event ↔ token
round-trip path — encoding (``note_event2event`` / ``encode_note_events``) and
the per-chunk note-list decode (``event2note_event`` / ``decode_tokens``) — is
exercised only by tests and lives here rather than as dead code in the package.
"""

from collections import Counter

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Event,
    NoteEvent,
    TieNoteEvent,
    build_event_vocab,
    sort_note_events,
    sort_tie_note_events,
)


def note_event2event(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    start_time: float = 0.0,
    frame_rate: int = 100,
) -> list[Event]:
    if tie_note_events is not None:
        sort_tie_note_events(tie_note_events)
    note_events.sort(
        key=lambda n: (
            round(n.time * frame_rate),
            n.is_drum,
            n.program,
            n.velocity,
            n.pitch,
        )
    )

    events = []
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    program_state = None

    if tie_note_events:
        for tne in tie_note_events:
            if tne.program != program_state:
                events.append(Event(type="program", value=tne.program))
                program_state = tne.program
            events.append(Event(type="pitch", value=tne.pitch))

    events.append(Event(type="tie", value=0))

    velocity_state = None
    for ne in note_events:
        if ne.is_drum and ne.velocity == 0:
            continue

        ne_tick = round(ne.time * frame_rate)
        if ne_tick > tick_state:
            shift_ticks = ne_tick - start_tick
            events.append(Event(type="shift", value=shift_ticks))
            tick_state = ne_tick
        elif ne_tick == tick_state:
            pass
        else:
            raise ValueError(
                f"NoteEvent tick {ne_tick} at time {ne.time} is smaller than tick_state {tick_state}."
            )

        if ne.is_drum and ne.velocity == 1:
            if velocity_state != 1 or velocity_state is None:
                events.append(Event(type="velocity", value=1))
                velocity_state = 1
            events.append(Event(type="drum", value=ne.pitch))
        else:
            if ne.program != program_state or program_state is None:
                events.append(Event(type="program", value=ne.program))
                program_state = ne.program
            if ne.velocity != velocity_state or velocity_state is None:
                events.append(Event(type="velocity", value=ne.velocity))
                velocity_state = ne.velocity
            events.append(Event(type="pitch", value=ne.pitch))

    return events


def encode_index_map(max_shift_steps: int) -> dict[tuple[str, int], int]:
    """Inverse of :func:`muscriptor.tokenizer.notes.build_event_vocab`."""
    return {
        (e.type, e.value): i for i, e in enumerate(build_event_vocab(max_shift_steps))
    }


def encode_note_events(
    note_events: list[NoteEvent],
    max_shift_steps: int,
    tie_note_events: list[TieNoteEvent] | None = None,
    start_time: float = 0.0,
    frame_rate: int = 100,
) -> list[int]:
    """Encode note events into model token indices (test counterpart of decode)."""
    events = note_event2event(note_events, tie_note_events, start_time, frame_rate)
    index = encode_index_map(max_shift_steps)
    return [index[(e.type, e.value)] for e in events]


def event2note_event(
    events: list[Event], start_time: float = 0.0, frame_rate: int = 100
) -> tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, ...]], Counter]:
    assert start_time >= 0.0

    tie_index = None
    program_state = None
    tie_note_events = []
    last_activity = set()
    err_cnt: Counter = Counter()

    for i, e in enumerate(events):
        try:
            if e.type == "tie":
                tie_index = i
                break
            if e.type == "shift":
                break
            elif e.type == "program":
                program_state = e.value
            elif e.type == "pitch":
                if program_state is None:
                    raise ValueError("Err/Missing prg in tie")
                tie_note_events.append(
                    TieNoteEvent(program=program_state, pitch=e.value)
                )
                last_activity.add((program_state, e.value))
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    try:
        if tie_index is None:
            raise ValueError("Err/Missing tie")
        else:
            events = events[tie_index + 1 :]
    except ValueError as ve:
        err_cnt[str(ve)] += 1
        return [], [], [], err_cnt

    note_events = []
    velocity_state = None
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    for e in events:
        try:
            if e.type == "shift":
                if e.value <= 0:
                    raise ValueError("Err/Negative shift")
                prev_tick_state = tick_state
                tick_state = start_tick + e.value
                if tick_state <= prev_tick_state:
                    raise ValueError("Err/Shift not strictly monotonic")
            elif e.type == "drum":
                note_events.append(
                    NoteEvent(
                        is_drum=True,
                        program=DRUM_PROGRAM,
                        time=tick_state / frame_rate,
                        velocity=1,
                        pitch=e.value,
                    )
                )
            elif e.type == "program":
                program_state = e.value
            elif e.type == "velocity":
                velocity_state = e.value
            elif e.type == "pitch":
                if program_state is None:
                    raise ValueError("Err/Missing prg")
                elif velocity_state is None:
                    raise ValueError("Err/Missing vel")
                if velocity_state > 0:
                    last_activity.add((program_state, e.value))
                elif velocity_state == 0 and (program_state, e.value) in last_activity:
                    last_activity.remove((program_state, e.value))
                else:
                    raise ValueError("Err/Note off without note on")
                note_events.append(
                    NoteEvent(
                        is_drum=False,
                        program=program_state,
                        time=tick_state / frame_rate,
                        velocity=velocity_state,
                        pitch=e.value,
                    )
                )
            elif e.type == "EOS":
                break
            elif e.type in ("PAD", "UNK"):
                continue
            elif e.type == "tie":
                if tick_state == start_tick:
                    raise ValueError("Err/Multi-tie type 1")
                else:
                    raise ValueError("Err/Multi-tie type 2")
            else:
                raise ValueError("Err/Unknown event")
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    sort_note_events(note_events)
    sort_tie_note_events(tie_note_events)
    return note_events, tie_note_events, list(last_activity), err_cnt


def decode_tokens(
    tokens: list[int],
    vocab: list[Event],
    start_time: float = 0.0,
    frame_rate: int = 100,
) -> tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, ...]], Counter]:
    """Decode model token indices into note events via :func:`event2note_event`.

    ``vocab`` is the table returned by :func:`build_event_vocab`.
    """
    events = []
    for idx in tokens:
        if idx < 0 or idx >= len(vocab):
            raise ValueError(f"Unknown event index: {idx}")
        events.append(vocab[idx])
    return event2note_event(events, start_time, frame_rate)
