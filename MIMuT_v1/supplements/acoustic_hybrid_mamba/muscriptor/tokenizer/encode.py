"""Supervised MT3 encoding for training and round-trip tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    Event,
    Note,
    NoteEvent,
    TieNoteEvent,
    sort_note_events,
    sort_tie_note_events,
)


@dataclass(frozen=True)
class EncodedChunk:
    start_time: float
    end_time: float
    target_ids: list[int]
    tie_count: int
    note_event_count: int


def representative_program(tokenizer: MT3Tokenizer, program: int) -> int:
    if program == DRUM_PROGRAM:
        return DRUM_PROGRAM
    for programs in tokenizer.group_program_map.values():
        if program in programs:
            return programs[0]
    raise ValueError(f"MIDI program {program} is outside the tokenizer taxonomy")


def instrument_group_ids(tokenizer: MT3Tokenizer, notes: list[Note]) -> list[int]:
    reverse = {
        program: group
        for group, programs in tokenizer.group_program_map.items()
        for program in programs
    }
    groups = set()
    for note in notes:
        if note.is_drum or note.program == DRUM_PROGRAM:
            groups.add(36)
        elif note.program in reverse:
            groups.add(reverse[note.program])
    return sorted(groups)


def note_event2event(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    start_time: float = 0.0,
    frame_rate: int = 100,
) -> list[Event]:
    tie_note_events = list(tie_note_events or [])
    sort_tie_note_events(tie_note_events)
    sort_note_events(note_events)

    events: list[Event] = []
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    program_state: int | None = None

    for tied in tie_note_events:
        if tied.program != program_state:
            events.append(Event("program", tied.program))
            program_state = tied.program
        events.append(Event("pitch", tied.pitch))
    events.append(Event("tie", 0))

    velocity_state: int | None = None
    for note in note_events:
        if note.is_drum and note.velocity == 0:
            continue
        tick = round(note.time * frame_rate)
        if tick > tick_state:
            shift = tick - start_tick
            events.append(Event("shift", shift))
            tick_state = tick
        elif tick < tick_state:
            raise ValueError(
                f"note event tick {tick} is before current tick {tick_state}"
            )

        if note.is_drum:
            if velocity_state != 1:
                events.append(Event("velocity", 1))
                velocity_state = 1
            events.append(Event("drum", note.pitch))
            continue

        if note.program != program_state:
            events.append(Event("program", note.program))
            program_state = note.program
        if note.velocity != velocity_state:
            events.append(Event("velocity", note.velocity))
            velocity_state = note.velocity
        events.append(Event("pitch", note.pitch))
    return events


def encode_note_events(
    tokenizer: MT3Tokenizer,
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    start_time: float = 0.0,
) -> list[int]:
    events = note_event2event(
        note_events,
        tie_note_events=tie_note_events,
        start_time=start_time,
        frame_rate=tokenizer.frame_rate,
    )
    try:
        return [tokenizer._token_index[(e.type, e.value)] for e in events]
    except KeyError as exc:
        raise ValueError(
            "event is outside the tokenizer vocabulary; check segment length "
            "and MIDI taxonomy"
        ) from exc


def encode_notes_chunk(
    tokenizer: MT3Tokenizer,
    notes: list[Note],
    start_time: float,
    duration: float = 5.0,
) -> EncodedChunk:
    """Encode one half-open 5-second block plus offsets on its right edge.

    Onsets belong to ``[start, end)``. Offsets belong to ``(start, end]`` so a
    note ending exactly at a block boundary is explicitly closed rather than
    leaking into the next block's tie state.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")
    end_time = start_time + duration
    ties: list[TieNoteEvent] = []
    events: list[NoteEvent] = []

    for note in notes:
        if not (0 <= note.pitch <= 127):
            raise ValueError(f"invalid MIDI pitch: {note.pitch}")
        is_drum = note.is_drum or note.program == DRUM_PROGRAM
        program = (
            DRUM_PROGRAM if is_drum else representative_program(tokenizer, note.program)
        )

        if not is_drum and note.onset < start_time < note.offset:
            ties.append(TieNoteEvent(program=program, pitch=note.pitch))

        if start_time <= note.onset < end_time:
            events.append(
                NoteEvent(
                    is_drum=is_drum,
                    program=program,
                    time=note.onset,
                    velocity=1,
                    pitch=note.pitch,
                )
            )
        if (
            not is_drum
            and start_time < note.offset <= end_time
            and note.offset > note.onset
        ):
            events.append(
                NoteEvent(
                    is_drum=False,
                    program=program,
                    time=note.offset,
                    velocity=0,
                    pitch=note.pitch,
                )
            )

    ids = encode_note_events(
        tokenizer,
        events,
        tie_note_events=ties,
        start_time=start_time,
    )
    ids.append(tokenizer.eos_id)
    return EncodedChunk(
        start_time=start_time,
        end_time=end_time,
        target_ids=ids,
        tie_count=len(ties),
        note_event_count=len(events),
    )


def tie_section_length(tokenizer: MT3Tokenizer, ids: list[int]) -> int:
    """Number of leading tokens forming the tie prologue, terminator included.

    Every encoded chunk starts with ``(program, pitch)…`` pairs followed by
    the ``tie`` terminator (an empty prologue is just the terminator).
    Returns 0 when no terminator is present (e.g. a pathological truncation).
    """
    tie_id = tokenizer._token_index[("tie", 0)]
    try:
        return ids.index(tie_id) + 1
    except ValueError:
        return 0


def corrupt_tie_section(
    tokenizer: MT3Tokenizer,
    ids: list[int],
    rng: random.Random,
    *,
    pitch_probability: float = 0.5,
    program_probability: float = 0.25,
) -> list[int]:
    """Return ``ids`` with the tie prologue perturbed, length preserved.

    Pitch tokens are substituted with a nearby pitch and program tokens with
    a different representative program, simulating the imperfect forced
    prelude the model receives at inference after its own mistakes.  Tokens
    at or past the ``tie`` terminator are never touched, so the MIDI body and
    the sequence layout stay intact.
    """
    prologue = tie_section_length(tokenizer, ids)
    if prologue <= 1:
        return list(ids)
    vocab = tokenizer._vocab
    representatives = sorted(
        {programs[0] for programs in tokenizer.group_program_map.values() if programs}
    )
    result = list(ids)
    for index in range(prologue - 1):  # exclude the tie terminator
        event = vocab[result[index]]
        if event.type == "pitch" and rng.random() < pitch_probability:
            shift = rng.choice((-2, -1, 1, 2))
            pitch = min(127, max(0, event.value + shift))
            if pitch != event.value:
                result[index] = tokenizer._token_index[("pitch", pitch)]
        elif event.type == "program" and rng.random() < program_probability:
            candidates = [p for p in representatives if p != event.value]
            if candidates:
                result[index] = tokenizer._token_index[
                    ("program", rng.choice(candidates))
                ]
    return result


def encode_contiguous_chunks(
    tokenizer: MT3Tokenizer,
    notes: list[Note],
    start_time: float,
    num_chunks: int,
    duration: float = 5.0,
) -> list[EncodedChunk]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    return [
        encode_notes_chunk(
            tokenizer,
            notes,
            start_time=start_time + index * duration,
            duration=duration,
        )
        for index in range(num_chunks)
    ]
