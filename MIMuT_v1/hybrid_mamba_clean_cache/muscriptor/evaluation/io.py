"""Read reference/prediction notes from standardized JSON or MIDI."""

from __future__ import annotations

from pathlib import Path

import mido

from muscriptor.data.schema import load_standardized_track
from muscriptor.tokenizer.notes import DRUM_PROGRAM, Note, sort_notes


def midi_to_notes(path: str | Path) -> list[Note]:
    midi = mido.MidiFile(path)
    tempo = 500_000
    seconds = 0.0
    programs = {channel: 0 for channel in range(16)}
    active: dict[tuple[int, int], list[tuple[float, int, bool]]] = {}
    notes: list[Note] = []
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "program_change":
            programs[message.channel] = message.program
        elif message.type == "note_on" and message.velocity > 0:
            is_drum = message.channel == 9
            program = DRUM_PROGRAM if is_drum else programs[message.channel]
            active.setdefault((message.channel, message.note), []).append(
                (seconds, program, is_drum)
            )
            if is_drum:
                notes.append(
                    Note(True, DRUM_PROGRAM, seconds, seconds + 0.01, message.note)
                )
        elif message.type in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            stack = active.get(key)
            if stack:
                onset, program, is_drum = stack.pop(0)
                if not is_drum:
                    notes.append(
                        Note(
                            False,
                            program,
                            onset,
                            max(seconds, onset + 0.01),
                            message.note,
                        )
                    )
    sort_notes(notes)
    return notes


def load_notes(path: str | Path) -> list[Note]:
    path = Path(path)
    if path.suffix.lower() in {".mid", ".midi"}:
        return midi_to_notes(path)
    return load_standardized_track(path).notes
