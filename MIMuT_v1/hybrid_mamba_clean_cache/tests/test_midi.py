"""Tests for muscriptor/utils/midi.py."""

import tempfile
from pathlib import Path

from mido import MidiFile

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.midi import notes_to_midi, save_midi


def _sample_notes():
    return [
        Note(is_drum=False, program=0, onset=0.0, offset=0.5, pitch=60),
        Note(is_drum=False, program=0, onset=0.5, offset=1.0, pitch=64),
        Note(is_drum=True, program=128, onset=0.0, offset=0.01, pitch=36),
    ]


def test_notes_to_midi_returns_midi_file():
    midi = notes_to_midi(_sample_notes())
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_has_tracks():
    midi = notes_to_midi(_sample_notes())
    assert len(midi.tracks) > 0


def test_notes_to_midi_custom_tempo():
    midi = notes_to_midi(_sample_notes(), tempo_bpm=90)
    assert isinstance(midi, MidiFile)


def test_notes_to_midi_empty_notes():
    midi = notes_to_midi([])
    assert isinstance(midi, MidiFile)


def test_save_midi_creates_file():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_save_midi_is_valid_midi():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "out.mid"
        save_midi(notes, path)
        loaded = MidiFile(str(path))
        assert len(loaded.tracks) > 0


def test_save_midi_string_path():
    notes = _sample_notes()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "out.mid")
        save_midi(notes, path)
        assert Path(path).exists()
