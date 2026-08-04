from muscriptor.tokenizer.encode import (
    encode_contiguous_chunks,
    representative_program,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note


def test_cross_boundary_note_has_tie_and_explicit_offset():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    note = Note(False, 40, 4.0, 6.0, 60)
    first, second = encode_contiguous_chunks(tokenizer, [note], 0.0, 2)
    program = representative_program(tokenizer, 40)
    tie = tokenizer.tie_section_token_ids([(program, 60)])
    assert first.target_ids[-1] == tokenizer.eos_id
    assert second.target_ids[: len(tie)] == tie
    decoded = [tokenizer._vocab[index] for index in second.target_ids]
    assert any(event.type == "shift" and event.value == 100 for event in decoded)
    assert any(event.type == "velocity" and event.value == 0 for event in decoded)


def test_programs_are_mapped_to_group_representatives():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    assert representative_program(tokenizer, 1) == 0
    assert representative_program(tokenizer, 57) == 57
