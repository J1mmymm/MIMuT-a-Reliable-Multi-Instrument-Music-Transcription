from dataclasses import replace

import torch

from muscriptor.data.dataset import (
    NUM_INSTRUMENT_GROUPS,
    NUM_MIDI_PITCHES,
    REENTRY_NONE_CLASS,
    TrainingBatch,
)
from muscriptor.events import ChunkBoundary, OpenNoteTracker
from muscriptor.models.config import ModelConfig
from muscriptor.models.training import _condition_attributes
from muscriptor.modules.conditioners import MelSpectrogramConditioner
from muscriptor.modules.streaming import (
    clone_model_state,
    increment_steps,
    init_states,
    iter_state_tensors,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.training.config import (
    BoundaryStateSupervisionConfig,
    CleanCacheTrainingConfig,
)
from muscriptor.training.train import LongContextTrainingModel
from muscriptor.transcription_model import _build_model


def _config(**overrides) -> ModelConfig:
    base = ModelConfig(
        dim=32,
        num_heads=4,
        num_layers=3,
        card=1395,
        backbone="local_transformer",
        position_encoding="rope",
        local_window=128,
        local_query_chunk=32,
        use_type_embeddings=True,
        correct_class_conditioning=True,
        learned_null_conditioning=True,
        num_dataset_condition_classes=8,
        clean_acoustic_cache=True,
        boundary_state_dim=8,
    )
    return replace(base, **overrides)


def _model():
    model = _build_model(torch.device("cpu"), _config()).eval()
    for conditioner in model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
    return model


def _state_snapshot(state):
    return {path: tensor.detach().clone() for path, tensor in iter_state_tensors(state)}


def _assert_snapshot(state, expected):
    actual = dict(iter_state_tensors(state))
    assert actual.keys() == expected.keys()
    for path, tensor in actual.items():
        torch.testing.assert_close(tensor, expected[path], rtol=0, atol=0)


def test_deep_state_clone_has_no_shared_tensor_storage():
    model = _model()
    state = init_states(model, batch_size=1, sequence_length=4096)
    clone = clone_model_state(state)
    for (left_path, left), (right_path, right) in zip(
        iter_state_tensors(state), iter_state_tensors(clone)
    ):
        assert left_path == right_path
        assert left.data_ptr() != right.data_ptr() or left.numel() == 0


def test_first_chunk_split_prefill_matches_reset_logits():
    torch.manual_seed(2)
    model = _model()
    prefix = torch.randn(1, 13, model.dim)
    events = torch.randn(1, 7, model.dim)

    reset_state = init_states(model, batch_size=1, sequence_length=128)
    reset_hidden = model.encode_embeddings(
        torch.cat([prefix, events], dim=1), model_state=reset_state
    )

    clean_state = init_states(model, batch_size=1, sequence_length=128)
    model.encode_embeddings(prefix, model_state=clean_state)
    increment_steps(model.transformer, clean_state, prefix.shape[1])
    decode_state = clone_model_state(clean_state)
    split_hidden = model.encode_embeddings(events, model_state=decode_state)
    torch.testing.assert_close(
        model.linear(split_hidden),
        model.linear(reset_hidden[:, -events.shape[1] :]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_2000_bad_event_tokens_cannot_mutate_clean_state():
    torch.manual_seed(3)
    model = _model()
    clean_state = init_states(model, batch_size=1, sequence_length=4096)
    prefix = torch.randn(1, 17, model.dim)
    model.encode_embeddings(prefix, model_state=clean_state)
    increment_steps(model.transformer, clean_state, prefix.shape[1])
    expected = _state_snapshot(clean_state)

    decode_state = clone_model_state(clean_state)
    model.encode_embeddings(torch.randn(1, 2000, model.dim), model_state=decode_state)
    increment_steps(model.transformer, decode_state, 2000)
    _assert_snapshot(clean_state, expected)


def test_different_bad_histories_cannot_change_next_chunk_from_clean_snapshot():
    torch.manual_seed(4)
    model = _model()
    clean = init_states(model, batch_size=1, sequence_length=4096)
    prefix = torch.randn(1, 9, model.dim)
    model.encode_embeddings(prefix, model_state=clean)
    increment_steps(model.transformer, clean, prefix.shape[1])
    for length in (31, 53):
        branch = clone_model_state(clean)
        model.encode_embeddings(torch.randn(1, length, model.dim), model_state=branch)

    next_prefix = torch.randn(1, 11, model.dim)
    left = clone_model_state(clean)
    right = clone_model_state(clean)
    left_hidden = model.encode_embeddings(next_prefix, model_state=left)
    right_hidden = model.encode_embeddings(next_prefix, model_state=right)
    torch.testing.assert_close(left_hidden, right_hidden, rtol=0, atol=0)


def test_clean_local_kv_contains_prefix_only():
    model = _model()
    clean = init_states(model, batch_size=1, sequence_length=4096)
    prefix = torch.randn(1, 23, model.dim)
    model.encode_embeddings(prefix, model_state=clean)
    increment_steps(model.transformer, clean, prefix.shape[1])
    clean_lengths = [
        state["key"].shape[2]
        for state in clean.values()
        if isinstance(state, dict) and "key" in state
    ]
    assert clean_lengths and set(clean_lengths) == {23}
    branch = clone_model_state(clean)
    model.encode_embeddings(torch.randn(1, 19, model.dim), model_state=branch)
    assert set(clean_lengths) == {23}
    assert all(
        state["key"].shape[2] == 23
        for state in clean.values()
        if isinstance(state, dict) and "key" in state
    )


def _training_batch(chunks: int = 2) -> TrainingBatch:
    target = torch.tensor([1317, 1, 0, 0], dtype=torch.long)
    targets = target.view(1, 1, -1).repeat(1, chunks, 1)
    active = torch.zeros(
        1, chunks, NUM_INSTRUMENT_GROUPS, NUM_MIDI_PITCHES, dtype=torch.bool
    )
    active[:, :, 0, 60] = True
    reentry = torch.full(
        (1, chunks, NUM_INSTRUMENT_GROUPS), REENTRY_NONE_CLASS, dtype=torch.long
    )
    valid = torch.ones_like(reentry, dtype=torch.bool)
    return TrainingBatch(
        waveform=torch.zeros(1, chunks, 1, 1600),
        target_ids=targets,
        target_lengths=torch.full((1, chunks), target.numel(), dtype=torch.long),
        instrument_groups=["9"],
        dataset_ids=torch.tensor([3]),
        track_ids=["x"],
        start_times=torch.tensor([0.0]),
        has_long_gap=torch.tensor([False]),
        target_loss_mask=torch.ones_like(targets, dtype=torch.bool),
        active_note_targets=active,
        reentry_targets=reentry,
        reentry_valid=valid,
    )


def test_audio_only_conditions_are_learned_null_metadata_tokens():
    model = _model()
    batch = _training_batch(chunks=1)
    attributes = _condition_attributes(
        batch,
        sample_rate=16_000,
        audio_condition_dropout=0.0,
        instrument_condition_dropout=1.0,
        dataset_condition_dropout=1.0,
    )
    assert attributes[0].text == {
        "instrument_group": None,
        "dataset_name": None,
    }
    tokenized = model.condition_provider.tokenize(attributes)
    assert tokenized["instrument_group"].item() == 1
    assert tokenized["dataset_name"].item() == 1
    encoded = model.condition_provider(tokenized)
    assert encoded["instrument_group"][1].item() == 1
    assert encoded["dataset_name"][1].item() == 1


def test_malformed_or_eos_failed_candidate_never_commits():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    tracker = OpenNoteTracker(tokenizer._vocab, tokenizer.frame_rate)
    candidate = tracker.fork()
    candidate.feed(ChunkBoundary(0.0, 5.0))
    shift = next(
        index
        for index, event in enumerate(tokenizer._vocab)
        if event.type == "shift" and event.value == 1
    )
    candidate.feed(shift)
    assert candidate.commit_ready is False
    assert tracker.open_keys() == []


def test_clean_branch_training_runs_and_boundary_heads_are_audio_causal():
    model = _model().train()
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    wrapper = LongContextTrainingModel(
        model,
        BoundaryStateSupervisionConfig(
            enabled=True, active_weight=0.1, reentry_weight=0.1
        ),
        tokenizer=tokenizer,
        clean_cache_config=CleanCacheTrainingConfig(
            enabled=True, symbolic_state_curriculum=True
        ),
    )
    logits, labels, token_count, active, reentry, modes = wrapper(
        _training_batch(), 0.0, 1.0, 1.0, global_step=20_000
    )
    assert logits.shape == (labels.numel(), model.card)
    assert token_count == labels.numel()
    assert active.shape == (1, 2, NUM_INSTRUMENT_GROUPS, NUM_MIDI_PITCHES)
    assert reentry.shape == (1, 2, NUM_INSTRUMENT_GROUPS, REENTRY_NONE_CLASS + 1)
    assert modes.sum().item() == 2
    loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    loss = loss + 0.1 * active.float().mean() + 0.1 * reentry.float().mean()
    loss.backward()
    assert model.boundary_projection.weight.grad is not None
