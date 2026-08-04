import torch

from muscriptor.data.dataset import (
    TrainingBatch,
    _record_window_gap_bins,
    boundary_state_targets,
)
from muscriptor.models.config import ModelConfig
from muscriptor.models.training import (
    IGNORE_INDEX,
    _condition_attributes,
    pack_training_batch,
)
from muscriptor.modules.conditioners import MelSpectrogramConditioner
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from muscriptor.tokenizer.notes import Note
from muscriptor.training.config import BoundaryStateSupervisionConfig
from muscriptor.training.train import LongContextTrainingModel
from muscriptor.transcription_model import _build_model


def test_training_pack_masks_conditions_and_runs_backbone():
    config = ModelConfig(
        dim=32,
        num_heads=4,
        num_layers=2,
        card=1393,
        backbone="local_transformer",
        local_window=64,
        local_query_chunk=16,
        use_type_embeddings=True,
        correct_class_conditioning=True,
        num_dataset_condition_classes=8,
    )
    model = _build_model(torch.device("cpu"), config)
    for conditioner in model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
    batch = TrainingBatch(
        waveform=torch.zeros(1, 1, 1, 1600),
        target_ids=torch.tensor([[[1317, 1, 0]]]),
        target_lengths=torch.tensor([[2]]),
        instrument_groups=["9"],
        dataset_ids=torch.tensor([0]),
        track_ids=["x"],
        start_times=torch.tensor([0.0]),
        has_long_gap=torch.tensor([False]),
    )
    packed = pack_training_batch(model, batch, condition_dropout=0.0)
    assert packed.labels.ne(IGNORE_INDEX).sum().item() == 2
    logits = model.forward_embeddings(packed.embeddings)
    assert logits.shape[:2] == packed.labels.shape
    assert torch.isfinite(logits).all()


def _tiny_batch() -> TrainingBatch:
    return TrainingBatch(
        waveform=torch.zeros(1, 1, 1, 1600),
        target_ids=torch.tensor([[[1317, 1, 0]]]),
        target_lengths=torch.tensor([[2]]),
        instrument_groups=["9"],
        dataset_ids=torch.tensor([3]),
        track_ids=["x"],
        start_times=torch.tensor([0.0]),
        has_long_gap=torch.tensor([False]),
    )


def test_formal_conditioning_keeps_audio_and_nulls_metadata():
    attributes = _condition_attributes(
        _tiny_batch(),
        sample_rate=16_000,
        audio_condition_dropout=0.0,
        instrument_condition_dropout=1.0,
        dataset_condition_dropout=1.0,
    )
    assert attributes[0].wav["self_wav"].length.item() == 1600
    assert attributes[0].text["instrument_group"] is None
    assert attributes[0].text["dataset_name"] is None


def test_boundary_targets_follow_tie_and_censoring_semantics():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    notes = [
        Note(False, 0, 0.0, 6.0, 60),
        Note(False, 0, 17.0, 18.0, 62),
    ]
    active, reentry, valid = boundary_state_targets(
        tokenizer,
        notes,
        boundaries=[5.0, 10.0, 50.0],
        track_duration=60.0,
    )
    assert active[0, 0, 60]
    assert not valid[0, 0]  # still active, so this is not a re-entry state
    assert valid[1, 0] and reentry[1, 0].item() == 1  # 7 s -> 5-10 s
    assert not valid[2, 0]  # only 10 s of annotated future: censored


def test_reentry_bins_use_instrument_groups_and_activity_episodes():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    notes = (
        Note(False, 0, 0.0, 20.0, 60),
        Note(False, 1, 5.0, 6.0, 64),
        Note(False, 1, 25.0, 26.0, 67),
    )
    bins = _record_window_gap_bins(
        notes,
        tokenizer=tokenizer,
        num_windows=1,
        context_duration=40.0,
        segment_duration=5.0,
    )
    assert bins == ["5-10"]  # same piano group; 25 - activity end 20 = 5 s


def test_reentry_boundary_classes_are_half_open_except_final_horizon():
    tokenizer = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    _, reentry, valid = boundary_state_targets(
        tokenizer,
        [
            Note(False, 0, 0.0, 0.5, 60),
            Note(False, 0, 5.5, 6.0, 62),
        ],
        boundaries=[0.5],
        track_duration=50.0,
    )
    assert valid[0, 0]
    assert reentry[0, 0].item() == 1  # exactly 5 s belongs to [5, 10)


def test_boundary_heads_read_chunk_end_hidden_states():
    config = ModelConfig(
        dim=32,
        num_heads=4,
        num_layers=2,
        card=1393,
        backbone="local_transformer",
        local_window=64,
        local_query_chunk=16,
        use_type_embeddings=True,
        correct_class_conditioning=True,
        num_dataset_condition_classes=8,
    )
    model = _build_model(torch.device("cpu"), config)
    for conditioner in model.condition_provider.conditioners.values():
        if isinstance(conditioner, MelSpectrogramConditioner):
            conditioner.log_timing = False
    wrapper = LongContextTrainingModel(
        model,
        BoundaryStateSupervisionConfig(
            enabled=True, active_weight=0.1, reentry_weight=0.1
        ),
    )
    batch = _tiny_batch()
    outputs = wrapper(batch, 0.0, 1.0, 1.0)
    assert outputs[3].shape == (1, 1, 37, 128)
    assert outputs[4].shape == (1, 1, 37, 5)
