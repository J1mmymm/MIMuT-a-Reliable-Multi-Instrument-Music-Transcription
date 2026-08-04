from pathlib import Path
import random

import numpy as np
import pytest
import torch

from muscriptor.data.augmentation import (
    AugmentationCatalogEntry,
    validate_augmentation_catalog,
)
from muscriptor.data.dataset import (
    BalancedWindowBatchSampler,
    TrainingBatch,
    TrainingExample,
    collate_training_examples,
)
from muscriptor.data.manifest import ManifestRecord
from muscriptor.models.training import _condition_attributes
from muscriptor.evaluation.metrics import evaluate_notes
from muscriptor.tokenizer.notes import Note
from muscriptor.training.config import load_experiment_config
from muscriptor.training.train import (
    _EMPTY_SOURCE_SEQUENCE_SHA256,
    _assert_new_run_output_is_empty,
    _context_counts_before,
    _context_for_stage_step,
    _distillation_loss,
    _scheduler_lambda,
    _restore_resume_rng,
    _tokenizer_contract,
    _update_source_sequence_hash,
)
from muscriptor.tokenizer.mt3 import MT3Tokenizer
from dataclasses import replace


def test_v2_main_curriculum_and_lr_floor_are_frozen():
    config = load_experiment_config("configs/main_312m_v2.yaml")
    assert sum(stage.steps for stage in config.stages) == 280_000
    assert config.stages[0].context_probabilities() == {1: 1.0}
    assert config.stages[1].context_probabilities() == {
        1: 0.50,
        2: 0.25,
        4: 0.15,
        8: 0.10,
    }
    assert config.stages[2].context_probabilities() == {
        1: 0.40,
        2: 0.25,
        4: 0.20,
        8: 0.15,
    }
    assert config.stages[3].context_probabilities() == {
        1: 0.10,
        2: 0.20,
        4: 0.30,
        8: 0.40,
    }
    assert config.optimizer.min_lr_ratio == 0.05
    assert config.model.position_encoding == "rope"
    assert config.model.prefill_mode == "auto"
    assert config.model.learned_null_conditioning is True
    assert config.conditioning.mode == "audio_only"


def test_v2_backbone_controls_share_identical_curriculum_and_optimizer():
    for prefix in ("main_312m", "ablation_106m"):
        baseline = load_experiment_config(f"configs/{prefix}_v2.yaml")
        expected_stages = [
            (
                stage.name,
                stage.steps,
                stage.context_probabilities(),
                stage.datasets,
                stage.dataset_weights,
            )
            for stage in baseline.stages
        ]
        for suffix in ("local_transformer", "pure_mamba", "transformer"):
            control = load_experiment_config(
                f"configs/{prefix}_v2_{suffix}.yaml"
            )
            assert control.optimizer == baseline.optimizer
            assert [
                (
                    stage.name,
                    stage.steps,
                    stage.context_probabilities(),
                    stage.datasets,
                    stage.dataset_weights,
                )
                for stage in control.stages
            ] == expected_stages


def test_v2_mechanism_controls_change_only_the_declared_axis():
    baseline = load_experiment_config("configs/ablation_106m_v2.yaml")

    fixed = load_experiment_config("configs/ablation_106m_v2_5s_only.yaml")
    assert fixed.optimizer == baseline.optimizer
    assert fixed.model == baseline.model
    assert fixed.conditioning == baseline.conditioning
    assert len(fixed.stages) == len(baseline.stages)
    for fixed_stage, baseline_stage in zip(fixed.stages, baseline.stages):
        assert (
            fixed_stage.name,
            fixed_stage.steps,
            fixed_stage.datasets,
            fixed_stage.dataset_weights,
            fixed_stage.augmentation,
        ) == (
            baseline_stage.name,
            baseline_stage.steps,
            baseline_stage.datasets,
            baseline_stage.dataset_weights,
            baseline_stage.augmentation,
        )
        assert fixed_stage.context_probabilities() == {1: 1.0}

    for window in (1024, 4096):
        control = load_experiment_config(
            f"configs/ablation_106m_v2_window_{window}.yaml"
        )
        assert control.optimizer == baseline.optimizer
        assert control.stages == baseline.stages
        assert replace(control.model, local_window=baseline.model.local_window) == (
            baseline.model
        )


def test_context_draw_is_stateless_and_resume_counts_match_suffix():
    stage = load_experiment_config("configs/main_312m_v2.yaml").stages[2]
    sequence = [
        _context_for_stage_step(stage, seed=3407, stage_step=step)
        for step in range(1000)
    ]
    resumed = [
        _context_for_stage_step(stage, seed=3407, stage_step=step)
        for step in range(437, 1000)
    ]
    assert resumed == sequence[437:]
    counts = _context_counts_before(stage, seed=3407, stage_step=437)
    assert counts == {chunks: sequence[:437].count(chunks) for chunks in counts}


def test_context_draw_tracks_declared_distribution_and_is_rank_independent():
    stage = load_experiment_config("configs/main_312m_v2.yaml").stages[3]
    draws = [
        _context_for_stage_step(stage, seed=3407, stage_step=step)
        for step in range(20_000)
    ]
    expected = stage.context_probabilities()
    for chunks, probability in expected.items():
        assert draws.count(chunks) / len(draws) == pytest.approx(
            probability, abs=0.012
        )
    # No rank enters the stateless key, so every DDP worker chooses the same
    # context before drawing its own examples.
    assert draws[:100] == [
        _context_for_stage_step(stage, seed=3407, stage_step=step)
        for step in range(100)
    ]


def test_cosine_schedule_retains_five_percent_floor():
    assert _scheduler_lambda(
        280_000,
        warmup=2_000,
        total_steps=280_000,
        min_lr_ratio=0.05,
    ) == pytest.approx(0.05)
    assert _scheduler_lambda(
        140_000,
        warmup=2_000,
        total_steps=280_000,
        min_lr_ratio=0.05,
    ) > 0.05


def test_distillation_vocab_contract_fails_closed():
    student = torch.randn(1, 2, 5)
    labels = torch.tensor([[1, -100]])
    teacher = torch.randn(1, 5)
    assert torch.isfinite(
        _distillation_loss(
            student,
            labels,
            teacher,
            temperature=2.0,
            shared_vocab_size=5,
        )
    )
    with pytest.raises(RuntimeError, match="do not cover"):
        _distillation_loss(
            student,
            labels,
            teacher,
            temperature=2.0,
            shared_vocab_size=6,
        )


def test_condition_dropout_is_sampled_once_per_sequence(monkeypatch):
    batch = TrainingBatch(
        waveform=torch.ones(1, 2, 1, 16),
        target_ids=torch.zeros(1, 2, 1, dtype=torch.long),
        target_lengths=torch.ones(1, 2, dtype=torch.long),
        instrument_groups=["0"],
        dataset_ids=torch.tensor([3]),
        track_ids=["x"],
        start_times=torch.tensor([0.0]),
        has_long_gap=torch.tensor([False]),
    )
    values = iter((0.9, 0.1, 0.9))

    def fake_rand(*_args, **kwargs):
        return torch.tensor(next(values), device=kwargs.get("device"))

    monkeypatch.setattr(torch, "rand", fake_rand)
    attributes = _condition_attributes(
        batch,
        sample_rate=16_000,
        audio_condition_dropout=0.5,
        instrument_condition_dropout=0.5,
        dataset_condition_dropout=0.5,
    )
    assert len(attributes) == 2
    assert all(item.wav["self_wav"].length.item() == 16 for item in attributes)
    assert all(item.text["instrument_group"] is None for item in attributes)
    assert all(item.text["dataset_name"] == "3" for item in attributes)


def test_truncated_structural_eos_is_masked_from_loss():
    example = TrainingExample(
        waveform=torch.zeros(1, 1, 16),
        target_chunks=[[10, 11, 0]],
        instrument_group=None,
        dataset_id=0,
        track_id="dense",
        start_time=0.0,
        has_long_gap=False,
        target_loss_masks=[[True, True, False]],
        original_target_lengths=[2500],
        truncated_chunks=[True],
    )
    batch = collate_training_examples([example])
    assert batch.target_loss_mask.tolist() == [[[True, True, False]]]
    assert batch.original_target_lengths.item() == 2500
    assert batch.truncated_chunks.item() is True


def test_augmented_sampler_resume_replays_indices_and_augmentation_seeds():
    class FakeDataset:
        augmentation_enabled = True
        records = [
            type(
                "Record",
                (),
                {"dataset": "d", "duration": 60.0, "is_multi_instrument": True},
            )()
        ]
        windows = [(0, index) for index in range(20)]
        long_gap_flags = [False] * 20

    full = list(
        BalancedWindowBatchSampler(FakeDataset(), batch_size=4, num_batches=8, seed=7)
    )
    resumed = list(
        BalancedWindowBatchSampler(
            FakeDataset(),
            batch_size=4,
            num_batches=5,
            seed=7,
            start_batch=3,
        )
    )
    assert resumed == full[3:]


def test_source_sequence_audit_is_order_sensitive_and_resumeable():
    first = TrainingBatch(
        waveform=torch.zeros(1, 1, 1, 4),
        target_ids=torch.zeros(1, 1, 1, dtype=torch.long),
        target_lengths=torch.ones(1, 1, dtype=torch.long),
        instrument_groups=[None],
        dataset_ids=torch.tensor([0]),
        track_ids=["track-a"],
        start_times=torch.tensor([5.0]),
        has_long_gap=torch.tensor([False]),
    )
    second = replace(first, track_ids=["track-b"])
    prefix = _update_source_sequence_hash(
        _EMPTY_SOURCE_SEQUENCE_SHA256,
        first,
        stage_index=1,
        stage_step=7,
        context_chunks=4,
    )
    resumed = _update_source_sequence_hash(
        prefix,
        second,
        stage_index=1,
        stage_step=8,
        context_chunks=4,
    )
    continuous = _update_source_sequence_hash(
        _update_source_sequence_hash(
            _EMPTY_SOURCE_SEQUENCE_SHA256,
            first,
            stage_index=1,
            stage_step=7,
            context_chunks=4,
        ),
        second,
        stage_index=1,
        stage_step=8,
        context_chunks=4,
    )
    swapped = _update_source_sequence_hash(
        _update_source_sequence_hash(
            _EMPTY_SOURCE_SEQUENCE_SHA256,
            second,
            stage_index=1,
            stage_step=7,
            context_chunks=4,
        ),
        first,
        stage_index=1,
        stage_step=8,
        context_chunks=4,
    )
    assert resumed == continuous
    assert swapped != continuous


def test_per_rank_rng_restore_replays_condition_sampling():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    state = {
        "rng_by_rank": [
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": None,
            }
        ]
    }
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    _restore_resume_rng(state, rank=0, device=torch.device("cpu"))
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected


def test_new_run_refuses_a_nonempty_output_directory(tmp_path: Path):
    config = load_experiment_config("configs/main_312m_v2.yaml")
    output = tmp_path / "run"
    output.mkdir()
    (output / "old-checkpoint").write_text("occupied")
    with pytest.raises(RuntimeError, match="non-empty output"):
        _assert_new_run_output_is_empty(
            replace(config, output_dir=str(output), resume=None)
        )


def test_legacy_config_defaults_model_protocol_and_infers_old_conditioning(tmp_path: Path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
name: legacy
manifest: manifest.jsonl
output_dir: run
model: {dim: 16, num_heads: 4, num_layers: 1, card: 64}
stages:
  - {name: fixed, steps: 2, context_chunks: 1}
condition_dropout: 0.2
""".strip()
    )
    config = load_experiment_config(path)
    assert config.model.position_encoding == "sinusoidal"
    assert config.model.prefill_mode == "step"
    assert config.model.learned_null_conditioning is False
    assert config.stages[0].context_probabilities() == {1: 1.0}
    assert config.conditioning.mode == "conditioned"


def test_tokenizer_contract_fingerprints_exact_event_mapping():
    config = load_experiment_config("configs/main_312m_v2.yaml")
    student = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001
    )
    changed = MT3Tokenizer(
        instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1000
    )
    assert _tokenizer_contract(student, config.model) != _tokenizer_contract(
        changed, config.model
    )


def test_augmentation_catalog_rejects_nontrain_lineage(tmp_path: Path):
    entry = AugmentationCatalogEntry(
        track_id="source",
        dataset="maestro",
        split="train",
        audio_path=str(tmp_path / "source.wav"),
        notes_path=str(tmp_path / "source.notes.json"),
        duration=60.0,
        instrument_groups=[0],
        group_id="piece",
        role="isolated_complete",
        is_drum_only=False,
        lineage=["maestro:piece"],
    )
    manifest = [
        ManifestRecord(
            track_id="heldout",
            dataset="maestro",
            split="test",
            audio_path="heldout.wav",
            notes_path="heldout.notes.json",
            duration=60.0,
            instrument_groups=[0],
            group_id="piece",
            is_multi_instrument=False,
        )
    ]
    with pytest.raises(ValueError, match="split leakage"):
        validate_augmentation_catalog([entry], manifest, check_files=False)


def test_official_mir_eval_and_elapsed_strata_are_reported():
    pytest.importorskip("mir_eval")
    notes = [Note(False, 0, 41.0, 42.0, 60)]
    result = evaluate_notes(notes, list(notes))
    assert result.official_mir_eval["available"] is True
    assert result.official_mir_eval["onset_offset"]["f1"] == pytest.approx(1.0)
    assert result.elapsed_time_strata["40-80"]["reference_notes"] == 1
