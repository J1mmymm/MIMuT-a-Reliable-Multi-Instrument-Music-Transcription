import json

import pytest
import soundfile as sf

from muscriptor.data.dataset import BalancedWindowBatchSampler
from muscriptor.data.manifest import (
    ManifestRecord,
    canonicalize_provenance_groups,
    discover_standardized_dataset,
    read_manifest,
    validate_no_leakage,
    write_manifest,
)
from muscriptor.evaluation.metrics import evaluate_notes
from muscriptor.data.urmp import discover_urmp
from muscriptor.evaluation.io import load_notes
from muscriptor.tokenizer.notes import Note


def test_manifest_discovers_audio_and_provenance_group(tmp_path):
    root = tmp_path / "choralebricks"
    audio = root / "audio" / "mixtures"
    notes = root / "standardized_notes" / "mixtures"
    audio.mkdir(parents=True)
    notes.mkdir(parents=True)
    sf.write(audio / "piece.wav", [0.0] * 16000, 16000)
    (notes / "piece.notes.json").write_text(
        json.dumps(
            {
                "notes": [
                    {
                        "onset": 0.1,
                        "offset": 0.5,
                        "pitch": 60,
                        "program": 40,
                        "is_drum": False,
                        "velocity": 64,
                    }
                ],
                "provenance": {"component_track_ids": ["b", "a"]},
            }
        )
    )
    records, issues = discover_standardized_dataset(root)
    assert not issues
    assert len(records) == 1
    assert records[0].group_id == "a|b"
    assert records[0].duration == 1.0


def test_urmp_adapter_uses_aligned_note_annotations_and_test_split(tmp_path):
    root = tmp_path / "urmp"
    piece = root / "urmp_yourmt3_16k" / "01_Jupiter_vn_vc"
    piece.mkdir(parents=True)
    sf.write(piece / "AuMix_01_Jupiter_vn_vc.wav", [0.0] * 16000, 16000)
    (piece / "Notes_1_vn_01_Jupiter.txt").write_text("0.1 440.0 0.4\n")
    (piece / "Notes_2_vc_01_Jupiter.txt").write_text("0.2 220.0 0.5\n")
    records, issues = discover_urmp(root, tmp_path / "cache")
    assert not issues
    assert len(records) == 1
    assert records[0].split == "test"
    loaded = load_notes(records[0].notes_path)
    assert {note.pitch for note in loaded} == {57, 69}


def test_leakage_check_rejects_group_in_two_splits():
    base = dict(
        track_id="x",
        dataset="d",
        audio_path="a",
        notes_path="n",
        duration=1.0,
        instrument_groups=[0],
        group_id="same",
        is_multi_instrument=False,
    )
    records = [
        ManifestRecord(split="train", **base),
        ManifestRecord(split="test", **{**base, "track_id": "y"}),
    ]
    try:
        validate_no_leakage(records)
    except ValueError as exc:
        assert "split leakage" in str(exc)
    else:
        raise AssertionError("expected split leakage to be rejected")


def test_component_overlap_is_collapsed_before_splitting():
    def record(track_id: str, group_id: str) -> ManifestRecord:
        return ManifestRecord(
            track_id=track_id,
            dataset="d",
            split="train",
            audio_path=f"/data/{track_id}.wav",
            notes_path=f"/data/{track_id}.notes.json",
            duration=1.0,
            instrument_groups=[0],
            group_id=group_id,
            is_multi_instrument="|" in group_id,
        )

    records = canonicalize_provenance_groups(
        [record("mix", "piece/a|piece/b"), record("stem", "piece/b")]
    )
    assert records[0].group_id == records[1].group_id
    assert records[0].split == records[1].split


def test_manifest_serializes_canonical_audio_notes_keys(tmp_path):
    record = ManifestRecord(
        track_id="x",
        dataset="d",
        split="train",
        audio_path="/a.wav",
        notes_path="/n.json",
        duration=1.0,
        instrument_groups=[3],
        group_id="x",
        is_multi_instrument=False,
    )
    path = tmp_path / "manifest.jsonl"
    write_manifest([record], path)
    raw = json.loads(path.read_text())
    assert raw["audio"] == "/a.wav"
    assert raw["notes"] == "/n.json"
    assert raw["instrument_classes"] == [3]
    assert read_manifest(path) == [record]


def test_long_metrics_separate_accuracy_from_consistency():
    reference = [
        Note(False, 40, 0.0, 0.5, 60),
        Note(False, 40, 12.0, 12.5, 62),
        Note(False, 40, 25.0, 25.5, 64),
    ]
    perfect = evaluate_notes(reference, list(reference))
    assert perfect.multi.f1 == 1.0
    assert perfect.instrument_attribution_accuracy == 1.0
    assert perfect.instrument_switch_error_rate == 0.0
    wrong = [Note(False, 56, note.onset, note.offset, note.pitch) for note in reference]
    result = evaluate_notes(reference, wrong)
    assert result.instrument_switch_error_rate == 0.0
    assert result.instrument_attribution_accuracy == 0.0
    assert result.multi.f1 == 0.0


def test_metrics_compare_mt3_instrument_groups_not_raw_gm_programs():
    reference = [Note(False, 1, 0.0, 0.5, 60)]
    grouped_prediction = [Note(False, 0, 0.0, 0.5, 60)]
    result = evaluate_notes(reference, grouped_prediction)
    assert result.multi.f1 == 1.0
    assert result.instrument_attribution_accuracy == 1.0


def test_program_aware_frame_metric_detects_wrong_instrument():
    reference = [Note(False, 40, 0.0, 1.0, 60)]
    prediction = [Note(False, 56, 0.0, 1.0, 60)]
    result = evaluate_notes(reference, prediction)
    assert result.frame.f1 == 1.0
    assert result.program_frame.f1 == 0.0


def test_reentry_metric_counts_missed_notes_in_recall_denominator():
    reference = [
        Note(False, 0, 0.0, 0.5, 60),
        Note(False, 0, 12.0, 12.5, 62),
    ]
    missed_reentry = [reference[0]]
    result = evaluate_notes(reference, missed_reentry)
    metric = result.long_gap_reidentification["10-20"]
    assert metric.reference == 1
    assert metric.predicted == 0
    assert metric.recall == 0.0
    assert metric.f1 == 0.0
    assert result.long_gap_reidentification["10+"].reference == 1
    assert result.long_gap_reidentification["10+"].recall == 0.0
    perfect = evaluate_notes(reference, list(reference))
    assert perfect.long_gap_reidentification["10-20"].f1 == 1.0
    assert perfect.long_gap_reidentification["10+"].f1 == 1.0


def test_boundary_error_metrics_separate_truncation_and_duplication():
    reference = [Note(False, 0, 4.0, 6.0, 60)]
    prediction = [
        Note(False, 0, 4.0, 5.0, 60),
        Note(False, 0, 5.01, 6.0, 60),
    ]
    result = evaluate_notes(reference, prediction)
    assert result.boundary_errors["omission"].count == 0
    assert result.boundary_errors["truncation"].count == 1
    assert result.boundary_errors["duplication"].count == 1


def test_boundary_strict_metric_includes_offsets_near_chunk_boundaries():
    reference = [Note(False, 0, 1.0, 5.0, 60)]
    result = evaluate_notes(reference, list(reference))
    assert result.boundary_multi.reference == 1
    assert result.boundary_multi.f1 == 1.0


def test_sampler_uses_sqrt_duration_with_probability_cap():
    class FakeDataset:
        records = [
            type(
                "Record",
                (),
                {
                    "dataset": dataset,
                    "duration": duration,
                    "is_multi_instrument": False,
                },
            )()
            for dataset, duration in (
                ("a", 3_600.0),
                ("b", 14_400.0),
                ("c", 32_400.0),
            )
        ]
        windows = [(0, 0), (1, 0), (2, 0)]
        long_gap_flags = [False, False, False]
        long_gap_bins = ["none", "none", "none"]

    sampler = BalancedWindowBatchSampler(
        FakeDataset(), batch_size=3, num_batches=1, seed=1
    )
    assert sampler.weights.tolist() == pytest.approx([0.30, 0.35, 0.35])


def test_sampler_reserves_at_least_half_a_batch_for_multi_instrument_windows():
    class FakeDataset:
        records = [
            type(
                "Record",
                (),
                {
                    "dataset": "d",
                    "duration": 60.0,
                    "is_multi_instrument": is_multi,
                },
            )()
            for is_multi in (True, False)
        ]
        windows = [(0, 0), (1, 0)]
        long_gap_flags = [False, False]
        long_gap_bins = ["none", "none"]

    sampler = BalancedWindowBatchSampler(
        FakeDataset(), batch_size=4, num_batches=5, seed=2
    )
    for batch in sampler:
        assert sum(index == 0 for index in batch) >= 2


def test_batch_sampler_resume_is_exact_suffix():
    class FakeDataset:
        records = [
            type(
                "Record",
                (),
                {
                    "dataset": "d",
                    "duration": 60.0,
                    "is_multi_instrument": True,
                },
            )()
        ]
        windows = [(0, index) for index in range(20)]
        long_gap_flags = [index % 3 == 0 for index in range(20)]

    full = list(
        BalancedWindowBatchSampler(FakeDataset(), batch_size=4, num_batches=8, seed=123)
    )
    resumed = list(
        BalancedWindowBatchSampler(
            FakeDataset(),
            batch_size=4,
            num_batches=5,
            seed=123,
            start_batch=3,
        )
    )
    assert resumed == full[3:]
