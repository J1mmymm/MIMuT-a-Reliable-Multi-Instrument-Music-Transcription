"""Tests for muscriptor/modules/conditioners.py — CPU only."""

import torch

from muscriptor.modules.conditioners import (
    ConditioningAttributes,
    WavCondition,
    ConditioningProvider,
    MelSpectrogramConditioner,
    ClassConditioner,
    nullify_all_conditions,
    nullify_wav,
)


# ---------------------------------------------------------------------------
# ConditioningAttributes
# ---------------------------------------------------------------------------


def test_conditioning_attributes_defaults():
    ca = ConditioningAttributes()
    assert ca.text == {}
    assert ca.wav == {}


def test_conditioning_attributes_getitem():
    ca = ConditioningAttributes(text={"prompt": "hello"})
    assert ca["text"] == {"prompt": "hello"}


def test_conditioning_attributes_properties():
    wav = WavCondition(
        wav=torch.zeros(1, 1, 1600),
        length=torch.tensor([1600]),
        sample_rate=[16000],
    )
    ca = ConditioningAttributes(
        text={"t": "x"},
        wav={"audio": wav},
    )
    assert "t" in ca.text_attributes
    assert "audio" in ca.wav_attributes


# ---------------------------------------------------------------------------
# nullify_wav
# ---------------------------------------------------------------------------


def test_nullify_wav_zeros_waveform():
    wav = WavCondition(
        wav=torch.randn(1, 1, 1600),
        length=torch.tensor([1600]),
        sample_rate=[16000],
    )
    null = nullify_wav(wav)
    assert null.wav.abs().sum() == 0.0
    assert null.length.item() == 0


# ---------------------------------------------------------------------------
# MelSpectrogramConditioner
# ---------------------------------------------------------------------------


def _make_mel_conditioner(output_dim=32):
    return MelSpectrogramConditioner(
        output_dim=output_dim,
        device="cpu",
        sample_rate=16000,
        n_fft=256,
        frame_rate=100,
        n_mel_bins=64,
    )


def test_mel_conditioner_output_shape():
    cond = _make_mel_conditioner(output_dim=32)
    wav_len = 16000  # 1 second at 16kHz
    wav = WavCondition(
        wav=torch.randn(1, 1, wav_len),
        length=torch.tensor([wav_len]),
        sample_rate=[16000],
    )
    tokens = cond.tokenize(wav)
    embed, mask = cond(tokens)
    assert embed.shape[-1] == 32
    assert embed.shape[0] == 1
    assert mask.shape[0] == 1


def test_mel_conditioner_mask_dtype():
    cond = _make_mel_conditioner()
    wav = WavCondition(
        wav=torch.randn(1, 1, 8000),
        length=torch.tensor([8000]),
        sample_rate=[16000],
    )
    _, mask = cond(cond.tokenize(wav))
    assert mask.dtype in (torch.bool, torch.float32, torch.int32)


# ---------------------------------------------------------------------------
# ClassConditioner
# ---------------------------------------------------------------------------


def _make_class_conditioner(num_classes=10, output_dim=16):
    return ClassConditioner(
        num_classes=num_classes, output_dim=output_dim, device="cpu"
    )


def test_class_conditioner_output_shape():
    cond = _make_class_conditioner(num_classes=10, output_dim=16)
    tokens = cond.tokenize(["3", "7"])
    embed, mask = cond(tokens)
    assert embed.shape == (2, 1, 16)
    assert mask.shape == (2, 1)


def test_class_conditioner_none_input():
    cond = _make_class_conditioner(num_classes=10, output_dim=16)
    tokens = cond.tokenize([None, None])
    embed, mask = cond(tokens)
    assert embed.shape[0] == 2


def test_correct_class_conditioner_learns_null_but_masks_sequence_padding():
    cond = ClassConditioner(
        num_classes=4,
        output_dim=3,
        device="cpu",
        correct_padding=True,
        learned_null=True,
    )
    tokens = cond.tokenize([None, "0 2"])
    assert tokens.tolist() == [[1, 0], [2, 4]]
    embed, mask = cond(tokens)
    assert mask.tolist() == [[1, 0], [1, 1]]
    embed.sum().backward()
    assert cond.embed.weight.grad[cond.null_idx].abs().sum() > 0
    assert cond.embed.weight.grad[cond.pad_idx].abs().sum() == 0


# ---------------------------------------------------------------------------
# ConditioningProvider
# ---------------------------------------------------------------------------


def test_conditioning_provider_tokenize_and_forward():
    mel = _make_mel_conditioner(output_dim=16)
    provider = ConditioningProvider({"audio": mel}, device="cpu")

    wav = WavCondition(
        wav=torch.randn(1, 1, 16000),
        length=torch.tensor([16000]),
        sample_rate=[16000],
    )
    attrs = [ConditioningAttributes(wav={"audio": wav})]
    tokenized = provider.tokenize(attrs)
    conditions = provider(tokenized)
    assert "audio" in conditions
    embed, mask = conditions["audio"]
    assert embed.shape[-1] == 16


# ---------------------------------------------------------------------------
# nullify_all_conditions
# ---------------------------------------------------------------------------


def test_nullify_all_conditions():
    wav = WavCondition(
        wav=torch.randn(1, 1, 1600),
        length=torch.tensor([1600]),
        sample_rate=[16000],
    )
    attrs = [ConditioningAttributes(wav={"audio": wav}, text={"prompt": "hi"})]
    result = nullify_all_conditions(attrs)
    assert result[0].wav["audio"].length.item() == 0
    assert result[0].text["prompt"] is None
    # original is untouched (deepcopy)
    assert attrs[0].wav["audio"].length.item() == 1600
