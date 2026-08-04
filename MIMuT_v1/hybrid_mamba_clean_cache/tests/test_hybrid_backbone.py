from pathlib import Path
from dataclasses import replace

import pytest
import torch

from muscriptor.models.config import (
    HYBRID_PRESETS,
    ModelConfig,
    estimate_total_parameters,
    matched_transformer_config,
)
from muscriptor.models.lm import LMModel
from muscriptor.modules.conditioners import ConditioningProvider
from muscriptor.modules.hybrid import (
    StreamingLocalAttention,
    StreamingMamba2,
    StreamingMixedBackbone,
)
from muscriptor.modules.streaming import (
    increment_steps,
    init_states,
    reorder_states,
    state_size_bytes,
)
from muscriptor.training.config import load_experiment_config
from tools.validate_formal_protocol import validate_matrix


def test_local_attention_streaming_matches_parallel_and_bounds_cache():
    torch.manual_seed(0)
    attention = StreamingLocalAttention(
        embed_dim=16,
        num_heads=4,
        window_size=5,
        query_chunk_size=3,
    ).eval()
    x = torch.randn(2, 13, 16)
    parallel = attention(x)

    state = init_states(attention, batch_size=2, sequence_length=100)
    streamed = []
    for index in range(x.shape[1]):
        streamed.append(attention(x[:, index : index + 1], model_state=state))
        increment_steps(attention, state)
    streamed = torch.cat(streamed, dim=1)
    torch.testing.assert_close(streamed, parallel, atol=1e-5, rtol=1e-5)
    assert state[""]["key"].shape[2] == 5
    before = state_size_bytes(state)
    reorder_states(attention, state, torch.tensor([1, 0]))
    assert state_size_bytes(state) == before


def test_local_backbone_streaming_matches_parallel():
    torch.manual_seed(1)
    config = ModelConfig(
        dim=16,
        num_heads=4,
        num_layers=3,
        card=64,
        backbone="local_transformer",
        local_window=32,
        local_query_chunk=4,
        use_type_embeddings=True,
    )
    model = StreamingMixedBackbone(config).eval()
    x = torch.randn(1, 11, 16)
    parallel = model(x)
    state = init_states(model, batch_size=1, sequence_length=100)
    pieces = []
    for size in (4, 3, 4):
        start = sum(piece.shape[1] for piece in pieces)
        value = model(x[:, start : start + size], model_state=state)
        pieces.append(value)
        increment_steps(model, state, increment=size)
    torch.testing.assert_close(torch.cat(pieces, dim=1), parallel, atol=1e-5, rtol=1e-5)


def test_rope_local_backbone_streaming_matches_parallel_at_long_offsets():
    torch.manual_seed(2)
    config = ModelConfig(
        dim=16,
        num_heads=4,
        num_layers=3,
        card=64,
        backbone="local_transformer",
        position_encoding="rope",
        local_window=32,
        local_query_chunk=4,
        use_type_embeddings=True,
    )
    model = StreamingMixedBackbone(config).eval()
    x = torch.randn(1, 12, 16)
    parallel = model(x)
    state = init_states(model, batch_size=1, sequence_length=200_000)
    streamed = []
    for start, end in ((0, 5), (5, 9), (9, 12)):
        streamed.append(model(x[:, start:end], model_state=state))
        increment_steps(model, state, increment=end - start)
    torch.testing.assert_close(
        torch.cat(streamed, dim=1), parallel, atol=1e-5, rtol=1e-5
    )

    attention = StreamingLocalAttention(
        embed_dim=16,
        num_heads=4,
        window_size=8,
        position_encoding="rope",
    ).eval()
    long_state = init_states(attention, batch_size=1, sequence_length=200_000)
    long_state[""]["offset"] = 100_000
    assert torch.isfinite(
        attention(torch.randn(1, 3, 16), model_state=long_state)
    ).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Mamba kernels")
@pytest.mark.parametrize(
    "dtype,atol,rtol",
    [
        # Chunk scan and repeated recurrent steps use different accumulation
        # orders; their FP32 continuation differs only at ~1e-4 while BF16 is
        # the production path.
        (torch.float32, 1e-4, 3e-3),
        (torch.bfloat16, 1e-2, 5e-2),
    ],
)
def test_chunk_prefill_matches_official_step_and_final_states(dtype, atol, rtol):
    pytest.importorskip("mamba_ssm")
    pytest.importorskip("causal_conv1d")
    torch.manual_seed(3)
    base = ModelConfig(
        dim=64,
        num_heads=4,
        num_layers=1,
        card=64,
        backbone="pure_mamba",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=32,
        prefill_mode="step",
    )
    step_model = StreamingMamba2(
        base, layer_idx=0, device="cuda", dtype=dtype
    ).eval()
    chunk_model = StreamingMamba2(
        replace(base, prefill_mode="chunk"),
        layer_idx=0,
        device="cuda",
        dtype=dtype,
    ).eval()
    chunk_model.load_state_dict(step_model.state_dict())
    prefix = torch.randn(1, 9, 64, device="cuda", dtype=dtype)
    continuation = torch.randn(1, 17, 64, device="cuda", dtype=dtype)
    step_state = init_states(step_model, batch_size=1, sequence_length=128)
    chunk_state = init_states(chunk_model, batch_size=1, sequence_length=128)
    with torch.no_grad():
        torch.testing.assert_close(
            step_model(prefix, model_state=step_state),
            chunk_model(prefix, model_state=chunk_state),
            atol=atol,
            rtol=rtol,
        )
        increment_steps(step_model, step_state, increment=prefix.shape[1])
        increment_steps(chunk_model, chunk_state, increment=prefix.shape[1])
        step_output = step_model(continuation, model_state=step_state)
        chunk_output = chunk_model(continuation, model_state=chunk_state)
    torch.testing.assert_close(chunk_output, step_output, atol=atol, rtol=rtol)
    head = torch.nn.Linear(64, 97, bias=False, device="cuda", dtype=dtype).eval()
    step_logits = head(step_output)
    chunk_logits = head(chunk_output)
    torch.testing.assert_close(chunk_logits, step_logits, atol=atol, rtol=rtol)
    assert torch.equal(
        chunk_logits[:, -1].argmax(dim=-1),
        step_logits[:, -1].argmax(dim=-1),
    )

    step_cache = step_state[""]["inference_params"].key_value_memory_dict[0]
    chunk_cache = chunk_state[""]["inference_params"].key_value_memory_dict[0]
    for chunk_value, step_value in zip(chunk_cache, step_cache):
        torch.testing.assert_close(chunk_value, step_value, atol=atol, rtol=rtol)


def test_hybrid_presets_hit_targets_and_matched_transformers():
    targets = {
        "hybrid-small": 106_000_000,
        "hybrid-medium": 312_000_000,
        "hybrid-large": 1_390_000_000,
    }
    for name, target in targets.items():
        hybrid_count = estimate_total_parameters(HYBRID_PRESETS[name])
        transformer_count = estimate_total_parameters(matched_transformer_config(name))
        assert abs(hybrid_count - target) / target < 0.03
        assert abs(hybrid_count - transformer_count) / transformer_count < 0.03


def test_experiment_config_inheritance_keeps_curriculum():
    config = load_experiment_config("configs/main_312m_transformer.yaml")
    assert config.model.backbone == "transformer"
    assert config.model.num_layers == 25
    assert [stage.context_chunks for stage in config.stages] == [1, 1, 2, 4, 8]
    assert config.condition_dropout.audio == 0.0
    assert config.condition_dropout.instrument == 1.0
    assert config.condition_dropout.dataset == 1.0


def test_boundary_state_candidate_config_is_explicitly_opt_in():
    base = load_experiment_config("configs/ablation_106m.yaml")
    candidate = load_experiment_config("configs/ablation_106m_bss_010.yaml")
    assert base.boundary_state_supervision.enabled is False
    assert candidate.boundary_state_supervision.enabled is True
    assert candidate.boundary_state_supervision.active_weight == 0.1
    assert candidate.boundary_state_supervision.reentry_weight == 0.1


def test_formal_matrix_is_audio_only_and_parameter_matched():
    result = validate_matrix(Path("configs/formal_experiment_matrix.yaml"))
    assert result["status"] == "ok", result["errors"]


def test_chunk_marker_advances_external_generation_state():
    config = ModelConfig(
        dim=16,
        num_heads=4,
        num_layers=1,
        card=16,
        backbone="local_transformer",
        local_window=32,
        use_type_embeddings=True,
    )
    model = LMModel(
        condition_provider=ConditioningProvider(
            conditioners={}, device=torch.device("cpu")
        ),
        card=16,
        dim=16,
        num_heads=4,
        model_config=config,
    ).eval()
    state = init_states(model, batch_size=1, sequence_length=16)
    list(
        model.generate(
            max_gen_len=3,
            num_samples=1,
            use_sampling=False,
            model_state=state,
        )
    )
    # chunk marker + BOS + two generated inputs; the final predicted token has
    # not yet been consumed by the backbone.
    assert int(state["transformer"]["offsets"].item()) == 4
