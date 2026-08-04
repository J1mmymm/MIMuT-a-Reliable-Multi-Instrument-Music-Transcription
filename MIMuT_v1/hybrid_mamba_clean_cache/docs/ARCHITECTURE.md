# Hybrid-Mamba architecture

## Sequence layout

Audio remains split into five-second, 16 kHz blocks. Training packs 1, 2, 4,
or 8 contiguous blocks into one causal sequence:

```text
[chunk marker, 501 mel frames, NULL instrument, NULL dataset, BOS, MIDI]
```

Every block keeps its own EOS and tie prelude. Notes crossing a boundary have
an explicit offset in the left/right block according to the half-open onset
rule and are restated in the next block's tie section. Chunk/type embeddings
distinguish the four modalities without adding symbols to MT3_FULL_PLUS.
Cross-entropy is masked everywhere except MIDI target positions.

Formal MIMuT training is audio-only: audio dropout is zero, while instrument
and dataset condition dropout are one. The class-conditioner modules remain in
the parameter-matched architecture and checkpoint format, but their inputs are
always the null class. Metadata-conditioned runs are oracle diagnostics, not
main results.

## Backbone

`hybrid_mamba` repeats:

```text
Mamba2 -> Mamba2 -> local causal attention
```

Each mixer has pre-norm and a residual `D -> 4D -> D` GELU feed-forward block.
Mamba2 is provided by the official `mamba-ssm` package with `d_state=128`,
`d_conv=4`, `expand=2`, and `headdim=64`. Local attention has a fixed 2048
token receptive window. FlashAttention supplies the CUDA sliding-window path;
chunked PyTorch SDPA is the correctness/testing fallback. Hybrid and
`local_transformer` never instantiate full-sequence attention.

The v2 experiment uses rotary position embeddings on local-attention Q/K and
does not add an absolute sinusoid to Mamba inputs. Cached keys are stored after
rotation, while the streamed absolute offset is used only to construct the
new Q/K phases. Legacy configurations default to the original sinusoidal path.

The presets and parameter-matched full-attention controls are:

| Scale | Hybrid | Estimated parameters | Matched Transformer |
|---|---:|---:|---:|
| small | D=768, L=13 | 108.1M | D=768, L=15, 109.6M |
| medium | D=1024, L=22 | 317.5M | D=1024, L=25, 319.2M |
| large | D=1536, L=44 | 1.409B | D=1536, L=49, 1.394B |

The estimates include mel, instrument, and dataset conditioners, token
embeddings, and the logits head. `muscriptor-check-cuda` verifies the formula
against the installed official Mamba2 implementation.

## Streaming

Training processes up to 40 seconds in parallel so gradients cross all eight
blocks. Inference processes blocks sequentially:

- Mamba convolution and SSM states persist for the entire song.
- Local-attention K/V state retains no more than 2048 tokens.
- The state is cleared only at a new song.
- A new block's multi-token condition prefix uses a vectorized causal-conv +
  SSD scan with both cached conv and SSM initial states. `step` remains the
  unavailable-kernel fallback and the autoregressive decoding path.
- Beam cache updates use the shared `reorder_state(indices)` interface.

Use `muscriptor transcribe --long-context carry` for Hybrid-Mamba and
`--long-context reset` for the required state ablation. `auto` carries Hybrid
state and preserves reset behavior for released Transformer checkpoints.

## Optional boundary-state supervision

The pre-registered candidate reads normalized hidden states at every packed
chunk end. Training-only heads predict the 37x128 active-note set and a
five-way re-entry target for each instrument group. Track tails without a full
40-second observed future are censored. The heads are stored only in trainer
state for exact resume and are omitted from inference safetensors.

## Compatibility

Released MuScriptor Transformer state-dict names and class-conditioning
behavior are unchanged. New checkpoints add a flat `config.json` understood by
the same loader, `tokenizer.json`, `taxonomy.json`, and safetensors metadata.
Training checkpoints additionally record optimizer/scheduler/RNG state and the
manifest SHA-256 for deterministic resumption. V2 signatures also include the
context distribution, LR floor, position/prefill modes, condition contract,
teacher provenance, and augmentation catalog hash. Changing a legacy shared
condition dropout to the formal three-way audio-only policy changes the
training signature, so an old conditional run cannot be resumed silently as a
formal MIMuT run.
