# Main 312M context-mix/RoPE v2

This implementation is a new experiment family. It must not resume a v1 or
unsigned checkpoint. No formal quality, throughput, or SOTA result is claimed
until the registered pilots and whole-track evaluation complete.

## Core configuration

`configs/main_312m_v2.yaml` fixes the 280k-step curriculum to:

- 100k Slakh 5-second steps;
- 50k Slakh context mix, 1/2/4/8 chunks = 50/25/15/10%;
- 100k real-data context mix = 40/25/20/15%;
- 30k long-context focus = 10/20/30/40%.

The global audio budget remains 320 seconds per optimizer step. Warmup is
2,000 steps and the single cosine schedule bottoms out at 5% of the base LR.
The formal condition is explicitly `audio_only`: audio is retained and both
metadata conditions are NULL in training and inference. V2 reserves a learned
NULL class distinct from sequence padding, so conditioned-ablation dropout and
the main audio-only contract both optimize the same inference-time embedding;
legacy model configs retain their original class tables. RoPE is applied only
inside bounded local attention; Mamba layers use recurrent order. Cached
multi-token prefixes use vectorized convolution plus SSD scan when the pinned
kernels are present, with `step` as the unavailable-kernel fallback.

## Strict augmentation catalog

Build and validate the train-only catalog before enabling an augmentation
config:

```bash
muscriptor-data augmentation-build \
  --manifest /path/to/manifests/all.jsonl \
  --slakh-root /path/to/data/slakh2100_redux \
  --cache-root /path/to/augmentation_cache/slakh_stems \
  --output /path/to/manifests/augmentation.jsonl

muscriptor-data augmentation-check \
  /path/to/manifests/augmentation.jsonl \
  --manifest /path/to/manifests/all.jsonl \
  --required-duration 40
```

The validator rejects non-train or unregistered lineage, non-whitelisted
datasets, mixtures, missing files, invalid labels, and catalogs without two
datasets plus four distinct groups supporting the maximum context. Each
augmentation seed is a hash of the global sampler seed, absolute distributed
sample position, and source group, so worker prefetch does not change replay
after resume.

## Frozen teacher setup

Download MuScriptor-large manually after accepting its gated access terms.
Keep `model.safetensors` and `config.json` together, then materialize a new-run
config. The helper computes and embeds the SHA256 rather than accepting an
unverified model path:

```bash
python tools/prepare_v2_experiment.py \
  --base configs/main_312m_v2.yaml \
  --output configs/generated/main_312m_v2_distill.yaml \
  --name main_312m_v2_distill_seed3407 \
  --output-dir /path/to/runs/main_312m_v2_distill_seed3407 \
  --teacher /path/to/teachers/muscriptor-large/model.safetensors \
  --teacher-revision PINNED_HF_REVISION
```

For the combined experiment use `configs/main_312m_v2_augmentation.yaml` as
the base and also pass `--augmentation-catalog`. Teacher forward uses
micro-batches of four chunks; the student batch and optimizer-step definition
are unchanged. The revision is mandatory, and runtime verification records the
weight/config hashes plus exact event-vocabulary and instrument-map hashes.

## Acceptance order

1. Run the v2 100-step single-GPU core pilot.
2. Run independent 100-step teacher and augmentation pilots.
3. Run the v2 1,000-step two-GPU preflight and resume replay check.
4. Run the registered 106M distillation, augmentation, combined, and
   2048-vs-4096 gates.
5. Start a fresh 280k Main 312M run only after all functional gates pass.

Concrete smoke bases are registered under the deployment `runtime_configs/`
directory. Materialize teacher-bearing files only after the local snapshot is
available:

```bash
# independent augmentation smoke
torchrun --standalone --nproc_per_node=1 -m muscriptor.training.train \
  --config runtime_configs/pilot_106m_ctxmix_rope_v2_augmentation_1gpu_100.yaml

# independent teacher smoke
python code/muscriptor-hybrid-mamba/tools/prepare_v2_experiment.py \
  --base runtime_configs/pilot_106m_ctxmix_rope_v2_1gpu_100.yaml \
  --output runtime_configs/generated/pilot_106m_v2_distill_1gpu_100.yaml \
  --name pilot_106m_v2_distill_1gpu_100 \
  --output-dir /path/to/runs/pilot_106m_v2_distill_1gpu_100 \
  --teacher /path/to/teachers/muscriptor-large/model.safetensors \
  --teacher-revision PINNED_HF_REVISION

# two-GPU combined preflight
python code/muscriptor-hybrid-mamba/tools/prepare_v2_experiment.py \
  --base runtime_configs/preflight_106m_ctxmix_rope_v2_augmentation_2gpu_1000.yaml \
  --output runtime_configs/generated/preflight_106m_v2_combined_2gpu_1000.yaml \
  --name preflight_106m_v2_combined_2gpu_1000 \
  --output-dir /path/to/runs/preflight_106m_v2_combined_2gpu_1000 \
  --teacher /path/to/teachers/muscriptor-large/model.safetensors \
  --teacher-revision PINNED_HF_REVISION \
  --augmentation-catalog /path/to/manifests/augmentation.jsonl
```

Fresh runs refuse non-empty output directories. Checkpoints retain per-context
draw counts and a per-rank source-sequence SHA256 chain; compare the final
chains between uninterrupted and resumed pilots. The validator also checks
finite CE/KL/gradient norm, positive KL, teacher freeze, and optimizer steps:

```bash
python tools/validate_v2_pilot.py --run runs/resumed_pilot \
  --continuous-reference runs/continuous_pilot
```

Evaluation now includes official `mir_eval` pitched-note onset/onset-offset
scores, custom whole-track long-horizon metrics, 40/80/160-second elapsed-time
strata with pooled support counts and separate whole-track bootstrap CIs,
dense-chunk support, and separate prefix/decode timing. Formal prediction and
benchmark commands default to `prelude_forcing=false`.

For the optimized prefill gate, create same-weight step/auto hardlinked
snapshots, benchmark both on the same audio and hardware, then compare:

```bash
python tools/prepare_prefill_benchmark.py \
  --checkpoint runs/model/checkpoint.safetensors \
  --output-root runs/prefill_gate_snapshots
muscriptor-eval benchmark -m runs/prefill_gate_snapshots/step/checkpoint.safetensors \
  --audio example.wav -o runs/prefill_step.json --lengths 10,20,40,80
muscriptor-eval benchmark -m runs/prefill_gate_snapshots/auto/checkpoint.safetensors \
  --audio example.wav -o runs/prefill_auto.json --lengths 10,20,40,80
muscriptor-eval compare-efficiency --baseline runs/prefill_step.json \
  --candidate runs/prefill_auto.json -o runs/prefill_gate.json
```

The comparison gate requires identical weight SHA256 and decoded token-stream
SHA256, then checks median carried-prefix speedup at 5/10/20/40 seconds is at
least 1.25x. If it fails, keep `prefill_mode: step` and make no prefill speed
claim.
