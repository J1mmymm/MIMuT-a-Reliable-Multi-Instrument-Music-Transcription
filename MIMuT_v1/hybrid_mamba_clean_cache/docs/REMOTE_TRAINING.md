# 2 + 1 × 96 GB Linux training checklist

The local/macOS test path exercises the tokenizer, data, state, attention
fallback, CLI, and legacy regressions. Official Mamba2 kernels, Blackwell
FlashAttention, DDP/FSDP, and memory measurements must be accepted on the
Linux training host.

## Environment

For RTX PRO 6000 Blackwell (SM120), use a PyTorch build linked against CUDA
12.8 or newer. In a fresh environment:

```bash
python -m pip install -U pip setuptools wheel ninja packaging
python -m pip install -e '.[train]'
python -m pip install causal-conv1d==1.6.2.post1 --no-build-isolation
python -m pip install mamba-ssm==2.3.2.post1 --no-build-isolation
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=12.0 \
  python -m pip install flash-attn==2.8.3.post1 --no-build-isolation
```

Do not install a second PyTorch wheel indirectly after selecting the
CUDA-compatible build. The pure PyTorch local-attention fallback remains
available for tests, but the reported efficiency experiment must show a
non-null `flash_attn` version in the acceptance report.

Compile and exercise forward, backward, retained-state multi-token prefixes,
streaming equivalence, cache bounds, and the official Mamba parameter layout:

```bash
muscriptor-check-cuda --output runs/cuda_acceptance.json
```

## Data and smoke test

```bash
muscriptor-data build /datasets --output manifests/all.jsonl --strict
muscriptor-data check manifests/all.jsonl
cp configs/smoke.yaml /tmp/smoke.yaml
# Set manifest/output_dir in /tmp/smoke.yaml.
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m muscriptor.training.train --config /tmp/smoke.yaml
```

The smoke run must complete both 5-second and 40-second stages, save
safetensors plus trainer state, and resume from `latest.json` without changing
the manifest hash or taxonomy.

## Main runs

```bash
scripts/train_2x96g.sh configs/main_312m.yaml
scripts/train_2x96g.sh configs/main_312m_transformer.yaml
```

Run the four 106M ablations using `configs/ablation_106m*.yaml`. The optional
large model uses `configs/large_1_39b.yaml`, which selects FSDP full-shard.

## Evaluation

For each checkpoint, produce the 2x2 state/prelude factorial in audio-only
mode. Oracle instruments are a separately labelled upper bound only:

```bash
muscriptor-eval predict -m CHECKPOINT --manifest manifests/all.jsonl \
  -o predictions/reset_no_prelude --no-oracle-instruments \
  --long-context reset --no-prelude-forcing
muscriptor-eval predict -m CHECKPOINT --manifest manifests/all.jsonl \
  -o predictions/carry_no_prelude --no-oracle-instruments \
  --long-context carry --no-prelude-forcing
muscriptor-eval predict -m CHECKPOINT --manifest manifests/all.jsonl \
  -o predictions/reset_prelude --no-oracle-instruments \
  --long-context reset --prelude-forcing
muscriptor-eval predict -m CHECKPOINT --manifest manifests/all.jsonl \
  -o predictions/carry_prelude --no-oracle-instruments \
  --long-context carry --prelude-forcing
muscriptor-eval compare --manifest manifests/all.jsonl \
  --predictions predictions/carry_no_prelude -o metrics/carry_no_prelude.json
```

YourMT3+ and official MuScriptor exports can use either
`predictions/<dataset>/<track_id>.mid` or standardized `.notes.json`. Benchmark
5/10/20/40/80/160 seconds with:

```bash
muscriptor-eval benchmark -m CHECKPOINT --audio example.wav \
  -o metrics/efficiency_carry.json --long-context carry
```

The result separates cold and stable runs and includes real-time factor, audio
throughput, allocated/reserved peak CUDA memory, persistent-state bytes, and
local K/V lengths. State bytes and K/V lengths must plateau rather than grow
with song duration. See `MIMUT_AAAI_PROTOCOL.md` for completeness gates,
paired bootstrap, and claim rules.
