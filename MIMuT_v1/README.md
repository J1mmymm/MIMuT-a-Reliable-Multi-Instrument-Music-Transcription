# MIMuT v1 historical source archive

> [!CAUTION]
> **This directory is an archived research record, not the latest MIMuT source
> tree.** The Hybrid-Mamba and Acoustic-Mamba/Hybrid-Decoder routes preserved
> here were explored during the first project iteration, but their observed
> transcription quality did not meet the project team's expectations. They are
> retained for provenance, reproducibility, comparison, and future ablations;
> they should not be interpreted as the current recommended architecture.

## Contents

```text
MIMuT_v1/
├── hybrid_mamba_clean_cache/       # Cleaned v1 baseline source snapshot
├── supplements/
│   ├── acoustic_hybrid_mamba/      # Historical overlay: changed/new files only
│   └── mimut_clean_cache_106m_from50k_control_15k_h800_resume.yaml
├── docs/
│   └── DESIGN_HISTORY.md           # Sanitized design rationale and evidence limits
└── SOURCE_MANIFEST.md              # Inclusion, exclusion, and cleaning record
```

The Acoustic-Hybrid-Mamba directory is intentionally stored as a supplement,
not as a standalone current release. Its paths mirror the v1 baseline; applying
the supplement over `hybrid_mamba_clean_cache/` reconstructs the historical
experimental branch represented by the supplied archive.

## Historical architecture lines

### 1. Hybrid-Mamba decoder

The first line placed recurrent Mamba2 state and bounded local causal attention
inside the autoregressive event decoder:

```text
5 s audio block
  -> mel/condition tokens
  -> Mamba2 -> Mamba2 -> local causal attention
  -> MT3_FULL_PLUS event tokens
```

With runtime `carry`, the decoder retained Mamba SSM state, convolution state,
and a bounded local-attention KV cache across five-second blocks. This allowed
later blocks to depend on earlier context, but the persistent state mixed prior
audio conditioning with previously generated MIDI tokens. Prediction errors
could therefore contaminate later blocks. With `reset`, the same checkpoint was
used while runtime decoder state was cleared at each block boundary.

### 2. Acoustic-Mamba plus Hybrid-Mamba decoder

The later supplement attempted to separate long-term acoustic memory from
autoregressive decoder memory:

```text
continuous audio
  -> 6-layer Acoustic Mamba2 encoder
       acoustic state carried across 5 s blocks
  -> per-block acoustic conditions
  -> 22-layer Hybrid-Mamba decoder
       Mamba2 -> Mamba2 -> local causal attention
       decoder state reset at each block
  -> MT3_FULL_PLUS event tokens
```

The historical design used a 4,096-token local-attention window and was reported
at the time as approximately 407.4M parameters. Compatible decoder, embedding,
conditioner, normalization, and output-head parameters could be warm-started
from an earlier Hybrid-Mamba checkpoint, while the new acoustic encoder was
initialized separately. Model weights are not included in this archive.

## Carry, reset, and prelude

`carry` and `reset` are inference-state policies, not different checkpoints.
They use the same learned parameters and differ only in whether temporary Mamba
and attention states survive a five-second boundary. Tie/prelude forcing is a
separate symbolic path: it can pass still-active `(program, pitch)` notes to the
next block even when neural decoder state is reset.

The Acoustic-Mamba supplement was designed so that `carry` referred only to the
acoustic encoder state, while autoregressive decoder state reset per block. This
was intended to reduce long-range pollution from erroneous MIDI predictions.
The intended benefit was not established as a successful final model result.

## Evidence boundary

- The source and configuration snapshots are historical and may require the
  dependency versions described by the archived project metadata.
- Earlier development notes reported CPU regression and CUDA smoke acceptance,
  but those claims were not independently rerun during this publication pass.
- The available trials did not provide satisfactory model quality for the
  project's goals; no state-of-the-art or superiority claim is made.
- No dataset, manifest, checkpoint, optimizer state, prediction output, or
  private runtime file is included.
- All machine-specific paths were replaced with `/path/to/...` templates.

## Data and weights

Dataset metadata and training manifests are maintained separately in
[J1mmymm/MIMuT_Data_v2](https://huggingface.co/datasets/J1mmymm/MIMuT_Data_v2).
Users must obtain every dataset and model weight under its own upstream license
and update the template paths before attempting to run this historical code.

## License

The archived MuScriptor-derived source retains the bundled MIT license and
upstream attribution in `hybrid_mamba_clean_cache/LICENSE`. The parent GitHub
repository's Apache-2.0 license does not relicense that historical source,
datasets, pretrained weights, or other third-party components.
