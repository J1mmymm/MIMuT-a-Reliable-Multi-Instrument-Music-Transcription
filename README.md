# MIMuT: Reliable Multi-Instrument Music Transcription

[![Project status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)](#project-status)
[![Current architecture: Hybrid-Mamba-Attention](https://img.shields.io/badge/current%20architecture-Hybrid--Mamba--Attention-6f42c1)](#current-prototype)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> [!IMPORTANT]
> **MIMuT is an ongoing research project, not a finished model release.** The work currently covers one experimental direction—a Hybrid-Mamba-Attention architecture—and a limited set of training and diagnostic runs. Formal model selection, matched baselines, multi-seed ablations, whole-dataset evaluation, and a complete code/weight release are still in progress.

## Overview

**MIMuT: Reliable Multi-Instrument Music Transcription** investigates robust automatic music transcription (AMT) from raw audio to instrument-aware note events and MIDI. The current prototype explores whether recurrent state-space modeling can carry musical context across long recordings while bounded causal attention preserves precise local token interactions.

The project is built as an experimental extension of [MuScriptor](https://github.com/muscriptor/muscriptor). It retains an autoregressive audio-to-event transcription pipeline while replacing the all-attention backbone with a repeated Hybrid-Mamba-Attention pattern.

## Project status

The repository is being prepared incrementally. At this stage, the public repository documents verified progress; cleaned source code, reproducible configurations, selected checkpoints, and complete evaluation artifacts will be added only after their release and licensing checks are finished.

| Component | Current status | Evidence boundary |
|---|---|---|
| Hybrid-Mamba-Attention backbone | Implemented | Mamba2 and bounded local causal attention are integrated in the experimental transcription model. |
| Main 312M Hybrid-Mamba baseline | Completed to 280,000 optimizer steps | The final checkpoint is `latest`, not a validation-selected “best” model. |
| Separate two-GPU recovery run | Stopped at a complete 256,000-step checkpoint | Recoverable training state was preserved; this run is not the formal context-mix/RoPE v2 experiment. |
| Context-mix/RoPE v2 implementation | Code and local tests completed | Formal Linux/CUDA training and quality/efficiency claims remain pending. |
| 106M Clean-Cache control continuation | Completed for 15,000 continuation steps | Retained as an engineering control, not a paper result. |
| 106M Clean-Cache distillation continuation | Completed for 5,000 continuation steps | Positive KL and a frozen teacher were verified, but this is a limited pilot. |
| Whole-validation and test evaluation | Not completed | No checkpoint is currently presented as a formal best model or state of the art. |

## Current prototype

The current experimental backbone repeats:

```text
5 s audio blocks
      │
      ▼
mel and condition embeddings
      │
      ▼
Mamba2 → Mamba2 → bounded local causal attention
      │                 (repeated across the decoder)
      ▼
MT3_FULL_PLUS event tokens
      │
      ▼
instrument-aware notes and MIDI
```

The implemented prototype includes:

- 16 kHz mono audio divided into five-second blocks;
- packed 5/10/20/40-second causal training contexts;
- recurrent Mamba2 state for cross-block history;
- bounded local causal attention with a 2,048-token window;
- an experimental RoPE path for local-attention queries and keys;
- `MT3_FULL_PLUS` event supervision with 1,395 output classes;
- an audio-only formal condition that removes oracle instrument and dataset metadata;
- BF16/DDP training, gradient checkpointing, resumable optimizer/trainer state, manifest hashes, and deterministic sampling records;
- whole-track and long-horizon evaluation utilities, including onset, onset-offset, boundary, instrument-switch, long-gap re-identification, and efficiency diagnostics.

The medium experimental configuration uses `D=1024`, 22 layers, 16 attention heads, and approximately 317.5M parameters. Parameter counts are configuration-dependent and will be frozen with the public checkpoint release.

## Completed training evidence

### Main 312M Hybrid-Mamba baseline

The completed baseline run used an NVIDIA H800 PCIe GPU and reached **280,000 optimizer steps**. Its curriculum consisted of:

1. 150k steps of five-second Slakh pretraining;
2. 100k steps of five-second multi-dataset training;
3. 10k steps each at 10, 20, and 40 seconds.

Eleven complete weight/trainer checkpoint pairs were verified from steps 260k through 280k, with no observed non-finite loss, cross-entropy, or gradient values. Checkpoints 274k, 278k, and 280k were retained as validation candidates. Because whole-validation evaluation has not been completed, step 280k is only the latest checkpoint and is not claimed to be the best checkpoint.

### Engineering continuations

A smaller 106M Clean-Cache branch was used to test controlled continuation and knowledge distillation from an existing 50k checkpoint. The no-distillation control completed 15k continuation steps. The distillation branch retained a complete 5k-step checkpoint after checks for positive distillation loss, frozen teacher parameters, finite metrics, and GPU memory headroom. These runs validate parts of the training pipeline but are not formal model comparisons.

## Diagnostic transcription

An intermediate 312M checkpoint was evaluated end to end on one Slakh2100 Redux validation track (`Track01501`) under an audio-only, no-oracle setting. The 192.9-second input was processed in 39 chunks and produced a structurally valid 190-second MIDI file with 5,529 matched note-on/note-off pairs.

For this single track, `mir_eval` onset $F_1$ was **0.2373** and onset-offset $F_1$ was **0.1217**. These values are included only as a pipeline diagnostic. They are not whole-validation results, do not establish generalization, and must not be interpreted as a comparison with existing AMT systems.

## Data

The current training manifest contains **8,454 records across 13 source datasets**. Individual experiments use explicit dataset allowlists and do not necessarily train on every source represented in the manifest.

- Dataset repository: [J1mmymm/MIMuT_Data_v2](https://huggingface.co/datasets/J1mmymm/MIMuT_Data_v2)
- Training manifest: [manifests/manifest_all.jsonl](https://huggingface.co/datasets/J1mmymm/MIMuT_Data_v2/blob/main/manifests/manifest_all.jsonl)

The dataset collection has mixed upstream licenses. The Apache-2.0 license of this code repository does **not** relicense any dataset, model weight, checkpoint, or third-party component. Users must follow the terms and attribution requirements of each upstream source.

## Evaluation principles

The intended formal protocol follows these rules:

- audio-only inputs for primary results;
- no oracle instrument labels or filename-derived conditions in the main table;
- validation-only checkpoint selection, with the test split reserved for final reporting;
- whole-track evaluation rather than isolated short-window scoring;
- matched Transformer, local-Transformer, and pure-Mamba controls;
- separate reporting of note accuracy, instrument attribution, long-gap re-identification, boundary behavior, and streaming efficiency;
- multiple seeds and confidence intervals before any improvement claim.

## Roadmap

- [x] Implement the initial Hybrid-Mamba-Attention backbone.
- [x] Complete a 280k-step Main 312M baseline training run.
- [x] Validate checkpoint/resume handling and end-to-end MIDI generation.
- [x] Implement context-mixed training, RoPE, strict audio-only conditioning, and extended evaluation code.
- [ ] Finish formal context-mix/RoPE v2 training.
- [ ] Run whole-validation checkpoint selection and freeze the test protocol.
- [ ] Complete matched Transformer/pure-Mamba baselines and multi-seed ablations.
- [ ] Benchmark long-context quality, streaming speed, cache size, and memory on matched hardware.
- [ ] Curate and publish reproducible source code, configurations, checkpoints, and result bundles.
- [ ] Release paper-ready results and a formal citation.

## Availability

The repository is currently a work-in-progress project page. Installation and inference commands will be added when the cleaned implementation and verified checkpoint package are publicly released. Until then, please treat architectural details and diagnostic numbers as provisional research documentation rather than a supported software release.

## Acknowledgements

This work builds on [MuScriptor](https://github.com/muscriptor/muscriptor) and the official [Mamba/Mamba2 implementation](https://github.com/state-spaces/mamba). Their original licenses and attribution requirements continue to apply to reused components.

## License

MIMuT-specific material in this repository is released under the [Apache License 2.0](LICENSE), unless a file states otherwise. Third-party source code, datasets, pretrained weights, and generated artifacts may use different licenses.

## Citation

MIMuT is not yet a completed paper or frozen release. A project citation will be added after the method, evaluation protocol, and public artifacts are finalized.

