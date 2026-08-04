# MIMuT: Reliable Multi-Instrument Music Transcription

[![Project status: work in progress](https://img.shields.io/badge/status-work%20in%20progress-orange)](#project-status)
[![Architecture direction: under revision](https://img.shields.io/badge/architecture%20direction-under%20revision-6f42c1)](#architecture-direction-update)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

> [!IMPORTANT]
> **MIMuT is an ongoing research project, not a finished model release.** The first implemented direction was a Hybrid-Mamba-Attention architecture. Training and diagnostic attempts to date did not achieve the transcription quality expected by the project team, so this route is being retained as an experimental baseline rather than treated as the final architecture. Project members are now exploring alternative architectures and training formulations. Formal model selection, matched baselines, multi-seed ablations, whole-dataset evaluation, and a complete code/weight release are still in progress.

## Overview

**MIMuT: Reliable Multi-Instrument Music Transcription** investigates robust automatic music transcription (AMT) from raw audio to instrument-aware note events and MIDI. The first prototype explored whether recurrent state-space modeling could carry musical context across long recordings while bounded causal attention preserved precise local token interactions.

The project is built as an experimental extension of [MuScriptor](https://github.com/muscriptor/muscriptor). Its initial research branch retained an autoregressive audio-to-event transcription pipeline while replacing the all-attention backbone with a repeated Hybrid-Mamba-Attention pattern. Because the observed transcription quality did not meet the team's expectations, the architecture is no longer treated as a settled final design.

## Project status

The repository is being prepared incrementally. At this stage, the public repository documents verified progress; cleaned source code, reproducible configurations, selected checkpoints, and complete evaluation artifacts will be added only after their release and licensing checks are finished. No replacement architecture has yet been selected as the final MIMuT design.

| Component | Current status | Evidence boundary |
|---|---|---|
| Hybrid-Mamba-Attention backbone | Implemented and evaluated as the initial research branch | Available training and diagnostic results did not meet the expected transcription quality; the branch is retained as a baseline while alternatives are explored. |
| Main 312M Hybrid-Mamba baseline | Completed to 280,000 optimizer steps | The final checkpoint is `latest`, not a validation-selected “best” model. |
| Separate two-GPU recovery run | Stopped at a complete 256,000-step checkpoint | Recoverable training state was preserved; this run is not the formal context-mix/RoPE v2 experiment. |
| Context-mix/RoPE v2 implementation | Code and local tests completed | Formal Linux/CUDA training and quality/efficiency claims remain pending. |
| 106M Clean-Cache control continuation | Completed for 15,000 continuation steps | Retained as an engineering control, not a paper result. |
| 106M Clean-Cache distillation continuation | Completed for 5,000 continuation steps | Positive KL and a frozen teacher were verified, but this is a limited pilot. |
| Whole-validation and test evaluation | Not completed | No checkpoint is currently presented as a formal best model or state of the art. |
| MIMuT v1 historical source archive | Published under [`MIMuT_v1/`](MIMuT_v1/) | Sanitized early Hybrid-Mamba and Acoustic-Hybrid-Mamba source is preserved for provenance only; it is not the latest or recommended architecture. |

## Architecture direction update

Preliminary training and end-to-end diagnostic results indicate that the Hybrid-Mamba route has not produced satisfactory model performance for the project's goals. This is a project-level design conclusion based on the attempts completed so far, not yet a benchmark-backed claim that the architecture is universally inferior to other AMT systems.

The project team is therefore broadening the architecture search. Future prototypes will be compared under a shared audio-only protocol, with particular attention to note accuracy, instrument attribution, cross-boundary stability, long-context reliability, and computational efficiency. The existing Hybrid-Mamba implementation, configurations, checkpoints, and diagnostics remain valuable as a reproducible baseline and ablation reference.


## Historical source archive

The cleaned source and configuration material from the first project iteration is preserved under [`MIMuT_v1/`](MIMuT_v1/). It includes the Hybrid-Mamba/Clean-Cache baseline and a partial Acoustic-Hybrid-Mamba overlay reconstructed from the supplied archive and development notes.

> [!WARNING]
> `MIMuT_v1/` is a historical record of architecture lines that did not achieve the expected model quality. It is not the latest MIMuT source tree, not a supported release, and not the architecture currently recommended for further development. No model weights, datasets, manifests, private runtime files, or unverified result claims are included.

## Current prototype

The evaluated Hybrid-Mamba experimental backbone repeats:

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
- `MT3_FULL_PLUS` event supervision with 1,393 output classes;
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
- [x] Complete a preliminary assessment and identify the quality limitations that prevent Hybrid-Mamba from being adopted as the final architecture.
- [x] Preserve the cleaned Hybrid-Mamba/Acoustic-Hybrid-Mamba v1 source and configurations as a historical baseline and ablation reference.
- [ ] Prototype and screen alternative architectures and training formulations under the same audio-only evaluation protocol.
- [ ] Select the next primary architecture using validation-set note accuracy, instrument attribution, boundary stability, long-context reliability, and efficiency criteria.
- [ ] Run whole-validation checkpoint selection and freeze the test protocol for the selected architecture.
- [ ] Complete matched baselines and multi-seed ablations before making comparative claims.
- [ ] Benchmark long-context quality, streaming speed, cache size, and memory on matched hardware.
- [ ] Curate and publish reproducible source code, configurations, checkpoints, and result bundles.
- [ ] Release paper-ready results and a formal citation.

## Availability

The repository is currently a work-in-progress project page. Installation and inference commands will be added when the cleaned implementation and verified checkpoint package are publicly released. Until then, please treat architectural details and diagnostic numbers as provisional research documentation rather than a supported software release or a commitment to Hybrid-Mamba as the final architecture. The source under `MIMuT_v1/` is archival and does not change that release status.

## Acknowledgements

This work builds on [MuScriptor](https://github.com/muscriptor/muscriptor) and the official [Mamba/Mamba2 implementation](https://github.com/state-spaces/mamba). Their original licenses and attribution requirements continue to apply to reused components.

## License

MIMuT-specific material in this repository is released under the [Apache License 2.0](LICENSE), unless a file states otherwise. Third-party source code, datasets, pretrained weights, and generated artifacts may use different licenses.

## Citation

MIMuT is not yet a completed paper or frozen release. A project citation will be added after the method, evaluation protocol, and public artifacts are finalized.

