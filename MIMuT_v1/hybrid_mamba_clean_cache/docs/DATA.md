# Data preparation

## Manifest

Build and validate the canonical JSONL manifest outside the source tree:

```bash
muscriptor-data build /datasets --output manifests/all.jsonl --strict
muscriptor-data check manifests/all.jsonl
```

Each row contains `track_id`, dataset, split, audio/notes paths, duration,
instrument groups, multi-instrument status, and a leakage-safe `group_id`.
Overlapping `provenance.component_track_ids` are collapsed with union-find so
stems and mixtures cannot land in different splits. Official path splits take
precedence; other groups use a deterministic 80/10/10 split. RWC-P, URMP,
Bach10, Dagstuhl ChoirSet, and PHENICX are forced to test.

The parser directly consumes the supplied `*.notes.json` schema. It also
converts:

- Slakh2100 Redux stem MIDI plus metadata into mixture note JSON.
- URMP aligned `Notes_*.txt` annotations; its unaligned score MIDI is not used.
- aligned WAV/MIDI pairs from preprocessed IDMT-SMT-Bass and STAR Drums.

Converted JSON is cached beside the manifest, never in the source dataset.
Unsupported or unmatched content is listed in `<manifest>.issues.jsonl`; a
dataset that yields zero records is always an issue.

## Sampling

Dataset mass is proportional to square-root duration and capped at 35%.
Explicit stage weights can retain 20% Slakh replay during real-data
fine-tuning. At least half of each batch is sampled from multi-instrument
mixtures when available. Windows are stratified by the largest same-instrument
silence/re-entry gap: none, 0–5, 5–10, 10–20, and 20+ seconds.

The global dataset-to-class mapping is computed once from the complete
manifest and reused across every curriculum stage. It is saved in every
checkpoint as provenance. Formal MIMuT inputs nevertheless map dataset class
to NULL during training and inference; the mapping must never act as a dataset
identity shortcut in the main experiment.
