# Acoustic-Hybrid-Mamba historical supplement

> [!WARNING]
> This is a **partial historical overlay**, not a standalone release and not the
> latest MIMuT implementation. The explored route did not achieve the quality
> expected by the project team.

This directory contains only files that were new or different relative to the
`MIMuT_v1/hybrid_mamba_clean_cache/` snapshot. To reconstruct the archived
experiment for inspection, copy these paths over a separate copy of that
baseline while preserving their relative locations.

The overlay records the experimental separation of:

- a six-layer Acoustic Mamba2 encoder whose acoustic state could persist across
  five-second blocks; and
- a 22-layer Hybrid-Mamba autoregressive decoder whose recurrent and KV state
  reset at every block.

The short `*_407m_*` YAML files are compatibility aliases for older filenames
that still contain `396m`. The alias names describe the later reported parameter
count; the older filenames are retained because historical launch automation
referenced them. Absolute user/server paths have been replaced with
`/path/to/...` templates.

No weights, manifests, optimizer states, runtime logs, or private deployment
files are included. The overlay has not been rerun as part of this archival
publication.
