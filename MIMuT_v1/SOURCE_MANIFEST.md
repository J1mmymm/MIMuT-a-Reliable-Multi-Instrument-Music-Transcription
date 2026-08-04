# Source manifest and cleaning record

## Included

- The Python package, training/evaluation code, tokenizer, configuration files,
  tests, technical documentation, helper scripts, dependency metadata, and
  license from the supplied Hybrid-Mamba/Clean-Cache source directory.
- Only changed or newly introduced Python/configuration files from the supplied
  Acoustic-Hybrid-Mamba archive.
- A sanitized continuation configuration and its separately supplied resume
  override; the directly referenced base configuration was included so the
  override is not orphaned.
- A rewritten design-history note derived from the supplied conversation.

## Excluded

- `__pycache__/`, `*.pyc`, `.DS_Store`, `__MACOSX/`, AppleDouble files, and
  timestamped `*.pre_*` backups;
- the automatically generated `uv.lock`; dependency intent remains available
  through `pyproject.toml` and `requirements-cuda12.txt`;
- model checkpoints, optimizer/trainer states, datasets, manifests, generated
  predictions, logs, caches, and temporary runtime reports;
- the raw conversation export, which contained personal server paths and SSH
  command history;
- upstream deployment workflows and the bundled public SSH key;
- the unrelated demonstration web frontend and media assets;
- paper drafts and result-table templates that could be mistaken for the current
  project submission.

## Sanitization

- Personal and machine-specific absolute paths were replaced with
  `/path/to/...` templates.
- The integration-test audio path can now be supplied through the
  `MUSCRIPTOR_TEST_AUDIO` environment variable.
- No access token, private key, password, dataset content, or checkpoint was
  found in or added to the publication tree.
- Archive notices were added at the version root, baseline root, and supplement
  root so these files cannot reasonably be mistaken for the latest MIMuT code.

## Provenance boundary

File inclusion preserves the supplied historical source content except for the
documented path sanitization, archive banners, broken-link cleanup, and removal
of excluded artifacts. Runtime and quality claims from the development
conversation are identified as historical reports and were not independently
revalidated during this publication pass.
