# MIMuT v1 design history

This note summarizes the supplied development conversation after removing
personal server locations, SSH details, transient process information, and
checkpoint files. It records historical reasoning rather than the current MIMuT
architecture.

## Why one checkpoint supports carry and reset

For a recurrent decoder, inference can be summarized as

## Why one checkpoint supports carry and reset

For a recurrent decoder, inference can be summarized as:

```math
h_t = f_{\theta}\left(x_t, h_{t-1}\right)
```

where:

- $\theta$ represents the model parameters stored in the checkpoint;
- $x_t$ is the input at step $t$;
- $h_{t-1}$ is the runtime state inherited from the previous step;
- $h_t$ is the updated runtime state.

where `theta` is stored in the checkpoint and `h_t` is temporary runtime state.
The checkpoint is identical in both modes:

- `reset`: initialize runtime state for every five-second block and discard it
  at the boundary;
- `carry`: initialize once per track and retain state between blocks.

In the historical Hybrid-Mamba decoder, the carried state included Mamba2 SSM
state, short-convolution state, and bounded local-attention KV state. Because
audio conditions and generated event tokens passed through the same decoder,
the state represented joint acoustic-symbolic history. This made useful context
available but also created an error-propagation path:

```text
incorrect MIDI token
  -> persistent decoder state
  -> later-block prediction
  -> additional errors
```

Reset cut this path but also removed learned neural history. Tie/prelude forcing
remained a third, explicit cross-block mechanism that transferred only notes
predicted to be still active at the boundary.

## Relationship to the MuScriptor baseline

The baseline MuScriptor-style path primarily used acoustic evidence from the
current five-second block. Without a prelude it carried neither historical
acoustic representations nor historical symbolic state. With a prelude it
carried a small symbolic boundary summary, not the prior audio representation.
Oracle instrument conditioning, when used diagnostically, was a global prior
rather than historical acoustic memory.

The historical comparison was therefore:

| Route | Historical acoustic representation | Historical symbolic information |
|---|---|---|
| MuScriptor without prelude | No | No |
| MuScriptor with prelude | No | Active-note boundary cues only |
| Hybrid-Mamba decoder with carry | Yes, mixed with generated tokens | Yes |
| Acoustic-Mamba with carry | Acoustic state only by design | Prelude optional |

## Acoustic/decoder state separation

The supplement introduced a six-layer Acoustic Mamba2 encoder whose state was
intended to persist across five-second blocks. A 22-layer Hybrid-Mamba decoder
then generated `MT3_FULL_PLUS` tokens, but its Mamba and local-attention state
was reset at every block. The design goal was to preserve compressed acoustic
history without allowing prior MIDI prediction errors to remain in the
long-lived state.

At the time, the configuration was reported as:

- six Acoustic Mamba2 encoder layers;
- 22 Hybrid-Mamba decoder layers following the repeated
  `Mamba2 -> Mamba2 -> local attention` pattern;
- a 4,096-token local-attention window;
- approximately 407,438,560 parameters;
- validation sampling every 2,000 steps in the referenced training setup.

Historical notes also report fixes for acoustic-state slot preservation during
decoder initialization and BF16 query/KV-cache dtype alignment. These are
development records, not fresh validation results from this publication pass.

## Warm start and distillation intent

The planned student initialization reused compatible components from an older
Hybrid-Mamba run:

```text
older Hybrid-Mamba checkpoint -> initialize compatible student decoder parts
MuScriptor checkpoint          -> provide teacher logits during distillation
ground-truth events            -> provide cross-entropy supervision
new Acoustic-Mamba layers      -> initialize separately
```

Warm start was intended to preserve learned event syntax, EOS/tie behavior,
embeddings, decoder layers, normalization, and output projection. It did not
restore the older optimizer state or global step. The public templates use
placeholder paths; no referenced checkpoint is included.

## Outcome and archival status

The architecture was explored but did not achieve the model quality expected by
the project team. It is therefore preserved under `MIMuT_v1/` solely as a
historical source record and possible future ablation baseline. It must not be
described as the latest MIMuT source tree, a successful final architecture, or a
validated improvement over MuScriptor or other AMT systems.
