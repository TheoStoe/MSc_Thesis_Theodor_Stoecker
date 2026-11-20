# SASRec PyTorch provenance

`sasrec.py` adapts the model architecture from:

- repository: <https://github.com/pmixer/SASRec.pytorch>
- revision: `fde8a9c08febc2c898dc269b1d98ab762c50d2be`
- model source: <https://github.com/pmixer/SASRec.pytorch/blob/fde8a9c08febc2c898dc269b1d98ab762c50d2be/python/model.py>
- sampler/training reference: <https://github.com/pmixer/SASRec.pytorch/blob/fde8a9c08febc2c898dc269b1d98ab762c50d2be/python/utils.py>
- upstream `python/model.py` SHA-256: `ECA47285450F76F179E7464AF46ED7ED84F585E42C9B08550A351378B49E2C77`
- inspected: 2026-08-14
- licence: Apache License 2.0; see `LICENSE-SASREC-APACHE-2.0`

The software citation supplied by the upstream project identifies Zan Huang's
`SASRec.pytorch` (2020). Research using the recommender should also cite the
original SASRec paper by Wang-Cheng Kang and Julian McAuley (ICDM 2018).

## Kept from the pinned PyTorch implementation

- item and absolute-position embeddings with padding index zero;
- item-embedding scaling by the square root of the hidden dimension;
- causal `torch.nn.MultiheadAttention` without an additional padding-key mask;
- two kernel-size-one convolutional feed-forward layers;
- selectable pre- or post-LayerNorm blocks and a final LayerNorm;
- masked positive/negative binary-cross-entropy training;
- one uniformly sampled non-history negative for each next-item target;
- Adam with betas `(0.9, 0.98)` and optional item-embedding-only L2;
- final-sequence-feature dot products for candidate scoring; and
- Xavier-normal matrix initialization with zeroed padding embeddings.

`norm_first=False` remains the default, matching the pinned command-line
implementation. The selected value is stored in settings, diagnostics,
checkpoints, and fingerprints.

## Deliberate simulator-interface adaptations

- The public methods implement `recommenders/base.py`: `fit`, `recommend`,
  `score_candidates`, state export/restore, clone, and fingerprint.
- External item identifiers are deterministically mapped to dense model indices
  so the reusable interface does not require consecutive dataset IDs.
- Prepared chronological histories replace the upstream text-file partitioner.
- A deterministic in-process sampler replaces `WarpSampler` and its background
  worker processes. Sampling remains uniform over items absent from the user's
  complete fitted history.
- Inference receives the current history explicitly, allowing history feedback
  without changing shared model weights.
- Ranking is over the caller's complete eligible candidate set rather than one
  held-out positive plus 100 sampled negatives.
- Empty histories use a deterministic popularity fallback because the simulator
  may activate previously unseen reserve users.
- Epoch count and batch size remain experiment configuration. The adapter keeps
  its earlier 201-epoch default instead of silently adopting upstream's 1000.
- Device-portable checkpoints and semantic fingerprints are local additions.

These adaptations mean the local file is not a verbatim copy and its saved
states are not interchangeable with upstream checkpoints. The distinct variant
identifier `sasrec_pmixer_pytorch_v1` prevents accidental restoration of older
local SASRec states.
