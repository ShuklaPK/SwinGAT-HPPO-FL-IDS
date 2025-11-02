# Assumptions

- Kaggle SDN‑IoT dataset requires user authentication; we provide a downloader stub with instructions.
- GAT is implemented in pure PyTorch using edge indices to avoid heavy compiled deps; Swin from `timm`.
- CI runs a tiny synthetic dataset and 1–2 training steps to verify plumbing; not a benchmark.
- Flan‑T5 is supported via `transformers`. Tests avoid heavy downloads by setting `FLAN_OFFLINE=1`.
