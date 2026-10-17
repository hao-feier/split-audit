# experiments/

Data prep, model training, and resumable GPU sweep launchers for the bi-temporal
Sentinel-2 mangrove change-detection study. 21 scripts (`.py`) + 15 resumable
`.sh` launchers.  Flat by design (referenced by name from the package
docstrings and the paper's reproduction commands). Run from the repo root.

## Pipeline stages (12)

- `exp0_region_selection.py` — Confirm study regions against GMW v4.1.12 extent (PLAN 1); emits `results/exp0_region_stats.json`.
- `exp1_fetch_s2.py` — Fetch S2 L2A scenes per region x epoch from the AWS element84 STAC API.
- `exp1_composite.py` — Multi-tile per-tile compositing of the fetched scenes (cloud-cleared).
- `exp1_labels.py` — GMW v4.1.12 timeseries -> UTM 10 m mangrove presence masks.
- `exp1_patch.py` — Build the single 10 m UTM reference grid and extract per-patch stacks.
- `exp2_split_v1.py` — Spatial (buffer-separated) train/test split, v1.
- `exp2_split_v2.py` — v2 split: enforces the >=2 km edge guarantee (consumed by `analysis/split_audit.py`).
- `exp3_train.py` — Core training loop for the five architectures (the main sweep).
- `exp3_random_split.py` — Counterfactual to `exp3_train.py`: random (non-spatial) split, the leaky arm.
- `exp3_e3c128.py` — E3-C: size-matched, overlap-free test of residual proximity.
- `exp3_ir_split.py` — I/R factorial cell: spatial-isolated train + random test (self-contained split generation).
- `exp3_block_split.py` — E3-B: overlap-free block-interleaved control.

## Classical / tuned baselines (4)

- `exp3_baseline_ml.py` — Random Forest + SVM pixel-level baselines (PLAN 4.3).
- `exp3_baseline_cva.py` — CVA and index-difference baselines (NDVI / norm / logreg).
- `exp3_baseline_ml_tuned.py` — Tuned Random Forest + SVM (fair baselines).
- `exp3_baseline_pcc.py` — PCC (post-classification comparison) baseline.

## Split generators (helpers called by launchers) (2)

- `make_random_split.py` — Emit the random-split cells used by the leaky-arm launchers.
- `make_block_split.py` — Emit the overlap-free block-interleaved split cells.

## Visualization & utilities (3)

- `exp4_visualize.py` — Publication-quality result visualizations (layout covered by `tests/test_exp4_layout.py`).
- `reeval_s2026.py` — Re-evaluate existing v2 seed-2026 checkpoints (no retraining; recovery of an OOM-lost metrics JSON).
- `watch_exp1_composite.py` — Dependency-free `/proc` memory watcher for `exp1_composite.py`.

## Resumable GPU sweep launchers (15)

All resumable (skip already-done runs), no auto-retry; re-aggregate on completion.

- `run_training_v2.sh`, `run_seed_sweep.sh`, `run_seed_sweep_gpu.sh` -> `exp3_train.py`
- `run_random_split_gpu.sh`, `run_fullarch_leaky_gpu.sh`, `run_arch_extension_gpu.sh` -> `exp3_random_split.py` (+ `make_random_split.py`)
- `run_ir_factorial_gpu.sh`, `run_uniform_counterfactual_gpu.sh` -> `exp3_ir_split.py` (+ `make_random_split.py`)
- `run_block_counterfactual_gpu.sh`, `run_e3b_gpu.sh` -> `exp3_block_split.py` (+ `make_block_split.py`)
- `run_e3c128_gpu.sh`, `run_e3c_roc_gpu.sh` -> `exp3_e3c128.py`
- `run_strict_scl.sh` -> full data-prep chain (`exp1_composite` -> `exp1_labels` -> `exp1_patch` -> `exp3_random_split.py`)
- `run_strict_scl_train.sh` -> `exp3_random_split.py` (strict-SCL-mask training)
- `run_uniform_topup_gpu.sh` -> `exp3_train.py` (uniform-protocol top-up runs)
