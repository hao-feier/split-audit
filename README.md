# When F1 and ROC-AUC Disagree: A 30-Point Metric Divergence Under Spatial Interleaving

A **spatial evaluation audit** of bi-temporal Sentinel-2 mangrove change detection.
This repository asks a measurement question, not a model question: **how much of a
reported change-detection score is produced by the train/test split design rather
than by the model?**

The manuscript (`paper/main.tex`, MDPI *Remote Sensing*, in preparation) is
checked number-by-number against `results/*.json` by
`analysis/verify_manuscript_numbers.py` (the check set grows with the manuscript).

## The finding in one table

Same imagery, same architecture, same optimiser, same labels — only the spatial
split assignment differs:

| Protocol | change `F1` | ROC-AUC |
|---|---|---|
| Interleaved random split (train and test patches share footprint) | **0.572** | 0.965 |
| Buffer-separated split (≥ 2 km) | **0.276** | 0.946 |
| **Gap** | **+29.6 pp** | **+1.9 pp** |

`F1` is pooled over the 81 of 90 runs that pass a pre-specified validity filter
(`F1_ch < 0.05`, applied symmetrically to both arms); ROC-AUC is defined even at
zero positive predictions and is pooled over all 90 runs.

The inflation is concentrated in the thresholded, recall-driven part of the
metric — TPR 0.19 → 0.48 at the fixed 0.5 threshold — while global pairwise
ranking barely moves. Test-set prevalence arithmetic accounts for only +2.7 pp;
the dominant component is the **joint spatial configuration** of training and
test samples.

The `F1` gap is 26.9–34.1 pp across the three split draws and 11.8–32.8 pp
across the ten architectures. Including all 90 runs gives +26.7 pp instead of
+29.6 pp and changes no conclusion.

## What this study does *not* claim

Stated explicitly, because the honest boundary is the contribution:

- **No architecture ranking.** The ten architectures share training data and
  recipe, so directional agreement is not independent replication.
- **No field accuracy.** All metrics measure agreement with Global Mangrove
  Watch, not field truth; the 2,400 designed interpretation points have not
  yet been interpreted.
- **No causal decomposition.** Overlap, near-neighbour proximity and morphology
  are bundled into one factor that the 512-px design cannot separate; the
  zero-overlap control at reduced 128-px geometry was underpowered.
- **No causal language.** Reported effects are design-factor contrasts.

## Study design

**Patches.** 512-px patches (5.12 km footprint) cut at 2.56 km steps, so
adjacent patches overlap by half their width. Vietnam supplies 76 patches
(40 train / 10 val / 10 test / 16 held out by the buffer); Beilunhe (18) and
Zhanjiang (15) are test-only.

**The contrast.** In-box patches are assigned to training and testing in two
ways, with everything else held fixed (imagery, labels, recipe, run seeds; both
arms of each contrast come from the same sweep):

- *interleaved* — a uniform random draw over all in-box patches: training and
  test patches spatially mixed and share footprint;
- *buffer-separated* — train and test patches at least 2 km apart.

**Weighting protocols.** The primary protocol trains with uniform class weights
(`--cw-mode none`, `results/uniform/`); an earlier sqrt-weight protocol
(`results/` top level) is retained as a sensitivity. All headline numbers are
primary.

**Experiments.** Each experiment is pre-specified (question, resampling unit,
what the design can and cannot identify); registration cards E1–E7 are in the
paper appendix (`tab:reg`).

| ID | Question |
|---|---|
| E1 | interleaved vs separated split (the headline contrast) |
| E2 | 2×2 train/test spatial factorial (prevalence arithmetic vs spatial) |
| E3 | overlap-free block-interleaved control, incl. reduced 128-px geometry |
| E4 | distance response under separation |
| E5 | effective-separation sensitivity (grid-quantised buffer size) |
| E6 | leave-one-cluster-out fragility |
| E7 | label-resolution tolerance sweep |

## Study regions (verified against GMW v4.1.12)

| Region | Box (WGS84) | Mangrove km², t1 → t2 | Role |
|---|---|---|---|
| Vietnam Red River Delta | 106.5–107.0 °E, 20.5–21.2 °N | 77.4 → 70.9 | training (A) |
| Beilunhe estuary, CN | 108.1–108.7 °E, 21.15–21.75 °N | 18.3 → 19.0 | held-out test (B) |
| Zhanjiang, CN | 110.05–110.85 °E, 21.0–21.5 °N | 14.6 → 19.1 | held-out test (C) |

Training and test regions are geographically disjoint and buffer-separated.
Evaluated change prevalence: 0.81–2.65 % of pixels across the three
reference-label sets (interleaved test draws: 2.9–3.8 %). Areas are the
processed label domain (`data/processed/<region>/label_stats.json`).

## Data

- **Imagery** — Sentinel-2 L2A (B02/B03/B04/B08/B11/B12 + SCL), AWS Open Data
  COG collection (Element 84 earth-search STAC API), dry-season cloud-filtered
  median composites (2019-12–2020-01, 2024-12–2025-01). Processing baselines
  differ by epoch, but the COG collection serves both on one offset-free
  0–10000 scale — a property of the COG reprocessing, not of the original
  products (tile audit in `results/s2_processing_baseline.json`). No
  cross-epoch radiometric harmonisation.
- **Labels** — Global Mangrove Watch v4.1.12 (Zenodo record `21346457`),
  Landsat-based, four classes: stable mangrove / loss / gain / stable
  background. Distributed CC BY 4.0 by the GMW consortium; its reported
  extent-level F1 of 0.928 is an extent, not a change, figure.
- **Second label source (sensitivity only)** — GMW v4.0.19 S2-based layer
  (Zenodo record `12756047`); an extent-level label-source check, never
  training truth.

All data from public archives (Copernicus open data; GMW CC BY 4.0); the
`LICENSE` file covers the code only. Google Earth Engine is not used.

## Reproducing

**Requirements.** Python 3.11+; ~100 GB free disk (≈60 GB raw S2, ≈25 GB
processed, ≈17 GB checkpoints); one GPU for the training sweeps (reference:
RTX 4080 Laptop 12 GB — the 120-run counterfactual is ≈8 h, the 50-run main
sweep ≈15–17 h, so a full re-run is on the order of 1–2 days; a 4-core CPU box
is ≈15× slower). The CPU install suffices for data preparation, evaluation,
table/figure regeneration and manuscript checking.

**The committed record.** `results/*.json` are the primary record: every
manuscript number is verified against them. The sweeps below are resumable (a
run is skipped when its metrics JSON exists), so re-running only fills gaps;
bit-exact reproduction of the committed numbers is not the goal, the verified
JSONs are.

### 1. Environment

```bash
conda create -n junzhe python=3.11 -y
conda run -n junzhe pip install -r requirements.txt
conda run -n junzhe pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU host: use your CUDA build (e.g. .../whl/cu121) instead of the CPU wheel
```

### 2. Data (public archives only: AWS open data + Zenodo, no GEE)

```bash
# for each region in {vietnam, beilunhe, zhanjiang} and epoch in {t1, t2}:
python experiments/exp1_fetch_s2.py --region vietnam --epoch t1
python experiments/exp1_composite.py --region vietnam --epoch t1
python experiments/exp1_labels.py     --region vietnam
python experiments/exp1_patch.py      --region vietnam
python experiments/exp2_split_v1.py                    # all regions; buffer >= 2 km
```

### 3. Training sweeps

Primary main sweep — 10 models × 5 seeds, uniform weights, written to
`results/uniform/` (the `MANGROVE_RESULTS_ROOT` below scopes every stage to
that directory):

```bash
export MANGROVE_RESULTS_ROOT=results/uniform
for model in unet unet_cbam unet_msa unet_cbam_msa deeplabv3plus \
             temporal_unet temporal_unet_gn fc_siam changeformer deeplabv3plus_pre; do
  for seed in 2026 2027 2028 2029 2030; do
    python experiments/exp3_train.py --model $model --seed $seed --cw-mode none
  done
done
```

Headline counterfactual — interleaved vs buffer-separated (120 runs, ≈8 h on
the reference GPU), resumable, re-aggregates on completion:

```bash
bash experiments/run_uniform_counterfactual_gpu.sh
```

Classical and index baselines (CVA, pixel Random Forest / SVM RBF, and the
post-classification-comparison arm) are wrapped in one resumable launcher:

```bash
bash experiments/run_baselines_gpu.sh
```

Per-family launchers, recipes and failure policy are documented in the headers
of `experiments/run_*.sh`.

### 4. Aggregation, tables, figures

```bash
# same MANGROVE_RESULTS_ROOT as the sweep:
python analysis/aggregate_results.py         # results/uniform/metrics_summary.json
python analysis/aggregate_ir_factorial.py    # results/uniform/ir_factorial.json
python analysis/make_latex_tables.py         # inlines the tables into paper/main.tex
python analysis/make_figures.py              # figures/fig1-7
```

### 5. Checking the manuscript (no training needed)

```bash
python analysis/verify_manuscript_numbers.py   # every headline number vs the result JSONs
python analysis/bib_audit.py                   # every reference cross-resolved
python -m pytest tests/ -q                     # metric conventions, bootstrap indexing,
                                               # mask morphology, multiple-comparison correction
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Layout

```
mangrove/      core package (config, paths, geo, datasets, metrics, models)
experiments/   data prep (exp1_*), splits (exp2_*), training & evaluation-design
               sweeps (exp3_*), visualisation (exp4_*), resumable run_*.sh launchers
analysis/      post-hoc analysis, figure/table generation, claim verification,
               figure-QA utilities
tests/         unit tests: change-F1 conventions, bootstrap index layout, label-mask
               morphology, F1-gap decomposition, multiple-comparison corrections
results/       metrics JSON (versioned), checkpoints/predictions (not versioned).
               Legacy sqrt-protocol runs at the top level; uniform-protocol runs
               under results/uniform/ (separate MANGROVE_RESULTS_ROOT)
figures/       manuscript figures, generated by analysis/make_figures.py
paper/         LaTeX manuscript (main.tex, refs.bib, cover_letter.tex)
logs/          run logs (not versioned)
```

## Citation

The manuscript is not yet published. Until it is, cite the repository:

```bibtex
@misc{split_audit_mangrove,
  title  = {When F1 and ROC-AUC Disagree: A 30-Point Metric Divergence Under
            Spatial Interleaving in Extreme-Imbalance Bi-Temporal Change Detection},
  author = {Yuan, Junzhe and Hao, Feier},
  year   = {2026},
  note   = {Manuscript, MDPI Remote Sensing (in preparation)}
}
```

## Status & limitations

- **Field interpretation** (2,400 designed points): tooling ready, not yet
  interpreted — so no field-accuracy claim is possible yet.
- **Zenodo deposit** of code and processed metrics: pending.
- Runnable but not reported in the current manuscript: strict-SCL mask
  sensitivity, tide-matched subset, and the sqrt-protocol full-architecture
  table.
