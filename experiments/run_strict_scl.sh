#!/usr/bin/env bash
# Strict-SCL-mask sensitivity.
#
# Re-composite Vietnam with the STRICT clean-surface mask (SCL 4-6 only:
# vegetation / not-vegetated / water; saturated, dark, shadow and unclassified
# pixels masked out), rebuild the patch features from those composites, then
# retrain unet x 3 seeds on the SAME 42-contrast-style both-arm protocol and
# compare the honest-vs-leaky gap to the current (loose-mask) result.
#
# Everything is written to SEPARATE roots so the current data/processed and
# results/uniform are untouched:
#   MANGROVE_PROCESSED_ROOT = data/processed_strict_scl
#   MANGROVE_RESULTS_ROOT   = results/strict_scl
#   MANGROVE_CLEAR_SCL      = 4,5,6
set -uo pipefail
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE
export MANGROVE_PROCESSED_ROOT=data/processed_strict_scl
export MANGROVE_CLEAR_SCL=4,5,6
export MANGROVE_RESULTS_ROOT=results/strict_scl
mkdir -p logs data/processed_strict_scl results/strict_scl

echo "[$(date '+%F %T')] strict-SCL pipeline start (CLEAR_SCL=4,5,6)"

# --- 1. strict composites (Vietnam, both epochs) ---
for epoch in t1 t2; do
  echo "[$(date '+%F %T')] composite vietnam/$epoch (strict)"
  if ! $PY experiments/exp1_composite.py --region vietnam --epoch "$epoch" \
        > "logs/strict_scl_comp_${epoch}.log" 2>&1; then
    echo "[$(date '+%F %T')] FAIL composite vietnam/$epoch"; tail -5 "logs/strict_scl_comp_${epoch}.log"; exit 1
  fi
done

# --- 2. labels (SCL-independent; re-run into the strict dir for self-containment) ---
echo "[$(date '+%F %T')] labels vietnam"
if ! $PY experiments/exp1_labels.py --region vietnam \
      > "logs/strict_scl_labels.log" 2>&1; then
  echo "[$(date '+%F %T')] FAIL labels"; tail -5 "logs/strict_scl_labels.log"; exit 1
fi

# --- 3. patch/feature extraction (strict composites + labels -> features.npz) ---
echo "[$(date '+%F %T')] patch/feature extraction vietnam"
if ! $PY experiments/exp1_patch.py --region vietnam \
      > "logs/strict_scl_patch.log" 2>&1; then
  echo "[$(date '+%F %T')] FAIL patch"; tail -5 "logs/strict_scl_patch.log"; exit 1
fi
echo "[$(date '+%F %T')] data rebuilt under $MANGROVE_PROCESSED_ROOT"

# --- 4. retrain unet, split-seed 1, 3 run-seeds (both arms per run:
#        random_test = leaky, isolated_test = honest) ---
for seed in 2026 2027 2028; do
  mf="results/strict_scl/random_split_k1/metrics_unet_s${seed}.json"
  if [ -f "$mf" ]; then echo "[$(date '+%F %T')] SKIP unet k1 s${seed}"; continue; fi
  echo "[$(date '+%F %T')] train unet k1 s${seed}"
  if ! $PY experiments/exp3_random_split.py --model unet --seed "$seed" \
        --split-seed 1 --out-subdir random_split_k1 \
        --batch-size 8 --epochs 100 --patience 15 --threads 2 \
        --cw-mode none --device auto \
        > "logs/strict_scl_unet_k1_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] FAIL unet k1 s${seed}"; tail -8 "logs/strict_scl_unet_k1_s${seed}.log"; exit 1
  fi
done

echo "[$(date '+%F %T')] strict-SCL unet k1 x3 DONE -> results/strict_scl/random_split_k1/"
