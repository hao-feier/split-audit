#!/usr/bin/env bash
# B4 strict-SCL sensitivity — TRAINING ONLY (data already built under
# data/processed_strict_scl with the same 76 patches as the baseline).
#
# DO NOT re-run exp1_composite/exp1_patch here: the strict dir holds strict-mask composites
# (features) + the ORIGINAL clear-fraction (patch selection) so the patch set is
# identical to the baseline. Re-compositing would overwrite that and drop the
# low-clear patches, breaking the index-based splits.
#
# Retrain unet x 3 seeds; exp3c reports BOTH arms per run:
#   random_test  = leaky (interleaved vietnam_test_rand1)
#   isolated_test= honest (buffer-isolated vietnam_test)
set -uo pipefail
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE
export MANGROVE_PROCESSED_ROOT=data/processed_strict_scl
export MANGROVE_RESULTS_ROOT=results/strict_scl
mkdir -p results/strict_scl logs

echo "[$(date '+%F %T')] strict-SCL training start (unet, split-seed 1, 3 seeds)"
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
  echo "[$(date '+%F %T')] OK unet k1 s${seed}"
done
echo "[$(date '+%F %T')] strict-SCL training DONE -> results/strict_scl/random_split_k1/"
