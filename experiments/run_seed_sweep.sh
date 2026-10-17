#!/bin/bash
# Stage 5 robustness (2026-09-21): 3-seed repeat (2026/2027/2028) of every DNN
# so the comparison + ablation tables report mean±std instead of one seed.
#
# Unified recipe for ALL runs: batch 2 (7 GB RAM host: bs>=4 OOMs when anything
# else is resident — measured 2026-09-21), 100 epochs max, early stop patience
# 15 on val mIoU, weighted-CE + dice, OMP_NUM_THREADS=2.
# The seed-2026 pilot runs (2026-09-20, 32-core/15.9 GB host, per-model batch
# 4-8) are archived as *_pilot2026* and superseded.
#
# Resumable: a (model, seed) is skipped when its metrics JSON already exists.
# ONLY ONE training process at a time on this 7 GB host — never run probes
# or inference concurrently with the sweep (OOM-killed runs, 2026-09-21).
set -u
cd /mnt/fu-4070/Junzhe
PY=/usr/bin/python3
export OMP_NUM_THREADS=2

SEEDS=${SEEDS:-"2026 2027 2028"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

run() {
  local name="$1" seed="$2"
  if [ -f "results/metrics_${name}_s${seed}.json" ]; then
    echo "[$(date '+%F %T')] SKIP $name seed=$seed (metrics exist)"
    return 0
  fi
  echo "[$(date '+%F %T')] START $name seed=$seed"
  if $PY experiments/exp3_train.py --model "$name" --seed "$seed" --batch-size 2 \
       > "logs/train_${name}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL $name seed=$seed (see logs/train_${name}_s${seed}.log)"
    return 1
  fi
}

for seed in $SEEDS; do
  for name in $MODELS; do
    run "$name" "$seed"
  done
done
echo "[$(date '+%F %T')] SEED_SWEEP_DONE"
