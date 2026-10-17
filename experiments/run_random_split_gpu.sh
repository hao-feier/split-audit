#!/bin/bash
# Random-split (spatially-correlated) control sweep on Fu4080-server GPU.
#
# Counterfactual to run_seed_sweep_gpu.sh: same 5 models x 3 seeds, SAME
# hyperparameters (batch 8 / 2-for-temporal, 100 epochs, patience 15, cw sqrt,
# lr 1e-3), the ONLY change is the train/val/test assignment (spatially
# interleaved random split instead of the 2 km buffer-isolated split).
#
# Isolation: outputs go to results/random_split/ (NOT the top-level
# results/metrics_*_s<seed>.json the paper aggregator globs).
#
# Robustness: resumable (skips a run whose metrics JSON already exists),
# ONE training process at a time, and NO auto-retry -- if a run fails or
# diverges it is logged and the loop STOPS for that run so we inspect before
# any re-run (per instruction: 跑飞不要重新跑).
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

SEEDS=${SEEDS:-"2026 2027 2028"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

# 1) build the random split once (idempotent, fixed RNG seed)
$PY experiments/make_random_split.py || { echo "FATAL: split gen failed"; exit 1; }
echo "---- split ready ----"

# 2) single pre-flight: confirm the GPU is free before launching
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

run() {
  local name="$1" seed="$2"
  local mf="results/random_split/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START $name seed=$seed (bs=$bs)"
  if $PY experiments/exp3_random_split.py --model "$name" --seed "$seed" \
        --batch-size $bs --device auto \
        > "logs/train_${name}_rand_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL $name seed=$seed (see logs/train_${name}_rand_s${seed}.log)"
    # no auto-retry: stop and let a human inspect before any re-run
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

rc=0
for seed in $SEEDS; do
  for name in $MODELS; do
    run "$name" "$seed" || { rc=1; break 2; }
  done
done
if [ "$rc" -eq 0 ]; then
  echo "[$(date '+%F %T')] RANDOM_SPLIT_SWEEP_DONE"
  $PY analysis/random_split_contrast.py || echo "WARN: comparison step failed"
else
  echo "[$(date '+%F %T')] RANDOM_SPLIT_SWEEP_INCOMPLETE (rc=$rc)"
fi
