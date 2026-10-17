#!/usr/bin/env bash
# Extend the spatial-interleaving counterfactual from 3 to 10 independently
# drawn train/test partitions so the leaky-minus-honest F1_change gap can carry
# a partition-level interval (see analysis/split_replicate_ci.py).
#
# Trains ONLY the R/R cell (random train + random test, cw=none) for splits
# k=1..10 x run-seeds 2026/2027/2028 x the 10 architectures, writing to
# results/uniform/random_split_k{ss}/. Splits k=1/2/3 already exist and are
# skipped, so this trains k=4..10 (210 runs, ~7 min/run on one 4080, ~24 h).
#
# Fault-tolerant: a single failing run is logged and the sweep CONTINUES (the
# base counterfactual launcher halts instead); failures are listed at the end
# so they can be re-run. Idempotent/resumable: re-launching skips completed
# runs. The I/R cell and its aggregation are NOT part of this CI extension and
# are intentionally not run here.
#
# Usage:
#   nohup bash experiments/run_split_replicate_ci.sh > logs/split_replicate_ci.launch.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2
export MANGROVE_RESULTS_ROOT=results/uniform
export MANGROVE_SPLITS_ROOT=data/splits

SS_LIST=${SS_LIST:-"1 2 3 4 5 6 7 8 9 10"}
SEEDS_R=${SEEDS_R:-"2026 2027 2028"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa unet_msa deeplabv3plus deeplabv3plus_pre changeformer fc_siam temporal_unet temporal_unet_gn"}

bs_for() {  # dual-input / temporal U-Net family needs a smaller batch on 12 GB
  case "$1" in
    changeformer|fc_siam|temporal_unet|temporal_unet_gn) echo 2 ;;
    *) echo 8 ;;
  esac
}

mkdir -p logs
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
FAILS=()
RAN=0; SKIPPED=0
for ss in $SS_LIST; do
  if [ ! -f "data/splits/vietnam_test_rand${ss}.json" ]; then
    echo "[$(date '+%F %T')] generating random split k=$ss"
    "$PY" experiments/make_random_split.py --seed "$ss" > "logs/make_random_split_k${ss}.log" 2>&1 \
      || { echo "[$(date '+%F %T')] FATAL: could not generate split k=$ss"; exit 1; }
  fi
  for seed in $SEEDS_R; do
    for name in $MODELS; do
      mf="results/uniform/random_split_k${ss}/metrics_${name}_s${seed}.json"
      if [ -f "$mf" ]; then
        SKIPPED=$((SKIPPED+1)); continue
      fi
      bs=$(bs_for "$name")
      echo "[$(date '+%F %T')] START k=$ss $name seed=$seed bs=$bs cw=none"
      if "$PY" experiments/exp3_random_split.py --model "$name" --seed "$seed" \
            --split-seed "$ss" --out-subdir "random_split_k${ss}" \
            --batch-size "$bs" --epochs 100 --patience 15 --threads 2 \
            --cw-mode none --device auto \
            > "logs/uniform_ci_${name}_randk${ss}_s${seed}.log" 2>&1; then
        echo "[$(date '+%F %T')] OK    k=$ss $name seed=$seed"
        RAN=$((RAN+1))
      else
        echo "[$(date '+%F %T')] FAIL  k=$ss $name seed=$seed (log: logs/uniform_ci_${name}_randk${ss}_s${seed}.log)"
        FAILS+=("k${ss}_${name}_s${seed}")
      fi
    done
  done
done

echo "=============================================================="
echo "[$(date '+%F %T')] RAN=$RAN  SKIPPED=$SKIPPED  FAILED=${#FAILS[@]}"
if [ "${#FAILS[@]}" -gt 0 ]; then
  echo "FAILED (re-run these): ${FAILS[*]}"
fi

if [ "${#FAILS[@]}" -eq 0 ]; then
  echo "[$(date '+%F %T')] computing partition-level CI over k=1..10"
  "$PY" analysis/split_replicate_ci.py 10 > logs/split_replicate_ci_result.log 2>&1 \
    && { echo "[$(date '+%F %T')] CI:"; sed -n '1,60p' logs/split_replicate_ci_result.log; } \
    || echo "[$(date '+%F %T')] WARN: CI aggregation failed"
  echo "[$(date '+%F %T')] SPLIT_REPLICATE_CI_DONE"
else
  echo "[$(date '+%F %T')] SPLIT_REPLICATE_CI_INCOMPLETE (see FAILED list)"
fi
