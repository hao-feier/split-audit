#!/bin/bash
# 5-seed uniform-weight top-up of the main
# comparison, so the PRIMARY model comparison can be reported under the
# scheme selected by the strictest training-only protocol (nested CV ->
# uniform) with the same 5 seeds (2026-2030) as the sqrt main sweep.
#
# Already present in results/uniform/ (3 seeds, 5 core models) are SKIPPED.
# New runs: 5 core models x seeds 2029/2030 (10) + unet_msa, deeplabv3plus_pre,
# temporal_unet_gn, fc_siam, changeformer x 5 seeds (25) = 35 runs.
#
# Recipe identical to the main sweep (experiments/run_seed_sweep_gpu.sh) plus
# --cw-mode none; outputs go to results/uniform/ (MANGROVE_RESULTS_ROOT).
# Run on Fu4080-server:
#   cd /home/lizhuo/Junzhe && setsid nohup bash experiments/run_uniform_topup_gpu.sh \
#     > logs/uniform_topup_gpu.log 2>&1 < /dev/null &
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

# name:batch_size
MODELS="unet:8 unet_cbam:8 unet_msa:8 unet_cbam_msa:8 deeplabv3plus:8 \
        deeplabv3plus_pre:8 temporal_unet:2 temporal_unet_gn:2 fc_siam:8 changeformer:8"

run() {
  local name="$1" seed="$2" bs="$3"
  if [ -f "results/uniform/metrics_${name}_s${seed}.json" ]; then
    echo "[$(date '+%F %T')] SKIP $name seed=$seed (metrics exist)"
    return 0
  fi
  echo "[$(date '+%F %T')] START $name seed=$seed bs=$bs cw=none"
  env MANGROVE_RESULTS_ROOT=results/uniform MANGROVE_SPLITS_ROOT=data/splits \
    $PY experiments/exp3_train.py --model "$name" --seed "$seed" \
      --epochs 100 --patience 15 --batch-size "$bs" --device auto \
      --threads 2 --cw-mode none \
      > "logs/uniform_${name}_s${seed}.log" 2>&1
  if [ $? -eq 0 ]; then
    echo "[$(date '+%F %T')] OK $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL $name seed=$seed (see logs/uniform_${name}_s${seed}.log)"
    return 1
  fi
}

for spec in $MODELS; do
  name="${spec%%:*}"; bs="${spec##*:}"
  for seed in 2026 2027 2028 2029 2030; do
    run "$name" "$seed" "$bs"
  done
done
echo "[$(date '+%F %T')] UNIFORM_TOPUP_DONE"
