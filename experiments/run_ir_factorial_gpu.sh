#!/bin/bash
# I/R factorial cell sweep on Fu4080-server GPU.
#
# Completes the 2x2 spatial train/test factorial:
#   I/I = main sweep (results/)        R/I + R/R = results/random_split_k*/
#   I/R = THIS sweep (results/ir_k{1,2,3}/)
#
# 3 random-test draws x 5 run seeds x 5 models = 75 runs. Training is
# identical to the main sweep (same spatial train/val zones, same recipe,
# cw sqrt); only the within-region test set differs (random, interleaved).
#
# Resumable (skips runs whose metrics JSON exists), ONE training process at
# a time, NO auto-retry: on failure the loop STOPS for inspection.
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

K_LIST=${K_LIST:-"1 2 3"}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

# pre-flight: GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

run() {
  local k="$1" name="$2" seed="$3"
  local mf="results/ir_k${k}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP k=$k $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START k=$k $name seed=$seed (bs=$bs)"
  if $PY experiments/exp3_ir_split.py --model "$name" --seed "$seed" \
        --ir-k "$k" --batch-size $bs --device auto \
        > "logs/train_${name}_ir${k}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK k=$k $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL k=$k $name seed=$seed (see logs/train_${name}_ir${k}_s${seed}.log)"
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

rc=0
for k in $K_LIST; do
  for seed in $SEEDS; do
    for name in $MODELS; do
      run "$k" "$name" "$seed" || { rc=1; break 3; }
    done
  done
done
if [ "$rc" -eq 0 ]; then
  echo "[$(date '+%F %T')] IR_FACTORSWEEP_DONE"
else
  echo "[$(date '+%F %T')] IR_FACTORSWEEP_STOPPED_WITH_ERROR"
fi
