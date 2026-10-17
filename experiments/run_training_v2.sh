#!/bin/bash
# Stage 3-4 rerun (2026-09-20): weighted-CE + dice loss, epochs 100, patience 15.
# ml_baselines already done (results/metrics_ml_baselines.json) — not rerun.
# temporal_unet at batch 4 (batch 8 was OOM-killed on 2026-09-19).
set -u
cd /home/lizhuo/Junzhe
PY=/usr/bin/python3.12

run() {
  local name="$1"; shift
  echo "[$(date '+%F %T')] START $name"
  if $PY "$@" > "logs/train_${name}_v2.log" 2>&1; then
    echo "[$(date '+%F %T')] OK $name"
  else
    echo "[$(date '+%F %T')] FAIL $name (see logs/train_${name}_v2.log)"
  fi
}

run unet experiments/exp3_train.py --model unet
run unet_cbam experiments/exp3_train.py --model unet_cbam
run unet_cbam_msa experiments/exp3_train.py --model unet_cbam_msa
run deeplabv3plus experiments/exp3_train.py --model deeplabv3plus
run temporal_unet experiments/exp3_train.py --model temporal_unet --batch-size 4
echo "[$(date '+%F %T')] ALL_TRAINING_V2_DONE"
