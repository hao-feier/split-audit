#!/bin/bash
# Stage 5 robustness (2026-09-21): 3-seed repeat (2026/2027/2028) of every DNN
# on Fu4080-server's RTX 4080 Laptop (12 GB) so the comparison + ablation
# tables report mean±std instead of one seed.
#
# Recipe: batch 8, 100 epochs max, early stop patience 15 on val mIoU,
# weighted-CE (--cw-mode sqrt, decided by the matched 25-epoch probe) + dice.
# ~6 s/epoch on the 4080 vs ~100 s/epoch on the 4-core CPU box, so the whole
# 15-run sweep takes ~5 h instead of ~30 h.
#
# Resumable: a (model, seed) is skipped when its metrics JSON already exists.
# Run ONE training process at a time (12 GB VRAM, single GPU).
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

SEEDS=${SEEDS:-"2026 2027 2028"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

run() {
  local name="$1" seed="$2"
  if [ -f "results/metrics_${name}_s${seed}.json" ]; then
    echo "[$(date '+%F %T')] SKIP $name seed=$seed (metrics exist)"
    return 0
  fi
  # temporal_unet is a dual-branch model and OOMs the 12 GB card at batch 8
  # (temporal fusion concatenates [a,b,|a-b|,a*b] at every decoder level).
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START $name seed=$seed"
  if $PY experiments/exp3_train.py --model "$name" --seed "$seed" --batch-size $bs \
       --device auto > "logs/train_${name}_s${seed}.log" 2>&1; then
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
echo "[$(date '+%F %T')] GPU_SEED_SWEEP_DONE"
