#!/bin/bash
# Architecture-extension GPU experiments (2026-09-24), single RTX 4080 Laptop, 12 GB.
#
# Stages (resumable: a (base, seed) is skipped when its metrics JSON exists;
# ONE training process at a time):
#   1. changeformer        x3 seeds  (Transformer-based CD baseline)
#   2. deeplabv3plus_pre   x3 seeds  (ImageNet-pretrained backbone control)
#   3. extra seeds 2029/2030 x 5 main models (seed count 3 -> 5)
#   4. deterministic audit, all 8 architectures: det + detc + nd
#   5. random-split sweep: 10 split seeds x 5 models x 3 run seeds
#
# Run:  setsid nohup bash experiments/run_arch_extension_gpu.sh > logs/arch_extension_gpu.log 2>&1 &
# Watch: tail -f logs/arch_extension_gpu.log
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

log() { echo "[$(date '+%F %T')] $*"; }

train() {  # train <base> <model> <seed> <bs> [extra args...]
  local base="$1" model="$2" seed="$3" bs="$4"; shift 4
  if [ -f "results/metrics_${base}_s${seed}.json" ]; then
    log "SKIP $base seed=$seed (metrics exist)"; return 0
  fi
  log "START $base seed=$seed bs=$bs $*"
  if $PY experiments/exp3_train.py --model "$model" --seed "$seed" --batch-size "$bs" \
       --device auto "$@" > "logs/train_${base}_s${seed}.log" 2>&1; then
    log "OK $base seed=$seed"
  else
    log "FAIL $base seed=$seed (see logs/train_${base}_s${seed}.log)"
    return 1
  fi
}

randtrain() {  # randtrain <split_seed> <model> <seed> <bs>
  local ss="$1" model="$2" seed="$3" bs="$4"
  local sub="random_split_k${ss}"
  local mf="results/${sub}/metrics_${model}_s${seed}.json"
  if [ -f "$mf" ]; then
    log "SKIP rand k=$ss $model seed=$seed (metrics exist)"; return 0
  fi
  $PY experiments/make_random_split.py --seed "$ss" > "logs/make_random_split_k${ss}.log" 2>&1
  log "START rand k=$ss $model seed=$seed bs=$bs"
  if $PY experiments/exp3_random_split.py --model "$model" --seed "$seed" \
       --batch-size "$bs" --device auto --split-seed "$ss" --out-subdir "$sub" \
       > "logs/train_${model}_randk${ss}_s${seed}.log" 2>&1; then
    log "OK rand k=$ss $model seed=$seed"
  else
    log "FAIL rand k=$ss $model seed=$seed (see log)"
    return 1
  fi
}

log "===== ARCH EXTENSION QUEUE START ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ---------------------------------------------------------------- 1. changeformer
for seed in 2026 2027 2028; do
  if ! train "changeformer" "changeformer" "$seed" 8; then
    if grep -qi "out of memory" "logs/train_changeformer_s${seed}.log" 2>/dev/null; then
      log "changeformer OOM at bs=8, retrying bs=4"
      rm -f "results/train_log_changeformer_s${seed}.jsonl"
      train "changeformer" "changeformer" "$seed" 4
    fi
  fi
done
log "STAGE 1 (changeformer) done"

# ---------------------------------------------------------------- 2. pretrained deeplab
for seed in 2026 2027 2028; do
  train "deeplabv3plus_pre" "deeplabv3plus_pre" "$seed" 8
done
log "STAGE 2 (deeplabv3plus_pre) done"

# ---------------------------------------------------------------- 3. extra seeds
for seed in 2029 2030; do
  for model in unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet; do
    bs=8; [ "$model" = "temporal_unet" ] && bs=2
    train "$model" "$model" "$seed" "$bs"
  done
done
log "STAGE 3 (seeds 2029/2030) done"

# ---------------------------------------------------------------- 4. det audit all 8
ARCHS="unet unet_cbam unet_msa unet_cbam_msa deeplabv3plus temporal_unet temporal_unet_gn fc_siam"
for arch in $ARCHS; do
  bs=8; case "$arch" in temporal_unet|temporal_unet_gn) bs=2;; esac
  train "${arch}_det"  "$arch" 2026 "$bs" --deterministic --tag det
  train "${arch}_detc" "$arch" 2026 "$bs" --deterministic --tag detc
  train "${arch}_nd"   "$arch" 2026 "$bs" --tag nd
done
log "STAGE 4 (deterministic audit, all architectures) done"

# ---------------------------------------------------------------- 5. random split sweep
for ss in 1 2 3 4 5 6 7 8 9 10; do
  for seed in 2026 2027 2028; do
    for model in unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet; do
      bs=8; [ "$model" = "temporal_unet" ] && bs=2
      randtrain "$ss" "$model" "$seed" "$bs" || { log "HALTED (no auto-retry)"; exit 1; }
    done
  done
done
log "ARCH_EXTENSION_QUEUE_DONE"
