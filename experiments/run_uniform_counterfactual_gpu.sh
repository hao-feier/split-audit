#!/bin/bash
# Uniform-weight (PRIMARY protocol, cw none) counterfactual sweep for the
# spatial 2x2 train/test factorial. Closes the protocol gap between the
# PRIMARY main sweep (results/uniform/, done 2026-09-28) and the headline
# inflation counterfactual, which so far exists only under the secondary
# sqrt protocol (results/random_split_k*, results/ir_k*).
#
# Cells (same split draws, same recipe as the sqrt-era runs; only
# --cw-mode none differs):
#   R/R + R/I : exp3_random_split.py, split draws ss=1..3,
#              5 models x 3 run seeds (2026-2028, parity with sqrt era)
#              -> results/uniform/random_split_k<ss>/
#                 (random_test = R/R, isolated_test = R/I)
#   I/R       : exp3_ir_split.py, test draws k=1..3,
#              5 models x 5 run seeds (2026-2030, parity with sqrt era)
#              -> results/uniform/ir_k<k>/  (ir_test = I/R)
#   I/I       : already done = results/uniform/ main sweep (10 models x 5 seeds)
#
# 45 + 75 = 120 runs, ~4 min/run on the RTX 4080 Laptop -> ~8 h.
# Resumable (skips runs whose metrics JSON exists); ONE training process at
# a time; NO auto-retry: on failure the loop HALTS for inspection.
# After the sweep, re-aggregates under the uniform root:
#   results/uniform/random_split_sweep.json, results/uniform/ir_factorial.json
#
# Run on Fu4080-server:
#   cd /home/lizhuo/Junzhe && setsid nohup bash experiments/run_uniform_counterfactual_gpu.sh \
#     > logs/uniform_counterfactual_gpu.log 2>&1 < /dev/null &
# Watch:
#   tail -f logs/uniform_counterfactual_gpu.log
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

SS_LIST=${SS_LIST:-"1 2 3"}              # random-split draws (R/R + R/I cell)
K_LIST=${K_LIST:-"1 2 3"}                # random-test draws  (I/R cell)
SEEDS_R=${SEEDS_R:-"2026 2027 2028"}      # run seeds, R cells (sqrt-era parity)
SEEDS_I=${SEEDS_I:-"2026 2027 2028 2029 2030"}  # run seeds, I/R cell (sqrt-era parity)
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

export MANGROVE_RESULTS_ROOT=results/uniform
export MANGROVE_SPLITS_ROOT=data/splits

# pre-flight: GPU status + split files present
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
for ss in $SS_LIST; do
  if [ ! -f "data/splits/vietnam_test_rand${ss}.json" ]; then
    echo "[$(date '+%F %T')] regenerating random split k=$ss"
    $PY experiments/make_random_split.py --seed "$ss" \
      > "logs/make_random_split_k${ss}.log" 2>&1
  fi
done

randrun() {  # <split_seed> <model> <run_seed>   (R/R + R/I cell)
  local ss="$1" name="$2" seed="$3"
  local mf="results/uniform/random_split_k${ss}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP R k=$ss $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START R k=$ss $name seed=$seed bs=$bs cw=none"
  if $PY experiments/exp3_random_split.py --model "$name" --seed "$seed" \
        --split-seed "$ss" --out-subdir "random_split_k${ss}" \
        --batch-size "$bs" --epochs 100 --patience 15 --threads 2 \
        --cw-mode none --device auto \
        > "logs/uniform_${name}_randk${ss}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK R k=$ss $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL R k=$ss $name seed=$seed (see logs/uniform_${name}_randk${ss}_s${seed}.log)"
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

irrun() {  # <draw_k> <model> <run_seed>   (I/R cell)
  local k="$1" name="$2" seed="$3"
  local mf="results/uniform/ir_k${k}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP I/R k=$k $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START I/R k=$k $name seed=$seed bs=$bs cw=none"
  if $PY experiments/exp3_ir_split.py --model "$name" --seed "$seed" \
        --ir-k "$k" --batch-size "$bs" --epochs 100 --patience 15 --threads 2 \
        --cw-mode none --device auto \
        > "logs/uniform_${name}_ir${k}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK I/R k=$k $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL I/R k=$k $name seed=$seed (see logs/uniform_${name}_ir${k}_s${seed}.log)"
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

rc=0
for ss in $SS_LIST; do
  for seed in $SEEDS_R; do
    for name in $MODELS; do
      randrun "$ss" "$name" "$seed" || { rc=1; break 3; }
    done
  done
done
if [ "$rc" -eq 0 ]; then
  for k in $K_LIST; do
    for seed in $SEEDS_I; do
      for name in $MODELS; do
        irrun "$k" "$name" "$seed" || { rc=1; break 3; }
      done
    done
  done
fi

if [ "$rc" -eq 0 ]; then
  echo "[$(date '+%F %T')] sweep complete -- re-aggregating under results/uniform/"
  $PY analysis/random_split_sweep.py > logs/uniform_random_split_sweep.log 2>&1 \
    || echo "[$(date '+%F %T')] WARN: random_split_sweep aggregation failed"
  $PY analysis/aggregate_ir_factorial.py > logs/uniform_ir_factorial.log 2>&1 \
    || echo "[$(date '+%F %T')] WARN: ir_factorial aggregation failed"
  echo "[$(date '+%F %T')] UNIFORM_COUNTERFACTUAL_DONE"
else
  echo "[$(date '+%F %T')] UNIFORM_COUNTERFACTUAL_STOPPED_WITH_ERROR"
fi
