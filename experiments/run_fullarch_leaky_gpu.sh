#!/bin/bash
# Extend the spatial-split inflation (tab:rand) to the 5 R2/R3 control
# architectures. The primary 5 (unet, unet_cbam, unet_cbam_msa, deeplabv3plus,
# temporal_unet) already have results/uniform/random_split_k{1,2,3}/; these 5 do
# not, so we train them under the SAME protocol as the existing counterfactual
# (experiments/run_uniform_counterfactual_gpu.sh):
#   exp3_random_split.py, --cw-mode none (uniform, primary), split draws ss=1..3,
#   5 models x 3 run seeds (2026-2028) -> results/uniform/random_split_k<ss>/
#   (random_test = leaky R/R, isolated_test = R/I)
#
# NOTE: exp3_random_split.py now resolves the temporal (in_ch=9) set from
#   mangrove.models.DUAL_INPUT_MODELS (was hard-coded to {temporal_unet}), so
#   temporal_unet_gn / fc_siam / changeformer get the correct dual-epoch input.
#   temporal-layout models use batch 2 (OOM headroom, as temporal_unet did);
#   the static ones (unet_msa, deeplabv3plus_pre) use batch 8.
#
# 45 runs, ~4 min/run on the RTX 4080 -> ~4-5 h. Resumable (skips runs whose
# metrics JSON exists); ONE training process at a time; NO auto-retry: on
# failure the loop HALTS for inspection.
#
# Run:
#   cd /home/lizhuo/Junzhe && setsid nohup bash experiments/run_fullarch_leaky_gpu.sh \
#     > logs/fullarch_leaky_gpu.log 2>&1 < /dev/null &
#   tail -f logs/fullarch_leaky_gpu.log
set -eu
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

SS_LIST=${SS_LIST:-"1 2 3"}
SEEDS=${SEEDS:-"2026 2027 2028"}
# temporal-layout (bs=2) first so any issue surfaces early, then the static ones
MODELS=${MODELS:-"temporal_unet_gn changeformer fc_siam unet_msa deeplabv3plus_pre"}

export MANGROVE_RESULTS_ROOT=results/uniform
export MANGROVE_SPLITS_ROOT=data/splits

echo "[$(date '+%F %T')] full-arch leaky sweep start: models=[$MODELS] ss=[$SS_LIST] seeds=[$SEEDS]"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
for ss in $SS_LIST; do
  if [ ! -f "data/splits/vietnam_test_rand${ss}.json" ]; then
    echo "[$(date '+%F %T')] regenerating random split k=$ss"
    $PY experiments/make_random_split.py --seed "$ss" > "logs/make_random_split_k${ss}.log" 2>&1
  fi
done

bs_for() {
  case "$1" in
    temporal_unet|temporal_unet_gn|fc_siam|changeformer) echo 2 ;;
    *) echo 8 ;;
  esac
}

randrun() {
  local ss="$1" name="$2" seed="$3"
  local mf="results/uniform/random_split_k${ss}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP k=$ss $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs; bs=$(bs_for "$name")
  echo "[$(date '+%F %T')] START k=$ss $name seed=$seed bs=$bs cw=none"
  if $PY experiments/exp3_random_split.py --model "$name" --seed "$seed" \
        --split-seed "$ss" --out-subdir "random_split_k${ss}" \
        --batch-size "$bs" --epochs 100 --patience 15 --threads 2 \
        --cw-mode none --device auto \
        > "logs/fullarch_${name}_randk${ss}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK k=$ss $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL k=$ss $name seed=$seed (see logs/fullarch_${name}_randk${ss}_s${seed}.log)"
    return 1
  fi
}

for name in $MODELS; do
  for ss in $SS_LIST; do
    for seed in $SEEDS; do
      randrun "$ss" "$name" "$seed"
    done
  done
done
echo "[$(date '+%F %T')] full-arch leaky sweep DONE -> results/uniform/random_split_k{1,2,3}/"
