#!/bin/bash
# E3-B: higher-power overlap-free block-interleaved control.
#
# WHY THIS REPLACES THE ORIGINAL E3 (4 seeds), rather than adding to it:
#   the original four block splits were NOT four independent samples --
#   blk12 and blk38 share the identical test set (patches 49-51). The sign-test
#   premise ("four independent draws") therefore did not hold in v4, and the
#   split-seed variance that dominates E3 was inflated by two seeds scoring the
#   same three patches. E3-B uses 16 splits with 16 DISTINCT test sets.
#
# DESIGN (frozen before any model is trained -- see logs/e3b_selection_lock.txt):
#   double stratification, zero selection freedom: every candidate with
#     - test-set size >= 5 patches (v4 used 2-6; size heterogeneity was the
#       dominant variance source), and
#     - achievable prevalence-match residual <= 0.5 pp
#   qualifies, and all 16 qualifiers are used. Residual across the locked set:
#   max 0.461 pp, mean 0.167 pp.
#
# UNITS: 16 split seeds x 3 architectures x 2 run seeds x 2 arms = 192 runs.
#   architectures: unet (cleanest main-sweep control), unet_cbam_msa (the
#   paper's reference model), temporal_unet (the only dual-encoder family, so
#   the input representation is not uniformly a 27-channel concat).
#   The architecture dimension is near-saturated already: the v4 within-model
#   residual was positive for all five original architectures (+4.7 to +7.7 pp),
#   so a 14 h five-architecture variant would buy no new information.
#
# SCALE CAVEAT (unchanged by any of this): the block arm runs at 18-25 training
# and 5-6 test patches against the main protocol's 40/10/10. That is a
# structural cost of patch geometry -- a non-overlapping interleaved split is
# not achievable at full scale (0 of 4000 reseeded permutations give zero
# overlap). More splits narrow the interval; they do NOT remove the scale
# mismatch, and nothing here licenses extrapolation to the full protocol.
#
# Resumable (skips runs whose metrics JSON exists); ONE training at a time;
# NO auto-retry: on failure the loop HALTS for inspection.
#
# Run:
#   cd /home/lizhuo/Junzhe && setsid nohup bash experiments/run_e3b_gpu.sh \
#     > logs/e3b_gpu.log 2>&1 < /dev/null &
#   tail -f logs/e3b_gpu.log
set -eu
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

MODELS=${MODELS:-"unet unet_cbam_msa temporal_unet"}
RUN_SEEDS=${RUN_SEEDS:-"2026 2027"}
# frozen split list, read back from the pre-flight record (never re-derived here)
SPLITS=$(/home/lizhuo/venvs/junzhe-gpu/bin/python -c "
import json;print(' '.join(str(s['split_seed']) for s in json.load(open('results/e3b_splits.json'))['splits']))")

export MANGROVE_RESULTS_ROOT=results/uniform
export MANGROVE_SPLITS_ROOT=data/splits

[ -f logs/e3b_selection_lock.txt ] || { echo "FATAL: selection lock missing"; exit 1; }

echo "[$(date '+%F %T')] E3-B start"
echo "[$(date '+%F %T')] models=[$MODELS] run_seeds=[$RUN_SEEDS]"
echo "[$(date '+%F %T')] splits=[$SPLITS]"
echo "[$(date '+%F %T')] total runs: $(echo $SPLITS | wc -w) x 2 arms x $(echo $MODELS | wc -w) models x $(echo $RUN_SEEDS | wc -w) run seeds"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

bs_for() {
  case "$1" in
    temporal_unet|temporal_unet_gn|fc_siam|changeformer) echo 2 ;;
    *) echo 8 ;;
  esac
}

onerun() {  # <split_seed> <model> <run_seed> <arm>
  local ss="$1" name="$2" seed="$3" arm="$4"
  # both arms score their partner: blk<->hon. That is what makes the
  # WITHIN-MODEL contrast available (same trained model, two test sets).
  local partner=""
  if [ "$arm" = "blk" ]; then partner="--partner-tag _hon${ss}"; else partner="--partner-tag _blk${ss}"; fi
  local mf="results/uniform/block_${arm}${ss}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP ${arm}${ss} $name s=$seed (metrics exist)"
    return 0
  fi
  local bs; bs=$(bs_for "$name")
  echo "[$(date '+%F %T')] START ${arm}${ss} $name s=$seed bs=$bs cw=none"
  # shellcheck disable=SC2086
  if $PY experiments/exp3_block_split.py --model "$name" --seed "$seed" \
        --split-tag "_${arm}${ss}" $partner \
        --out-subdir "block_${arm}${ss}" \
        --batch-size "$bs" --epochs 100 --patience 15 --num-workers 2 \
        --cw-mode none --threads 2 --device auto \
        > "logs/e3b_${arm}${ss}_${name}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK ${arm}${ss} $name s=$seed"
  else
    echo "[$(date '+%F %T')] FAIL ${arm}${ss} $name s=$seed (see logs/e3b_${arm}${ss}_${name}_s${seed}.log)"
    return 1
  fi
}

for ss in $SPLITS; do
  for name in $MODELS; do
    for seed in $RUN_SEEDS; do
      onerun "$ss" "$name" "$seed" blk
      onerun "$ss" "$name" "$seed" hon
    done
  done
done
echo "[$(date '+%F %T')] E3-B DONE -> results/uniform/block_{blk,hon}<ss>/"