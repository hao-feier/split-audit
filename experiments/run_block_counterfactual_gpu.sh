#!/bin/bash
# Non-overlapping, block-interleaved counterfactual
# with a SIZE-MATCHED, PREVALENCE-MATCHED honest control.
#
# This is the overlap-free counterfactual with a
# non-overlapping random split. It is only feasible at reduced scale, and the
# feasibility probe is itself a result (see experiments/make_block_split.py):
#   * 0 of 4000 reseeded random permutations of the 60-patch pool give zero
#     train/test footprint overlap (each patch overlaps a mean of 4 others, so
#     40 train patches necessarily cover the region);
#   * a buffered random split (drop any patch within 0.512 km of a differently
#     classed patch) keeps ZERO test patches;
#   * the block design keeps 19-25 train / 3-6 test against the protocol's
#     40/10/10.
#
# Design: for each split seed, two arms with IDENTICAL counts, IDENTICAL
# recipe, and ZERO footprint overlap in both:
#   _blk<ss>  block-interleaved; every test patch lies INSIDE the training
#             extent; a one-block void band guarantees no overlap
#   _hon<ss>  the isolated protocol's own zones, subsampled to the same counts
#             and matched on test-set change prevalence; buffer-isolated
# blk - hon therefore isolates spatial ARRANGEMENT with neither pixel overlap,
# training-set size, nor case mix as a confound. Each run additionally scores
# the partner arm with the same model, giving a within-model contrast too.
#
# 4 seeds x 5 models x 2 run seeds x 2 arms = 80 runs, ~5 min/run (the training
# set is 19-25 patches, so 3 batches/epoch) -> ~7 h.
#
# Resumable (skips runs whose metrics JSON exists); ONE training process at a
# time; NO auto-retry: on failure the loop HALTs for inspection.
#
# Run:
#   cd /home/lizhuo/Junzhe && setsid nohup bash experiments/run_block_counterfactual_gpu.sh \
#     > logs/block_counterfactual_gpu.log 2>&1 < /dev/null &
# Watch: tail -f logs/block_counterfactual_gpu.log
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python

SS_LIST=${SS_LIST:-"12 17 38 62"}
SEEDS=${SEEDS:-"2026 2027"}
MODELS=${MODELS:-"unet unet_cbam unet_cbam_msa deeplabv3plus temporal_unet"}

export MANGROVE_RESULTS_ROOT=results/uniform
export MANGROVE_SPLITS_ROOT=data/splits

echo "[$(date '+%F %T')] pre-flight GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# regenerate the block + matched-honest splits if any is missing
NEED_GEN=0
for ss in $SS_LIST; do
  for arm in blk hon; do
    for sp in train val test; do
      [ -f "data/splits/vietnam_${sp}_${arm}${ss}.json" ] || NEED_GEN=1
    done
  done
done
if [ "$NEED_GEN" -eq 1 ]; then
  echo "[$(date '+%F %T')] regenerating block splits for: $SS_LIST"
  $PY experiments/make_block_split.py --n 4 > logs/make_block_split.log 2>&1 \
    || { echo "[$(date '+%F %T')] FATAL: make_block_split failed"; exit 1; }
fi

run_one() {  # <arm> <ss> <model> <run_seed>
  local arm="$1" ss="$2" name="$3" seed="$4"
  local sub="block_${arm}${ss}"
  local tag="_${arm}${ss}"
  local partner="_hon${ss}"
  [ "$arm" = "hon" ] && partner="_blk${ss}"
  local mf="results/uniform/${sub}/metrics_${name}_s${seed}.json"
  if [ -f "$mf" ]; then
    echo "[$(date '+%F %T')] SKIP $arm k=$ss $name seed=$seed (metrics exist)"
    return 0
  fi
  local bs=8
  [ "$name" = "temporal_unet" ] && bs=2
  echo "[$(date '+%F %T')] START $arm k=$ss $name seed=$seed bs=$bs cw=none"
  if $PY experiments/exp3_block_split.py --model "$name" --seed "$seed" \
        --split-tag "$tag" --partner-tag "$partner" --out-subdir "$sub" \
        --batch-size "$bs" --epochs 100 --patience 15 --threads 2 \
        --cw-mode none --device auto \
        > "logs/block_${name}_${arm}${ss}_s${seed}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK $arm k=$ss $name seed=$seed"
  else
    echo "[$(date '+%F %T')] FAIL $arm k=$ss $name seed=$seed (see logs/block_${name}_${arm}${ss}_s${seed}.log)"
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

rc=0
for ss in $SS_LIST; do
  for arm in blk hon; do
    for name in $MODELS; do
      for seed in $SEEDS; do
        run_one "$arm" "$ss" "$name" "$seed" || { rc=1; break 4; }
      done
    done
  done
done

if [ "$rc" -eq 0 ]; then
  echo "[$(date '+%F %T')] sweep complete -- aggregating"
  $PY analysis/aggregate_block_split.py > logs/block_aggregate.log 2>&1 \
    || echo "[$(date '+%F %T')] WARN: aggregate_block_split failed"
  echo "[$(date '+%F %T')] BLOCK_COUNTERFACTUAL_DONE"
else
  echo "[$(date '+%F %T')] BLOCK_COUNTERFACTUAL_STOPPED_WITH_ERROR"
fi
