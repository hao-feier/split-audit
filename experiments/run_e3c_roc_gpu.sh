#!/usr/bin/env bash
# Re-run the full 90-cell E3-C sweep capturing ROC-AUC / AP alongside F1.
# Why a re-run and not a re-score: exp3_e3c128.py did not save checkpoints and
# a confusion matrix cannot yield ROC-AUC, so the softmax has to be captured
# during evaluation. Training is not bit-deterministic on this GPU (same seed
# moved change-F1 by 0.003 on a trial re-run), so F1 and ROC must come from the
# SAME run; the original F1 values are archived in results/e3c_f1_orig/ so the
# run-to-run drift can be reported instead of hidden.
set -u
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
cd "$(dirname "$0")/.."
for model in unet unet_cbam_msa temporal_unet; do
  for draw in 0 1 2 3 4; do
    for arm in scattered separated; do
      for seed in 2026 2027 2028; do
        out="results/e3c/metrics_${model}_d${draw}_${arm}_s${seed}.json"
        if [ -f "$out" ] && grep -q test_scores "$out"; then continue; fi
        echo "[$(date +%H:%M:%S)] $model d$draw $arm s$seed"
        "$PY" experiments/exp3_e3c128.py --model "$model" --draw "$draw" \
            --arm "$arm" --seed "$seed" 2>&1 | grep -E "F1ch|Error|Traceback"
      done
    done
  done
done
echo "SWEEP_DONE $(date +%H:%M:%S)"
