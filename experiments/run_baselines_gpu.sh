#!/bin/bash
# Classic/index baseline sweep (Stage 3; paper tab:main baselines + sensitivity).
#
# Wraps the four baseline scripts into one resumable, no-auto-retry launcher,
# aligned with the DNN sweep launchers (run_seed_sweep_gpu.sh / run_ir_factorial_gpu.sh):
#   CVA (NDVI / Norm index difference)               -> results/metrics_cva_baselines.json
#   pixel Random Forest + SVM RBF, no weighting      -> results/metrics_ml_baselines.json
#   pixel Random Forest + SVM RBF, balanced weighting-> results/metrics_ml_baselines_weighted.json
#   tuned Random Forest (manual grid)                -> results/metrics_tuned_baselines.json
#   PCC (per-epoch presence U-Net), 3 seeds          -> results/metrics_pcc_s{seed}.json
#
# CVA / pixel-ML / tuned-ML are CPU (numpy + sklearn, a few minutes each);
# PCC trains a 2-branch U-Net per seed (GPU, same recipe as the main sweep:
# bs 8, patience 15, weighted CE). Resumable: a step is skipped when its
# output JSON already exists. One process at a time, NO auto-retry: on
# failure the loop STOPS for inspection.
set -u
cd /home/lizhuo/Junzhe
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
export OMP_NUM_THREADS=2

SEEDS=${SEEDS:-"2026 2027 2028"}
PCC_CW=${PCC_CW:-sqrt}          # sqrt (default) | none | inv -- matches the PCC script's tag
case "$PCC_CW" in
  none) PCC_TAG="_none" ;;
  inv)  PCC_TAG="_inv" ;;
  *)    PCC_TAG="" ;;
esac

# pre-flight: GPU status (PCC is the only GPU step; CPU steps run regardless)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true

# step <label> <out-json> <command...> : skip if the output exists, else run+log+halt-on-fail
step() {
  local label="$1" out="$2"; shift 2
  if [ -f "$out" ]; then
    echo "[$(date '+%F %T')] SKIP $label (exists: $out)"
    return 0
  fi
  echo "[$(date '+%F %T')] START $label"
  if "$@" > "logs/baseline_${label}.log" 2>&1; then
    echo "[$(date '+%F %T')] OK $label"
  else
    echo "[$(date '+%F %T')] FAIL $label (see logs/baseline_${label}.log)"
    echo "[$(date '+%F %T')] HALTED (no auto-retry) -- inspect the log above"
    return 1
  fi
}

rc=0
step cva           results/metrics_cva_baselines.json            $PY experiments/exp3_baseline_cva.py            || rc=1
[ "$rc" -eq 0 ] && step ml_none    results/metrics_ml_baselines.json            $PY experiments/exp3_baseline_ml.py --cw none      || rc=1
[ "$rc" -eq 0 ] && step ml_weighted results/metrics_ml_baselines_weighted.json $PY experiments/exp3_baseline_ml.py --cw balanced || rc=1
[ "$rc" -eq 0 ] && step ml_tuned   results/metrics_tuned_baselines.json         $PY experiments/exp3_baseline_ml_tuned.py       || rc=1

for seed in $SEEDS; do
  [ "$rc" -eq 0 ] && step "pcc_s${seed}" "results/metrics_pcc${PCC_TAG}_s${seed}.json" \
    $PY experiments/exp3_baseline_pcc.py --seeds "$seed" --cw-mode "$PCC_CW" || rc=1
  [ "$rc" -ne 0 ] && break
done

if [ "$rc" -eq 0 ]; then
  echo "[$(date '+%F %T')] BASELINE_SWEEP_DONE"
else
  echo "[$(date '+%F %T')] BASELINE_SWEEP_STOPPED_WITH_ERROR"
fi
exit "$rc"
