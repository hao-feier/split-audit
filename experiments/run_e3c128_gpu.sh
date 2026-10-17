#!/usr/bin/env bash
# E3-C: size-matched, overlap-free test of residual proximity at 128 px.
# 3 architectures x 5 split draws x 2 arms x 3 run seeds = 90 runs.
# Resumable: an existing metrics file means the run already completed.
set -u
PY=/home/lizhuo/venvs/junzhe-gpu/bin/python
ROOT=/home/lizhuo/Junzhe
OUT=$ROOT/results/e3c
LOG=$ROOT/logs/e3c128.log
mkdir -p "$OUT" "$ROOT/logs"
echo "=== E3-C start $(date -Is) ===" >> "$LOG"
n=0; skip=0; fail=0
for model in unet unet_cbam_msa temporal_unet; do
  for draw in 0 1 2 3 4; do
    for arm in scattered separated; do
      for seed in 2026 2027 2028; do
        f=$OUT/metrics_${model}_d${draw}_${arm}_s${seed}.json
        if [ -f "$f" ]; then skip=$((skip+1)); continue; fi
        n=$((n+1))
        if ! $PY "$ROOT/experiments/exp3_e3c128.py" --model "$model" \
             --draw "$draw" --arm "$arm" --seed "$seed" >> "$LOG" 2>&1; then
          fail=$((fail+1)); echo "  FAIL $model d$draw $arm s$seed" >> "$LOG"
        fi
      done
    done
  done
  echo "  [$model] done; ran=$n skipped=$skip failed=$fail" >> "$LOG"
done
echo "=== E3-C end $(date -Is): ran=$n skipped=$skip failed=$fail ===" >> "$LOG"
