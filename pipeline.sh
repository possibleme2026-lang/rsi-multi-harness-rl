#!/usr/bin/env bash
# Full experiment pipeline, in dependency order, with fail-fast.
#
# Written as one script because the stages are strictly sequential and the
# expensive ones (difficulty scan, two training runs, two evaluation sweeps)
# each take tens of minutes. Running them by hand invites the classic error of
# starting stage N+1 from a stale stage-N artifact.
#
# Stage order and why:
#   0. guards          cheap static checks; a harness that advertises a tool it
#                      never implemented would score low for a reason that has
#                      nothing to do with the model
#   1. smokes          correctness gates; abort before any GPU time
#   2. scan            measures pass rates for every (harness, task) cell,
#                      which train.py needs to drop zero-gradient rows
#   3. train single    baseline arm: one harness, free to overfit it
#   4. train multi     treatment arm: four harnesses, cannot
#   5. eval            baseline + both arms, on held-out tasks AND the
#                      held-out harness, then prints the ablation
#
# Usage:
#   bash pipeline.sh                      # full run
#   STEPS=20 N_EVAL=4 bash pipeline.sh
#   SKIP_SCAN=1 bash pipeline.sh          # reuse an existing scan
#   SKIP_TRAIN=1 bash pipeline.sh         # re-evaluate existing checkpoints

set -euo pipefail

cd "$(dirname "$0")"

RUN=./run.sh
STEPS="${STEPS:-40}"
N_SCAN="${N_SCAN:-8}"
N_EVAL="${N_EVAL:-4}"
OUT="${MULTIHARNESS_OUT:-$PWD/outputs}"
mkdir -p "$OUT"

TRAIN_TASKS="t1-01,t1-02,t1-03,t1-04,t1-05,t1-06,t1-07,t1-08,t2-01,t2-02,t2-03,t2-04,t3-01,t3-02,t3-03,t3-04"

banner() { printf '\n\n########## %s ##########\n\n' "$1"; }

banner "STAGE 0/5  static guards (no model, no GPU)"
# Tool-surface guard first: it is the cheapest check and it protects the
# meaning of every number downstream. A harness whose GUIDANCE advertises a
# tool it never implemented scores low for a reason that has nothing to do
# with the model — and that would be read as a capability result.
bash "$RUN" scripts/guard_tool_surface.py

banner "STAGE 1/5  correctness gates (no model)"
bash "$RUN" tests/test_path_errors.py
bash "$RUN" tests/test_shell_timeout.py
bash "$RUN" tests/test_scan_tooling.py
bash "$RUN" tests/smoke_env.py
bash "$RUN" tests/smoke_trl.py

banner "STAGE 2/5  difficulty scan (${N_SCAN} rollouts/cell, 4 harnesses x 16 tasks)"
if [ "${SKIP_SCAN:-0}" = "1" ] && [ -f "$OUT/scan_all.json" ]; then
  echo "SKIP_SCAN=1 and $OUT/scan_all.json exists -> reusing it"
else
  bash "$RUN" scripts/probe.py \
    --n "$N_SCAN" --tasks "$TRAIN_TASKS" --out "$OUT/scan_all.json"
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  banner "STAGE 3/5  train — SINGLE harness (baseline arm)"
  bash "$RUN" scripts/train.py \
    --mode single --steps "$STEPS" --scan "$OUT/scan_all.json" \
    --tag "single-s${STEPS}"

  banner "STAGE 4/5  train — MULTI harness (treatment arm)"
  bash "$RUN" scripts/train.py \
    --mode multi --steps "$STEPS" --scan "$OUT/scan_all.json" \
    --tag "multi-s${STEPS}"
else
  echo "SKIP_TRAIN=1 -> reusing checkpoints under $OUT"
fi

banner "STAGE 5/5  eval — baseline + single + multi in ONE process"
# All three arms run in a single invocation so the ablation table is produced
# with a shared seed and shared harness instances. Splitting this into three
# commands would lose the comparison and make the arms non-paired.
bash "$RUN" scripts/eval.py \
  --baseline \
  --adapter "$OUT/train-single-s${STEPS}/final" \
  --adapter "$OUT/train-multi-s${STEPS}/final" \
  --n "$N_EVAL" --out "$OUT/eval_ablation.json"

banner "PIPELINE COMPLETE"
echo "artifacts in $OUT:"
ls -1 "$OUT"
