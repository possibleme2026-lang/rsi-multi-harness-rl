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
#   3. curriculum      reads the scan and plans which tasks to move; with
#                      APPLY_CURRICULUM=1 it regenerates them and gates them
#   4. train single    baseline arm: one harness, free to overfit it
#   5. train multi     treatment arm: four harnesses, cannot
#   6. eval            baseline + both arms, on held-out tasks AND the
#                      held-out harness, then prints the ablation
#
# Usage:
#   bash pipeline.sh                      # full run
#   STEPS=20 N_EVAL=4 bash pipeline.sh
#   SKIP_SCAN=1 bash pipeline.sh          # reuse an existing scan
#   SKIP_TRAIN=1 bash pipeline.sh         # re-evaluate existing checkpoints
#   APPLY_CURRICULUM=1 bash pipeline.sh   # also regenerate the flagged tasks
#
# The curriculum stage plans by default and regenerates only on request,
# because regeneration runs the four gates on a fresh batch and that is real
# shell work. The *plan* is always produced: a steering rule that is computed
# and never reported is indistinguishable from one that does not exist, which
# is what this stage was added to fix.

set -euo pipefail

cd "$(dirname "$0")"

RUN=./run.sh
STEPS="${STEPS:-40}"
N_SCAN="${N_SCAN:-8}"
N_EVAL="${N_EVAL:-4}"
OUT="${MULTIHARNESS_OUT:-$PWD/outputs}"
mkdir -p "$OUT"

# Read the train split from the suite rather than repeating it here. A literal
# copy drifts the moment a task is added: the scan would keep measuring 16 cells
# while `suite.py` trains on 17, and nothing would fail — the extra task would
# simply never be scanned, so `train.py` would never see its pass rate and would
# silently include it as a zero-gradient row.
TRAIN_TASKS="$(bash "$RUN" -c 'import sys; sys.path.insert(0, "src"); from multiharness.tasks.suite import TRAIN_TASK_IDS; print(",".join(TRAIN_TASK_IDS))')"
if [ -z "$TRAIN_TASKS" ]; then
  echo "FATAL: could not read TRAIN_TASK_IDS from the suite" >&2
  exit 1
fi
echo "train split: $TRAIN_TASKS"

banner() { printf '\n\n########## %s ##########\n\n' "$1"; }

banner "STAGE 0/6  static guards (no model, no GPU)"
# Tool-surface guard first: it is the cheapest check and it protects the
# meaning of every number downstream. A harness whose GUIDANCE advertises a
# tool it never implemented scores low for a reason that has nothing to do
# with the model — and that would be read as a capability result.
bash "$RUN" scripts/guard_tool_surface.py

banner "STAGE 1/6  correctness gates (no model)"
bash "$RUN" tests/test_path_errors.py
bash "$RUN" tests/test_shell_timeout.py
bash "$RUN" tests/test_scan_tooling.py
bash "$RUN" tests/smoke_env.py
bash "$RUN" tests/smoke_trl.py

banner "STAGE 2/6  difficulty scan (${N_SCAN} rollouts/cell, 4 harnesses x 16 tasks)"
if [ "${SKIP_SCAN:-0}" = "1" ] && [ -f "$OUT/scan_all.json" ]; then
  echo "SKIP_SCAN=1 and $OUT/scan_all.json exists -> reusing it"
else
  bash "$RUN" scripts/probe.py \
    --n "$N_SCAN" --tasks "$TRAIN_TASKS" --out "$OUT/scan_all.json"
fi

banner "STAGE 3/6  curriculum — read the scan, plan the moves"
# `rsi_loop.py` generates its own batch and writes it to $OUT/rsi/batch.json,
# then plans against whatever scan it is pointed at. The default scan
# ($OUT/scan_all.json) measures the *shipped* 24-task suite, which shares no
# ids with a generated batch, so the plan will correctly report zero moves and
# say why. That is not a failure — it is the honest answer to "steer a batch
# you never measured".
#
# To close the loop for real, scan the generated batch first:
#   bash "$RUN" scripts/probe.py --from-batch "$OUT/rsi/batch.json" \
#     --n "$N_SCAN" --out "$OUT/rsi/scan_batch.json"
#   CURRICULUM_SCAN="$OUT/rsi/scan_batch.json" bash pipeline.sh
# The cost is n x 4 harnesses x batch tasks of rollout time, and it is the
# price of steering on evidence rather than on the shipped suite's numbers.
CURRICULUM_ARGS=(--task-batch "${TASK_BATCH:-12}" --scan "${CURRICULUM_SCAN:-$OUT/scan_all.json}")
if [ "${APPLY_CURRICULUM:-0}" = "1" ]; then
  CURRICULUM_ARGS+=(--apply-curriculum)
fi
bash "$RUN" scripts/rsi_loop.py "${CURRICULUM_ARGS[@]}"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  banner "STAGE 4/6  train — SINGLE harness (baseline arm)"
  bash "$RUN" scripts/train.py \
    --mode single --steps "$STEPS" --scan "$OUT/scan_all.json" \
    --tag "single-s${STEPS}"

  banner "STAGE 5/6  train — MULTI harness (treatment arm)"
  bash "$RUN" scripts/train.py \
    --mode multi --steps "$STEPS" --scan "$OUT/scan_all.json" \
    --tag "multi-s${STEPS}"
else
  echo "SKIP_TRAIN=1 -> reusing checkpoints under $OUT"
fi

banner "STAGE 6/6  eval — baseline + single + multi in ONE process"
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
