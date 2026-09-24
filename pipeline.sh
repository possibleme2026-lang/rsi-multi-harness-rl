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
#   3. curriculum      generates the batch, scans THAT batch, then steers it;
#                      with APPLY_CURRICULUM=1 it also regenerates the flagged
#                      tasks and gates them
#   4. train single    baseline arm: one harness, free to overfit it
#   5. train multi     treatment arm: four harnesses, cannot
#   6. eval            baseline + both arms, on held-out tasks AND the
#                      held-out harness, then prints the ablation
#
# Usage:
#   bash pipeline.sh                      # full run
#   STEPS=20 N_EVAL=4 bash pipeline.sh
#   SKIP_SCAN=1 bash pipeline.sh          # reuse an existing scan
#   SKIP_BATCH_SCAN=1 bash pipeline.sh    # reuse an existing scan of the batch
#   SKIP_TRAIN=1 bash pipeline.sh         # re-evaluate existing checkpoints
#   APPLY_CURRICULUM=1 bash pipeline.sh   # also regenerate the flagged tasks
#   TRAIN_ON_BATCH=0 bash pipeline.sh     # train on the shipped suite instead
#
# Stage 3 is the one that closes the loop, and the ordering in it is the
# mechanism rather than a detail: a batch has to exist before it can be
# measured, and measured before it can be steered. Earlier releases generated
# the batch and steered it with a scan of the shipped suite, which shares no
# task ids, so the curriculum correctly reported zero moves on every run while
# the training arms never saw a generated task at all.

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

banner "STAGE 3/6  curriculum — generate the batch, scan IT, then steer on that"
# The curriculum steers a batch by its measured difficulty. That requires a scan
# of the batch — not a scan of the shipped suite. This stage used to generate
# the batch and then hand the curriculum $OUT/scan_all.json, which measures the
# shipped 16 ids; the batch shares none of them, so every run reported
# `move_count: 0` with the reason "the scan measures 16 task ids, none of which
# are in the batch of 2". The wiring was correct and the loop was open.
#
# The fix is ordering, and ordering is the whole mechanism: the batch must exist
# before it can be measured, and it must be measured before it can be steered.
# Generation is deterministic in --seed, so `--batch-only` writes exactly the
# batch the full run would have written.
BATCH_SEED="${BATCH_SEED:-11}"
TASK_BATCH="${TASK_BATCH:-12}"
ENV_BATCH_N="${ENV_BATCH_N:-0}"

CURRICULUM_SCAN="$OUT/scan_all.json"
if [ "${SKIP_BATCH_SCAN:-0}" = "1" ] && [ -f "$OUT/rsi/scan_batch.json" ]; then
  echo "SKIP_BATCH_SCAN=1 and $OUT/rsi/scan_batch.json exists -> reusing it"
  CURRICULUM_SCAN="$OUT/rsi/scan_batch.json"
else
  echo "-- generating the batch (seed=$BATCH_SEED, tasks=$TASK_BATCH, env=$ENV_BATCH_N)"
  bash "$RUN" scripts/rsi_loop.py --batch-only \
    --task-batch "$TASK_BATCH" --env-batch "$ENV_BATCH_N" --seed "$BATCH_SEED"

  # Cost: n x 4 harnesses x batch tasks of rollout time. It is the price of
  # steering on evidence rather than on the shipped suite's numbers, and there
  # is no cheaper way to know how hard a generated task is.
  echo "-- scanning the generated batch (this is the part that closes the loop)"
  bash "$RUN" scripts/probe.py --from-batch "$OUT/rsi/batch.json" \
    --n "$N_SCAN" --out "$OUT/rsi/scan_batch.json"
  CURRICULUM_SCAN="$OUT/rsi/scan_batch.json"
fi

# The curriculum's scan must cover the ids the batch actually has, or the plan
# is empty for a reason that has nothing to do with difficulty. Fail loudly here
# rather than printing "0 moves" as if it were a finding about the tasks.
bash "$RUN" - "$CURRICULUM_SCAN" "$OUT/rsi/batch.json" <<'PY'
import json, sys
scan = json.loads(open(sys.argv[1], encoding="utf-8").read())
batch = json.loads(open(sys.argv[2], encoding="utf-8").read())
scan_ids = {str(r.get("task_id")) for r in scan.get("records", [])}
batch_ids = {str(t["id"]) for t in batch}
overlap = scan_ids & batch_ids
print(f"scan ids {len(scan_ids)}  batch ids {len(batch_ids)}  overlap {len(overlap)}")
if not overlap:
    sys.exit(f"FATAL: {sys.argv[1]} measures none of the {len(batch_ids)} batch ids; "
             f"the curriculum would plan zero moves and the scan would filter nothing")
PY

CURRICULUM_ARGS=(--task-batch "$TASK_BATCH" --seed "$BATCH_SEED" --scan "$CURRICULUM_SCAN")
if [ "${APPLY_CURRICULUM:-0}" = "1" ]; then
  CURRICULUM_ARGS+=(--apply-curriculum)
fi
bash "$RUN" scripts/rsi_loop.py "${CURRICULUM_ARGS[@]}"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  # Training reads the same scan the curriculum steered on, and the batch it
  # measures. Before this, both arms trained on the shipped 16 ids with a filter
  # that matched nothing -- 64 rows over 128 steps is 16 epochs, which is
  # memorisation, and the flat reward curve past step 65 was that and not a
  # capability limit.
  TRAIN_BATCH_ARGS=()
  TRAIN_SCAN_ARGS=()
  if [ "${TRAIN_ON_BATCH:-1}" = "1" ]; then
    TRAIN_BATCH_ARGS=(--batch "$OUT/rsi/batch.json" --require-signal)
    TRAIN_SCAN_ARGS=(--scan "$CURRICULUM_SCAN")
  else
    echo "TRAIN_ON_BATCH=0 -> training on the shipped suite with $OUT/scan_all.json"
    TRAIN_SCAN_ARGS=(--scan "$OUT/scan_all.json")
  fi

  banner "STAGE 4/6  train — SINGLE harness (baseline arm)"
  bash "$RUN" scripts/train.py \
    --mode single --steps "$STEPS" "${TRAIN_SCAN_ARGS[@]}" "${TRAIN_BATCH_ARGS[@]}" \
    --tag "single-s${STEPS}"

  banner "STAGE 5/6  train — MULTI harness (treatment arm)"
  bash "$RUN" scripts/train.py \
    --mode multi --steps "$STEPS" "${TRAIN_SCAN_ARGS[@]}" "${TRAIN_BATCH_ARGS[@]}" \
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
