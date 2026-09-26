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
#   3. curriculum      generates the batch, scans THAT batch, steers it,
#                      regenerates the flagged tasks, scans the STEERED batch,
#                      and reports the alignment before/after. The rescan is the
#                      step that makes the axis closed rather than merely wired:
#                      without it the curriculum's output stops at a file and
#                      the training arms read the pre-steer batch.
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
#   APPLY_CURRICULUM=0 bash pipeline.sh   # plan the curriculum but do not apply
#   STEER_AND_RESCAN=0 bash pipeline.sh   # apply, but train on the pre-steer batch
#   TRAIN_ON_BATCH=0 bash pipeline.sh     # train on the shipped suite instead
#   TAG_SUFFIX=-steer bash pipeline.sh    # write train-*-steer/ instead of overwriting
#   SKIP_STEERED_SCAN=1 bash pipeline.sh  # reuse an existing rescan of the steered batch
#
# TAG_SUFFIX exists so a second full run can be made without destroying the
# first one's checkpoints. The tag decides the output directory, and the
# evaluation reads the checkpoints back by tag, so a run that steers the batch
# and a run that does not would otherwise collide on `train-multi-s48` and on
# `eval_ablation.json` -- the second silently replacing the first, which is the
# failure this repository has been fixing all along: an artifact that looks
# current and describes a different experiment.
#
# Stage 3 is the one that closes the loop, and the ordering in it is the
# mechanism rather than a detail: a batch has to exist before it can be
# measured, measured before it can be steered, and steered before it can be
# trained on. Each of those three orderings was missing in turn, and each time
# the loop looked wired while nothing reached the gradient.
#
# N_SCAN is 64 rather than 8 because of arithmetic, not taste. A band is only
# assigned when the whole Wilson interval falls inside one band, and the
# out-of-reach boundary is 0.1: a cell with zero passes needs n >= 35 before its
# upper bound drops below that, and 64 before it does so with room to spare.
# Below 35 every cell is `unresolved`, the plan holds every task, and the
# curriculum reports zero moves for a reason that has nothing to do with the
# tasks. The stage prints a warning when N_SCAN is below the threshold.

set -euo pipefail

cd "$(dirname "$0")"

RUN=./run.sh
STEPS="${STEPS:-40}"
# 64, not 8: see the header. Below 35 the curriculum cannot place a zero-pass
# cell in any band, so every task is held and the axis is inert.
N_SCAN="${N_SCAN:-64}"
N_EVAL="${N_EVAL:-4}"
# Empty by default, so the documented invocation keeps producing the documented
# artifacts. A non-empty suffix redirects every tag-dependent path at once, which
# is why it is a suffix on the tag rather than a separate `--out` per script: the
# trainer names its directory from the tag and the evaluator reads it back by
# tag, so the two only stay consistent if one variable moves both.
TAG_SUFFIX="${TAG_SUFFIX:-}"
OUT="${MULTIHARNESS_OUT:-$PWD/outputs}"
mkdir -p "$OUT"

# The threshold is a property of the band definition (out-of-reach is p < 0.1)
# and the Wilson interval, not a tunable. Restated here so a reader who lowers
# N_SCAN sees why the curriculum went quiet, rather than concluding the tasks
# are on target.
N_SCAN_MIN_FOR_BANDS=35
if [ "$N_SCAN" -lt "$N_SCAN_MIN_FOR_BANDS" ]; then
  echo "WARNING: N_SCAN=$N_SCAN is below $N_SCAN_MIN_FOR_BANDS." >&2
  echo "         A zero-pass cell's Wilson upper bound is z^2/(n+z^2), which stays" >&2
  echo "         above the 0.1 out-of-reach boundary until n >= $N_SCAN_MIN_FOR_BANDS." >&2
  echo "         Every cell will be 'unresolved', the curriculum will hold every" >&2
  echo "         task, and its zero moves will describe the scan and not the tasks." >&2
fi

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
APPLY="${APPLY_CURRICULUM:-1}"
if [ "$APPLY" = "1" ]; then
  CURRICULUM_ARGS+=(--apply-curriculum)
fi
bash "$RUN" scripts/rsi_loop.py "${CURRICULUM_ARGS[@]}"

# ---- the step that closes the loop -----------------------------------------
# `--apply-curriculum` writes the steered batch to batch_steered.json. Until this
# stage read it back, the regenerated tasks stopped at that file: the curriculum
# moved a task, the four gates confirmed the replacement was solvable, and the
# training arms went on reading `batch.json`. Reachable, never reached.
#
# So the steered batch is scanned and becomes the training input. It has to be
# scanned rather than reused because the ids are new — a reparameterised task is
# a different task, and its predecessor's pass rate says nothing about it. The
# signal filter would find no measurement for any row and keep them all, which
# is the silent state `--require-signal` exists to refuse.
#
# When nothing moved, batch_steered.json is a copy of batch.json and the
# existing scan still describes it exactly, so the rescan is skipped rather than
# paid for. The decision is read from the artifact, not inferred from a log line.
TRAIN_BATCH_PATH="$OUT/rsi/batch.json"
TRAIN_SCAN_PATH="$CURRICULUM_SCAN"
STEERED_MOVED="$(bash "$RUN" - "$OUT/rsi/curriculum.json" <<'PY'
import json, sys
try:
    cur = json.loads(open(sys.argv[1], encoding="utf-8").read())
except Exception:
    print(0); raise SystemExit
print(int(cur.get("steered", {}).get("moved", 0)))
PY
)"

if [ "${STEER_AND_RESCAN:-1}" = "1" ] && [ "$APPLY" = "1" ]; then
  # The scan the report compares against. It is the rescan when something moved,
  # and the original scan when nothing did -- in that case the steered batch is a
  # byte-for-byte copy of batch.json, so the same measurements describe it
  # exactly and paying for a second scan would buy the same numbers.
  AFTER_SCAN="$CURRICULUM_SCAN"

  if [ "${STEERED_MOVED:-0}" -gt 0 ]; then
    # SKIP_STEERED_SCAN reuses a rescan that already exists. It is not a
    # convenience: a scan is ~30 minutes of GPU on this batch, and the one case
    # that most needs it -- a run that stopped *after* the rescan, in the report
    # or in training -- is exactly the case where paying for it twice teaches
    # nothing. Reusing it is safe because the batch it measures is on disk and
    # the scan names the ids it measured; the overlap guard below re-checks that
    # the two still agree rather than trusting the file's existence.
    if [ "${SKIP_STEERED_SCAN:-0}" = "1" ] && [ -f "$OUT/rsi/scan_batch_steered.json" ]; then
      echo "-- SKIP_STEERED_SCAN=1 and the rescan exists -> reusing it"
    else
      echo "-- $STEERED_MOVED task(s) moved -> scanning the steered batch and training on IT"
      bash "$RUN" scripts/probe.py --from-batch "$OUT/rsi/batch_steered.json" \
        --n "$N_SCAN" --out "$OUT/rsi/scan_batch_steered.json"
    fi
    TRAIN_BATCH_PATH="$OUT/rsi/batch_steered.json"
    TRAIN_SCAN_PATH="$OUT/rsi/scan_batch_steered.json"
    AFTER_SCAN="$TRAIN_SCAN_PATH"

    # Same guard as above, for the same reason: the steered batch carries new
    # hashed ids, and a scan that misses them would train on unmeasured rows.
    bash "$RUN" - "$TRAIN_SCAN_PATH" "$TRAIN_BATCH_PATH" <<'PY'
import json, sys
scan = json.loads(open(sys.argv[1], encoding="utf-8").read())
batch = json.loads(open(sys.argv[2], encoding="utf-8").read())
scan_ids = {str(r.get("task_id")) for r in scan.get("records", [])}
batch_ids = {str(t["id"]) for t in batch}
overlap = scan_ids & batch_ids
print(f"steered scan ids {len(scan_ids)}  batch ids {len(batch_ids)}  overlap {len(overlap)}")
if not overlap:
    sys.exit(f"FATAL: {sys.argv[1]} measures none of the {len(batch_ids)} steered ids")
PY
  else
    echo "-- no tasks moved (see move_diagnosis in curriculum.json) -> reusing the scan"
    echo "   the steered batch is a copy of the original, so the same scan describes it"
  fi

  # The before/after comparison, run on both paths. This is the only number in
  # the repository that the task axis produces and that could come out negative,
  # and a zero-move plan is itself a result -- it says the batch was already on
  # target or that the scan was too coarse to see it. Writing the report in both
  # cases is what makes "no moves" an artifact with a number rather than a
  # missing file that reads as "the step was skipped".
  bash "$RUN" scripts/loop_report.py \
    --before-batch "$OUT/rsi/batch.json" \
    --before-scan "$CURRICULUM_SCAN" \
    --after-batch "$OUT/rsi/batch_steered.json" \
    --after-scan "$AFTER_SCAN" \
    --out "$OUT/rsi/loop_closed.json"
else
  echo "STEER_AND_RESCAN=$STEER_AND_RESCAN or APPLY_CURRICULUM=$APPLY -> training on the pre-steer batch"
fi

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  # Training reads the same scan the curriculum steered on, and the batch it
  # measures. Before this, both arms trained on the shipped 16 ids with a filter
  # that matched nothing -- 64 rows over 128 steps is 16 epochs, which is
  # memorisation, and the flat reward curve past step 65 was that and not a
  # capability limit.
  #
  # TRAIN_BATCH_PATH / TRAIN_SCAN_PATH were decided in stage 3: the steered
  # batch when the curriculum moved something, the original otherwise.
  TRAIN_BATCH_ARGS=()
  TRAIN_SCAN_ARGS=()
  if [ "${TRAIN_ON_BATCH:-1}" = "1" ]; then
    TRAIN_BATCH_ARGS=(--batch "$TRAIN_BATCH_PATH" --require-signal)
    TRAIN_SCAN_ARGS=(--scan "$TRAIN_SCAN_PATH")
  else
    echo "TRAIN_ON_BATCH=0 -> training on the shipped suite with $OUT/scan_all.json"
    TRAIN_SCAN_ARGS=(--scan "$OUT/scan_all.json")
  fi

  banner "STAGE 4/6  train — SINGLE harness (baseline arm)"
  bash "$RUN" scripts/train.py \
    --mode single --steps "$STEPS" "${TRAIN_SCAN_ARGS[@]}" "${TRAIN_BATCH_ARGS[@]}" \
    --tag "single-s${STEPS}${TAG_SUFFIX}"

  banner "STAGE 5/6  train — MULTI harness (treatment arm)"
  bash "$RUN" scripts/train.py \
    --mode multi --steps "$STEPS" "${TRAIN_SCAN_ARGS[@]}" "${TRAIN_BATCH_ARGS[@]}" \
    --tag "multi-s${STEPS}${TAG_SUFFIX}"
else
  echo "SKIP_TRAIN=1 -> reusing checkpoints under $OUT"
fi

banner "STAGE 6/6  eval — baseline + single + multi in ONE process"
# All three arms run in a single invocation so the ablation table is produced
# with a shared seed and shared harness instances. Splitting this into three
# commands would lose the comparison and make the arms non-paired.
bash "$RUN" scripts/eval.py \
  --baseline \
  --adapter "$OUT/train-single-s${STEPS}${TAG_SUFFIX}/final" \
  --adapter "$OUT/train-multi-s${STEPS}${TAG_SUFFIX}/final" \
  --n "$N_EVAL" --out "$OUT/eval_ablation${TAG_SUFFIX}.json"

banner "PIPELINE COMPLETE"
echo "artifacts in $OUT:"
ls -1 "$OUT"
