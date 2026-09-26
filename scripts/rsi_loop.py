#!/usr/bin/env python3
# Copyright 2026 The rsi-multi-harness-rl Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the RSI loop and write the artifacts the figures read.

    ./run.sh scripts/rsi_loop.py --rounds 6
    ./run.sh scripts/rsi_loop.py --rounds 6 --task-batch 40

What this actually does
-----------------------
Two things, both of which write evidence rather than claims.

**The task axis.** Generates a batch of tasks from the parameter space, runs
all four validation gates on each, and writes the accept/reject counts. This is
the part that can be done without a model: gates V1–V3 need only a shell, and
V4's structural half needs nothing at all. The result is a real accept rate for
a real batch, not an assertion that the generator works.

**The harness axis.** Runs the evolutionary loop over the harness edit space.
Each round proposes edits within the annealed budget, and scores the candidate
against the incumbent. Scoring here is *measured*, not simulated: every
candidate is rolled out on the task batch and its pass rate is the score.

Two modes
---------
``--score ledger-replay`` (default when no model is available)
    Scores candidates with a deterministic stand-in derived from the edit
    itself, so the loop's *machinery* — budget, guard, floor, ledger, prune —
    runs end to end without a GPU. The ledger it produces is real in the sense
    that matters for testing the machinery, and it is labelled as replayed in
    the artifact so no figure can present it as a model measurement.

``--score rollout``
    Rolls the candidate harness out against the model on the task batch. This
    is the honest number and it needs a GPU and a loaded model. It is
    substantially slower, so it is opt-in rather than the default.

The distinction is recorded in the output under ``score_mode``, because a
figure that showed a replayed trajectory as if it were a measured one would be
the exact kind of unearned claim this project exists to avoid.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiharness._bootstrap import outputs_root  # noqa: E402
from multiharness.rsi import curriculum as cu  # noqa: E402
from multiharness.rsi import (  # noqa: E402
    env_adapter,
    envtask,
    harbor_export,
    task_gen,
    validate,
)
from multiharness.rsi import harness_evolve as he  # noqa: E402
from multiharness.rsi.ledger import Ledger, edit_budget  # noqa: E402
from multiharness.rsi.stats import noise_floor  # noqa: E402

#: Artifacts land in ``<repo>/outputs/rsi``, or ``$MULTIHARNESS_OUT/rsi``.
#:
#: This used to be ``ROOT / "outputs" / "rsi"``, which ignored
#: ``MULTIHARNESS_OUT`` while ``probe.py``, ``train.py`` and ``eval.py`` all
#: honoured it — so a redirected pipeline wrote its scan, checkpoints and eval
#: to one place and this loop's artifacts to another, and the documented
#: override silently applied to only part of a run. ``outputs_root`` is the
#: single definition; the ``rsi`` subdirectory is this script's own namespace.
OUT = outputs_root() / "rsi"


# --------------------------------------------------------------------------
# the environment axis
# --------------------------------------------------------------------------


def run_env_axis(
    n_envs: int,
    seed: int,
    root: Path,
    *,
    n_records: int,
    n_distractors: int,
    n_steps: int,
    export: bool,
) -> dict:
    """Synthesise stateful environment tasks, gate them, and optionally export them.

    This is the axis the string task generator cannot provide. Where
    ``run_task_axis`` produces prompts whose only effect is the contents of one
    file, this produces **stateful systems** the agent acts on through named
    tools, and grades the state it leaves behind. The three discriminability
    checks below are the analogue of gates V1/V2 for a state-based reward, and
    they are strictly stronger than the string versions:

    *reference*  the reference trace must score exactly 1.0 — otherwise the
                 environment is inconsistent with its own checkpoints;
    *initial*    the untouched state must score 0.0 — otherwise the task is
                 already solved, or a checkpoint passes vacuously;
    *distractor* a trace applied to the *wrong* record must score strictly less
                 than the reference — otherwise the distractors are decorative
                 and an agent that cannot read the state scores the same as one
                 that can. This third check has no analogue in the string
                 generator, because a string task has nothing to confuse.

    Every one of those three was violated by an early version of the generator,
    which is why they are run on the whole batch rather than spot-checked. The
    measured failures are recorded in ``envgen``/``envtask``'s docstrings.
    """
    print("=" * 74)
    print("ENVIRONMENT AXIS — synthesise stateful tasks, then try to falsify each")
    print("=" * 74)
    t0 = time.time()
    tasks = envtask.generate_env_batch(
        n_envs,
        seed=seed,
        n_records=n_records,
        n_distractors=n_distractors,
        n_steps=n_steps,
    )
    cov = envtask.summarise_env_batch(tasks)
    print(f"synthesised {cov['total']} environment tasks in {time.time() - t0:.1f}s")
    print(f"  by domain        : {cov['by_domain']}")
    print(f"  edges by reason  : {cov['edges_by_reason']}   (data_flow / precondition / state)")
    print(f"  checkpoints      : min {cov['checkpoints_min']}  max {cov['checkpoints_max']}  "
          f"mean {cov['checkpoints_mean']}")
    print(f"  with distractors : {cov['with_distractors']}")
    print()

    t0 = time.time()
    rows: list[dict] = []
    bad = {"reference": 0, "initial": 0, "distractor": 0, "ambiguous": 0}
    for t in tasks:
        cps = list(t.checkpoints)
        ref_state = envtask.apply_trace(t.instance, list(t.trace))
        ref = envtask.grade_state(t.instance, ref_state, cps)
        init = envtask.grade_state(t.instance, t.instance.initial_state, cps)
        wrong_state = envtask.plausible_wrong_state(t.instance, list(t.trace))
        wrong = envtask.grade_state(t.instance, wrong_state, cps)

        ok_ref = ref == 1.0
        ok_init = init == 0.0
        ok_wrong = wrong < ref
        if not ok_ref:
            bad["reference"] += 1
        if not ok_init:
            bad["initial"] += 1
        if not ok_wrong:
            bad["distractor"] += 1

        rows.append({
            "task_id": t.task_id,
            "domain": t.instance.spec.domain,
            "chain": list(t.instance.chain),
            "checkpoints": t.checkpoint_count,
            "grade_reference": ref,
            "grade_initial": init,
            "grade_distractor": wrong,
            "discriminates": ok_ref and ok_init and ok_wrong,
        })
    print(f"discriminability checked in {time.time() - t0:.1f}s")
    n_ok = sum(1 for r in rows if r["discriminates"])
    print(f"  reference scores 1.0    : {len(rows) - bad['reference']}/{len(rows)}")
    print(f"  initial scores 0.0      : {len(rows) - bad['initial']}/{len(rows)}")
    print(f"  distractor scores less  : {len(rows) - bad['distractor']}/{len(rows)}")
    print(f"  all three               : {n_ok}/{len(rows)}")
    print()

    # The trust boundary is audited on every task, not sampled: the leak it
    # prevents (the oracle readable from the agent's own image) is silent, and a
    # sampled audit would report a clean package while shipping a leaky one.
    audit = harbor_export.audit_batch(tasks)
    print(f"harbor package audit: {audit['ok']}/{audit['total']} clean")
    if audit["problems_by_kind"]:
        print(f"  problems: {audit['problems_by_kind']}")
    if audit["warnings"]:
        print(f"  warnings: {audit['warnings']}")
    print()

    exported: list[str] = []
    if export:
        dest = root / "harbor"
        paths = harbor_export.write_harbor_batch(tasks, dest)
        exported = [str(p) for p in paths]
        print(f"exported {len(exported)} Harbor task packages to {dest}")
        print(f"  e.g. {exported[0]}")
        print()

    # The environment batch is dumped in the *harness-pool* shape, not the
    # Harbor shape, and the difference is the whole point of this file. The
    # Harbor packages are the deliverable — what a container runner consumes.
    # The harness pool runs on the host, and `probe.py --from-batch` reads a list
    # of task dicts. Without this dump there is no way to scan an environment
    # batch, so the cross-harness gap could only ever be measured on string
    # tasks — which is the weaker finding the environment axis exists to
    # replace. Written unconditionally: it is small, and a run that exported
    # nothing is exactly the run most likely to want it.
    env_batch = env_adapter.as_harness_batch(tasks)
    env_batch_path = root / "env_batch.json"
    env_batch_path.write_text(json.dumps(env_batch, indent=2), encoding="utf-8")
    print(f"wrote {env_batch_path}  ({len(env_batch)} environment tasks; scan it with "
          f"`probe.py --from-batch {env_batch_path}`)")

    return {
        "coverage": cov,
        "tasks": rows,
        "discriminable": n_ok,
        "failures": bad,
        "audit": audit,
        "exported": exported,
        "env_batch": str(env_batch_path),
    }


# --------------------------------------------------------------------------
# the task axis
# --------------------------------------------------------------------------


def run_task_axis(batch_size: int, seed: int, root: Path) -> dict:
    """Generate a batch and put every task through the four gates."""
    print("=" * 74)
    print("TASK AXIS — generate a batch, then try to falsify each task")
    print("=" * 74)
    t0 = time.time()
    batch = task_gen.generate_batch(batch_size, seed=seed)
    cov = task_gen.summarise_batch(batch)
    print(f"generated {cov['total']} tasks in {time.time() - t0:.1f}s")
    print(f"  by tier       : {cov['by_tier']}")
    print(f"  by verify mode: {cov['by_verify_mode']}   (what the tasks use)")
    print(f"  mode requested: {cov['by_verify_mode_requested']}   (what the parameters asked for)")
    print(f"  mode overrides: {cov['mode_overrides']}   (short answer: a digest would be brute-forceable)")
    print(f"  by step count : {cov['by_steps']}")
    print(f"  with a source file: {cov['with_source_file']}")
    print()

    t0 = time.time()
    verdicts = validate.validate_batch(batch, roots=root / "gates")
    summary = validate.summarise_validation(verdicts)
    print(f"validated in {time.time() - t0:.1f}s")
    print(f"  accepted {summary['accepted']}/{summary['total']}")
    print(f"  failures by gate: {summary['failures_by_gate'] or 'none'}")
    for tier, d in sorted(summary["by_tier"].items()):
        print(f"    {tier}: {d['accepted']}/{d['total']}")
    print()

    # The rejection details are what make a failure actionable, so they are
    # written out rather than reduced to a count.
    rejected = [
        {"task_id": v.task_id, "tier": v.tier, "failed_gates": v.failed_gates,
         "details": {g.gate: g.detail for g in v.gates if not g.passed}}
        for v in verdicts
        if not v.accepted
    ]
    return {
        "coverage": cov,
        "summary": summary,
        "rejected": rejected,
        "verdicts": [v.as_dict() for v in verdicts],
    }


# --------------------------------------------------------------------------
# the curriculum axis
# --------------------------------------------------------------------------


def run_curriculum_axis(
    scan_path: Path,
    batch: list[dict],
    root: Path,
    *,
    alpha: float,
    k_min: float,
    max_moves: int | None,
    apply: bool,
) -> dict:
    """Turn a measured scan into the next batch, and gate the result.

    This is the call site ``band.steer`` never had. Before this function
    existed, ``steer`` was reachable only from its own tests: the module
    docstring said the task generator consumes it and the README said it
    "already returns the override", but no script ever asked it for one. A
    curriculum that is never executed is a claim, not a mechanism.

    The scan is read from disk rather than passed in, because the honest
    dependency is a *file*: the pass rates that justify moving a task have to
    come from a run that happened, and accepting them as an argument would let
    a caller pass a fabricated dict. ``source`` records the path, so a plan can
    always be traced back to the measurement behind it.

    The regenerated tasks go through the same four gates as every other task.
    That is not ceremony — a regenerated task is exactly the case where a
    generator bug produces something unsolvable, since it moves parameters to
    the edge of the space (payload 16, escape 0.6, three steps), and the oracle
    gate is what catches a reference solution that no longer works there.
    """
    print("=" * 74)
    print("CURRICULUM AXIS — read the scan, move the tasks that are not teaching")
    print("=" * 74)

    if not scan_path.is_file():
        print(f"  no scan at {scan_path} — skipping (the curriculum needs a measurement)")
        print()
        return {"skipped": f"no scan at {scan_path}"}
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    # Collapse raw rollout records into per-cell pass counts, the same way
    # tools/plot.py does, so the plan and the figures cannot disagree about
    # what the scan says.
    agg: dict[tuple[str, str], dict] = {}
    for r in scan.get("records", []):
        key = (str(r.get("harness", "")), str(r.get("task_id", "")))
        c = agg.setdefault(key, {"harness": key[0], "task_id": key[1], "passes": 0, "n": 0})
        c["n"] += 1
        c["passes"] += int(float(r.get("reward", 0.0)) >= 1.0)
    cells = list(agg.values())

    if not cells:
        print("  the scan has no records — nothing to steer on")
        print()
        return {"skipped": "scan has no records"}

    tasks_by_id = {t["id"]: t for t in batch}

    # Whether the scan and the batch describe the *same tasks*. They need not:
    # the scan shipped with this repository measures the 24-task suite
    # (`t1-01`), while a generated batch carries hashed ids (`t1-8f87ad9e`), and
    # the two sets do not intersect. When they do not, no task can be moved —
    # `steer` moves a parameter vector, and there is no parameter vector for an
    # id that is not in the batch.
    #
    # Reported as an explicit count rather than left to surface as `moves: 0`,
    # because zero moves has several causes and they need different responses:
    # a mismatched id set means "scan the batch you intend to steer", while
    # every cell being frontier means "nothing to do, this batch is on target".
    scan_ids = {str(r.get("task_id", "")) for r in scan.get("records", [])}
    overlap = scan_ids & set(tasks_by_id)
    id_match = bool(overlap)

    plan = cu.plan_regeneration(
        cells,
        tasks_by_id=tasks_by_id,
        alpha=alpha,
        k_min=k_min,
        source=str(scan_path),
        max_moves=max_moves,
    )

    print(f"  scan          : {scan_path}  ({len(cells)} cells, {len(scan_ids)} task ids)")
    print(f"  id overlap    : {len(overlap)} of {len(scan_ids)} scan ids are in the batch")
    if not id_match:
        print("      -> the scan measures a different task set; no task can be moved.")
        print("         Scan the batch you intend to steer (probe.py --tasks <batch>).")
    print(f"  alpha         : {plan.alpha}  (argmax of the GRPO signal curve at G={cu.DEFAULT_G})")
    print(f"  signal at alpha: {plan.signal_at_target:.4f}")
    print(f"  batch mean p  : {plan.mean_pass_rate:.4f}  -> {plan.misalignment}")
    print(f"  moves         : {plan.move_count}  {plan.by_direction or ''}")
    print(f"  held          : {plan.held_count}")
    reasons: dict[str, int] = {}
    for h in plan.held:
        key = h["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items()):
        print(f"      {k}: {v}")

    result: dict = {
        "scan": str(scan_path),
        "id_overlap": len(overlap),
        "scan_task_ids": len(scan_ids),
        "ids_match_batch": id_match,
        "plan": plan.as_dict(),
        "applied": False,
    }

    # The batch-level alignment, before any move. `plan.as_dict()` already
    # carries one, but it is computed over *resolved cells only*; this is the
    # same quantity over every measured task, which is what a before/after
    # comparison against the steered batch needs. The two are reported side by
    # side rather than merged, because a reader who sees them disagree should be
    # able to tell that the difference is the denominator.
    measured = {
        str(r["task_id"]): int(r["passes"]) / int(r["n"])
        for r in cells
        if int(r.get("n", 0)) > 0
    }
    result["alignment_before"] = round(
        cu.batch_alignment(batch, measured, alpha=plan.alpha, beta=plan.beta), 6
    )
    result["alignment_after"] = "unmeasured — the steered batch has not been scanned"

    if not id_match:
        # Stated in the artifact, not only on stdout: a reader of
        # curriculum.json has to be able to tell "nothing needed moving" from
        # "nothing could be moved", and those look identical in `move_count`.
        result["skipped"] = (
            f"the scan measures {len(scan_ids)} task ids, none of which are in the "
            f"batch of {len(tasks_by_id)}; no task can be steered"
        )

    # Zero moves has three causes and they need three different responses. Only
    # the first one used to be visible at the top level; the other two surfaced
    # as `move_count: 0` with the reason buried in `held[].reason`, which reads
    # exactly like "the batch is already on target" — the one interpretation
    # that requires doing nothing. The other two require scanning more or
    # scanning the right tasks, and a silent zero is how the previous instance
    # of this defect survived.
    if plan.move_count == 0:
        if not id_match:
            result["move_diagnosis"] = "ids_mismatch: scan the batch you intend to steer"
        else:
            held_reasons = {h.get("reason", "").split(":")[0] for h in plan.held}
            if "unresolved" in held_reasons:
                result["move_diagnosis"] = (
                    "all_unresolved: every interval spans two bands — this is a "
                    "statement about the scan, not the tasks; scan more (n>=35 to "
                    "place a 0-pass cell, n=64 to place it comfortably)"
                )
            else:
                result["move_diagnosis"] = (
                    "all_frontier: every measured cell is in the band training "
                    "should use — nothing to move, this batch is on target"
                )

    if not apply:
        print("  (plan only — pass --apply-curriculum to regenerate and gate)")
        print()
        return result

    if not id_match:
        print("  refusing to apply: there is nothing to apply a plan to")
        print()
        return result

    fresh = cu.execute_plan(plan, tasks_by_id)
    print(f"  regenerated   : {len(fresh)} tasks")
    if fresh:
        verdicts = validate.validate_batch(fresh, roots=root / "gates_regen")
        summary = validate.summarise_validation(verdicts)
        print(f"  regen accepted: {summary['accepted']}/{summary['total']}")
        print(f"  failures      : {summary['failures_by_gate'] or 'none'}")
        result["applied"] = True
        result["regenerated"] = {
            "count": len(fresh),
            "summary": summary,
            "coverage": task_gen.summarise_batch(fresh),
            "rejected": [
                {"task_id": v.task_id, "failed_gates": v.failed_gates,
                 "details": {g.gate: g.detail for g in v.gates if not g.passed}}
                for v in verdicts if not v.accepted
            ],
        }
        # The honest limit, stated where the number is: this measures whether
        # the regenerated tasks are *well-formed*, not whether they are better.
        # The alignment claim needs a rescan, and until one happens this is a
        # statement about solvability rather than about learning.
        result["measured_effect"] = (
            "none — the regenerated batch is gate-valid, which is a solvability "
            "claim. Whether it moved toward alpha needs a rescan of these tasks."
        )
        print("  NOTE: gate-valid is not the same as better aligned. "
              "The alignment claim needs a rescan.")

    # The steered batch is written for *every* apply, including a zero-move one.
    #
    # `fresh` above is the answer to "what did the curriculum produce"; it is
    # ordered by the plan and contains only the moved tasks. What the next round
    # needs is the whole batch with those tasks swapped in, which is a different
    # object, and it is the one that has to exist on disk for the loop to close.
    # Without this file the regenerated tasks stop at `curriculum.json` and the
    # training run reads the pre-steer batch — the loop is reachable and never
    # reached, which is the defect this function was rewritten to fix.
    #
    # Written unconditionally so downstream has one path rather than two: when
    # nothing moved this is a byte-identical copy of `batch.json`, which is the
    # correct input for the next stage either way.
    steered = cu.steered_batch(plan, batch)
    steered_path = OUT / "batch_steered.json"
    steered_path.write_text(json.dumps(steered, indent=2), encoding="utf-8")
    kept = sum(1 for t in steered if "regenerated_from" not in t)
    # `applied` means "the curriculum was executed", not "something moved".
    # A zero-move plan is still executed, and its correct output is the batch
    # unchanged — so the flag is set here rather than inside the `if fresh`
    # block above, where it would report `applied: false` for a run that did
    # apply a (vacuous) plan and did write the file downstream reads.
    result["applied"] = True
    result["steered"] = {
        "path": str(steered_path),
        "total": len(steered),
        "moved": len(steered) - kept,
        "kept": kept,
    }
    result["provenance"] = [
        {"from": t["regenerated_from"], "to": t["id"], "band": t["regenerated_band"],
         "direction": t["regenerated_direction"]}
        for t in steered
        if "regenerated_from" in t
    ]
    print(f"  steered batch : {steered_path}  "
          f"({len(steered)} tasks: {len(steered) - kept} moved, {kept} kept)")
    print(f"      -> scan it, then train on it: "
          f"probe.py --from-batch {steered_path}")
    print()
    return result


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def replayed_score(state: he.HarnessState, rng: random.Random, base: float) -> float:
    """A deterministic stand-in score for one harness state.

    Built so that the loop's machinery is exercised in a way that resembles a
    real run: edits that add actionable guidance help a little, the tool-drop
    edit is a coin flip, and every measurement carries sampling noise. That is
    enough for the budget, the guard, the noise floor, the ledger and the prune
    rule to all be reached.

    It is **not** a model measurement and the artifact says so. Its only
    purpose is to let the loop be run and tested without a GPU; the honest
    numbers come from ``--score rollout``.
    """
    help_by_edit = {
        "guidance+=submit_echo": 0.09,
        "guidance+=retry_hint": 0.11,
        "guidance+=exactness": 0.07,
        "guidance+=one_line": 0.05,
        "prompt-=verbose_preamble": 0.04,
        "output_plumbing+=explicit_path": 0.08,
        "context_mgmt+=keep_last_error": 0.03,
        "client_tool-=read_file": -0.01,
    }
    gain = sum(help_by_edit.get(name, 0.0) for name in state.history)
    # Diminishing returns, so the loop cannot climb forever on one component.
    gain = 0.30 * (1.0 - pow(2.718281828, -gain / 0.15))
    noise = rng.gauss(0.0, 0.035)
    return max(0.0, min(1.0, base + gain + noise))


def rollout_score(state: he.HarnessState, tasks: list[dict], *, n: int, agent) -> float:
    """Roll a candidate harness out against the model. The honest number.

    The driver takes an environment *class*, so the state is materialised via
    :func:`rsi.harness_evolve.to_env_class` — that conversion is the single
    point where the evolver's data becomes executable.

    The generated batch is registered first. ``Agent.run`` resolves ``task_id``
    through ``core.get_task``, which reads a process-global registry populated
    by ``tasks.suite.load()`` — the *shipped* 24 tasks. Generated ids live only
    in the batch list, so without this the first rollout dies on
    ``KeyError: unknown task_id 't4-c1ff0a35'``. It also means the suite has to
    be loaded, because a *fresh process* starts with an empty registry and
    ``reset`` would fail on the shipped ids for the same reason.

    The task dicts are registered as-is: the generator's output is documented
    as a superset of what ``register_tasks`` expects, and ``core.verify`` reads
    ``verify`` / ``expected`` / ``check_script`` straight off the task, so no
    translation is needed or wanted here.
    """
    from multiharness.harnesses.core import TASKS, register_tasks
    from multiharness.tasks import load as _load_suite

    _load_suite()
    fresh = [t for t in tasks if t["id"] not in TASKS]
    if fresh:
        register_tasks(fresh)

    cls = he.to_env_class(state, _BASE_CLASS[state.name])
    rewards: list[float] = []
    for t in tasks:
        rewards.extend(r.reward for r in agent.run(cls, t["id"], n=n))
    return sum(rewards) / len(rewards) if rewards else 0.0


#: Base classes by harness name, resolved lazily so importing this module does
#: not require torch.
_BASE_CLASS: dict[str, type] = {}


def _resolve_bases() -> None:
    if _BASE_CLASS:
        return
    from multiharness.harnesses.pool import ALL_HARNESSES

    _BASE_CLASS.update(ALL_HARNESSES)


# --------------------------------------------------------------------------
# the harness axis
# --------------------------------------------------------------------------


def run_harness_axis(
    rounds: int,
    *,
    score_mode: str,
    tasks: list[dict],
    n_rollouts: int,
    seed: int,
    b_min: int,
    b_max: int,
    agent=None,
) -> dict:
    print("=" * 74)
    print(f"HARNESS AXIS — evolve the interface, score_mode={score_mode}")
    print("=" * 74)

    rng = random.Random(seed)
    ledger = Ledger(OUT / "ledger.jsonl")
    _resolve_bases()

    # Start from the real harnesses as they exist, so the evolution is a
    # continuation of the shipped interfaces rather than of a blank slate.
    states: dict[str, he.HarnessState] = {}
    for name in he.EVOLVABLE_HARNESSES:
        cls = _BASE_CLASS.get(name)
        if cls is None:
            continue
        env = cls()
        tools = tuple(sorted(m.__name__ for m in _public_tools(env)))
        base = he.HarnessState(name=name, guidance=env.GUIDANCE, tools=tools)
        base.score = _score(base, score_mode, rng, tasks, n_rollouts, agent)
        states[name] = base
        print(f"  {name:<18} baseline score {base.score:.3f}  tools={tools}")

    # The floor is sized from the observed spread rather than assumed. With no
    # repeated measurements to pool (replayed mode produces one score per
    # state) it falls back to a fixed fraction, and that fallback is reported
    # rather than hidden, because the floor is what decides every accept.
    if score_mode == "rollout":
        per_cell = _reward_samples(states, tasks, n_rollouts, agent)
        floor = noise_floor(per_cell) if per_cell else 0.05
        floor_source = "bootstrap over measured rollouts"
    else:
        floor = 0.05
        floor_source = "fixed fallback (replayed scoring produces one score per state)"
    print(f"  noise floor {floor:.4f}  ({floor_source})")
    print()

    trajectory: list[float] = []
    for t in range(rounds):
        budget = edit_budget(t, rounds, b_min, b_max)
        for hname, incumbent in list(states.items()):
            # Stall is per harness: one interface exhausting its neighbourhood
            # says nothing about whether another has.
            allow_pruned = ledger.stalled(w=3, delta=floor, harness=hname)
            proposals = he.propose_edits(incumbent, ledger, budget, rng=rng, allow_pruned=allow_pruned)
            if not proposals:
                continue
            for edit in proposals:
                cand = he.apply_edit(incumbent, edit)
                ok, why = he.guard_candidate(cand)
                if not ok:
                    ledger.append(
                        he.EditRecord(
                            round=t, candidate_id=f"{hname}-r{t}-{edit.name}", edit=edit.name,
                            harness=hname, components=(edit.component,),
                            score_before=incumbent.score, score_after=incumbent.score,
                            delta=0.0, floor=floor, accepted=False, reason=f"guard: {why}",
                        )
                    )
                    continue
                cand.score = _score(cand, score_mode, rng, tasks, n_rollouts, agent)
                accepted, rec = he.judge_candidate(
                    cand, incumbent, floor=floor, round_index=t,
                    candidate_id=f"{hname}-r{t}-{edit.name}", edit=edit,
                    meta={"score_mode": score_mode, "rationale": edit.rationale},
                )
                ledger.append(rec)
                if accepted:
                    states[hname] = cand
                    incumbent = cand
        best = max(s.score for s in states.values())
        trajectory.append(best)
        print(
            f"  round {t}: budget={budget}  best={best:.3f}  "
            f"attempts={ledger.attempted()}  accepted={len(ledger.accepted_edits())}"
        )

    summary = ledger.summary()
    print()
    print(f"  attempted {summary['attempted']}, accepted {summary['accepted']} "
          f"({summary['accept_rate']:.0%})")
    print(f"  rejected worse={summary['rejected_worse']}, "
          f"within-noise={summary['rejected_within_noise']}")
    print(f"  yield per component: {summary['yield_per_component']}")
    print(f"  prune set: {summary['prune_set'] or 'none'}")
    print()

    return {
        "score_mode": score_mode,
        "rounds": rounds,
        "floor": floor,
        "floor_source": floor_source,
        "ledger": summary,
        "trajectory": trajectory,
        "final_states": {k: v.as_dict() for k, v in states.items()},
        "winning_edits": {
            k: [r.edit for r in ledger.records if r.accepted and r.harness == k] for k in states
        },
    }


def _public_tools(env) -> list:
    import inspect

    return [
        m for n, m in inspect.getmembers(env, predicate=inspect.ismethod)
        if not n.startswith("_") and n not in ("reset", "get_reward")
    ]


def _score(state, mode, rng, tasks, n_rollouts, agent) -> float:
    if mode == "rollout":
        return rollout_score(state, tasks, n=n_rollouts, agent=agent)
    return replayed_score(state, rng, base=0.25)


def _reward_samples(states, tasks, n_rollouts, agent) -> list[list[float]]:
    """Every reward from the baseline rollouts, pooled for the noise floor."""
    out: list[list[float]] = []
    for s in states.values():
        try:
            cls = he.to_env_class(s, _BASE_CLASS[s.name])
            rs: list[float] = []
            for t in tasks:
                rs.extend(r.reward for r in agent.run(cls, t["id"], n=n_rollouts))
            if rs:
                out.append(rs)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the RSI loop and record its evidence.")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--task-batch", type=int, default=30, help="tasks to generate and validate")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--score", choices=["ledger-replay", "rollout"], default="ledger-replay")
    ap.add_argument("--n-rollouts", type=int, default=8, help="rollouts per task when --score rollout")
    ap.add_argument("--max-turns", type=int, default=4, help="matches scripts/probe.py")
    ap.add_argument("--max-new-tokens", type=int, default=192, help="matches scripts/probe.py")
    ap.add_argument("--b-min", type=int, default=1)
    ap.add_argument("--b-max", type=int, default=3)
    # The curriculum axis. Off by default for `--apply-curriculum` because
    # regeneration runs the four gates on a fresh batch, which is real shell
    # work; the *plan* is always computed, because a plan that is never printed
    # is the state this script was in before this flag existed.
    ap.add_argument(
        "--scan",
        default=None,
        help="scan artifact to steer from (default: $MULTIHARNESS_OUT/scan_all.json)",
    )
    ap.add_argument("--apply-curriculum", action="store_true",
                    help="regenerate the flagged tasks and gate them, not just plan")
    ap.add_argument("--curriculum-alpha", type=float, default=cu.DEFAULT_ALPHA)
    ap.add_argument("--curriculum-k-min", type=float, default=cu.DEFAULT_K_MIN)
    ap.add_argument("--curriculum-max-moves", type=int, default=8,
                    help="cap on tasks regenerated in one round; 0 means no cap")
    # The environment axis. Off by default because it is a different *kind* of
    # task (stateful, graded on a final state) rather than a different batch of
    # the same kind, and a run should say which one it measured.
    ap.add_argument("--env-batch", type=int, default=0,
                    help="synthesise this many stateful environment tasks (0 = skip)")
    ap.add_argument("--env-records", type=int, default=4,
                    help="relevant records per synthesised environment")
    ap.add_argument("--env-distractors", type=int, default=3,
                    help="irrelevant records; the knob that makes reading the state necessary")
    ap.add_argument("--env-steps", type=int, default=4,
                    help="tool-chain length, i.e. the depth of the dependency path")
    ap.add_argument("--env-export", action="store_true",
                    help="write Harbor task packages under outputs/rsi/harbor")
    ap.add_argument("--batch-only", action="store_true",
                    help="write rsi/batch.json and rsi/env_batch.json, then stop. Exists so "
                         "the batch can be scanned *before* the curriculum reads a scan: "
                         "steering needs a scan of the batch it is steering, and the batch "
                         "must therefore exist first. Generation is deterministic in --seed, "
                         "so the later full run reproduces this exact batch.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.batch_only:
        # The two-file dance this replaces was a comment in pipeline.sh telling
        # the reader to run `probe.py --from-batch` and then re-run the whole
        # pipeline with CURRICULUM_SCAN set. Nobody did, so the curriculum read
        # a scan of the shipped suite, found no shared ids, and reported zero
        # moves on every run. A closed loop that has to be assembled by hand is
        # an open loop.
        OUT.mkdir(parents=True, exist_ok=True)
        string_batch = task_gen.generate_batch(args.task_batch, seed=args.seed)
        (OUT / "batch.json").write_text(json.dumps(string_batch, indent=2), encoding="utf-8")
        print(f"wrote {OUT / 'batch.json'}  ({len(string_batch)} tasks)")

        if args.env_batch > 0:
            env_tasks = envtask.generate_env_batch(
                args.env_batch,
                seed=args.seed,
                n_records=args.env_records,
                n_distractors=args.env_distractors,
                n_steps=args.env_steps,
            )
            env_batch = env_adapter.as_harness_batch(env_tasks)
            (OUT / "env_batch.json").write_text(
                json.dumps(env_batch, indent=2), encoding="utf-8"
            )
            print(f"wrote {OUT / 'env_batch.json'}  ({len(env_batch)} environment tasks)")

        print("\nnow scan them before steering on them:")
        print(f"  ./run.sh scripts/probe.py --from-batch {OUT / 'batch.json'} \\")
        print(f"      --n <n> --out {OUT / 'scan_batch.json'}")
        return 0

    # Start each run from an empty ledger. Appending to a previous run's ledger
    # would make `tried()` claim edits were attempted when they were attempted
    # in a different experiment, and the yield statistics would be a mixture.
    ledger_path = OUT / "ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    agent = None
    if args.score == "rollout":
        print("loading model for rollout scoring ...")
        from multiharness.rollout import Agent

        agent = Agent(max_turns=args.max_turns, max_new_tokens=args.max_new_tokens)
        print(f"model: {agent.model_id} on {agent.device}\n")

    task_result = run_task_axis(args.task_batch, args.seed, OUT)
    (OUT / "validation.json").write_text(json.dumps(task_result, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'validation.json'}\n")

    # The environment axis runs before the harness axis because its artifact is
    # the benchmark the harness numbers are read against: "0.03 on generated
    # string tasks" and "0.03 on stateful environment tasks" are different
    # findings, and a run that reported only the first would be claiming the
    # weaker one.
    if args.env_batch > 0:
        env_result = run_env_axis(
            args.env_batch,
            args.seed,
            OUT,
            n_records=args.env_records,
            n_distractors=args.env_distractors,
            n_steps=args.env_steps,
            export=args.env_export,
        )
        (OUT / "env_validation.json").write_text(
            json.dumps(env_result, indent=2), encoding="utf-8"
        )
        print(f"wrote {OUT / 'env_validation.json'}\n")

    # The batch is generated once and shared: the curriculum steers *this*
    # batch, and the harness axis scores against it. Generating a second batch
    # for the harness axis would mean the two axes optimised against different
    # task sets in the same round, and a plan naming a task id that the harness
    # axis never saw.
    batch = task_gen.generate_batch(args.task_batch, seed=args.seed)

    # Write the batch so it can be scanned. A generated batch and the shipped
    # suite share no task ids, so the curriculum's scan has to be a scan of
    # *this* batch; `probe.py --from-batch` reads this file. Without it the
    # curriculum can only ever read a scan of the shipped 16 ids, find none of
    # them in the batch, and refuse to move anything — correct behaviour on
    # inputs that cannot produce a result.
    batch_path = OUT / "batch.json"
    batch_path.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(f"wrote {batch_path}  ({len(batch)} tasks; scan it with "
          f"`probe.py --from-batch {batch_path}`)\n")

    scan_path = Path(args.scan) if args.scan else outputs_root() / "scan_all.json"
    curriculum_result = run_curriculum_axis(
        scan_path,
        batch,
        OUT,
        alpha=args.curriculum_alpha,
        k_min=args.curriculum_k_min,
        max_moves=args.curriculum_max_moves or None,
        apply=args.apply_curriculum,
    )
    (OUT / "curriculum.json").write_text(json.dumps(curriculum_result, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'curriculum.json'}\n")

    harness_result = run_harness_axis(
        args.rounds,
        score_mode=args.score,
        tasks=batch,
        n_rollouts=args.n_rollouts,
        seed=args.seed,
        b_min=args.b_min,
        b_max=args.b_max,
        agent=agent,
    )
    (OUT / "harness.json").write_text(json.dumps(harness_result, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'harness.json'}")

    print()
    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    print(f"  tasks accepted      : {task_result['summary']['accepted']}/{task_result['summary']['total']}")
    print(f"  edits attempted     : {harness_result['ledger']['attempted']}")
    print(f"  edits accepted      : {harness_result['ledger']['accepted']}")
    print(f"  score mode          : {harness_result['score_mode']}")
    if "plan" not in curriculum_result:
        print(f"  curriculum          : skipped ({curriculum_result['skipped']})")
    else:
        p = curriculum_result["plan"]
        print(f"  curriculum moves    : {p['move_count']} {p['by_direction'] or ''}")
        print(f"  curriculum held     : {p['held_count']}")
        if not curriculum_result.get("ids_match_batch", True):
            print(f"  curriculum ids      : MISMATCH — {curriculum_result['skipped']}")
        if curriculum_result.get("applied"):
            r = curriculum_result["regenerated"]
            print(f"  regenerated accepted: {r['summary']['accepted']}/{r['summary']['total']}")
    if harness_result["score_mode"] != "rollout":
        print("  NOTE: the harness trajectory is replayed, not measured. "
              "Run with --score rollout for the honest number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
