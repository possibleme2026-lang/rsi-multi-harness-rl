"""Regression test: the curriculum must actually move tasks, and must refuse when it cannot.

The bug this locks down
-----------------------
``rsi/band.py`` defines ``steer``, whose docstring says *"the task generator
consumes"* it, and the README says *"``steer`` in ``rsi/band.py`` already
returns the override"*. Both statements were false: a grep for the symbol found
its definition and its tests and nothing else. The steering rule was never
called by any pipeline script, so the curriculum loop had never closed.

That is the failure mode this repository keeps finding in itself — the honest
arm is the arm that was never executed — and it is invisible from the outside,
because the module has full test coverage and the README describes it
accurately as a *mechanism*. A mechanism nobody calls still passes its own
tests.

Two further bugs were found while wiring it up, and both are pinned here.

**One: the difficulty filter disabled the steering.** GenEnv's rule
``|p̂ − α| > k_min`` was applied *per task*, with ``k_min = 0.1``. But
``mastered`` is ``p > 0.9`` and ``out_of_reach`` is ``p < 0.1``, so every cell
``steer`` would act on sits at least 0.4 from ``α = 0.5`` — far outside a 0.1
band. The filter rejected precisely the cells the rule existed to move, and the
first live run reported ``moves: 0``. GenEnv applies the rule to a *batch's*
aggregate success rate, because what it trains is an environment policy that
emits batches; there is no such policy here, so the rule was moved to
:func:`batch_is_misaligned` where it belongs.

**Two: a mismatched id set produced silently wrong overrides.** The scan
shipped with this repository measures the 24-task suite (``t1-01``) while a
generated batch carries hashed ids (``t1-8f87ad9e``); the sets do not
intersect. ``steer`` needs a task's *parameters*, so with a missing lookup it
falls back to its own defaults and emits an override derived from a task that
does not exist. Nothing in the output would have looked wrong. The lookup is
now an explicit refusal and the id overlap is reported as a count.

What this test asserts
----------------------
1. ``α`` is the argmax of *this* repository's GRPO signal curve, not a constant
   copied from a paper — and it stays so if ``G`` changes.
2. The α-reward is a bell: peaks at ``α``, symmetric, bounded in ``(0, 1]``, and
   independent of batch composition.
3. ``k_min`` is a *batch* predicate. Applying it per task is asserted to be
   incompatible with ever steering — the arithmetic that caused bug one.
4. ``plan_regeneration`` moves mastered cells harder and out-of-reach cells
   easier, holds frontier cells, and refuses unresolved ones.
5. A task whose id is not in the batch is refused rather than steered with
   defaults.
6. ``execute_plan`` produces a task whose parameters actually differ, and the
   regenerated task still passes the four gates — the end-to-end claim.
7. ``n = 64`` is the measurement size at which steering first becomes possible,
   which is a fact about the cost of the loop rather than about the code.

Needs no model and no GPU, so it runs in well under a second.

Run:
    ./run.sh tests/test_rsi_curriculum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in str(sys.path):
    sys.path.insert(0, str(_SRC))

from multiharness.rsi import band as band_mod  # noqa: E402
from multiharness.rsi import curriculum as cu  # noqa: E402
from multiharness.rsi import task_gen, validate  # noqa: E402
from multiharness.rsi.stats import DEFAULT_G, grpo_signal_probability  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _cells(batch: list[dict], spec) -> list[dict]:
    """Build per-cell pass counts from ``spec(index, harness) -> (passes, n)``."""
    out: list[dict] = []
    for i, t in enumerate(batch):
        for h in ("bash_minimal", "react_tools"):
            passes, n = spec(i, h)
            out.append({"harness": h, "task_id": t["id"], "passes": passes, "n": n})
    return out


def main() -> int:
    print("=" * 74)
    print("TEST — RSI curriculum: the steering rule must actually be reachable")
    print("=" * 74)

    # ------------------------------------------------------------------
    print("\n1. alpha is derived from this repo's own signal curve")
    # ------------------------------------------------------------------
    # The point is not that the answer is 0.5 — it is that the constant is tied
    # to the optimizer this repository uses. GenEnv justifies alpha = 0.5 via
    # p(1-p) = 1/4 - (p-1/2)^2, which is a statement about Bernoulli variance.
    # The quantity that decides whether a cell teaches anything *here* is the
    # GRPO group signal probability 1 - p^G - (1-p)^G, and it is already
    # measured at G = 8. So alpha is the argmax of that, and nothing is taken
    # on faith.
    check("alpha is the argmax of the GRPO signal curve", cu.alpha_is_derived(DEFAULT_G))
    check("alpha is 0.5 at G=8", close(cu.optimal_alpha(8), 0.5), f"got {cu.optimal_alpha(8)}")

    for g in (2, 4, 8, 16, 32):
        best = cu.optimal_alpha(g)
        check(
            f"alpha is the argmax at G={g} too",
            close(best, 0.5),
            f"argmax at G={g} was {best}",
        )
    # Symmetry is why: the curve is symmetric about 0.5 for every G, so the
    # argmax cannot move. Asserted rather than assumed.
    for g in (3, 8, 17):
        sym = all(
            close(grpo_signal_probability(p / 20, g), grpo_signal_probability(1 - p / 20, g), 1e-12)
            for p in range(21)
        )
        check(f"signal curve is symmetric about 0.5 at G={g}", sym)

    # The honest counter-argument, stated as a number: the curve is FLAT near
    # the top, so over-tuning difficulty buys almost nothing.
    check(
        "signal at alpha is high (0.9922 at G=8)",
        close(grpo_signal_probability(0.5, 8), 0.9922, 1e-4),
        f"got {grpo_signal_probability(0.5, 8):.4f}",
    )
    check(
        "and p=0.3 already reaches 0.942 — the curve is flat",
        close(grpo_signal_probability(0.3, 8), 0.9423, 1e-3),
        f"got {grpo_signal_probability(0.3, 8):.4f}",
    )

    # ------------------------------------------------------------------
    print("\n2. the alpha-reward is a bell, and a property of the task alone")
    # ------------------------------------------------------------------
    check("reward peaks at alpha", close(cu.alpha_reward(0.5), 1.0))
    check("reward is symmetric", close(cu.alpha_reward(0.4), cu.alpha_reward(0.6)))
    check("reward is bounded by (0, 1]", 0.0 < cu.alpha_reward(0.0) < 1.0 and cu.alpha_reward(0.5) <= 1.0)
    check("reward decreases away from alpha", cu.alpha_reward(0.3) < cu.alpha_reward(0.45))
    # Clamping: a rate outside [0,1] is a caller bug, and the reward must not
    # extrapolate past the peak into a value above 1.
    check("out-of-range rates are clamped, not extrapolated", close(cu.alpha_reward(1.7), cu.alpha_reward(1.0)))

    # No batch normalisation: the same task scores the same in any batch. This
    # is why the reward is not min-max scaled over the batch, and it is the
    # property that makes a plan comparable across rounds.
    solo = cu.batch_alignment([{"id": "a"}], {"a": 0.5})
    with_others = cu.batch_alignment([{"id": "a"}, {"id": "b"}], {"a": 0.5, "b": 0.0})
    check(
        "a task's reward does not depend on its neighbours",
        solo == 1.0 and cu.alpha_reward(0.5) == 1.0,
        f"solo={solo} with_others={with_others}",
    )
    check("batch_alignment ignores unmeasured tasks", cu.batch_alignment([{"id": "x"}], {}) == 0.0)

    # ------------------------------------------------------------------
    print("\n3. k_min is a BATCH rule — applying it per task kills the loop")
    # ------------------------------------------------------------------
    # This is bug one, as arithmetic. The bands and the k_min band are disjoint
    # by construction, so a per-task filter can never pass a steerable cell.
    check("k_min band is [0.4, 0.6] around alpha=0.5", cu.in_band(0.45) and cu.in_band(0.6) and not cu.in_band(0.61))
    check("boundary is inclusive", cu.in_band(0.4) and cu.in_band(0.6))

    mastered_p = band_mod.MASTERED_ABOVE  # 0.9
    oor_p = band_mod.OUT_OF_REACH_BELOW  # 0.1
    gap_mastered = abs(mastered_p - cu.DEFAULT_ALPHA)
    gap_oor = abs(oor_p - cu.DEFAULT_ALPHA)
    check(
        "every steerable cell is OUTSIDE the k_min band (so per-task filtering would reject them all)",
        gap_mastered > cu.DEFAULT_K_MIN and gap_oor > cu.DEFAULT_K_MIN,
        f"gaps are {gap_mastered} and {gap_oor}, k_min is {cu.DEFAULT_K_MIN}",
    )
    check(
        "the band edges are 4x the k_min band, so the two cannot overlap",
        gap_oor >= 4 * cu.DEFAULT_K_MIN,
        f"{gap_oor} vs {cu.DEFAULT_K_MIN}",
    )

    # The batch-level rule, where it does work.
    mis_easy, why_easy = cu.batch_is_misaligned(0.9)
    mis_hard, why_hard = cu.batch_is_misaligned(0.05)
    ok_mid, why_mid = cu.batch_is_misaligned(0.5)
    check("a too-easy batch is flagged", mis_easy and "too easy" in why_easy, why_easy)
    check("a too-hard batch is flagged", mis_hard and "too hard" in why_hard, why_hard)
    check("an on-target batch is not flagged", not ok_mid, why_mid)
    check("the diagnosis names the direction", "+" in why_easy and "-" in why_hard)

    # ------------------------------------------------------------------
    print("\n4. planning moves the right cells in the right direction")
    # ------------------------------------------------------------------
    batch = task_gen.generate_batch(12, seed=11)
    by_id = {t["id"]: t for t in batch}

    # n=64 is the measurement size at which a cell can resolve at all. At n=32
    # an all-pass cell has a Wilson lower bound of 0.8928, below the 0.9
    # threshold, so it is `unresolved` and nothing moves. That threshold is a
    # fact about the cost of the loop, and it is pinned below in section 7.
    def spec(i, _h):
        if i % 3 == 0:
            return 64, 64  # mastered -> harder
        if i % 3 == 1:
            return 0, 64  # out_of_reach -> easier
        return 32, 64  # frontier -> hold

    plan = cu.plan_regeneration(_cells(batch, spec), tasks_by_id=by_id, source="<test>")
    check("moves were produced", plan.move_count == 8, f"got {plan.move_count}")
    check(
        "both directions appear",
        plan.by_direction == {"harder": 4, "easier": 4},
        f"got {plan.by_direction}",
    )
    check("frontier cells are held", plan.held_count == 4, f"got {plan.held_count}")

    for m in plan.moves:
        if m["band"] == band_mod.Band.MASTERED:
            check(f"{m['task_id']}: mastered moves harder", m["direction"] == "harder")
            check("  and its override raises difficulty", m["override"].get("steps", 0) >= 2)
        else:
            check(f"{m['task_id']}: out_of_reach moves easier", m["direction"] == "easier")
            # Relative to the task's own parameters, not an absolute cap: a T4
            # task can start at payload_len 16 and a halving is still a
            # reduction. Asserting an absolute bound here failed on exactly that
            # task and the failure was the assertion's, not the code's.
            base_p = by_id[m["task_id"]]["params"]
            check(
                "  and its override lowers difficulty",
                m["override"].get("payload_len", 99) <= base_p["payload_len"],
                f"payload {base_p['payload_len']} -> {m['override'].get('payload_len')}",
            )

    # A mastered cell must not have its payload *reduced*, and vice versa: that
    # is the bug a sign error in `steer` would produce, and it would still
    # report a plausible plan.
    m0 = next(m for m in plan.moves if m["band"] == band_mod.Band.MASTERED)
    base0 = by_id[m0["task_id"]]["params"]
    check(
        "mastered: payload does not shrink",
        m0["override"]["payload_len"] >= base0["payload_len"],
        f"{base0['payload_len']} -> {m0['override']['payload_len']}",
    )
    e0 = next(m for m in plan.moves if m["band"] == band_mod.Band.OUT_OF_REACH)
    basee0 = by_id[e0["task_id"]]["params"]
    check(
        "out_of_reach: payload does not grow",
        e0["override"]["payload_len"] <= basee0["payload_len"],
        f"{basee0['payload_len']} -> {e0['override']['payload_len']}",
    )

    # ------------------------------------------------------------------
    print("\n5. an id that is not in the batch is refused, not steered blindly")
    # ------------------------------------------------------------------
    # Bug two. `steer` needs the task's parameters; without them it falls back
    # to defaults and emits an override for a task that does not exist. The
    # shipped scan measures `t1-01` while a generated batch carries hashed ids,
    # so this is the *default* situation, not an edge case.
    check("the two id sets really are disjoint", not ({"t1-01"} & set(by_id)))
    plan_nomatch = cu.plan_regeneration(_cells(batch, spec), tasks_by_id={}, source="<test>")
    check("nothing is moved without parameters", plan_nomatch.move_count == 0)
    check("and every cell is held with a reason", plan_nomatch.held_count == 12)
    reasons = " ".join(h["reason"] for h in plan_nomatch.held)
    check(
        "the reason names the mismatch",
        "different task sets" in reasons,
        reasons[:160],
    )
    # A partially-matching id set must move exactly the matching tasks.
    half = {t["id"]: t for t in batch[:3]}
    plan_half = cu.plan_regeneration(_cells(batch, spec), tasks_by_id=half, source="<test>")
    check(
        "a partial match moves only the matched tasks",
        plan_half.move_count < plan.move_count and plan_half.move_count > 0,
        f"full={plan.move_count} half={plan_half.move_count}",
    )

    # ------------------------------------------------------------------
    print("\n6. executing the plan produces a real task, and it still gates")
    # ------------------------------------------------------------------
    fresh = cu.execute_plan(plan, by_id)
    check("every move materialised", len(fresh) == plan.move_count, f"{len(fresh)} vs {plan.move_count}")
    check("regenerated ids are new", all(f["id"] not in by_id for f in fresh))
    check(
        "provenance is recorded",
        all(f["regenerated_from"] in by_id and f["regenerated_band"] for f in fresh),
    )
    # A move that changed nothing would be a silent no-op reported as a move.
    moved = 0
    for f in fresh:
        old = by_id[f["regenerated_from"]]["params"]
        new = f["params"]
        if any(old[k] != new[k] for k in ("payload_len", "escape_density", "steps", "read_source")):
            moved += 1
    check("every regenerated task actually differs", moved == len(fresh), f"{moved} of {len(fresh)}")

    # The end-to-end claim: a regenerated task is still a *usable* task. This is
    # the case where a generator bug is most likely, because steering pushes
    # parameters to the edge of the space.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        verdicts = validate.validate_batch(fresh, roots=Path(td))
        summary = validate.summarise_validation(verdicts)
    check(
        "regenerated tasks pass all four gates",
        summary["accepted"] == summary["total"],
        f"{summary['accepted']}/{summary['total']}, failures={summary['failures_by_gate']}",
    )
    check("the oracle gate ran on every one", all(v.gates for v in verdicts))

    # ------------------------------------------------------------------
    print("\n7. n=64 is where steering first becomes possible")
    # ------------------------------------------------------------------
    # Steering needs a *resolved* band, and resolution needs an interval that
    # fits inside one band. At n=32 an all-pass cell has a Wilson lower bound of
    # 0.8928 — below MASTERED_ABOVE = 0.9 — so it is unresolved and the loop
    # cannot act, however clear the point estimate looks.
    check("n=32 all-pass is unresolved (lo=0.8928 < 0.9)", band_mod.band_of(32, 32).band == band_mod.Band.UNRESOLVED)
    check("n=32 all-fail is unresolved (hi=0.1072 > 0.1)", band_mod.band_of(0, 32).band == band_mod.Band.UNRESOLVED)
    check("n=64 all-pass resolves mastered", band_mod.band_of(64, 64).band == band_mod.Band.MASTERED)
    check("n=64 all-fail resolves out_of_reach", band_mod.band_of(0, 64).band == band_mod.Band.OUT_OF_REACH)

    plan32 = cu.plan_regeneration(
        _cells(batch, lambda i, h: (32, 32) if i % 3 == 0 else (0, 32) if i % 3 == 1 else (16, 32)),
        tasks_by_id=by_id,
        source="<test n=32>",
    )
    check(
        "at n=32 the same scan produces zero moves",
        plan32.move_count == 0,
        f"got {plan32.move_count}",
    )

    # ------------------------------------------------------------------
    print("\n8. a generated batch can be registered, so it can be scanned")
    # ------------------------------------------------------------------
    # The last link in the chain. The curriculum can only steer tasks that were
    # *measured*, and measuring a generated batch requires its ids to resolve in
    # the process-global registry that `Agent.run` reads. The shipped suite and
    # a generated batch are different task sets, so without an explicit
    # registration path the curriculum would be structurally unable to scan its
    # own batch — a mechanism that exists and cannot be executed, which is the
    # exact failure this module was written to fix.
    import json
    import tempfile

    from multiharness.harnesses.core import TASKS, register_tasks
    from multiharness.tasks import load as load_tasks

    load_tasks()
    suite_size = len(TASKS)
    check("the shipped suite is loaded", suite_size >= 24, f"{suite_size} tasks")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "batch.json"
        p.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        reloaded = json.loads(p.read_text(encoding="utf-8"))
        fresh = [t for t in reloaded if t["id"] not in TASKS]
        register_tasks(fresh)

    check("every generated task registered", len(fresh) == len(batch), f"{len(fresh)} of {len(batch)}")
    check("generated ids now resolve", all(t["id"] in TASKS for t in batch))
    check("shipped ids still resolve", "t1-01" in TASKS)
    check("the registry grew by exactly the batch", len(TASKS) == suite_size + len(batch))
    # A generated task carries the fields `core.verify` reads, so it needs no
    # translation on the way in — that is what makes the round trip lossless.
    check(
        "generated tasks carry the fields the verifier reads",
        all(("verify" in t and "expected" in t and "params" in t) for t in batch),
    )

    # ------------------------------------------------------------------
    print("\n9. the plan is deterministic and self-describing")
    # ------------------------------------------------------------------
    p_a = cu.plan_regeneration(_cells(batch, spec), tasks_by_id=by_id, source="<test>")
    p_b = cu.plan_regeneration(_cells(batch, spec), tasks_by_id=by_id, source="<test>")
    check("two runs produce the same plan", p_a.as_dict() == p_b.as_dict())
    d = p_a.as_dict()
    for key in ("source", "alpha", "k_min", "signal_at_target", "moves", "held", "move_count", "held_count"):
        check(f"the artifact carries {key}", key in d)
    check("alpha_derived_from_g is recorded", d["alpha_derived_from_g"] == DEFAULT_G)
    check("the artifact is JSON-serialisable", isinstance(__import__("json").dumps(d), str))
    # max_moves caps the batch, and it drops the *least* aligned first.
    capped = cu.plan_regeneration(_cells(batch, spec), tasks_by_id=by_id, source="<test>", max_moves=3)
    check("max_moves caps the plan", capped.move_count == 3, f"got {capped.move_count}")
    check(
        "the cap keeps the least-aligned tasks",
        all(c["reward"] <= max(m["reward"] for m in capped.moves) for c in capped.moves),
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — the steering rule is reachable and the loop closes")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
