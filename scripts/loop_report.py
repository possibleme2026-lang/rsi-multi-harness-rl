"""Did the curriculum actually move the batch? The closed loop's one number.

The RSI loop has four axes, and the task axis is the one where "it works" is
easiest to assert and hardest to check. Every part of it passes its tests:

* ``band.steer`` returns an override,
* ``curriculum.plan_regeneration`` turns measurements into moves,
* ``task_gen.reparameterise`` turns a move into a well-formed task,
* the four gates accept the result.

And none of that says the batch got *better*, because "better" is a claim about
a measurement that has not been taken. The regenerated tasks are validated for
solvability, which is a different property: a task can be perfectly solvable and
still sit outside the band where GRPO has any signal to amplify.

This script takes the two measurements and reports the difference. It is the
only place in the repository where the task axis produces a number that could
come out *negative* — every other check is a pass/fail on well-formedness, and
a suite of pass/fail checks cannot detect a curriculum that moves tasks in the
wrong direction.

Run:
    ./run.sh scripts/loop_report.py \\
        --before-batch outputs/rsi/batch.json \\
        --before-scan  outputs/rsi/scan_batch.json \\
        --after-batch  outputs/rsi/batch_steered.json \\
        --after-scan   outputs/rsi/scan_batch_steered.json

Costs nothing and needs no model: both scans already exist by the time this
runs, so this is arithmetic over files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness._bootstrap import outputs_root
from multiharness.rsi import curriculum as cu


def _pass_rates(scan_path: Path) -> dict[str, float]:
    """Per-task pass rate, pooled over harnesses.

    Pooled rather than per-cell because the question here is about the *batch*,
    and a batch has one row per (harness, task) pair whose difficulty is a
    property of the task. Averaging the cells would let a task measured by four
    harnesses count four times, which is what the batch already does when it
    builds rows — so pooling the records and then dividing by task is the
    quantity that matches the training set.
    """
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    agg: dict[str, list[int]] = {}
    for r in scan.get("records", []):
        tid = str(r.get("task_id", ""))
        if not tid:
            continue
        c = agg.setdefault(tid, [0, 0])
        c[0] += int(float(r.get("reward", 0.0)) >= 1.0)
        c[1] += 1
    return {tid: p / n for tid, (p, n) in agg.items() if n > 0}


def _counts(scan_path: Path) -> dict[str, tuple[int, int]]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    agg: dict[str, tuple[int, int]] = {}
    for r in scan.get("records", []):
        tid = str(r.get("task_id", ""))
        if not tid:
            continue
        p, n = agg.get(tid, (0, 0))
        agg[tid] = (p + int(float(r.get("reward", 0.0)) >= 1.0), n + 1)
    return agg


def _band_mix(counts: dict[str, tuple[int, int]], task_ids: list[str]) -> dict[str, int]:
    """How the batch splits across difficulty bands, over the tasks it contains.

    ``band_of`` returns a ``BandedCell`` — the band plus the interval that
    justifies it — so the band is read off ``.band``. Taking the object itself
    would work until the first ``sorted()``, and then fail on a comparison
    between two cells rather than on anything to do with difficulty.
    """
    out: dict[str, int] = {}
    for tid in task_ids:
        pn = counts.get(tid)
        if pn is None:
            out["unmeasured"] = out.get("unmeasured", 0) + 1
            continue
        band = cu.band_mod.band_of(pn[0], pn[1]).band
        out[band] = out.get(band, 0) + 1
    return out


def _params_mix(batch: list[dict]) -> dict[str, dict]:
    """Distribution of the difficulty knobs, for the before/after comparison.

    Only over tasks that carry ``params`` (generated string tasks). Environment
    tasks do not, and are counted separately rather than silently averaged into
    a distribution they are not part of.
    """
    keys = ("payload_len", "escape_density", "steps")
    out: dict[str, dict] = {k: {} for k in keys}
    out["verify_mode"] = {}
    n_with = 0
    for t in batch:
        p = t.get("params")
        if not p:
            continue
        n_with += 1
        for k in keys:
            v = p.get(k)
            out[k][str(v)] = out[k].get(str(v), 0) + 1
        vm = p.get("verify_mode")
        out["verify_mode"][str(vm)] = out["verify_mode"].get(str(vm), 0) + 1
    out["_tasks_with_params"] = n_with
    out["_tasks_without_params"] = len(batch) - n_with
    return out


def _tier_mix(batch: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in batch:
        tier = str(t.get("tier", "?"))
        out[tier] = out.get(tier, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--before-batch", required=True, help="the batch the scan steered")
    ap.add_argument("--before-scan", required=True, help="its scan")
    ap.add_argument("--after-batch", required=True, help="rsi/batch_steered.json")
    ap.add_argument("--after-scan", required=True, help="a scan of the steered batch")
    ap.add_argument("--out", default=None, help="default: outputs/rsi/loop_closed.json")
    args = ap.parse_args()

    before_batch = json.loads(Path(args.before_batch).read_text(encoding="utf-8"))
    after_batch = json.loads(Path(args.after_batch).read_text(encoding="utf-8"))
    before_scan = Path(args.before_scan)
    after_scan = Path(args.after_scan)
    out_path = Path(args.out) if args.out else outputs_root() / "rsi" / "loop_closed.json"

    before_rates = _pass_rates(before_scan)
    after_rates = _pass_rates(after_scan)
    before_counts = _counts(before_scan)
    after_counts = _counts(after_scan)

    before_ids = [t["id"] for t in before_batch]
    after_ids = [t["id"] for t in after_batch]

    # The headline number. `batch_alignment` is the mean α-reward over the
    # tasks that have a measurement, and α is the pass rate GRPO has the most
    # signal at — so a rise means the batch moved toward learnable, which is the
    # only thing the task axis is trying to do.
    before_align = cu.batch_alignment(before_batch, before_rates)
    after_align = cu.batch_alignment(after_batch, after_rates)
    delta = after_align - before_align

    # How many tasks actually changed. Read off `regenerated_from`, which the
    # curriculum writes, rather than by diffing ids — a reparameterised task can
    # in principle collide with an existing id, and then a diff would undercount
    # exactly the case that needs attention.
    moved = [t for t in after_batch if "regenerated_from" in t]
    kept = len(after_batch) - len(moved)
    directions: dict[str, int] = {}
    for t in moved:
        d = str(t.get("regenerated_direction", "?"))
        directions[d] = directions.get(d, 0) + 1

    # ---- the treatment and the control -------------------------------------
    # A steered batch contains two populations that must not be pooled. The
    # moved tasks are the treatment: their before and after are two *different*
    # tasks, so a change in their α-reward is attributable to the steering rule.
    # The kept tasks are a control: they are the same task measured twice, so any
    # change in their α-reward is sampling noise and nothing else.
    #
    # Pooling them was a real defect in the first version of this script. In the
    # first measured run the whole-batch delta came out negative (-0.0031) and
    # the script called it "the steering rule is moving tasks in a direction its
    # own band definition disagrees with". That was wrong: every moved task had
    # a delta >= 0, and the entire negative came from three kept tasks drifting
    # by a rollout or two (58/256 -> 54/256 on a task that was never touched).
    # The control group was reporting the noise floor and the script read it as
    # a treatment effect. The fix is to compute the two deltas separately and
    # let the verdict rest on the treatment, with the control printed as the
    # scale against which the treatment has to be read.
    def _delta(task: dict) -> tuple[float, float, float] | None:
        """``(before_p, after_p, delta_alpha_reward)`` for one after-batch task."""
        src = task.get("regenerated_from", task["id"])
        if src not in before_rates or task["id"] not in after_rates:
            return None
        pb = before_rates[src]
        pa = after_rates[task["id"]]
        return pb, pa, cu.alpha_reward(pa) - cu.alpha_reward(pb)

    per_task: list[tuple[str, str, float, float, float, bool]] = []
    for t in after_batch:
        d = _delta(t)
        if d is None:
            continue
        per_task.append((t.get("regenerated_from", t["id"]), t["id"], d[0], d[1], d[2],
                         "regenerated_from" in t))

    moved_rows = [r for r in per_task if r[5]]
    kept_rows = [r for r in per_task if not r[5]]
    delta_moved = sum(r[4] for r in moved_rows) / len(moved_rows) if moved_rows else 0.0
    delta_kept = sum(r[4] for r in kept_rows) / len(kept_rows) if kept_rows else 0.0

    before_measured = sum(1 for tid in before_ids if tid in before_rates)
    after_measured = sum(1 for tid in after_ids if tid in after_rates)

    print("=" * 74)
    print("CLOSED-LOOP REPORT — did the curriculum move the batch toward learnable?")
    print("=" * 74)
    print(f"  before batch : {args.before_batch}  ({len(before_batch)} tasks)")
    print(f"  before scan  : {args.before_scan}  ({before_measured} tasks measured)")
    print(f"  after  batch : {args.after_batch}  ({len(after_batch)} tasks)")
    print(f"  after  scan  : {args.after_scan}  ({after_measured} tasks measured)")
    print()
    print(f"  moved        : {len(moved)}  {directions or ''}")
    print(f"  kept         : {kept}  (on the frontier, or unmeasured)")
    print()
    print(f"  alignment    : before {before_align:.4f}  ->  after {after_align:.4f}"
          f"   delta {delta:+.4f}   (whole batch)")
    # The two populations, kept apart. The treatment is the number the steering
    # rule is answerable for; the control is the noise floor that number has to
    # clear before it means anything.
    if moved_rows:
        print(f"  moved delta  : {delta_moved:+.4f}   over {len(moved_rows)} replaced tasks"
              f"   (treatment)")
    if kept_rows:
        print(f"  kept  delta  : {delta_kept:+.4f}   over {len(kept_rows)} untouched tasks"
              f"   (control — same task twice, so this is noise)")
    print()

    # The bands, before and after. This is the number that says whether the
    # move was in the right direction for the *right reason*: a delta driven by
    # pushing mastered tasks further out of reach would look the same in the
    # alignment figure if the pool were small enough.
    before_mix = _band_mix(before_counts, before_ids)
    after_mix = _band_mix(after_counts, after_ids)
    bands = sorted(set(before_mix) | set(after_mix))
    print(f"  {'band':<16}{'before':>8}{'after':>8}")
    for b in bands:
        print(f"  {b:<16}{before_mix.get(b, 0):>8}{after_mix.get(b, 0):>8}")
    print()

    before_tiers = _tier_mix(before_batch)
    after_tiers = _tier_mix(after_batch)
    tiers = sorted(set(before_tiers) | set(after_tiers))
    print(f"  {'tier':<16}{'before':>8}{'after':>8}   (sampling origin, not a difficulty read)")
    for t in tiers:
        print(f"  {t:<16}{before_tiers.get(t, 0):>8}{after_tiers.get(t, 0):>8}")
    print()

    params = _params_mix(after_batch)
    print("  parameter mix after steering:")
    for k in ("payload_len", "escape_density", "steps", "verify_mode"):
        dist = params.get(k, {})
        shown = "  ".join(f"{v}x{n}" for v, n in sorted(dist.items(), key=lambda kv: kv[0]))
        print(f"    {k:<16} {shown or '(none)'}")
    print()

    # The interpretation is derived, not asserted. Each branch names what the
    # numbers can and cannot support, and the case that is easiest to oversell —
    # a zero delta — gets the most careful wording, because "the curriculum did
    # nothing" and "the curriculum could not do anything" are different findings
    # and only one of them is about the generator.
    #
    # The verdict rests on `delta_moved`, the treatment effect, not on the pooled
    # `delta`. The pooled number includes the control group, so it can be
    # negative while every task the steering rule actually touched improved —
    # which is exactly what the first measured run showed. Reading the pooled
    # number as a verdict means reading sampling noise as a defect.
    if not moved:
        interp = (
            "no tasks were moved, so the delta above is 0 by construction and "
            "carries no evidence either way — the steered batch is the original "
            "batch, compared against its own scan. This is a statement about the "
            "scan or about the batch already being on target (see move_diagnosis "
            "in curriculum.json), not about the generator."
        )
    elif delta_moved > 0:
        interp = (
            f"positive on the treatment: the {len(moved_rows)} replaced tasks moved "
            "toward the band GRPO has signal at. This is the task axis doing what it "
            "claims. It is a statement about *alignment*, not yet about learning — "
            "whether the policy improves on the steered batch is the training run's "
            "question. Read it against the control: the untouched tasks' delta is "
            "the noise floor, and the treatment has to be larger than it to mean "
            "anything."
        )
    elif delta_moved < 0:
        interp = (
            f"negative on the treatment: the {len(moved_rows)} replaced tasks moved "
            "away from the learnable band. The steering rule is moving tasks in a "
            "direction its own band definition disagrees with — a real defect, and "
            "the reason this script exists rather than another pass/fail gate."
        )
    else:
        # Zero has several causes and they are not equivalent. Distinguished by
        # whether the *before* batch already sat where it should.
        before_frontier = before_mix.get(cu.band_mod.Band.FRONTIER, 0)
        if before_frontier == len(before_ids):
            interp = (
                "zero, and the batch was already entirely on the frontier — nothing "
                "needed to move. A correct no-op, not a failure."
            )
        elif before_mix.get(cu.band_mod.Band.OUT_OF_REACH, 0) > 0:
            interp = (
                "zero on the treatment despite moving tasks out of the out-of-reach "
                "band. The moves reduced difficulty but not enough to cross into the "
                "frontier — the steer step size (a single step per knob) is too small "
                "for the gap that was measured. Widen the step or scan at a larger n."
            )
        else:
            interp = (
                "zero on the treatment with tasks moved. The moves cancelled in "
                "aggregate — some easier, some harder, netting out. Read the band "
                "table above rather than the alignment delta alone."
            )

    # The pooled delta can disagree with the treatment, and when it does the
    # disagreement is the finding rather than a detail: it means the control
    # group moved more than the treatment did, so no steering conclusion can be
    # drawn at this n. Said explicitly so the pooled number is never quoted alone.
    if moved and kept_rows and (delta > 0) != (delta_moved > 0):
        interp += (
            f" The whole-batch delta ({delta:+.4f}) disagrees in sign with the "
            f"treatment delta ({delta_moved:+.4f}) because the {len(kept_rows)} "
            "untouched tasks moved further than the replaced ones — that is "
            "sampling noise in the control, and it means the treatment is not "
            "distinguishable from the noise floor at this n."
        )

    print("  interpretation:")
    for line in interp.split(". "):
        if line.strip():
            print(f"    {line.strip().rstrip('.')}.")
    print()

    payload = {
        "before": {
            "batch": args.before_batch,
            "scan": args.before_scan,
            "tasks": len(before_batch),
            "measured": before_measured,
            "alignment": round(before_align, 6),
            "bands": before_mix,
            "tiers": before_tiers,
        },
        "after": {
            "batch": args.after_batch,
            "scan": args.after_scan,
            "tasks": len(after_batch),
            "measured": after_measured,
            "alignment": round(after_align, 6),
            "bands": after_mix,
            "tiers": after_tiers,
            "params": params,
        },
        "moved": len(moved),
        "kept": kept,
        "directions": directions,
        # The pooled delta, and the treatment/control split it hides. The pooled
        # number is kept for continuity with the printed table, but the verdict
        # and the exit code rest on `alignment_delta_moved`: the kept tasks are a
        # control group (the same task measured twice), so their movement is the
        # noise floor and pooling it into the headline lets noise decide the sign.
        "alignment_delta": round(delta, 6),
        "alignment_delta_moved": round(delta_moved, 6),
        "alignment_delta_kept": round(delta_kept, 6),
        "moved_measured": len(moved_rows),
        "kept_measured": len(kept_rows),
        "per_task": [
            {
                "from": r[0],
                "to": r[1],
                "before_pass_rate": round(r[2], 6),
                "after_pass_rate": round(r[3], 6),
                "delta_alpha_reward": round(r[4], 6),
                "moved": r[5],
            }
            for r in per_task
        ],
        # Whether the delta is evidence. With zero moves the steered batch *is*
        # the original and is compared against its own scan, so the delta is 0
        # for arithmetic reasons rather than because the batch was measured to be
        # well-placed. Flagged in the payload so a reader (or a figure) cannot
        # pick up the number without the caveat that belongs to it.
        "delta_is_evidence": bool(moved),
        "interpretation": interp,
        "alpha": cu.DEFAULT_ALPHA,
        "alpha_derived_from_g": cu.DEFAULT_G,
        "note": (
            "alignment is the mean α-reward over measured tasks, where α is the "
            "pass rate with maximal GRPO signal. It is a statement about the "
            "batch's position in difficulty space, not about policy learning. "
            "`alignment_delta_moved` is the treatment effect; "
            "`alignment_delta_kept` is the noise floor from the untouched control "
            "tasks and is not a result."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # A negative *treatment* delta is the one outcome this script is built to
    # catch, so it is worth a non-zero exit: a CI job or a pipeline can gate on
    # it. The pooled delta is deliberately not used here — the first measured run
    # had a negative pooled delta with a non-negative treatment, and exiting on
    # the pooled number would have failed the pipeline over sampling noise in
    # tasks the steering rule never touched. Zero is not an error either; it has
    # innocent explanations, listed above.
    return 1 if delta_moved < 0 else 0


if __name__ == "__main__":
    sys.exit(main())
