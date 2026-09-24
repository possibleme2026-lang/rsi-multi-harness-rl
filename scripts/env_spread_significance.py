"""Is the cross-harness spread a signal or two lucky samples?

The environment scan reports ``G3 cross-harness spread`` as the largest
per-task difference in pass rate between harnesses. After the guidance fix it
reads **0.20**, which crosses the gate. But 0.20 came from **two** rollouts out
of 96: ``react_tools`` scored 0.20 and 0.10 on two tasks where the other three
harnesses scored 0.00. Two positives, n=2 per cell.

That is exactly the situation the repository's own ``stats`` module exists to
adjudicate, and reporting ``0.20`` as a measured gap without running it through
that module would be the same error as the ``81.2%`` tool-call rate — a number
that crossed a threshold while meaning nothing.

So this script asks three questions the raw spread cannot answer:

1. **Is the spread above the noise floor?** The floor is the smallest
   difference not explained by sampling noise, computed from the pooled reward
   distribution (``stats.noise_floor``).
2. **What is the confidence interval on each cell's rate?** An observed ``1/2``
   has a Wilson interval of roughly ``[0.09, 0.91]``; if that overlaps the
   zeros beside it, the spread is inside the interval, not outside it.
3. **How many rollouts would it take to detect a difference this size?** The
   minimum detectable effect at this ``n``, and the ``n`` needed for the
   observed effect (``stats.minimum_detectable_effect``,
   ``stats.rollouts_for_effect``). This is the number that turns "we measured
   a gap" into "we measured a gap, or we measured too little".

Run:
    ./run.sh scripts/env_spread_significance.py
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
from multiharness.rsi import stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default=None,
                    help="scan artifact (default: $MULTIHARNESS_OUT/rsi/env_scan.json)")
    args = ap.parse_args()

    path = Path(args.scan) if args.scan else outputs_root() / "rsi" / "env_scan.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    n_per_cell = data.get("n", 2)
    harnesses = sorted(data["matrix"].keys())

    print("=" * 78)
    print("IS THE CROSS-HARNESS SPREAD REAL?")
    print("=" * 78)
    print(f"scan    : {path}")
    print(f"model   : {data.get('model')}")
    print(f"rollouts: {len(records)}   ({n_per_cell} per cell, "
          f"{len(harnesses)} harnesses)")
    print()

    # ---- per-task spread, with intervals on the cells that differ --------
    task_ids = sorted({r["task_id"] for r in records})
    print("per-task spread, and the intervals behind it")
    print(f"  {'task':<16}" + "".join(f"{h[:11]:>13}" for h in harnesses)
          + f"{'spread':>9}")
    worst = ("", 0.0)
    for tid in task_ids:
        cells = {}
        for h in harnesses:
            rs = [r["reward"] for r in records
                  if r["harness"] == h and r["task_id"] == tid]
            cells[h] = (sum(rs), len(rs), sum(rs) / len(rs) if rs else 0.0)
        rates = [v[2] for v in cells.values()]
        spread = max(rates) - min(rates)
        if spread > worst[1]:
            worst = (tid, spread)
        row = "".join(f"{cells[h][2]:>12.2f} " for h in harnesses)
        print(f"  {tid:<16}{row}{spread:>8.2f}")
    print(f"\n  largest spread: {worst[1]:.2f} on {worst[0]}")
    print()

    # ---- the intervals on the cells that produce the spread --------------
    print(f"the cells behind the largest spread ({worst[0]})")
    print(f"  {'harness':<18}{'rate':>7}{'95% Wilson':>22}{'verdict':>16}")
    intervals = {}
    for h in harnesses:
        rs = [r["reward"] for r in records
              if r["harness"] == h and r["task_id"] == worst[0]]
        passes = sum(1 for r in rs if r > 0)
        lo, hi = stats.wilson_interval(passes, len(rs))
        intervals[h] = (lo, hi)
        verdict = "above 0" if lo > 0.0 else "indistinct from 0"
        print(f"  {h:<18}{passes / len(rs):>7.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>22}{verdict:>16}")
    print()

    # ---- does the best interval clear the zeros beside it? ---------------
    best = max(harnesses, key=lambda h: intervals[h][0])
    lo, _hi = intervals[best]
    print(f"  the best cell is {best}: its interval starts at {lo:.2f}.")
    if lo <= 0.0:
        print("  It does NOT exclude 0.00, so the spread is inside the sampling")
        print("  interval of the zeros next to it — this is not yet evidence of a gap.")
    else:
        print("  It excludes 0.00, so this cell is above the floor.")
    print()

    # ---- noise floor and power -------------------------------------------
    pooled: list[list[float]] = []
    for h in harnesses:
        for tid in task_ids:
            pooled.append([r["reward"] for r in records
                           if r["harness"] == h and r["task_id"] == tid])
    floor = stats.noise_floor(pooled)
    print(f"noise floor (z=2, pooled)   : {floor:.4f}")
    print(f"observed largest spread     : {worst[1]:.4f}"
          f"   {'ABOVE floor' if worst[1] > floor else 'BELOW floor'}")
    print()

    # ---- what would it take ----------------------------------------------
    # `p_bar` is the pooled rate across the two arms being compared. Using the
    # overall rate is the conservative choice: it is the rate a null of "no
    # difference" would assume.
    nonempty = [r["reward"] for r in records]
    p_bar = sum(1 for x in nonempty if x > 0) / len(nonempty)
    print(f"pooled pass rate            : {p_bar:.4f}  "
          f"({sum(1 for x in nonempty if x > 0)}/{len(nonempty)})")
    if 0.0 < p_bar < 1.0:
        mde = stats.minimum_detectable_effect(n_per_cell, max(p_bar, 0.01))
        print(f"minimum detectable effect   : {mde:.4f} at n={n_per_cell}/arm")
        if worst[1] < mde:
            print("  the observed spread is BELOW what this n could detect, so the")
            print("  design cannot distinguish 'no gap' from 'a gap this size'.")
        need = stats.rollouts_for_effect(max(worst[1], 1e-3), max(p_bar, 0.01))
        print(f"rollouts/arm for the observed effect: {need}"
              f"   ({need / max(n_per_cell, 1):.0f}x the current {n_per_cell})")
    else:
        print("  p_bar is on a floor, so a power calculation is undefined here —")
        print("  the honest statement is that the base rate is ~0 and the design")
        print("  has no power at any n until the task is easier or the model larger.")
    print()

    print("verdict")
    if lo <= 0.0 and worst[1] <= floor:
        print("  The spread is not separable from noise at this n. Report the")
        print("  environment axis as having produced partial credit — which it now")
        print("  has — and NOT as having measured a cross-harness gap.")
    else:
        print("  The spread clears at least one of the two checks above; see the")
        print("  individual lines for which, and size the claim to that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
