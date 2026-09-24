"""Regression test: the statistics layer must not manufacture certainty.

The bug this locks down
-----------------------
``scripts/train.py`` filtered training rows with a two-way split::

    SIGNAL_LO, SIGNAL_HI = 0.05, 0.95
    keep = row if SIGNAL_LO < p < SIGNAL_HI else drop

and its docstring justified the drop by claiming cells outside that band are
*"provably wasted"*. Two separate things are wrong with that.

**One: the arithmetic.** GRPO gives a group of ``G`` completions a gradient
only when their rewards disagree, so a group is dead with probability
``p**G + (1-p)**G``. At ``p = 0.05, G = 8`` that is **0.663** — a third of the
groups still carry gradient. Only ``p = 0`` and ``p = 1`` are provably wasted;
every value strictly between them carries some signal.

**Two: the measurement.** A cell's rate is estimated from a finite sample. An
observed ``0/8`` has a 95% upper confidence bound of **0.312**, and a cell
whose true rate is 0.30 shows ``0/8`` about 5.8% of the time. "We measured
zero" is evidence about the *sample*, not about the cell. The old split
silently converted that ambiguity into "drop", which biases the training set
towards tasks the model already finds easy and makes the suite look easier
than it is.

What this test asserts
----------------------
1. The GRPO arithmetic is what it claims, including that it is strictly
   between 0 and 1 for every ``0 < p < 1``.
2. Wilson intervals stay inside ``[0, 1]``, stay non-degenerate at
   ``passes = 0``, and are wider than the Wald interval they replace — the
   whole point is to be less certain.
3. ``rule_of_three_upper`` uses ``(1 - confidence)`` as its base, not
   ``confidence``; the two differ by 49x at ``n = 8``.
4. ``classify_cell`` refuses to call a cell dead on thin evidence, and refuses
   to call it live on a narrow-but-straddling interval.
5. ``summarise_cells`` reports how many rows the old filter would have dropped
   on evidence that did not support it.

Needs no model and no GPU, so it runs in well under a second.

Run:
    ./run.sh tests/test_rsi_stats.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.rsi.stats import (  # noqa: E402
    SIGNAL_HI,
    SIGNAL_LO,
    CellVerdict,
    bootstrap_se,
    classify_cell,
    grpo_dead_probability,
    grpo_signal_probability,
    minimum_detectable_effect,
    noise_floor,
    power_two_proportion,
    rollouts_for_dead,
    rollouts_for_effect,
    rule_of_three_upper,
    summarise_cells,
    wilson_interval,
)

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print("=" * 74)
    print("TEST — RSI statistics: no certainty from thin evidence")
    print("=" * 74)

    # ------------------------------------------------------------------
    print("\n1. GRPO dead-group probability  p^G + (1-p)^G")
    # ------------------------------------------------------------------
    # The number the old threshold got wrong: at p=0.05 and G=8 a cell is
    # dead 66.3% of the time, not 100%.
    d05 = grpo_dead_probability(0.05, 8)
    check("P(dead group) at p=0.05, G=8 is 0.663, not 1.0", close(d05, 0.6634, 1e-3), f"got {d05:.6f}")
    check(
        "so 33.7% of its groups still carry gradient",
        close(grpo_signal_probability(0.05, 8), 0.3366, 1e-3),
        f"got {grpo_signal_probability(0.05, 8):.6f}",
    )

    # Only the two endpoints are provably wasted. This is the assertion that
    # would have caught the docstring claim.
    interior = [p / 100 for p in range(1, 100)]
    worst = min(grpo_signal_probability(p, 8) for p in interior)
    check(
        "signal is strictly positive for every 0 < p < 1",
        worst > 0.0,
        f"min signal over p in (0,1) was {worst:.2e}",
    )
    check("p=0 is provably dead", close(grpo_dead_probability(0.0, 8), 1.0))
    check("p=1 is provably dead", close(grpo_dead_probability(1.0, 8), 1.0))
    check("p=0.5 maximises signal", close(grpo_signal_probability(0.5, 8), 1.0 - 2 * 0.5**8))
    # Symmetry: p and 1-p are equally (un)informative.
    check("symmetric in p <-> 1-p", close(grpo_dead_probability(0.3, 8), grpo_dead_probability(0.7, 8)))

    # ------------------------------------------------------------------
    print("\n2. Wilson interval: non-degenerate where Wald collapses")
    # ------------------------------------------------------------------
    lo0, hi0 = wilson_interval(0, 8)
    check("0/8 has a non-zero upper bound", hi0 > 0.0, f"got {hi0!r}")
    # Wilson and the rule of three are two different intervals and do not have
    # to agree exactly. Wilson centres on a shrunk point estimate
    # (z^2/2)/(n+z^2) = 0.1622 at n=8, which pushes its upper bound to 0.324;
    # the rule of three conditions on the observation and gives 0.312. Both
    # are valid, so the assertion is agreement to a few points, not equality.
    #
    # The 0.1622 is worth a comment of its own: this comment previously said
    # 0.0545, which is the shrinkage for the *lower* limit of a Wilson interval
    # and not the centre. Nothing computed it, so it survived; it is now pinned
    # below rather than restated.
    check("0/8 upper bound is near the rule of three (0.312)", close(hi0, 0.312, 2e-2), f"got {hi0:.6f}")
    check(
        "the Wilson centre is the shrinkage (z^2/2)/(n+z^2) = 0.1622, not 0",
        close((lo0 + hi0) / 2.0, 0.1622, 1e-3),
        f"got {(lo0 + hi0) / 2.0:.4f}",
    )
    check(
        "Wilson is the more conservative of the two here",
        hi0 > rule_of_three_upper(8),
        f"{hi0:.4f} vs {rule_of_three_upper(8):.4f}",
    )
    check("0/8 lower bound is 0", close(lo0, 0.0))
    # Wald would return [0, 0] here: p_hat=0 makes the sqrt term vanish. That
    # is a claim of exact certainty from 8 samples, which is the error.
    wald_half = 1.96 * math.sqrt(0.0 * 1.0 / 8)
    check("Wald would have said [0, 0] — the false certainty", close(wald_half, 0.0))
    check("Wilson is strictly wider than Wald here", (hi0 - lo0) > 2 * wald_half)

    for passes, n in [(0, 8), (1, 8), (4, 8), (8, 8), (0, 32), (1, 32), (32, 32), (3, 100)]:
        lo, hi = wilson_interval(passes, n)
        ok = 0.0 <= lo <= hi <= 1.0
        check(f"interval for {passes}/{n} stays inside [0,1]", ok, f"got [{lo:.4f}, {hi:.4f}]")

    check("n=0 degenerates to the whole range", wilson_interval(0, 0) == (0.0, 1.0))
    # More data must mean a narrower interval, or the scan is pointless.
    widths = [wilson_interval(1, n)[1] - wilson_interval(1, n)[0] for n in (8, 32, 128, 512)]
    check(
        "interval narrows monotonically with n",
        all(a > b for a, b in zip(widths, widths[1:], strict=False)),
        f"{widths}",
    )
    # An all-pass cell is dead too: zero variance at the top end. But 8/8 only
    # reaches lo=0.676, which does NOT clear SIGNAL_HI=0.95 — you need far more
    # data to prove a cell is saturated. This is the same lesson as 0/8.
    loP, hiP = wilson_interval(8, 8)
    check("8/8 does not clear the top threshold", loP < SIGNAL_HI, f"lo={loP:.4f}")
    check("8/8 is therefore not provably dead either", classify_cell(8, 8).verdict is CellVerdict.UNDER_MEASURED)
    check(
        "8/8 still carries 65% signal at its lower end",
        grpo_signal_probability(loP, 8) > 0.6,
        f"{grpo_signal_probability(loP, 8):.3f}",
    )

    # ------------------------------------------------------------------
    print("\n3. rule_of_three_upper: the base is (1 - confidence)")
    # ------------------------------------------------------------------
    r8 = rule_of_three_upper(8)
    check("exact upper bound at n=8 is 0.312", close(r8, 0.31234, 1e-4), f"got {r8:.6f}")
    # The slip this guards: using `confidence` as the base gives 1 - 0.95^0.125
    # = 0.0064, which is a lower tail probability wearing an upper bound's
    # name. It looks like a plausible float, which is why it needs a test.
    wrong = 1.0 - 0.95 ** (1.0 / 8)
    check("the wrong base would give 0.0064 — 49x too small", close(wrong, 0.00639, 1e-4), f"{wrong:.6f}")
    check("the two differ by more than an order of magnitude", r8 / wrong > 40, f"ratio {r8 / wrong:.1f}")
    check("exact bound is below the 3/n approximation", r8 < 3 / 8, f"{r8:.4f} vs {0.375:.4f}")
    check("the two agree to within 25% at n=8", abs(r8 - 3 / 8) / r8 < 0.25, f"{(3 / 8 - r8) / r8:.1%}")
    check("bound shrinks with n", rule_of_three_upper(32) < r8)
    check("n=0 is uninformative, not zero", rule_of_three_upper(0) == 1.0)
    # A larger confidence must widen the bound.
    check("higher confidence widens the bound", rule_of_three_upper(8, 0.99) > r8)
    # Consistency with Wilson: the two should agree closely for 0/n.
    check("agrees with Wilson upper bound at 0/8", abs(rule_of_three_upper(8) - wilson_interval(0, 8)[1]) < 0.02)

    # ------------------------------------------------------------------
    print("\n4. classify_cell: three verdicts, decided on the interval")
    # ------------------------------------------------------------------
    # The headline: 0/8 must NOT be called dead. This is the exact row the old
    # filter dropped, and the exact row it was wrong about.
    c = classify_cell(0, 8, harness="bash_minimal", task_id="t1-04")
    check("0/8 is not called DEAD", c.verdict is not CellVerdict.DEAD, f"got {c.verdict}")
    check("0/8 is called UNDER_MEASURED", c.verdict is CellVerdict.UNDER_MEASURED, f"got {c.verdict}")
    check("0/8 is flagged as needing more data", c.needs_more_data)
    check("0/8 records its interval", close(c.lo, 0.0) and close(c.hi, 0.3244, 2e-3), f"[{c.lo:.4f}, {c.hi:.4f}]")

    # The narrow-but-straddling trap: 0/32 has a 0.107-wide interval, which a
    # width-based shortcut would call LIVE even though the point estimate is
    # in the dead zone and the top of the interval is 53% signal.
    c32 = classify_cell(0, 32)
    check("0/32 is still UNDER_MEASURED, not LIVE", c32.verdict is CellVerdict.UNDER_MEASURED, f"got {c32.verdict}")
    check("0/32 interval is narrower than 0/8", (c32.hi - c32.lo) < (hi0 - lo0))
    check(
        "0/32 would have been 53% signal at its upper end",
        grpo_signal_probability(c32.hi, 8) > 0.5,
        f"{grpo_signal_probability(c32.hi, 8):.3f}",
    )

    # Death requires enough data to exclude signal outright.
    cdead = classify_cell(0, 128)
    check("0/128 IS called DEAD", cdead.verdict is CellVerdict.DEAD, f"got {cdead.verdict} (hi={cdead.hi:.4f})")
    check("0/128 is not flagged for more data", not cdead.needs_more_data)
    check("0/128 upper bound is below the signal floor", cdead.hi < SIGNAL_LO, f"hi={cdead.hi:.4f}")

    # The boundary itself, because the docstring used to claim "roughly n >= 128"
    # and nothing checked it. 73 is the first n whose Wilson upper bound at zero
    # passes falls below SIGNAL_LO; 72 misses by 0.0006. The gap is not cosmetic:
    # 73 rollouts is a laptop-sized scan, 128 is a redesign.
    check("0/73 IS called DEAD — the true boundary", classify_cell(0, 73).verdict is CellVerdict.DEAD)
    check("0/72 is NOT yet DEAD", classify_cell(0, 72).verdict is CellVerdict.UNDER_MEASURED)
    check(
        "the boundary is where the upper bound crosses SIGNAL_LO",
        wilson_interval(0, 73)[1] < SIGNAL_LO < wilson_interval(0, 72)[1],
        f"72->{wilson_interval(0, 72)[1]:.5f} 73->{wilson_interval(0, 73)[1]:.5f}",
    )
    # One observed pass needs more data than zero, because 1/n must clear the
    # threshold while the interval still has an upper tail above it.
    check("1/110 IS called DEAD", classify_cell(1, 110).verdict is CellVerdict.DEAD)
    check("1/109 is NOT yet DEAD", classify_cell(1, 109).verdict is CellVerdict.UNDER_MEASURED)

    # An all-pass cell is equally dead, at the other end.
    ctop = classify_cell(128, 128)
    check("128/128 IS called DEAD", ctop.verdict is CellVerdict.DEAD, f"got {ctop.verdict} (lo={ctop.lo:.4f})")

    # A genuinely live cell.
    clive = classify_cell(11, 32)
    check("11/32 IS called LIVE", clive.verdict is CellVerdict.LIVE, f"got {clive.verdict}")
    check("11/32 interval sits inside the band", clive.lo >= SIGNAL_LO and clive.hi <= SIGNAL_HI)
    check("11/32 carries substantial signal", clive.signal_prob > 0.95, f"{clive.signal_prob:.4f}")

    # Near the edges: p=0.5 with plenty of data is live; the same rate with
    # almost no data is under-measured, because containment in a 0.90-wide
    # band is vacuous when the interval is nearly that wide.
    check("16/32 LIVE", classify_cell(16, 32).verdict is CellVerdict.LIVE)
    check("1/2 UNDER_MEASURED", classify_cell(1, 2).verdict is CellVerdict.UNDER_MEASURED)
    check("2/4 UNDER_MEASURED", classify_cell(2, 4).verdict is CellVerdict.UNDER_MEASURED)
    # 1/2's interval is [0.09, 0.91] — contained in [0.05, 0.95], yet 0.82
    # wide. Containment alone would have called it live; the width cap is what
    # stops two rollouts from being reported as a settled result.
    c12 = classify_cell(1, 2)
    check(
        "1/2 really is contained in the band",
        c12.lo >= SIGNAL_LO and c12.hi <= SIGNAL_HI,
        f"[{c12.lo:.3f}, {c12.hi:.3f}]",
    )
    check("but it is 0.82 wide — wider than half the band", (c12.hi - c12.lo) > (SIGNAL_HI - SIGNAL_LO) / 2)
    check("10/20 LIVE (n=20 is plenty at p=0.5)", classify_cell(10, 20).verdict is CellVerdict.LIVE)
    # The measured boundary of the width cap: 8/16 clears it, 7/15 does not.
    check(
        "8/16 LIVE — the first n that clears the cap",
        classify_cell(8, 16).verdict is CellVerdict.LIVE,
        f"{classify_cell(8, 16).verdict}",
    )
    check(
        "7/15 UNDER_MEASURED — just short",
        classify_cell(7, 15).verdict is CellVerdict.UNDER_MEASURED,
        f"{classify_cell(7, 15).verdict}",
    )
    check("max_width is overridable", classify_cell(1, 2, max_width=0.99).verdict is CellVerdict.LIVE)

    # Input validation.
    for bad, label in [((0, 0), "n=0"), ((9, 8), "passes > n")]:
        try:
            classify_cell(*bad)
            check(f"rejects {label}", False, "no exception raised")
        except ValueError:
            check(f"rejects {label}", True)

    # The verdict must be derivable from the interval alone — no hidden state.
    band = SIGNAL_HI - SIGNAL_LO
    for passes, n in [(0, 8), (3, 8), (11, 32), (0, 128), (128, 128), (5, 16), (1, 2), (10, 20)]:
        cs = classify_cell(passes, n)
        expected = (
            CellVerdict.DEAD
            if (cs.hi < SIGNAL_LO or cs.lo > SIGNAL_HI)
            else (
                CellVerdict.LIVE
                if (cs.lo >= SIGNAL_LO and cs.hi <= SIGNAL_HI and (cs.hi - cs.lo) <= band / 2)
                else CellVerdict.UNDER_MEASURED
            )
        )
        check(f"{passes}/{n} verdict follows from its interval", cs.verdict is expected, f"{cs.verdict} vs {expected}")

    # ------------------------------------------------------------------
    print("\n5. summarise_cells: naming what the old filter discarded")
    # ------------------------------------------------------------------
    cells = [classify_cell(*pn) for pn in [(0, 8), (0, 32), (11, 32), (16, 32), (0, 128), (128, 128), (1, 8)]]
    s = summarise_cells(cells)
    check("total counts every cell", s["total"] == 7, f"{s}")
    check("live count", s["live"] == 2, f"{s}")
    check("dead count", s["dead"] == 2, f"{s}")
    check("under-measured count", s["under_measured"] == 3, f"{s}")
    check("verdicts partition the set", s["live"] + s["dead"] + s["under_measured"] == s["total"])
    check("empty input is handled", summarise_cells([])["total"] == 0)
    check("mean signal probability is in [0,1]", 0.0 <= s["mean_signal_prob"] <= 1.0, f"{s['mean_signal_prob']}")

    # ------------------------------------------------------------------
    print("\n6. bootstrap_se and noise_floor: how big is noise")
    # ------------------------------------------------------------------
    check("no rewards means no error", bootstrap_se([]) == 0.0)
    check("a constant cell has zero error", close(bootstrap_se([1.0] * 16), 0.0, 1e-12))
    # se of the mean for a Bernoulli(p) with n draws is sqrt(p(1-p)/n); the
    # bootstrap should land near it.
    rewards = [1.0] * 5 + [0.0] * 11  # p = 0.3125, n = 16
    se = bootstrap_se(rewards, reps=4000, seed=7)
    analytic = math.sqrt(0.3125 * 0.6875 / 16)
    check("bootstrap se tracks the analytic se", abs(se - analytic) / analytic < 0.15, f"{se:.4f} vs {analytic:.4f}")
    check("se is reproducible under a fixed seed", close(se, bootstrap_se(rewards, reps=4000, seed=7)))
    check("a different seed gives a nearby answer", abs(se - bootstrap_se(rewards, reps=4000, seed=8)) < 0.02)

    # The noise floor must be a real barrier: a gain smaller than it is not a
    # gain. Two identical arms must be inside the band.
    arm = [[1.0] * 5 + [0.0] * 11 for _ in range(4)]
    floor = noise_floor(arm)
    check("floor is positive for a mixed cell", floor > 0.0, f"{floor}")
    # The floor pools cells, so it must be sized against the standard error of
    # the *pooled* sample, not of one cell. Four copies of a 16-rollout cell
    # give 64 rollouts, and se falls as 1/sqrt(n). The comparison must use the
    # same number of bootstrap reps, or it compares two Monte Carlo estimates
    # rather than the formula — 2000 vs 4000 reps differ by about 2% here.
    pooled = [r for cell in arm for r in cell]
    se_pooled = bootstrap_se(pooled, reps=2000, seed=7)
    check("pooled se is smaller than one cell's se", se_pooled < se, f"{se_pooled:.4f} vs {se:.4f}")
    check(
        "floor is exactly z*sqrt(2) times the pooled se",
        close(floor, 2.0 * math.sqrt(2.0) * se_pooled, 1e-9),
        f"{floor:.6f} vs {2.0 * math.sqrt(2.0) * se_pooled:.6f}",
    )
    check("floor is at least 2x the pooled se", floor > 2.0 * se_pooled, f"{floor:.4f} vs {se_pooled:.4f}")
    check("floor grows with z", noise_floor(arm, z=4.0) > floor)
    check("floor is empty for no data", noise_floor([]) == 0.0)
    # A cleaner cell has a smaller floor.
    clean = [[1.0] * 16 for _ in range(4)]
    check("a deterministic cell has a zero floor", close(noise_floor(clean), 0.0, 1e-12))

    # ------------------------------------------------------------------
    print("\npower — the numbers that decide whether a null is a finding")
    # ------------------------------------------------------------------
    # These are quoted in the README and the CHANGELOG as the reason the
    # ablation settles nothing: an MDE far larger than the observed effect.
    # Nothing computed them until now, so the published pair could not be
    # checked against code at all. Pinned against the measured ablation, with
    # p_bar pooled over the two arms being compared (single, multi).
    p_bar = (34 / 128 + 39 / 128) / 2
    check(
        "the ablation's pooled rate is what the docs assume",
        close(p_bar, 0.285156, 1e-6),
        f"{p_bar}",
    )
    mde = minimum_detectable_effect(128, p_bar)
    check(
        "MDE at n=128/arm is 0.1581, as the README states",
        close(mde, 0.1581, 5e-5),
        f"{mde:.6f} — the README previously said 0.1572, which no definition produced",
    )
    check(
        "the observed effect is about a quarter of the MDE",
        close(0.0391 / mde, 0.247, 0.005),
        f"{0.0391 / mde:.4f}",
    )
    check(
        "the run had ~10% power, which is why it could not settle the question",
        close(power_two_proportion(0.0391, 128, p_bar), 0.103, 0.005),
        f"{power_two_proportion(0.0391, 128, p_bar):.4f}",
    )
    need = rollouts_for_effect(0.0391, p_bar)
    check(
        "80% power at the observed effect needs ~2,094 rollouts/arm, 16x the run",
        abs(need - 2094) <= 3,
        f"{need} per arm = {need / 128:.1f}x",
    )
    # The closed form and the bisection must agree, or one of them is wrong.
    closed = 1.959963985 + 0.8416212336
    closed *= math.sqrt(2.0 * p_bar * (1.0 - p_bar) / 128)
    check(
        "bisection agrees with the closed form at this sample size",
        close(mde, closed, 5e-4),
        f"bisection {mde:.6f} vs closed form {closed:.6f}",
    )
    check("power rises with n", power_two_proportion(0.0391, 512, p_bar) > power_two_proportion(0.0391, 128, p_bar))
    check(
        "power rises with the effect",
        power_two_proportion(0.10, 128, p_bar) > power_two_proportion(0.0391, 128, p_bar),
    )
    check("a degenerate rate has no defined MDE", math.isnan(minimum_detectable_effect(128, 0.0)))

    # ------------------------------------------------------------------
    # The DEAD boundary, asked for rather than restated.
    #
    # `classify_cell`'s docstring says zero passes needs n >= 73 and one pass
    # needs n >= 110. Those were prose for a while, and prose is where the
    # wrong "roughly n >= 128" lived. `rollouts_for_dead` exists so the number
    # can be asked for; these assertions keep it equal to the boundary the
    # verdict function actually applies, so the two cannot drift apart.
    # ------------------------------------------------------------------
    print("\n-- rollouts_for_dead: the boundary, computed rather than quoted --")
    check("zero passes needs n >= 73", rollouts_for_dead(0) == 73, str(rollouts_for_dead(0)))
    check("one pass needs n >= 110", rollouts_for_dead(1) == 110, str(rollouts_for_dead(1)))
    check(
        "n=73 is the first zero-pass cell the verdict calls DEAD",
        classify_cell(0, 73).verdict is CellVerdict.DEAD
        and classify_cell(0, 72).verdict is not CellVerdict.DEAD,
    )
    check(
        "the helper agrees with classify_cell at its own boundary",
        rollouts_for_dead(0) == 73 and wilson_interval(0, 73)[1] < SIGNAL_LO,
        f"wilson(0,73) hi={wilson_interval(0, 73)[1]:.5f}",
    )
    check(
        "the boundary is monotone in the observed passes",
        rollouts_for_dead(0) < rollouts_for_dead(1) < rollouts_for_dead(2),
    )
    try:
        rollouts_for_dead(-1)
        _neg_rejected = False
    except ValueError:
        _neg_rejected = True
    check("a negative pass count is rejected", _neg_rejected)

    # The held-out term of the shipped ablation, pooled over its three arms:
    # 0/96. This is the number that makes `gap == mean(train)` an identity, and
    # it is DEAD rather than merely under-measured -- a positive finding that
    # the harness sits outside the learnable band, not a call for more data.
    print("\n-- the shipped held-out term: 0/96 pooled is DEAD, not 'needs more data' --")
    _held = classify_cell(0, 96, harness="codex_style", task_id="(pooled over arms)")
    check("0/96 pools to a DEAD verdict", _held.verdict is CellVerdict.DEAD, _held.verdict)
    check("0/96 upper bound clears the floor", _held.hi < SIGNAL_LO, f"{_held.hi:.4f}")
    check("0/96 upper bound is 0.0385", abs(_held.hi - 0.0385) < 5e-5, f"{_held.hi:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — the statistics layer refuses to manufacture certainty")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
