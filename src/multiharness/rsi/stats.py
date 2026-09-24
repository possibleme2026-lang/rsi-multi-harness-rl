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

"""Statistics for deciding which cells can teach the policy anything.

The problem this module exists to fix
-------------------------------------
GRPO gives a group of ``G`` completions a gradient only when their rewards
disagree. With a binary reward and pass probability ``p``, a group carries no
signal with probability

    P(dead group) = p^G + (1 - p)^G

At ``G = 8`` this is 0.663 at ``p = 0.05`` — so a cell at ``p = 0.05`` still
produces a usable gradient in roughly **a third of its groups**. The pipeline
used to assert the opposite (``train.py``: *"cells with p <= 0.05 or
p >= 0.95 are provably wasted"*) and dropped every such cell. Only ``p = 0``
and ``p = 1`` are provably wasted; everything strictly between them carries
some signal.

The second half of the problem is measurement error. A cell's pass rate is
estimated from ``n`` rollouts, and an observed ``0/8`` has a 95% upper
confidence bound of **0.375**. A cell whose true rate is 0.30 shows ``0/8``
about 5.8% of the time, so "we measured zero" is not evidence that the cell is
dead — it is evidence that we measured too little. The old two-way split
(keep / drop) silently converted that ambiguity into "drop", which biases the
training set towards tasks the model already finds easy and makes the suite
look easier than it is.

What replaces it
----------------
:func:`classify_cell` returns one of three verdicts:

``DEAD``
    The confidence interval sits entirely below the signal floor. Provably
    uninformative, and safe to drop.
``LIVE``
    The interval overlaps the region where groups can disagree.
``UNDER_MEASURED``
    The interval is too wide to tell. **These are kept and flagged**, never
    dropped: an under-measured cell is a reason to scan more, not a licence to
    discard data.

Every function here is pure and standard-library only, so the core CI job
exercises them without installing anything.

Adapted from two ideas in ``google-research/rrsi``: the noise-adjusted floor
(``rrsi/selection.py``, Algorithm 2) and the bootstrap standard error used to
calibrate it (``rrsi/calibrate.py``). The mapping is not one-to-one — RRSI
calibrates a band for comparing two *harness* scores, while this calibrates a
band for classifying a single *cell* — so the code is written fresh against
this repository's data shapes rather than copied.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "SIGNAL_LO",
    "SIGNAL_HI",
    "CellVerdict",
    "CellStats",
    "wilson_interval",
    "rule_of_three_upper",
    "grpo_signal_probability",
    "grpo_dead_probability",
    "bootstrap_se",
    "noise_floor",
    "classify_cell",
    "summarise_cells",
]

#: Below this pass rate a cell contributes almost nothing to a group.
SIGNAL_LO = 0.05
#: Above this pass rate likewise. Symmetric by construction.
SIGNAL_HI = 1.0 - SIGNAL_LO

#: Default group size, matching ``train.py --num-generations``.
DEFAULT_G = 8


class CellVerdict(StrEnum):
    """What a measured (harness, task) cell tells us about learnability."""

    LIVE = "live"
    DEAD = "dead"
    UNDER_MEASURED = "under_measured"


@dataclass(frozen=True)
class CellStats:
    """One cell's measurement and the verdict drawn from it."""

    harness: str
    task_id: str
    passes: int
    n: int
    #: Point estimate, ``passes / n``.
    p_hat: float
    #: 95% Wilson interval. Preferred over normal-approximation Wald because
    #: it stays inside [0, 1] and remains meaningful at ``passes = 0``, which
    #: is exactly the case this module has to reason about.
    lo: float
    hi: float
    verdict: CellVerdict
    #: Probability that a group of ``G`` carries a gradient at ``p_hat``.
    signal_prob: float
    #: True when the interval is wider than the region that matters, so the
    #: verdict is a statement about the measurement rather than the cell.
    needs_more_data: bool

    def as_dict(self) -> dict:
        return {
            "harness": self.harness,
            "task_id": self.task_id,
            "passes": self.passes,
            "n": self.n,
            "p_hat": round(self.p_hat, 6),
            "ci95": [round(self.lo, 6), round(self.hi, 6)],
            "verdict": str(self.verdict),
            "signal_prob": round(self.signal_prob, 6),
            "needs_more_data": self.needs_more_data,
        }


def _z_for(confidence: float) -> float:
    """Two-sided z for a confidence level, without scipy.

    Only the levels this project uses are tabulated; anything else falls back
    to the normal approximation of the inverse CDF, which is accurate enough
    for a confidence band whose whole purpose is to be conservative.
    """
    table = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    if confidence in table:
        return table[confidence]
    # Acklam-style rational approximation of the inverse normal CDF.
    p = 1.0 - (1.0 - confidence) / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wilson_interval(passes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the textbook Wald interval (``p_hat +/- z*sqrt(p(1-p)/n)``)
    because Wald degenerates at the boundaries: at ``passes = 0`` it returns
    ``[0, 0]``, claiming certainty that the rate is exactly zero from a finite
    sample. That is precisely the false confidence this module exists to
    remove, so the estimator cannot be the one that manufactures it.
    """
    if n <= 0:
        return (0.0, 1.0)
    z = _z_for(confidence)
    p = passes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three_upper(n: int, confidence: float = 0.95) -> float:
    """Upper bound on a rate observed zero times in ``n`` trials.

    Solve ``(1 - p)**n = 1 - confidence`` for ``p``: if the true rate were
    ``p``, seeing zero successes in ``n`` trials has probability ``(1-p)**n``,
    so the largest ``p`` still consistent with the observation is

        ``1 - (1 - confidence)**(1/n)``

    The familiar ``3/n`` is its large-``n`` approximation, and at the sample
    sizes this project runs (``n = 8..32``) the approximation is off by enough
    to change a verdict — at ``n = 8`` it gives 0.375 against the exact 0.312.

    Note the base is ``1 - confidence``, not ``confidence``. Writing
    ``1 - confidence**(1/n)`` yields 0.0064 at ``n = 8``, which is the *lower*
    tail probability masquerading as an upper bound on ``p`` and is 49x too
    small; the assertion at ``n = 8`` in ``tests/test_rsi_stats.py`` exists
    because that slip is easy to make and produces a plausible-looking float.
    """
    if n <= 0:
        return 1.0
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def grpo_dead_probability(p: float, g: int = DEFAULT_G) -> float:
    """``p^G + (1-p)^G``: chance a group of ``G`` has zero reward variance.

    With a binary reward, zero variance means every advantage in the group is
    exactly zero, so the group contributes nothing to the gradient.
    """
    p = min(1.0, max(0.0, float(p)))
    return p**g + (1.0 - p) ** g


def grpo_signal_probability(p: float, g: int = DEFAULT_G) -> float:
    """``1 - P(dead group)``: chance a group of ``G`` carries any gradient."""
    return 1.0 - grpo_dead_probability(p, g)


def bootstrap_se(
    rewards: list[float],
    *,
    reps: int = 2000,
    seed: int = 7,
    statistic=None,
) -> float:
    """Bootstrap standard error of the mean over one cell's rollouts.

    Resamples the observed rewards with replacement ``reps`` times and returns
    the population standard deviation of the resampled means. Used to size the
    noise floor: comparing two arms that differ by less than this is comparing
    two draws from the same distribution.

    ``seed`` is fixed so a figure regenerated from the same artifact is
    byte-identical, which is what lets CI assert on the output.
    """
    if not rewards:
        return 0.0
    rng = random.Random(seed)
    n = len(rewards)
    fn = statistic or (lambda xs: sum(xs) / len(xs))
    means = []
    for _ in range(reps):
        means.append(fn([rewards[rng.randrange(n)] for _ in range(n)]))
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / len(means)
    return math.sqrt(var)


def noise_floor(
    rewards_per_cell: list[list[float]],
    *,
    z: float = 2.0,
    reps: int = 2000,
    seed: int = 7,
) -> float:
    """Smallest score difference that is not explained by sampling noise.

    Follows ``rrsi/calibrate.py``: the standard error of a *difference* between
    two independent evaluations of the same thing is ``sqrt(2) * se``, and the
    band is ``z`` standard deviations of that. An unchanged arm clears the band
    about 95% of the time at ``z = 2``, so a gain smaller than the band is not
    evidence of a gain.

    Cells are pooled rather than averaged so that a cell with more rollouts
    carries proportionally more weight — the same convention the evaluation
    sweep uses.
    """
    pooled: list[float] = []
    for cell in rewards_per_cell:
        pooled.extend(cell)
    if not pooled:
        return 0.0
    se = bootstrap_se(pooled, reps=reps, seed=seed)
    # sqrt(2) for the difference of two independent estimates.
    return z * math.sqrt(2.0) * se


def classify_cell(
    passes: int,
    n: int,
    *,
    harness: str = "",
    task_id: str = "",
    g: int = DEFAULT_G,
    lo_threshold: float = SIGNAL_LO,
    hi_threshold: float = SIGNAL_HI,
    confidence: float = 0.95,
    max_width: float | None = None,
) -> CellStats:
    """Decide whether a measured cell can teach the policy anything.

    The decision is made on the **confidence interval**, not the point
    estimate, and it needs two independent conditions to call a cell live —
    containment *and* precision. Either one alone is fooled:

    ``DEAD``
        The interval excludes the signal region entirely, so no amount of
        training on this cell can move a group's reward variance.
    ``LIVE``
        The interval is contained in the signal region **and** is narrow
        enough that the containment is informative.
    ``UNDER_MEASURED``
        Anything else.

    Why containment alone is not enough: the band ``[0.05, 0.95]`` is 0.90
    wide, so ``1/2`` — an interval of ``[0.09, 0.91]`` from two rollouts —
    is *contained* in it. Two samples cannot localise anything, and calling
    that cell live would be the same error as the old ``0/8 -> dead`` in the
    opposite direction. So containment is paired with a width cap.

    Why the cap is half the band: the interval must be narrower than the
    region being reasoned about, or the measurement cannot distinguish points
    inside that region from each other. The default is therefore
    ``(SIGNAL_HI - SIGNAL_LO) / 2 = 0.45``. It is a real cost — at ``p = 0.5``
    it takes ``n >= 16`` to call a cell live (``8/16`` is the first that
    clears it; ``7/15`` does not) — and it is stated here rather than hidden so
    the price of an honest verdict is visible.

    ``DEAD`` is harder still: observing zero passes and still excluding the
    signal region takes ``n >= 73`` — that is the first ``n`` whose Wilson
    upper bound at ``passes = 0`` drops below ``SIGNAL_LO``
    (``wilson_interval(0, 73)[1] = 0.04999``; ``n = 72`` gives 0.0506). With a
    single pass observed it takes ``n >= 110``, because ``1/n`` has to fall
    below the threshold with room for the interval's upper tail.
    ``tests/test_rsi_stats.py`` pins both boundaries.

    An earlier version of this docstring said "roughly ``n >= 128``". That was
    a guess that survived because no test checked it; the true boundary is 73,
    and the gap matters here — 73 is reachable on a laptop, 128 is where you
    stop and redesign instead.
    """
    if n <= 0:
        raise ValueError("classify_cell needs at least one rollout")
    if not 0 <= passes <= n:
        raise ValueError(f"passes={passes} is impossible out of n={n}")

    if max_width is None:
        max_width = (hi_threshold - lo_threshold) / 2.0

    p_hat = passes / n
    lo, hi = wilson_interval(passes, n, confidence)
    signal_prob = grpo_signal_probability(p_hat, g)

    # A cell is dead only when the whole interval sits in the region where a
    # group can essentially never disagree. Note this is about `signal`, not
    # about `p`: `p >= 0.95` is just as useless as `p <= 0.05`, because a
    # group of all-pass completions has zero variance too.
    if hi < lo_threshold or lo > hi_threshold:
        verdict = CellVerdict.DEAD
        needs_more = False
    # Live needs the interval inside the band *and* tight enough that being
    # inside it means something.
    elif lo >= lo_threshold and hi <= hi_threshold and (hi - lo) <= max_width:
        verdict = CellVerdict.LIVE
        needs_more = False
    else:
        # Either the interval covers a threshold, or it is so wide that
        # containment is vacuous. Both are "we do not know yet".
        verdict = CellVerdict.UNDER_MEASURED
        needs_more = True

    return CellStats(
        harness=harness,
        task_id=task_id,
        passes=passes,
        n=n,
        p_hat=p_hat,
        lo=lo,
        hi=hi,
        verdict=verdict,
        signal_prob=signal_prob,
        needs_more_data=needs_more,
    )


def summarise_cells(cells: list[CellStats]) -> dict:
    """Aggregate verdicts, and say how many cells a bigger scan would settle.

    ``resolvable_by_rescan`` counts the ``UNDER_MEASURED`` cells whose
    interval is still wide — those are the ones more rollouts would decide. A
    cell can be under-measured and *not* resolvable only if its point estimate
    sits outside the useful band, in which case more data would move it to
    ``DEAD``, which is also a resolution; the count therefore tracks all of
    them, and the caller should read it as "more data changes this verdict".
    """
    by = {v: 0 for v in CellVerdict}
    for c in cells:
        by[c.verdict] += 1
    under = [c for c in cells if c.verdict is CellVerdict.UNDER_MEASURED]
    return {
        "total": len(cells),
        "live": by[CellVerdict.LIVE],
        "dead": by[CellVerdict.DEAD],
        "under_measured": by[CellVerdict.UNDER_MEASURED],
        "resolvable_by_rescan": len(under),
        # The headline the old code got wrong: how many cells were being
        # called dead on evidence that did not support it.
        "dropped_on_thin_evidence": sum(
            1 for c in under if c.p_hat <= SIGNAL_LO or c.p_hat >= SIGNAL_HI
        ),
        "mean_signal_prob": (sum(c.signal_prob for c in cells) / len(cells)) if cells else 0.0,
    }
