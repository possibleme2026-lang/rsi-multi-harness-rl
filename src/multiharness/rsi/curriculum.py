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

"""Closing the curriculum loop: turning a band into the next batch of tasks.

Why this module exists
----------------------
``band.py`` says where a cell sits on the difficulty curve and
``task_gen.reparameterise`` can move a task along the parameter axes. Both
existed, both were tested, and **neither was ever called by a pipeline script.**
``band.steer``'s docstring claimed "the task generator consumes" it, and the
README claimed "``steer`` in ``rsi/band.py`` already returns the override" —
while a grep for the symbol found only its own definition and its tests.

That is the same failure this repository keeps finding in itself: the honest
arm is the arm that was never executed. A steering rule that no script invokes
is not a curriculum, it is a comment with a return type. This module is the
missing call site, plus the measurement that decides when to call it.

The α-Curriculum Reward, and why α = 0.5 is derived rather than borrowed
-----------------------------------------------------------------------
The difficulty-alignment signal follows GenEnv (arXiv:2512.19682):

    R_env(p̂) = exp(−β (p̂ − α)²)

a bell centred on a target success rate ``α``, so the environment is rewarded
for producing tasks the policy solves about ``α`` of the time. GenEnv uses
``α = 0.5`` and justifies it via the identity ``p(1−p) = 1/4 − (p − 1/2)²``.

**This repository does not need that identity, and taking it on faith would be
the wrong move.** The quantity this codebase already reasons about is not the
variance of a Bernoulli but :func:`rsi.stats.grpo_signal_probability` — the
probability that a group of ``G`` carries *any* gradient at all, which is
``1 − p^G − (1−p)^G``. That is the thing that actually determines whether a
cell teaches anything under GRPO, and it is already measured here at ``G = 8``.

So ``α`` is **solved for**: :func:`optimal_alpha` maximises the GRPO signal
probability over ``p`` and returns the argmax. The answer comes out at 0.5, and
:func:`alpha_is_derived` asserts it — but the derivation is the point. It means
the constant is tied to *this* repository's group size and reward shape, and
would move if either did, instead of being a number copied from a paper that
uses a different optimizer.

At ``G = 8`` the signal curve is strikingly flat near the top: ``p = 0.5`` gives
0.9922, and even ``p = 0.3`` gives 0.9424. That flatness is the honest
counter-argument to over-tuning difficulty, and it is why the difficulty filter
below is loose rather than tight.

The difficulty filter, and a design bug worth recording
------------------------------------------------------
GenEnv excludes a batch from the environment update when ``|p̂ − α| > k_min``,
with ``k_min = 0.1``. The first version of this module applied that rule
**per task**, and it silently disabled the entire curriculum: ``mastered`` is
``p > 0.9`` and ``out_of_reach`` is ``p < 0.1``, so every cell that ``steer``
would act on sits at least 0.4 away from ``α = 0.5`` — far outside a 0.1 band.
The filter rejected exactly the cells the steering rule existed to move, and the
first run reported ``moves: 0`` on a scan that had 16 cells.

The mistake was applying a *batch-level* rule per task. In GenEnv the reward and
the filter both score **one batch's aggregate success rate**, because the thing
being trained is an environment policy that emits batches. There is no
environment policy here — the "environment" is a parameter vector and the
curriculum is a rule — so the batch-level quantity has no per-task analogue.

The rule is therefore split by scope, which is where it belonged:

* **Per task** — :func:`rsi.band.band_of` decides the band, and the band decides
  whether and which way to move. Nothing else filters.
* **Per batch** — :func:`batch_alignment` reports the mean α-reward, and
  :func:`batch_is_misaligned` fires GenEnv's ``k_min`` rule at the level it was
  written for: if the whole batch's mean pass rate is more than ``k_min`` from
  ``α``, the diagnosis is "regenerate the batch", not "nudge four tasks".

Both are reported on every plan. The per-task band is the action; the batch
alignment is the context that says whether the action is enough.

What this module does *not* do
------------------------------
It does not generate tasks itself, and it does not score anything. It reads the
scan artifact, decides which cells to move and in which direction, and returns
a plan. ``scripts/rsi_loop.py`` executes the plan through
:func:`rsi.task_gen.reparameterise` and validates the result through the same
four gates every other task goes through. Keeping the decision separate from
the generation is what makes the decision testable without a model.
"""

from __future__ import annotations

import math

from . import band as band_mod
from . import task_gen
from .stats import DEFAULT_G, grpo_signal_probability

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA",
    "DEFAULT_K_MIN",
    "alpha_reward",
    "optimal_alpha",
    "alpha_is_derived",
    "in_band",
    "difficulty_gap",
    "signal_at_alpha",
    "plan_regeneration",
    "execute_plan",
    "batch_alignment",
    "batch_is_misaligned",
    "CurriculumPlan",
]

#: Target success rate. Derived, not assumed — see :func:`optimal_alpha`.
DEFAULT_ALPHA = 0.5
#: Sharpness of the bell. GenEnv's default; kept because the reward's *value*
#: is only ever compared against itself, so the scale does not matter, only the
#: ordering it induces.
DEFAULT_BETA = 10.0
#: Exclude a batch from the decision when it is this far from ``α``.
DEFAULT_K_MIN = 0.1


def alpha_reward(p_hat: float, *, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> float:
    """``exp(−β (p̂ − α)²)`` — how well a batch's difficulty is aligned.

    The value is in ``(0, 1]`` by construction, peaking at 1 when
    ``p̂ = α``. No batch-level normalisation is applied, and that is deliberate:
    GenEnv's form is already bounded, and a min-max over the batch would make
    the reward depend on which *other* tasks happened to be sampled, so the same
    task would be rewarded differently in two runs that differ only in their
    batch composition. A difficulty signal has to be a property of the task, not
    of its neighbours.
    """
    p = min(1.0, max(0.0, float(p_hat)))
    return math.exp(-beta * (p - alpha) ** 2)


def optimal_alpha(g: int = DEFAULT_G, *, steps: int = 20001) -> float:
    """The ``p`` that maximises the GRPO group signal probability.

    Scanned rather than solved in closed form. ``1 − p^G − (1−p)^G`` is
    symmetric about 0.5 and unimodal on ``[0, 1]``, so a fine grid is exact to
    the grid spacing and needs no calculus; the closed form would be an
    exercise in implicit differentiation for a constant that is computed once.

    The returned value is rounded to the grid it was found on. With
    ``steps = 20001`` the spacing is ``5e-5``, so the result is reported as
    exactly ``0.5`` rather than as ``0.5000250000000001`` — a target difficulty
    that carries float noise would make every downstream comparison of two
    plans depend on the noise.
    """
    best_p, best_v = 0.5, -1.0
    for i in range(steps):
        p = i / (steps - 1)
        v = grpo_signal_probability(p, g)
        if v > best_v:
            best_p, best_v = p, v
    # Snap to the nearest 1e-3: the curve is flat enough at the top that
    # anything finer is measuring the grid, not the function.
    return round(best_p, 3)


def alpha_is_derived(g: int = DEFAULT_G, *, alpha: float = DEFAULT_ALPHA) -> bool:
    """Whether the module's ``α`` is the argmax of this repo's own signal curve.

    Exposed as a predicate so a test can assert the *relationship* rather than
    the number. If someone changes ``DEFAULT_G`` and forgets to re-derive
    ``α``, this returns False and the test fails — which is the failure mode
    worth catching, since a stale ``α`` would silently mis-steer the curriculum
    while every individual number still looked reasonable.
    """
    return abs(optimal_alpha(g) - alpha) < 1e-9


def in_band(p_hat: float, *, alpha: float = DEFAULT_ALPHA, k_min: float = DEFAULT_K_MIN) -> bool:
    """Whether a pass rate is close enough to the target to count as aligned.

    **Scope matters here.** This is a *batch-level* predicate — GenEnv's rule
    applies to the aggregate success rate of a batch, not to one task. It is
    deliberately not used to filter individual tasks in
    :func:`plan_regeneration`: ``mastered`` and ``out_of_reach`` are by
    definition far from ``α``, so filtering tasks by this would reject every
    cell the steering rule exists to move. See the module docstring for the run
    that caught this.

    The boundary is inclusive: ``|p̂ − α| == k_min`` is *inside*. GenEnv's text
    says "exclude when greater than", and making the edge exclusive would mean
    a batch at exactly the threshold is dropped, which is the more surprising
    of the two readings for no benefit.
    """
    return abs(min(1.0, max(0.0, float(p_hat))) - alpha) <= k_min


def difficulty_gap(p_hat: float, *, alpha: float = DEFAULT_ALPHA) -> float:
    """Signed distance from the target. Negative means too hard.

    Sign matters for reporting and is the reason this is not ``abs``: a run
    whose batches are uniformly too easy needs a different fix from one that is
    uniformly too hard, and an absolute gap cannot tell the two apart in a log.
    """
    return min(1.0, max(0.0, float(p_hat))) - alpha


def signal_at_alpha(*, alpha: float = DEFAULT_ALPHA, g: int = DEFAULT_G) -> float:
    """The GRPO group signal probability at the target difficulty.

    Reported alongside every plan so a reader can see how much signal the
    target is actually buying, rather than having to trust that 0.5 is good.
    """
    return grpo_signal_probability(alpha, g)


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


class CurriculumPlan:
    """A regeneration plan: which tasks to move, and which to leave alone.

    Serializable so it can be written next to the scan it was derived from and
    checked by a figure or a test without re-running anything.
    """

    def __init__(
        self,
        *,
        moves: list[dict],
        held: list[dict],
        alpha: float,
        beta: float,
        k_min: float,
        signal_at_target: float,
        source: str,
        mean_pass_rate: float = 0.0,
        misaligned: bool = False,
        misalignment: str = "",
    ) -> None:
        self.moves = moves
        self.held = held
        self.alpha = alpha
        self.beta = beta
        self.k_min = k_min
        self.signal_at_target = signal_at_target
        #: Which artifact the plan was derived from, so a stale plan is visible.
        self.source = source
        #: Mean observed pass rate over resolved cells — the batch-level
        #: quantity GenEnv's k_min rule actually applies to.
        self.mean_pass_rate = mean_pass_rate
        self.misaligned = misaligned
        self.misalignment = misalignment

    @property
    def move_count(self) -> int:
        return len(self.moves)

    @property
    def held_count(self) -> int:
        return len(self.held)

    @property
    def by_direction(self) -> dict[str, int]:
        """How many moves go each way. ``hold`` is not counted — it is ``held``."""
        out: dict[str, int] = {}
        for m in self.moves:
            out[m["direction"]] = out.get(m["direction"], 0) + 1
        return out

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "alpha": self.alpha,
            "beta": self.beta,
            "k_min": self.k_min,
            "alpha_derived_from_g": DEFAULT_G,
            "signal_at_target": round(self.signal_at_target, 6),
            "mean_pass_rate": round(self.mean_pass_rate, 6),
            "batch_alignment": round(alpha_reward(self.mean_pass_rate, alpha=self.alpha, beta=self.beta), 6),
            "misaligned": self.misaligned,
            "misalignment": self.misalignment,
            "moves": self.moves,
            "held": self.held,
            "move_count": self.move_count,
            "by_direction": self.by_direction,
            "held_count": self.held_count,
        }


def _band_mean_pass_rate(banded: list[band_mod.BandedCell]) -> float:
    """Mean observed pass rate over cells that were actually placed.

    Unresolved cells are excluded rather than treated as 0.5 or as their point
    estimate. Including a point estimate from an interval that spans two bands
    would let the noisiest cells drive the target, which is the opposite of what
    the interval machinery is for.
    """
    placed = [b for b in banded if b.resolved]
    if not placed:
        return 0.0
    return sum(b.p_hat for b in placed) / len(placed)


def plan_regeneration(
    cells: list[dict],
    *,
    tasks_by_id: dict[str, dict] | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    k_min: float = DEFAULT_K_MIN,
    source: str = "",
    max_moves: int | None = None,
) -> CurriculumPlan:
    """Decide which tasks to regenerate, from measured per-cell pass rates.

    ``cells`` is the shape ``scripts/probe.py`` and ``tools/plot.py`` already
    use: a list of dicts with ``harness``, ``task_id``, ``passes`` and ``n``.

    Three outcomes per task, and the middle one is the one that matters:

    * **move** — the cell resolved to ``mastered`` or ``out_of_reach``, so
      :func:`rsi.band.steer` names a direction and the task is regenerated with
      :func:`rsi.task_gen.reparameterise`. Each move records the band, the
      override, and the *reason*, so the plan is auditable without the scan.
    * **hold** — the cell resolved to ``frontier``. Steering would be undirected
      drift; this is where training should happen.
    * **held for measurement** — the cell is ``unresolved``, so no band is
      supported and no direction can be named. The ``k_min`` filter is *not*
      applied here; see :func:`in_band` for why.

    A task is aggregated **across harnesses** by taking its maximum pass rate.
    That choice is stated because the alternative is defensible: a task that one
    harness solves and another cannot is measuring the harness axis, which is
    this repository's subject, so regenerating it would delete the very signal
    the experiment exists to produce. Taking the max means a task is only moved
    when *no* trainable harness finds it useful, which is the conservative rule.
    """
    if tasks_by_id is None:
        tasks_by_id = {}

    # Aggregate to one entry per task, keeping the per-harness detail for the
    # record. A task's cells are only meaningful together: a single harness at
    # 0/32 says nothing about the task if three others are at 16/32.
    per_task: dict[str, list[dict]] = {}
    for c in cells:
        tid = str(c.get("task_id", c.get("task", "")))
        if not tid:
            continue
        per_task.setdefault(tid, []).append(c)

    moves: list[dict] = []
    held: list[dict] = []

    for tid in sorted(per_task):
        records = per_task[tid]
        banded = [
            band_mod.band_of(
                int(r.get("passes", 0)),
                int(r.get("n", 0)),
                harness=str(r.get("harness", "")),
                task_id=tid,
            )
            for r in records
            if int(r.get("n", 0)) > 0
        ]
        if not banded:
            continue

        best = max(banded, key=lambda b: b.p_hat)
        entry: dict = {
            "task_id": tid,
            "p_hat_max": round(best.p_hat, 6),
            "p_hat_mean": round(sum(b.p_hat for b in banded) / len(banded), 6),
            "harnesses": len(banded),
            "bands": sorted({b.band for b in banded}),
            "ci95_of_best": [round(best.lo, 6), round(best.hi, 6)],
            "alpha_gap": round(difficulty_gap(best.p_hat, alpha=alpha), 6),
            "reward": round(alpha_reward(best.p_hat, alpha=alpha, beta=beta), 6),
        }

        # A move needs the task's *parameters*, because `steer` moves a
        # parameter vector and nothing else. Without them `steer` falls back to
        # its own defaults and emits an override derived from a task that does
        # not exist — a silently wrong answer, which is worse than no answer.
        #
        # This is not hypothetical. The shipped scan measures the 24-task suite
        # (`t1-01`), while a generated batch carries hashed ids
        # (`t1-8f87ad9e`); the two sets do not intersect, so every lookup
        # missed. The first version of this function would have emitted moves
        # built from default parameters and nothing in the output would have
        # looked wrong.
        base = tasks_by_id.get(tid)
        if base is None or "params" not in base:
            entry["reason"] = (
                "no parameters for this task id — the scan and the batch are "
                "different task sets; scan the batch you intend to steer"
            )
            held.append(entry)
            continue

        # The only filter on a task is whether its band could be resolved. An
        # unresolved cell is a statement about the measurement, and the response
        # is to scan more rather than to move the task.
        if not best.resolved:
            entry["reason"] = "unresolved: the interval spans two bands, scan more"
            held.append(entry)
            continue

        override = band_mod.steer(best.band, base["params"])
        if not override:
            entry["reason"] = f"{best.band}: hold, this is where training should happen"
            entry["band"] = best.band
            held.append(entry)
            continue

        entry["band"] = best.band
        entry["direction"] = "harder" if best.band == band_mod.Band.MASTERED else "easier"
        entry["override"] = override
        entry["reason"] = f"{best.band} at p={best.p_hat:.3f}: {entry['direction']}"
        moves.append(entry)

    # A cap, so one bad scan cannot regenerate the entire suite in a single
    # round. Sorted by reward ascending — the *least* aligned tasks move first,
    # which is the order the α-reward induces and the reason it is computed
    # rather than decorative.
    if max_moves is not None and len(moves) > max_moves:
        moves.sort(key=lambda m: m["reward"])
        moves = moves[:max_moves]

    # The batch-level check, computed over resolved cells only. Including an
    # unresolved cell's point estimate would let the noisiest measurements drive
    # the diagnosis, which is the opposite of what the interval is for.
    placed = [
        band_mod.band_of(int(r.get("passes", 0)), int(r.get("n", 0)))
        for r in cells
        if int(r.get("n", 0)) > 0
    ]
    mean_p = _band_mean_pass_rate(placed)
    misaligned, diagnosis = batch_is_misaligned(mean_p, alpha=alpha, k_min=k_min)

    return CurriculumPlan(
        moves=moves,
        held=held,
        alpha=alpha,
        beta=beta,
        k_min=k_min,
        signal_at_target=signal_at_alpha(alpha=alpha),
        source=source,
        mean_pass_rate=mean_p,
        misaligned=misaligned,
        misalignment=diagnosis,
    )


def execute_plan(plan: CurriculumPlan, tasks_by_id: dict[str, dict]) -> list[dict]:
    """Materialise a plan's moves into new tasks.

    Kept here rather than in the script so the transformation from "a band" to
    "a task" is one function with one test, instead of a loop inside
    ``main()`` that only a full pipeline run can exercise.

    A move whose task id is not in ``tasks_by_id`` is skipped rather than
    raising: the plan and the batch are written by different steps, and a
    mismatch should be visible as a shortfall in the returned length, not as a
    crash that loses the rest of the plan.
    """
    out: list[dict] = []
    for m in plan.moves:
        base = tasks_by_id.get(m["task_id"])
        if base is None:
            continue
        fresh = task_gen.reparameterise(base, **m["override"])
        fresh["regenerated_from"] = m["task_id"]
        fresh["regenerated_band"] = m["band"]
        out.append(fresh)
    return out


def batch_alignment(tasks: list[dict], pass_rates: dict[str, float], *, alpha: float = DEFAULT_ALPHA,
                    beta: float = DEFAULT_BETA) -> float:
    """Mean α-reward over a batch, for tracking alignment across rounds.

    A batch is scored by the α-reward of each task's measured pass rate, so the
    number is comparable between rounds even as the tasks change. Tasks with no
    measurement are excluded rather than counted as 0, which would make an
    under-scanned round look badly aligned.
    """
    vals = [alpha_reward(pass_rates[t["id"]], alpha=alpha, beta=beta) for t in tasks if t["id"] in pass_rates]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def batch_is_misaligned(
    mean_pass_rate: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    k_min: float = DEFAULT_K_MIN,
) -> tuple[bool, str]:
    """GenEnv's ``k_min`` rule, applied where it was written: to a whole batch.

    Returns whether the batch is misaligned and the diagnosis. A batch whose
    mean pass rate is more than ``k_min`` from ``α`` is not a batch that needs
    four tasks nudged — it is a batch that is systematically too easy or too
    hard, and the response is to regenerate it against the target rather than to
    hill-climb individual cells.

    Returning the direction is the useful part: "regenerate, too easy" and
    "regenerate, too hard" call for opposite overrides, and a bare boolean would
    throw away the only actionable content of the check.
    """
    gap = difficulty_gap(mean_pass_rate, alpha=alpha)
    if abs(gap) <= k_min:
        return False, f"aligned: mean pass rate within {k_min} of {alpha}"
    direction = "too easy" if gap > 0 else "too hard"
    return True, f"misaligned: mean pass rate {mean_pass_rate:.3f} is {direction} ({gap:+.3f} from {alpha})"
