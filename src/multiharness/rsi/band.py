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

"""Where each cell sits on the difficulty curve — the curriculum's steering signal.

:mod:`rsi.stats` answers "can this cell teach anything", which is a question
about *reward variance*. This module answers a different one: "how hard is this
cell for the current policy", which is a question about *position*. A cell can
be perfectly live and still be useless for training — at ``p = 0.5`` it carries
the most signal there is, but a curriculum that only ever trains at ``p = 0.5``
never moves.

Three bands, following SPADE's regret banding:

``mastered``
    Above ``mastered_above``. The policy already succeeds; there is almost
    nothing left to learn and the cell mostly contributes a near-constant
    reward.
``frontier``
    Between the two thresholds. Hard enough to be worth training, easy enough
    that success is reachable.
``out_of_reach``
    Below ``out_of_reach_below``. The policy essentially never succeeds. Under
    GRPO this is the *trap*: a binary reward of all-zeros carries no gradient,
    so an out-of-reach cell is not merely hard, it is inert.

The trap is why this module exists rather than a plain threshold on the pass
rate. The measured n=8 scan found T2 at exactly 0.000 across all four harnesses
and read that as "the task is too hard". The n=32 rescan shows the same cells
at 0/32 — but 0/32 still has an upper confidence bound of 0.107, so the honest
statement is that they are *below the reachable band*, not that they are
impossible. :func:`band_of` therefore refuses to place a cell whose interval
spans two bands, and reports it as ``unresolved`` instead. That is the same
principle as the three-way verdict in ``stats.py``, applied to position rather
than to signal.

The steering rule
-----------------
:func:`steer` is what the task generator consumes: given a band, it says which
direction to move a task's parameters. That is the whole curriculum mechanism —
the generator does not need to know what makes a task hard, only whether the
last one was too easy or too hard, and the loop closes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .stats import wilson_interval

__all__ = [
    "Band",
    "MASTERED_ABOVE",
    "OUT_OF_REACH_BELOW",
    "band_of",
    "band_matrix",
    "band_counts",
    "frontier_cells",
    "steer",
]

#: Above this pass rate the policy has essentially solved the cell.
MASTERED_ABOVE = 0.9
#: Below this pass rate the policy essentially never succeeds, so a binary
#: reward is constant and the cell is inert rather than merely difficult.
OUT_OF_REACH_BELOW = 0.1


class Band(str):
    """Difficulty band. A ``str`` subclass so it serialises to JSON directly."""

    MASTERED = "mastered"
    FRONTIER = "frontier"
    OUT_OF_REACH = "out_of_reach"
    #: The interval spans two bands, so no single band is supported.
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class BandedCell:
    """One cell's position, with the interval that justifies it."""

    harness: str
    task_id: str
    passes: int
    n: int
    p_hat: float
    lo: float
    hi: float
    band: str

    @property
    def resolved(self) -> bool:
        return self.band != Band.UNRESOLVED

    def as_dict(self) -> dict:
        return {
            "harness": self.harness,
            "task_id": self.task_id,
            "passes": self.passes,
            "n": self.n,
            "p_hat": round(self.p_hat, 6),
            "ci95": [round(self.lo, 6), round(self.hi, 6)],
            "band": self.band,
        }


def _band_of_point(p: float) -> str:
    if p > MASTERED_ABOVE:
        return Band.MASTERED
    if p < OUT_OF_REACH_BELOW:
        return Band.OUT_OF_REACH
    return Band.FRONTIER


def band_of(
    passes: int,
    n: int,
    *,
    harness: str = "",
    task_id: str = "",
    mastered_above: float = MASTERED_ABOVE,
    out_of_reach_below: float = OUT_OF_REACH_BELOW,
    confidence: float = 0.95,
) -> BandedCell:
    """Place a cell on the difficulty curve, or refuse to if the data cannot.

    A band is only reported when the whole confidence interval falls inside it.
    Otherwise the verdict is ``unresolved``, and the caller is expected to scan
    more. The refusal is the point: at ``n = 8`` almost every interesting cell
    is unresolved, and reporting a band anyway is how "T2 is too hard" got into
    a README on the strength of eight rollouts.
    """
    if n <= 0:
        raise ValueError("band_of needs at least one rollout")
    if not 0 <= passes <= n:
        raise ValueError(f"passes={passes} is impossible out of n={n}")

    lo, hi = wilson_interval(passes, n, confidence)
    # Every point in the interval must land in the same band. Checking the two
    # endpoints is sufficient because the bands are ordered and the interval is
    # contiguous: if both ends are frontier, so is everything between them.
    lo_band = _band_of_point(lo)
    hi_band = _band_of_point(hi)
    band = lo_band if lo_band == hi_band else Band.UNRESOLVED

    return BandedCell(
        harness=harness,
        task_id=task_id,
        passes=passes,
        n=n,
        p_hat=passes / n,
        lo=lo,
        hi=hi,
        band=band,
    )


def band_matrix(
    cells: list[dict],
    *,
    harnesses: list[str] | None = None,
    task_ids: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """``{harness: {task_id: band}}`` from raw cell records.

    Accepts the shape ``scripts/probe.py`` writes, so the figures and the
    README read the same artifact the loop does rather than a re-derived copy.
    """
    out: dict[str, dict[str, str]] = {}
    for c in cells:
        h = str(c.get("harness", ""))
        t = str(c.get("task_id", c.get("task", "")))
        if harnesses is not None and h not in harnesses:
            continue
        if task_ids is not None and t not in task_ids:
            continue
        bc = band_of(int(c.get("passes", 0)), int(c.get("n", 0)), harness=h, task_id=t)
        out.setdefault(h, {})[t] = bc.band
    return out


def band_counts(banded: list[BandedCell]) -> dict[str, int]:
    """How many cells are in each band, including ``unresolved``.

    ``unresolved`` is reported rather than folded into another band because the
    count is the honest headline: it is the number of cells the current scan
    cannot yet place.
    """
    counts = {
        Band.MASTERED: 0,
        Band.FRONTIER: 0,
        Band.OUT_OF_REACH: 0,
        Band.UNRESOLVED: 0,
    }
    for b in banded:
        counts[b.band] = counts.get(b.band, 0) + 1
    return counts


def frontier_cells(banded: list[BandedCell]) -> list[BandedCell]:
    """The cells worth training on: resolved, and in the frontier band."""
    return [b for b in banded if b.band == Band.FRONTIER]


def steer(
    band: str,
    params: dict,
    *,
    difficulty: float | None = None,
) -> dict:
    """Which direction to move a task's parameters, given its band.

    Returns a dict of *suggested overrides* for
    :func:`rsi.task_gen.reparameterise`. Only the parameters that plausibly
    move difficulty are touched, and each move is a single step along the axis
    rather than a jump, so the next measurement stays attributable to the
    change that caused it.

    The rule per band:

    ``mastered``
        Harder. Lengthen the payload, raise the escape density, add a step,
        and stop giving the content away in the prompt.
    ``out_of_reach``
        Easier. Shorten the payload, remove escaping, drop to one step, and
        ship the content in a file so it can be read rather than transcribed.
    ``frontier``
        Hold. This is where training should happen; moving it would be
        undirected drift.
    ``unresolved``
        Hold, and scan more — the band is a statement about the measurement.

    ``difficulty`` is accepted for logging and is not used in the decision, so
    that a caller cannot accidentally make the move depend on two different
    notions of difficulty.
    """
    p = dict(params)
    length = int(p.get("payload_len", 4))
    esc = float(p.get("escape_density", 0.0))
    steps = int(p.get("steps", 1))

    if band == Band.MASTERED:
        return {
            "payload_len": min(16, length * 2),
            "escape_density": min(0.6, round(esc + 0.15, 3)),
            "steps": min(3, steps + 1),
            "read_source": False,
        }
    if band == Band.OUT_OF_REACH:
        return {
            "payload_len": max(1, length // 2),
            "escape_density": max(0.0, round(esc - 0.2, 3)),
            "steps": max(1, steps - 1),
            "read_source": True,
        }
    return {}
