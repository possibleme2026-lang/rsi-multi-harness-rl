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

"""An append-only ledger of every self-modification the loop ever tried.

Why a ledger rather than a log
------------------------------
A self-improving loop that keeps only its current state cannot answer the
questions that matter about it. Which kinds of edit actually pay off? Is the
loop exploring or is it stuck re-proposing the same change? Did a rejection
happen because the change was bad or because the measurement was too noisy to
see it? A final score answers none of these, and a free-text log answers them
only by reading it.

So every *attempted* edit gets a record — accepted or not — with enough
evidence attached to re-derive the decision later:

    round           which iteration proposed it
    edit            the descriptor: what changed
    components      which parts of the harness it touched
    score_before    the incumbent's score at proposal time
    score_after     the candidate's score
    delta           the difference
    floor           the noise-adjusted floor the delta had to clear
    decision        accepted / rejected
    reason          why, in one line

Keeping rejected edits is the point. A ledger of successes would make the loop
look far more efficient than it is and would hide exactly the signal that
drives exploration: a run of rejections means the current neighbourhood is
exhausted.

Adapted from ``rrsi/history.py``. Two things are deliberately different. RRSI
records one entry per accepted *edit* and reconstructs the rest from the
trajectory; this records every attempt, because the harness axis here has a
smaller edit space and the rejections carry proportionally more information.
And the pruning rule here is stated in terms of *yield* — accepted edits per
attempt — rather than RRSI's ``yield_g``, so that a component which is easy to
edit but never helps is pruned as readily as one that is hard to edit.

The file is JSONL, one object per line, appended and never rewritten. A loop
that can rewrite its own history can make its history say anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "COMPONENTS",
    "STRUCTURAL_COMPONENTS",
    "EditRecord",
    "Ledger",
    "edit_budget",
    "stall_flag",
]


#: The parts of a harness an edit can touch. Adapted from RRSI's component
#: vocabulary (``rrsi/components.py``) to the surface this repository actually
#: has: a harness here is guidance text, a tool set, and a submission
#: convention, so the vocabulary is those three plus the context policy.
COMPONENTS = (
    "prompt",           # GUIDANCE text
    "client_tool",      # which tools are exposed
    "output_plumbing",  # how the answer gets submitted
    "context_mgmt",     # how much of the transcript the model sees
)

#: Components whose change alters the *shape* of the interface rather than its
#: wording. A novelty bonus applies to edits here, because changing the tool
#: set explores a genuinely different region while rewording guidance does not.
STRUCTURAL_COMPONENTS = ("client_tool", "output_plumbing")


@dataclass(frozen=True)
class EditRecord:
    """One attempted self-modification and the evidence for its verdict."""

    round: int
    candidate_id: str
    #: Human-readable descriptor, e.g. ``guidance+=retry_hint``.
    edit: str
    #: Which harness was edited.
    harness: str
    components: tuple[str, ...]
    score_before: float
    score_after: float
    delta: float
    floor: float
    accepted: bool
    reason: str = ""
    #: Free-form extras (task subset, rollout counts, ...).
    meta: dict = field(default_factory=dict)

    @property
    def cleared_floor(self) -> bool:
        return self.delta > self.floor

    def as_dict(self) -> dict:
        d = {
            "round": self.round,
            "candidate_id": self.candidate_id,
            "edit": self.edit,
            "harness": self.harness,
            "components": list(self.components),
            "score_before": round(self.score_before, 6),
            "score_after": round(self.score_after, 6),
            "delta": round(self.delta, 6),
            "floor": round(self.floor, 6),
            "accepted": self.accepted,
            "reason": self.reason,
        }
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EditRecord:
        return cls(
            round=int(d["round"]),
            candidate_id=str(d["candidate_id"]),
            edit=str(d["edit"]),
            harness=str(d.get("harness", "")),
            components=tuple(d.get("components", ())),
            score_before=float(d["score_before"]),
            score_after=float(d["score_after"]),
            delta=float(d["delta"]),
            floor=float(d["floor"]),
            accepted=bool(d["accepted"]),
            reason=str(d.get("reason", "")),
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------
# the annealed edit budget
# --------------------------------------------------------------------------


def edit_budget(t: int, T: int, b_min: int = 1, b_max: int = 3) -> int:
    """How many independent edits a proposal may carry at round ``t`` of ``T``.

    A cosine anneal from ``b_max`` down to ``b_min``: early rounds may change
    several things at once, late rounds may change one. The schedule encodes a
    real trade-off. Early on, a multi-part change is the fastest way to find a
    better region, and there is enough budget left to recover from a bad one.
    Late, a multi-part change is unattributable — if the score moves you cannot
    say which part moved it — and the run is nearly over, so an unforced
    regression is expensive.

    This bounds ``||z||_0``, the number of independent edits in one proposal,
    and nothing else. It is not a step size and not a score threshold.

    From ``rrsi/schedule.py``, kept numerically identical so that a figure
    showing the schedule is comparable across the two codebases::

        b_t = ceil(b_min + (b_max - b_min) * 0.5 * (1 + cos(pi * t / T)))

    ``t = 0`` gives ``b_max``; ``t = T`` gives ``b_min``.
    """
    if T <= 0:
        return b_min
    raw = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    return int(math.ceil(round(raw, 9)))


def stall_flag(trajectory: list[float], t: int, w: int = 3, delta: float = 0.0) -> bool:
    """True when the last ``w`` rounds failed to improve on the best so far.

    Used to switch the loop from exploitation to exploration. RRSI computes the
    same thing over a trajectory; the difference here is that ``delta`` is the
    noise floor, so a run of *measured* gains too small to clear the floor
    still counts as a stall. Without that, a loop can look like it is climbing
    while it is only sampling noise.
    """
    if t < w:
        return False
    window = trajectory[max(0, t - w) : t]
    if len(window) < w:
        return False
    best_before = max(trajectory[: max(1, t - w)])
    return max(window) <= best_before + delta


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL record of every attempted edit.

    In-memory plus a file. The file is opened in append mode on every write
    rather than held open, so a crash mid-run leaves a valid ledger of
    everything up to the last completed attempt.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[EditRecord] = []
        if self.path is not None and self.path.is_file():
            self._load()

    # -- persistence ---------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.records.append(EditRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                # A truncated final line is expected after a hard kill. Skipping
                # it is right; silently skipping a *middle* line would not be,
                # so the count of dropped lines is tracked instead of ignored.
                continue

    def append(self, rec: EditRecord) -> EditRecord:
        self.records.append(rec)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec.as_dict(), sort_keys=True) + "\n")
        return rec

    # -- queries -------------------------------------------------------

    def tried(self, harness: str | None = None) -> set[str]:
        """Edit descriptors already attempted, optionally for one harness only.

        The ``harness`` filter is not a convenience. An edit is a change to
        *one* harness's interface, and each harness is a separate search — the
        same guidance edit can help ``bash_minimal`` and hurt ``json_strict``,
        and both facts are worth learning. Without the filter the loop would
        learn "somebody already tried this" and stop, collapsing four
        independent searches into one shared one and leaving the later rounds
        with nothing to propose.
        """
        if harness is None:
            return {r.edit for r in self.records}
        return {r.edit for r in self.records if r.harness == harness}

    def attempted(self) -> int:
        return len(self.records)

    def accepted_edits(self) -> list[EditRecord]:
        return [r for r in self.records if r.accepted]

    def rejected_edits(self) -> list[EditRecord]:
        return [r for r in self.records if not r.accepted]

    def best_score(self) -> float:
        """The incumbent's score, taken as the highest accepted score_after."""
        acc = self.accepted_edits()
        return max((r.score_after for r in acc), default=0.0)

    def component_counts(self, harness: str | None = None) -> dict[str, int]:
        """Accepted edits per component. Drives the novelty bonus.

        Scoped to one harness when given, for the same reason :meth:`tried` is:
        novelty is a property of a single harness's history. A component that
        was accepted on ``react_tools`` is still unexplored on ``json_strict``.
        """
        counts = {c: 0 for c in COMPONENTS}
        for r in self.accepted_edits():
            if harness is not None and r.harness != harness:
                continue
            for c in r.components:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def novelty(self, components: tuple[str, ...], harness: str | None = None) -> int:
        """How many *structural* components this edit touches that are untouched.

        RRSI's ``novelty`` counts components the incumbent has never had an
        accepted edit on. Only structural components count here: a harness
        whose guidance was reworded five times has not explored five regions,
        and rewarding that as novelty would push the loop to keep rewriting
        prose instead of changing the interface.
        """
        counts = self.component_counts(harness)
        return sum(1 for c in components if c in STRUCTURAL_COMPONENTS and counts.get(c, 0) == 0)

    def yield_per_component(self, harness: str | None = None) -> dict[str, float]:
        """Accepted edits per attempt, per component.

        The quantity pruning is based on. An attempt is attributed to every
        component it touched, so a component that only ever appears in
        multi-part changes is judged on those changes' outcomes.
        """
        attempts = {c: 0 for c in COMPONENTS}
        wins = {c: 0 for c in COMPONENTS}
        for r in self.records:
            if harness is not None and r.harness != harness:
                continue
            for c in r.components:
                attempts[c] = attempts.get(c, 0) + 1
                if r.accepted:
                    wins[c] = wins.get(c, 0) + 1
        return {c: (wins[c] / attempts[c] if attempts[c] else 0.0) for c in attempts}

    def prune_set(
        self,
        min_attempts: int = 3,
        max_yield: float = 0.0,
        harness: str | None = None,
    ) -> list[str]:
        """Components to stop proposing: tried enough times, never once worked.

        ``min_attempts`` guards against pruning on one unlucky sample. The rule
        is deliberately blunt — a component is dropped only when it has *never*
        been accepted — because the alternative (drop below a yield threshold)
        would discard a component that works occasionally, and the whole reason
        this project exists is that discarding on thin evidence is the mistake
        the old pipeline made.
        """
        yields = self.yield_per_component(harness)
        attempts: dict[str, int] = {c: 0 for c in COMPONENTS}
        for r in self.records:
            if harness is not None and r.harness != harness:
                continue
            for c in r.components:
                attempts[c] = attempts.get(c, 0) + 1
        return [
            c
            for c in COMPONENTS
            if attempts.get(c, 0) >= min_attempts and yields.get(c, 0.0) <= max_yield
        ]

    def stalled(self, w: int = 3, delta: float = 0.0, harness: str | None = None) -> bool:
        """Whether the accepted-score trajectory has flattened."""
        traj = [r.score_after for r in self.accepted_edits() if harness is None or r.harness == harness]
        return stall_flag(traj, len(traj), w=w, delta=delta)

    # -- reporting -----------------------------------------------------

    def render(self, n: int = 12) -> str:
        """A markdown table of the last ``n`` attempts, for the README."""
        rows = self.records[-n:]
        if not rows:
            return "_no edits attempted_"
        out = [
            "| round | harness | edit | components | before | after | delta | floor | verdict |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in rows:
            out.append(
                f"| {r.round} | `{r.harness}` | `{r.edit}` | {', '.join(r.components)} | "
                f"{r.score_before:.3f} | {r.score_after:.3f} | {r.delta:+.3f} | {r.floor:.3f} | "
                f"{'accepted' if r.accepted else 'rejected'} |"
            )
        return "\n".join(out)

    def summary(self) -> dict:
        acc = self.accepted_edits()
        rej = self.rejected_edits()
        return {
            "attempted": self.attempted(),
            "accepted": len(acc),
            "rejected": len(rej),
            "accept_rate": (len(acc) / self.attempted()) if self.records else 0.0,
            "best_score": self.best_score(),
            # Rejections that were nonetheless above zero delta: the loop saw a
            # gain it could not distinguish from noise. A large count here means
            # the noise floor is the binding constraint, not the edit space.
            "rejected_within_noise": sum(1 for r in rej if 0.0 < r.delta <= r.floor),
            "rejected_worse": sum(1 for r in rej if r.delta <= 0.0),
            "yield_per_component": {k: round(v, 3) for k, v in self.yield_per_component().items()},
            "prune_set": self.prune_set(),
            "rounds": max((r.round for r in self.records), default=-1) + 1,
        }
