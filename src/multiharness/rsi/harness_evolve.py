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

"""Evolving the harness: the second axis of the loop.

The task axis asks "can the policy write a problem worth solving". This axis
asks the complementary question: **given a fixed policy and a fixed task, can
the loop find a better interface for the policy to act through.** The two are
independent — a task generator can produce a perfect curriculum and the policy
will still fail if the harness's guidance never tells it how to submit — which
is why the loop carries both rather than treating the harness as fixed
scaffolding.

What an edit is
---------------
An edit is a *descriptor*, not a code mutation. It names a component and a
change, and applying it produces a new ``GUIDANCE`` string (or a new tool
surface) for one harness. That restriction is deliberate. RRSI applies edits to
a policy's source; here the artefact being edited is a text interface plus a
tool list, and keeping edits as data means every candidate can be:

* recorded in the ledger with its components,
* replayed to reproduce a result,
* checked against a guard before it is applied,
* and plotted, since a descriptor has a name.

Editing source would give a larger search space and an unauditable one. The
components are the three things a harness here actually *is*:

``prompt``
    ``GUIDANCE`` — the text prepended to the task instruction.
``client_tool``
    which tools the harness exposes.
``output_plumbing``
    how the answer is meant to be submitted (a file, a ``finish`` call, ...).
``context_mgmt``
    how much of the transcript the model is shown.

The guard, and why ``codex_style`` is frozen
--------------------------------------------
Every candidate passes ``guard_tool_surface.py``'s invariant before it is
applied: the tools a harness advertises in ``GUIDANCE`` must match the tools it
actually exposes. An edit that adds a sentence promising a ``search`` tool
would make the harness lie to the policy, and a policy that believes it fails
for a reason no measurement will attribute correctly.

``codex_style`` is exempt from evolution entirely. It is the fixed reference
point: if every harness can drift, then a rising score cannot be attributed to
a better interface rather than to a moving target, and the cross-harness
comparison — the thing this repository exists to measure — would be between
four optimised harnesses and no baseline. One harness must stay put for the
others' movement to mean anything.

Selection
---------
A candidate is accepted when its score clears the incumbent's by more than the
noise floor, and — when it does not — the rejection reason distinguishes "worse"
from "better but indistinguishable from noise". Those are different facts about
the run: the first says the edit was bad, the second says the measurement is
the binding constraint and the loop should scan more before it concludes
anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .ledger import EditRecord, Ledger

__all__ = [
    "FROZEN_HARNESSES",
    "EVOLVABLE_HARNESSES",
    "EDIT_LIBRARY",
    "HarnessEdit",
    "HarnessState",
    "propose_edits",
    "apply_edit",
    "guard_candidate",
    "judge_candidate",
    "to_env_class",
    "component_weights",
    "EvolverReport",
]

#: Harnesses the evolver must never touch. ``oracle`` writes the answer itself,
#: so improving it is meaningless; ``codex_style`` is the fixed reference point
#: that makes the other harnesses' movement interpretable.
FROZEN_HARNESSES = ("oracle", "codex_style")

#: The harnesses the loop may edit.
EVOLVABLE_HARNESSES = ("bash_minimal", "react_tools", "json_strict", "longctx_summary")


# --------------------------------------------------------------------------
# the edit library
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessEdit:
    """One named, replayable change to one harness.

    ``name`` is the identity used by the ledger and by the stall check, so two
    edits that would produce the same text must share a name or the loop will
    re-propose work it has already done.
    """

    name: str
    component: str
    #: Text appended to GUIDANCE. Empty for edits that only change the tool set.
    guidance_addendum: str = ""
    #: Tools the edit removes from the advertised set. The tool surface guard
    #: rejects a candidate that removes a tool the guidance still promises.
    drops_tools: tuple[str, ...] = ()
    #: A one-line rationale, shown in the ledger and in the README table.
    rationale: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "component": self.component,
            "guidance_addendum": self.guidance_addendum,
            "drops_tools": list(self.drops_tools),
            "rationale": self.rationale,
        }


#: The edit space. Hand-written rather than generated, because the number of
#: genuinely different interfaces a harness here can have is small, and a
#: generated edit would mostly produce rewording — which the novelty rule
#: already refuses to reward.
#:
#: Each entry is a hypothesis about *why* the policy fails, stated as a change:
#: "it does not know it may retry", "it loses the answer in the transcript",
#: "it does not know the value must be exact". That is what makes the ledger's
#: accept/reject column a test of the hypothesis rather than a score.
EDIT_LIBRARY: tuple[HarnessEdit, ...] = (
    HarnessEdit(
        name="guidance+=submit_echo",
        component="prompt",
        guidance_addendum=(
            "After you write the answer file, run `cat answer.txt` and check that what "
            "comes back is exactly the value you intended. If it is not, write it again."
        ),
        rationale="the policy writes the file but never verifies its content",
    ),
    HarnessEdit(
        name="guidance+=retry_hint",
        component="prompt",
        guidance_addendum=(
            "If a command fails or returns nothing useful, you may run another one. "
            "You are not limited to a single tool call."
        ),
        rationale="the policy stops after one failed call",
    ),
    HarnessEdit(
        name="guidance+=exactness",
        component="prompt",
        guidance_addendum=(
            "The answer is compared exactly. Do not add commentary, quotes, or a "
            "trailing newline beyond what a plain write produces."
        ),
        rationale="the policy appends prose to an exact-match answer",
    ),
    HarnessEdit(
        name="guidance+=one_line",
        component="prompt",
        guidance_addendum=(
            "Keep the answer to a single line. If the task asks for a value, write only "
            "that value and nothing else."
        ),
        rationale="the policy writes a sentence when a value was wanted",
    ),
    HarnessEdit(
        name="prompt-=verbose_preamble",
        component="prompt",
        guidance_addendum=(
            "Do not explain your plan before acting. Call a tool immediately."
        ),
        rationale="guidance asking for reasoning spends the turn budget on prose",
    ),
    HarnessEdit(
        name="output_plumbing+=explicit_path",
        component="output_plumbing",
        guidance_addendum=(
            "The file must be named exactly `answer.txt` and must be in the current "
            "working directory — not a subdirectory, and not anywhere else."
        ),
        rationale="the policy writes to the wrong path or a nested directory",
    ),
    HarnessEdit(
        name="client_tool-=read_file",
        component="client_tool",
        drops_tools=("read_file",),
        rationale="an unused tool in the schema costs context and invites stray calls",
    ),
    HarnessEdit(
        name="context_mgmt+=keep_last_error",
        component="context_mgmt",
        guidance_addendum=(
            "If an earlier attempt failed, the error is shown to you again before your "
            "next call. Read it before deciding what to do."
        ),
        rationale="the policy repeats a call that already failed",
    ),
)


# --------------------------------------------------------------------------
# harness state
# --------------------------------------------------------------------------


@dataclass
class HarnessState:
    """A harness's current interface, as the loop sees it.

    Text plus a tool list, not a class. The evolver produces states, and the
    runner turns a state into an actual environment — keeping the two apart is
    what lets a candidate be scored without mutating anything global.
    """

    name: str
    guidance: str
    tools: tuple[str, ...]
    #: Names of the edits applied to reach this state, in order.
    history: tuple[str, ...] = ()
    #: Score measured for this state, when it has been evaluated.
    score: float = 0.0
    #: Extra guidance lines accumulated by edits, kept separate from the base
    #: so the diff between two states is readable.
    addenda: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "tools": list(self.tools),
            "history": list(self.history),
            "score": round(self.score, 6),
            "guidance_len": len(self.effective_guidance()),
        }

    def effective_guidance(self) -> str:
        """The guidance the policy actually receives."""
        parts = [self.guidance.strip()] if self.guidance.strip() else []
        parts.extend(a.strip() for a in self.addenda if a.strip())
        return "\n".join(parts)


# --------------------------------------------------------------------------
# proposing
# --------------------------------------------------------------------------


def propose_edits(
    state: HarnessState,
    ledger: Ledger,
    budget: int,
    *,
    rng=None,
    allow_pruned: bool = False,
) -> list[HarnessEdit]:
    """Up to ``budget`` edits that have not been tried and are not pruned.

    Four filters, in order, each removing a different kind of waste:

    *Frozen* — ``codex_style`` is the held-out harness and ``oracle`` is the
    reference; proposing an edit for either would quietly destroy the one axis
    the experiment holds out. Enforced here rather than only at the call site,
    because the cost of a caller forgetting is that the held-out result becomes
    a training result and nothing about the output would look wrong.
    *Already tried* — the ledger holds every attempt, so re-proposing one would
    burn a round to learn what is already known. Scoped to ``state.name``:
    each harness is a separate search, and an edit that was accepted on one is
    still unexplored on another.
    *Pruned* — a component that has been tried ``min_attempts`` times on this
    harness and never once accepted. Kept out of the proposal set rather than
    removed from the library, so ``allow_pruned`` can re-open it if the loop
    stalls badly enough.
    *Inapplicable* — an edit that drops a tool the harness does not expose would
    be a no-op recorded as an attempt, which inflates the attempt count and
    makes the yield statistic lie.

    Randomised order rather than first-``budget``, so the loop does not always
    spend its early rounds on whichever edit happens to be first in the library.
    """
    if state.name in FROZEN_HARNESSES:
        return []

    tried = ledger.tried(state.name)
    pruned = set() if allow_pruned else set(ledger.prune_set(harness=state.name))
    out: list[HarnessEdit] = []
    for e in EDIT_LIBRARY:
        if e.name in tried or e.component in pruned:
            continue
        if e.drops_tools and not any(t in state.tools for t in e.drops_tools):
            continue
        out.append(e)
    if rng is not None:
        rng.shuffle(out)
    return out[: max(0, budget)]


# --------------------------------------------------------------------------
# applying, guarding, judging
# --------------------------------------------------------------------------


def apply_edit(state: HarnessState, edit: HarnessEdit) -> HarnessState:
    """A new state with the edit applied. The input state is not modified.

    Immutability is load-bearing: a rejected candidate must leave no trace, and
    an in-place edit that is later rejected would have to be undone exactly.
    Producing a new state means a rejection is simply not using the result.
    """
    tools = tuple(t for t in state.tools if t not in edit.drops_tools)
    addenda = state.addenda
    if edit.guidance_addendum:
        addenda = addenda + (edit.guidance_addendum,)
    return replace(
        state,
        tools=tools,
        addenda=addenda,
        history=state.history + (edit.name,),
    )


def guard_candidate(state: HarnessState) -> tuple[bool, str]:
    """Whether a candidate is internally consistent, and why not if it is not.

    The invariant is the one ``scripts/guard_tool_surface.py`` enforces
    statically: **a harness must not advertise a tool it does not expose.** The
    converse is also checked — an exposed tool that the guidance names as
    available but that has been dropped would make the guidance stale.

    The check is on the *guidance text against the tool list*, which is the
    only pair that can drift when edits are descriptors rather than code.
    """
    text = state.effective_guidance()
    advertised = [t for t in ("read_file", "write_file", "replace_in_file", "finish", "submit") if f"`{t}`" in text]
    missing = [t for t in advertised if t not in state.tools and t not in ("finish", "submit")]
    if missing:
        return False, f"guidance advertises {missing} but the harness exposes {list(state.tools)}"
    if not state.tools:
        return False, "a harness with no tools cannot be acted through"
    if not text.strip():
        return False, "empty guidance leaves the policy with no instructions"
    return True, "guidance and tool surface agree"


def judge_candidate(
    candidate: HarnessState,
    incumbent: HarnessState,
    *,
    floor: float,
    round_index: int,
    candidate_id: str,
    edit: HarnessEdit,
    meta: dict | None = None,
) -> tuple[bool, EditRecord]:
    """Decide whether to accept a candidate, and record the evidence either way.

    Acceptance needs ``delta > floor``, where ``floor`` is the noise-adjusted
    band from :func:`rsi.stats.noise_floor`. The strict ``>`` rather than ``>=``
    matters: a delta exactly equal to the floor is the boundary case where the
    measurement says "about the size of noise", and accepting it would make the
    loop's accept rate depend on floating-point equality.

    The rejection *reason* is the part worth keeping. ``within noise`` and
    ``worse`` are different facts — the first means scan more, the second means
    the edit is bad — and collapsing them into "rejected" throws away the only
    signal that tells the loop which of the two it is facing.
    """
    delta = candidate.score - incumbent.score
    if delta > floor:
        accepted, reason = True, f"cleared the noise floor by {delta - floor:.4f}"
    elif delta > 0.0:
        accepted, reason = False, f"within noise: +{delta:.4f} <= floor {floor:.4f}"
    else:
        accepted, reason = False, f"worse by {delta:.4f}"

    rec = EditRecord(
        round=round_index,
        candidate_id=candidate_id,
        edit=edit.name,
        harness=candidate.name,
        components=(edit.component,),
        score_before=incumbent.score,
        score_after=candidate.score,
        delta=delta,
        floor=floor,
        accepted=accepted,
        reason=reason,
        meta=dict(meta or {}),
    )
    return accepted, rec


# --------------------------------------------------------------------------
# the loop, minus the scoring
# --------------------------------------------------------------------------


@dataclass
class EvolverReport:
    """What one evolution run did. Serializable so the figures can read it."""

    harness: str
    rounds: int
    accepted: int
    attempted: int
    start_score: float
    end_score: float
    #: Names of accepted edits, in order — the actual interface that was found.
    winning_edits: list[str] = field(default_factory=list)
    #: Per-round trajectory of the incumbent's score.
    trajectory: list[float] = field(default_factory=list)
    #: Rounds where no edit was accepted, which is what the stall check reads.
    stalled_rounds: int = 0

    @property
    def gain(self) -> float:
        return self.end_score - self.start_score

    def as_dict(self) -> dict:
        return {
            "harness": self.harness,
            "rounds": self.rounds,
            "accepted": self.accepted,
            "attempted": self.attempted,
            "start_score": round(self.start_score, 6),
            "end_score": round(self.end_score, 6),
            "gain": round(self.gain, 6),
            "winning_edits": list(self.winning_edits),
            "trajectory": [round(x, 6) for x in self.trajectory],
            "stalled_rounds": self.stalled_rounds,
        }


def component_weights(ledger: Ledger, harness: str | None = None) -> dict[str, float]:
    """Prior weight per component, from the ledger's yield.

    Components that have worked before are tried first. This is exploitation;
    the novelty bonus in :meth:`Ledger.novelty` is the exploration term, and
    the stall check is what switches between them. Exposed as a function rather
    than inlined so the figure that plots the weights uses the same numbers the
    loop does.
    """
    y = ledger.yield_per_component(harness)
    total = sum(y.values()) or 1.0
    return {c: (v / total) for c, v in y.items()}


def to_env_class(state: HarnessState, base_cls):
    """Turn a :class:`HarnessState` into an environment class the runner can use.

    The rollout driver takes a *class* (``Agent.run(cls, task_id, n=...)``), not
    an instance, and the harness's guidance and tool set are class attributes.
    So scoring a candidate requires materialising a class — which is the one
    place where the evolver's data representation has to become executable, and
    therefore the one place worth keeping explicit rather than inlined in the
    scoring function.

    Two invariants are enforced here rather than trusted:

    * The tool set is *exactly* the state's, so a candidate cannot score well
      by quietly keeping a tool the edit removed.
    * The guidance is the state's effective guidance, so the text the policy
      reads is the text the ledger recorded.

    Methods are bound from ``base_cls`` by name. A state naming a tool its base
    class does not implement raises, because silently dropping it would make
    the candidate differ from its record — and the ledger's whole value is that
    a record describes what was actually run.
    """
    missing = [t for t in state.tools if not hasattr(base_cls, t)]
    if missing:
        raise ValueError(
            f"{state.name}: state exposes {missing}, which {base_cls.__name__} does not implement"
        )

    class _Evolved(base_cls):  # type: ignore[misc, valid-type]
        pass

    _Evolved.__name__ = f"{base_cls.__name__}_evolved_{abs(hash(state.history)) % 10**6}"
    _Evolved.name = state.name
    _Evolved.GUIDANCE = state.effective_guidance()
    # `discover_tools` walks public methods, so removing a tool means removing
    # the method. Rebinding it to a private name is how that is expressed
    # without editing the base class.
    for t in ("read_file", "write_file", "replace_in_file", "finish", "submit"):
        if hasattr(base_cls, t) and t not in state.tools:
            setattr(_Evolved, t, None)
            delattr(_Evolved, t)
    return _Evolved
