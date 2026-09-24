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

"""Tasks built along a dependency path, graded on the state they leave behind.

This is EnvScaler's ScenGenerator (arXiv:2601.05808, §3) with the graph from
``envgen`` underneath it. The two decisions worth defending are *what a task
is* and *what a reward is*.

What a task is
--------------
A task is a **path through the tool dependency graph**, materialised as three
artifacts:

1. an **instruction** in the domain's language, naming the goal but not the
   route;
2. an **environment** — the executable tools the agent can call, plus the
   initial state;
3. a **reference solution** — the tool calls that solve it, in order.

This is SPADE's "generate the whole trial, not just the question", and it is
the property that makes the reward checkable rather than asserted. The reference
is written as *tool calls against the environment*; the reward is computed from
the *state those calls produce*. They are two independent derivations of the
same final state, and ``validate_state`` requires them to agree.

What a reward is
----------------
Reward is the **fraction of checkpoints satisfied**, not a boolean. EnvScaler
decomposes the task into K verifiable conditions and grades
``reward = (1/K) * sum(1[f_k(S_final)])``. Three reasons that matters here:

**It is process-agnostic.** A checkpoint reads only the final state, so every
path that reaches the goal passes. Grading the call sequence instead would fail
a correct agent that took a different route, and would fail it *invisibly* —
the task would simply look harder. This repository already commits to a
harness-agnostic verifier; a state-based reward is that same commitment applied
one level up.

**It carries gradient where a boolean does not.** ``stats.py`` computes the
GRPO group signal as ``1 - p^G - (1-p)^G``, which vanishes as ``p`` approaches
0 or 1. A 0.5B model on a hard multi-step task sits near ``p = 0``, where a
boolean reward makes every group identical and the gradient is exactly zero.
Fractional credit moves ``p`` off the floor. The checkpoint count is therefore
a *training* parameter, not a reporting detail, and the module exposes
``checkpoint_count`` so a run can state what it chose.

**It localises failure.** "3 of 5 checkpoints" says which part of the state is
wrong. "0.0" says nothing, and a suite of zeros is indistinguishable from a
broken verifier — the exact confusion gates V1/V2 exist to prevent.

The failure mode this must avoid
--------------------------------
The GEF survey (arXiv:2511.09586, §5.2) names **generator-verifier asymmetry**
as the hazard of synthesized environments: a generator weaker than its verifier
emits tasks that look solvable and are not. Here generator and verifier are the
same module, so the asymmetry has a checkable form — *the checkpoints must
discriminate*:

* they must **all pass** on the reference's final state (otherwise the task is
  unsolvable and every rollout is wasted);
* they must **not all pass** on the initial state (otherwise the task is already
  solved and teaches nothing);
* they must **not all pass** on a *plausible wrong* final state — one where the
  agent acted on a distractor record instead of the target. This third one is
  what distractors are for, and without it the environment would be a prompt
  with extra steps.

All three are asserted in ``tests/test_rsi_envgen.py``, on generated instances
rather than on a fixture, because the property has to hold for the whole
distribution and not for one example.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from . import envgen

__all__ = [
    "CHECKPOINT_KINDS",
    "Checkpoint",
    "EnvTask",
    "build_checkpoints",
    "reference_trace",
    "apply_trace",
    "grade_state",
    "generate_env_task",
    "generate_env_batch",
    "summarise_env_batch",
    "plausible_wrong_state",
]

#: The checkpoint families. Each is a *different question about the same final
#: state*, and a task carries one of each kind that applies rather than K copies
#: of one kind — K copies of "did you change the right field" would give
#: fractional credit for a single success, which is a reward-shaped lie.
CHECKPOINT_KINDS = (
    "target_field",      # the target record's field holds the requested value
    "target_only",       # no *other* record was changed
    "log_recorded",      # the change was recorded in the audit log
    "meta_consistent",   # the summary counter agrees with the log
    "order_observable",  # the state reflects one specific ordering of the chain
)


@dataclass(frozen=True)
class Checkpoint:
    """One verifiable condition on the final state.

    ``check`` is a small declarative spec rather than a Python callable, for two
    reasons. It can be serialised into the Harbor task package, where the
    verifier runs in a *different container* from the generator and cannot
    import this module. And it can be evaluated by the same code on both sides,
    so the exported verifier and the in-process one cannot drift — the drift
    would be invisible, since both would keep returning plausible scores.
    """

    kind: str
    #: Arguments the evaluator needs. Kept flat and JSON-able.
    args: dict = field(default_factory=dict)
    doc: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "args": self.args, "doc": self.doc}


def _records_by_id(state: dict) -> dict:
    return {r["_id"]: r for r in state.get("records", [])}


def evaluate(checkpoint: Checkpoint, state: dict) -> bool:
    """Evaluate one checkpoint against a final state. Total: never raises.

    Delegates to ``harnesses.core.evaluate_checkpoint``, which is the single
    implementation. There used to be two copies of this predicate — one here and
    one inlined into the exported verifier — and a third in ``core.verify`` was
    about to be written. Three copies of a reward function is not a style
    problem: each would keep returning a *plausible* score after the others
    changed, so a drift would be invisible in every number the repository
    reports. The dependency arrow ``rsi -> harnesses`` makes one implementation
    possible, and ``tests/test_rsi_envgen.py`` asserts the inlined copy agrees.
    """
    from ..harnesses.core import evaluate_checkpoint

    return evaluate_checkpoint(checkpoint.as_dict(), state)


def _evaluate(cp: Checkpoint, state: dict) -> bool:
    a = cp.args
    if cp.kind == "target_field":
        rec = _records_by_id(state).get(a["id"])
        if rec is None:
            return False
        # `fields` is a mapping, and *every* entry must hold. It is one
        # checkpoint rather than one per field because the fields belong to a
        # single action the agent was asked to take; splitting them would give
        # partial credit for a partially-applied single change, which inflates
        # the reward without distinguishing any behaviour.
        return all(str(rec.get(f)) == str(v) for f, v in a["fields"].items())

    if cp.kind == "target_only":
        changed = set(state.get("_changed", []))
        return changed == {a["id"]}

    if cp.kind == "log_recorded":
        entries = state.get("log", [])
        return any(e.get("id") == a["id"] and e.get("field") == a["field"] for e in entries)

    if cp.kind == "meta_consistent":
        meta = state.get("meta", {})
        entries = state.get("log", [])
        # `>= 1` is load-bearing, not defensive. The condition is "the counter
        # equals the number of log entries", which on an untouched state is
        # `0 == 0` — **trivially true**. Measured: the initial state scored 0.2
        # to 0.25 instead of 0.0, entirely from this checkpoint passing for
        # doing nothing. That is the V2 failure (a reward that fires without an
        # answer) arriving through the one checkpoint whose predicate has a
        # vacuous solution.
        #
        # Requiring at least one entry makes the checkpoint mean what its name
        # says: the counter is consistent *and* there is something to be
        # consistent about. A state with no change cannot satisfy a condition
        # about a change having been recorded.
        return len(entries) >= 1 and int(meta.get("changed", -1)) == len(entries)

    if cp.kind == "order_observable":
        # The state records the order in which the chain's mutations landed.
        # Comparing the *sequence* rather than the set is what makes the order
        # a real constraint instead of a decoration — and it is why a task built
        # on a `precondition` edge can be failed by reversing two calls.
        #
        # The `order` argument is never empty (the checkpoint is only built for
        # chains with >= 2 mutations), but the *state's* order can be, and the
        # comparison is guarded against a vacuous pass the same way
        # `meta_consistent` is: an untouched state has no order, and "no order
        # equals no order" would otherwise be true for free.
        want = list(a["order"])
        got = list(state.get("_order", []))
        return len(got) > 0 and got == want

    raise ValueError(f"unknown checkpoint kind {cp.kind!r}")


def build_checkpoints(instance: envgen.EnvInstance, *, trace: list[dict]) -> list[Checkpoint]:
    """The K conditions a correct final state must satisfy.

    Built from the *trace* — the reference's tool calls — rather than from the
    instruction text. Grading against the instruction would mean that rewording
    a prompt changed the reward, and would make the reward a function of the
    generator's prose style rather than of the environment's semantics.

    The field is read off the trace for the same reason, and it is the fix for a
    bug rather than a stylistic choice: the spec does not know which mutator the
    chain will pick, so a field taken from the spec is wrong whenever the chain
    picks a different one. The symptom was subtle — reference 0.8, initial state
    0.4, both wrong in the same direction — which is exactly the shape of bug
    that survives review.

    Only the *last* mutation's field is graded, and earlier mutations are graded
    through ``order_observable``. Grading every mutation's field would make
    ``target_field`` a conjunction that reads as one condition, so a task with
    three mutations would lose a third of its credit for one mistake — the
    checkpoint count would stop meaning what it says.
    """
    target = instance.initial_state["_target"]
    tid = target["id"]

    muts = [c for c in trace if c.get("kind") == "mutate"]
    final = apply_trace(instance, trace)

    # Every field the reference changes, in the order it changes them, with
    # duplicates collapsed. Grading only the *last* one was a bug: the
    # instruction names all of them (see `_instruction_for`), so a compliant
    # agent would be graded on a subset of what it was asked for and would score
    # below 1.0 for doing exactly what the task said.
    graded_fields: list[str] = []
    for call in muts:
        f = call.get("field")
        if f and f not in graded_fields:
            graded_fields.append(f)

    cps: list[Checkpoint] = []
    if graded_fields:
        final_rec = _records_by_id(final).get(tid, {})
        want = {f: final_rec.get(f) for f in graded_fields}
        cps.append(
            Checkpoint(
                "target_field",
                {"id": tid, "fields": want},
                f"record {tid} has " + ", ".join(f"{f}={v!r}" for f, v in want.items()),
            )
        )
    cps.append(
        Checkpoint(
            "target_only",
            {"id": tid},
            f"no record other than {tid} was modified",
        )
    )
    if muts:
        cps.append(
            Checkpoint(
                "log_recorded",
                {"id": tid, "field": graded_fields[-1] if graded_fields else None},
                "the change appears in the audit log",
            )
        )
        cps.append(
            Checkpoint(
                "meta_consistent",
                {},
                "the summary counter equals the number of log entries",
            )
        )
    order = [c["tool"] for c in muts]
    if len(order) >= 2:
        cps.append(
            Checkpoint(
                "order_observable",
                {"order": order},
                f"mutations landed in the order {order}",
            )
        )
    return cps


# --------------------------------------------------------------------------
# executing a trace against a state
# --------------------------------------------------------------------------


def apply_trace(instance: envgen.EnvInstance, trace: list[dict]) -> dict:
    """Apply a sequence of tool calls to the instance's state, returning the result.

    A pure function of ``(instance, trace)``, deliberately: the reference
    solution is graded by applying its trace here, and the agent's work is
    graded by reading the state *it* produced. Both go through this same
    evaluator, so a difference in score cannot come from a difference in how
    the state was interpreted.

    The tool semantics implemented here are the ones the exported environment's
    shell tools implement. Two implementations of one semantics is a real risk,
    and it is bounded rather than ignored: ``tests/test_rsi_envgen.py`` runs a
    reference trace through both and asserts the resulting states are equal.
    """
    state = json.loads(json.dumps(instance.initial_state))
    state.setdefault("_changed", [])
    state.setdefault("_order", [])

    by_id = _records_by_id(state)
    for call in trace:
        tool = call["tool"]
        if tool == "reset_log":
            state["log"] = []
            continue
        if tool == "bulk_update":
            for rec in state["records"]:
                rec[call["field"]] = call["value"]
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
            state["log"].append({"id": "*", "field": call["field"], "value": call["value"]})
            state["meta"]["changed"] = len(state["log"])
            state["_order"].append(tool)
            continue
        if tool == "archive":
            rec = by_id.get(call["id"])
            if rec is not None:
                rec["archived"] = True
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
                state["log"].append({"id": rec["_id"], "field": "archived", "value": True})
                state["meta"]["changed"] = len(state["log"])
                state["_order"].append(tool)
            continue
        if tool == "restore":
            rec = by_id.get(call["id"])
            if rec is not None:
                rec["archived"] = False
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
                state["log"].append({"id": rec["_id"], "field": "archived", "value": False})
                state["meta"]["changed"] = len(state["log"])
                state["_order"].append(tool)
            continue
        if tool == "archive_all":
            # Touches every record, like `bulk_update` — and for the same reason
            # it is *callable but never in a reference*: it violates
            # `target_only` by construction. It is implemented here anyway,
            # because a tool the agent can call must have defined semantics in
            # the evaluator, or a rollout that calls it would crash grading
            # rather than score.
            for rec in state["records"]:
                rec["archived"] = True
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
            state["log"].append({"id": "*", "field": "archived", "value": True})
            state["meta"]["changed"] = len(state["log"])
            state["_order"].append(tool)
            continue
        if tool == "pin":
            rec = by_id.get(call["id"])
            if rec is not None:
                rec["_pin"] = call["value"]
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
                state["log"].append({"id": rec["_id"], "field": "_pin", "value": call["value"]})
                state["meta"]["changed"] = len(state["log"])
                state["_order"].append(tool)
            continue
        if tool.startswith("set_"):
            field_name = tool[len("set_"):]
            rec = by_id.get(call["id"])
            if rec is not None:
                rec[field_name] = call["value"]
                if rec["_id"] not in state["_changed"]:
                    state["_changed"].append(rec["_id"])
                state["log"].append({"id": rec["_id"], "field": field_name, "value": call["value"]})
                state["meta"]["changed"] = len(state["log"])
                state["_order"].append(tool)
            continue
        if tool in ("get_item", "get_ticket", "get_account", "get_sensor", "list_items",
                    "list_tickets", "list_accounts", "list_sensors", "audit",
                    "count_by_sku", "count_by_tid", "count_by_aid", "count_by_sid"):
            continue
        raise ValueError(f"trace names unknown tool {tool!r}")

    state["meta"]["total"] = len(state["records"])
    return state


def reference_trace(instance: envgen.EnvInstance) -> list[dict]:
    """The tool calls that solve the task, derived from the chain.

    Only the chain's *mutations* become calls; the queries are what the agent
    would use to discover the target, and a reference that had to query first
    would be testing the reference's own reading comprehension. The reference is
    allowed to know the target — it is the oracle — and the task is to find out
    whether the *agent* can.

    Two things are derived rather than assumed, and both were bugs when they
    were assumed:

    **The field comes from the tool name.** ``set_qty`` mutates ``qty``; the
    chain decides which mutator it uses, so the field is a property of the trace
    and not of the spec. Recording it on the spec instead made the checkpoints
    grade a field the reference never touched — measured: reference scored 0.8
    instead of 1.0, and the initial state scored 0.4 instead of 0.0.

    **The value must differ from the current one.** Setting a field to the value
    it already holds is a no-op: the state does not change, so the checkpoints
    cannot distinguish a correct solution from doing nothing, and the task
    silently becomes unmeasurable. Measured before this was handled: with
    ``f1`` starting at ``"closed"`` for odd-indexed records, roughly half the
    seeds produced a reference that changed nothing. The value is therefore
    chosen by *reading the target's current value* and picking a different one.

    Returns ``[]`` when the chain contains no mutation, which the caller must
    treat as a generation failure rather than as an empty solution: an empty
    trace produces an empty checkpoint set, and an empty checkpoint set grades
    ``0/0``, which some callers would read as a perfect score.
    """
    target = instance.initial_state["_target"]
    tid = target["id"]
    rec = next((r for r in instance.initial_state["records"] if r["_id"] == tid), {})

    # The target's field values **as the trace will leave them**, updated as the
    # chain is walked. Reading `rec` directly was a bug for every chain that
    # touches one field twice: `('get_item', 'archive', 'restore')` decides
    # whether `restore` changes anything, and with a frozen snapshot of the
    # *initial* state it saw `archived=False`, concluded "not archived, so
    # restoring changes nothing", and skipped — while the `archive` immediately
    # before it had just set it to `True`. Measured: 34 of 60 tasks at
    # `n_steps=2` had a 3-tool chain and a 1-step reference, and the cause was
    # this one stale read.
    #
    # The two tools are not redundant — they are inverses, and a chain that uses
    # both is a legitimate two-step task (archive it, then un-archive it). What
    # was wrong was deciding the second step was a no-op by looking at the state
    # before the first.
    current = {k: v for k, v in rec.items() if not k.startswith("_")}

    trace: list[dict] = []
    for name in instance.chain:
        if name in envgen.NOT_REFERENCE_SAFE:
            # The exclusion list lives in `envgen` because `expand_chain` needs
            # the same one and the import arrow does not point back. A second
            # copy here is what made `n_steps` a fake knob: the chain was built
            # from tools this function then refused to use, so a 4-step chain
            # could yield a 1-step reference while `difficulty()` reported 4.
            continue
        if name.startswith("set_"):
            field_name = name[len("set_"):]
            value = _different_value(current.get(field_name))
            trace.append(
                {
                    "tool": name,
                    "kind": "mutate",
                    "id": tid,
                    "field": field_name,
                    "value": value,
                }
            )
            current[field_name] = value
        elif name == "pin":
            # `_pin` starts at `None` for every record, so unlike `set_*` this
            # step is *always* observable — there is no starting value that makes
            # it a no-op. That is why it exists: a fourth step that could be a
            # no-op half the time would put `n_steps=4` back where it was, with
            # the chain saying 4 and the reference grading 3.
            trace.append(
                {"tool": name, "kind": "mutate", "id": tid, "field": "_pin", "value": "pinned"}
            )
            current["_pin"] = "pinned"
        elif name == "archive":
            if current.get("archived") is True:
                # Already archived *at this point in the chain*, so the call
                # would be a no-op and contribute nothing observable. Skipping
                # is the honest response: the chain step exists but is not a
                # step the reference has to take.
                continue
            trace.append(
                {"tool": name, "kind": "mutate", "id": tid, "field": "archived", "value": True}
            )
            current["archived"] = True
        elif name == "restore":
            if current.get("archived") is not True:
                continue
            trace.append(
                {"tool": name, "kind": "mutate", "id": tid, "field": "archived", "value": False}
            )
            current["archived"] = False
        else:
            # A *query* in the chain is skipped, not an error: the chain records
            # how the environment is entered, and the reference is allowed to
            # know where the target is. A query contributes no mutation, so it
            # contributes no checkpoint — which is exactly why `expand_chain`
            # does not *append* queries and counts mutations as the step budget.
            #
            # An unhandled **mutation** is an error, and raising here is the
            # point. This function used to fall through silently, which is the
            # same failure that made `n_steps` a fake knob twice: a chain tool
            # with no reference call inflates the reported difficulty while the
            # graded solution stays short, and nothing in the output says so.
            # Measured: `touch_meta` sat in the chain and produced no call,
            # giving 24 of 60 tasks a chain of 3 with a reference of 1.
            #
            # A mutation that cannot appear in a reference belongs in
            # `envgen.NOT_REFERENCE_SAFE`, which keeps it out of the chain and
            # leaves it callable by the agent. A mutation that *can* appear needs
            # a branch above. There is no third case.
            if envgen.graph_of(instance.spec).get("kinds", {}).get(name) == "mutate":
                raise ValueError(
                    f"chain mutation {name!r} has no reference semantics; either "
                    f"add a branch here or list it in envgen.NOT_REFERENCE_SAFE"
                )
            continue
    return trace


def _different_value(current) -> str:
    """A value guaranteed to differ from *current*.

    Kept deliberately tiny and total: the point is only that the reference
    changes something, so that "the reference scored 1.0" is evidence the
    checkpoints can see a change rather than evidence they ignore one.
    """
    if current == "closed":
        return "open"
    return "closed"


def grade_state(instance: envgen.EnvInstance, state: dict, checkpoints: list[Checkpoint]) -> float:
    """Fraction of checkpoints satisfied. The reward, and the only reward.

    ``0/0`` returns 0.0 rather than 1.0. That choice is not obvious and matters:
    a task whose checkpoint list is empty is a generation bug, and scoring it 1.0
    would make the bug look like the easiest task in the suite — which is
    precisely the V2 failure (a reward that always fires) arriving through a
    different door.
    """
    if not checkpoints:
        return 0.0
    passed = sum(1 for cp in checkpoints if evaluate(cp, state))
    return passed / len(checkpoints)


def plausible_wrong_state(instance: envgen.EnvInstance, trace: list[dict]) -> dict:
    """The state a plausible *wrong* attempt leaves: everything applied to a distractor.

    This is the discriminator check the module docstring describes. The wrong
    attempt is not random — it is the same trace with the target record replaced
    by an irrelevant one, which is exactly the mistake an agent makes when it
    cannot read the state well enough to tell which record the instruction
    meant. That is the behaviour distractors exist to induce, so it is the
    behaviour the checkpoints must be able to see.

    When the environment has no distractors the function returns the initial
    state unchanged, and the check degenerates to "does the checkpoint set
    reject doing nothing" — already covered by the initial-state check. Callers
    should therefore only assert on this when ``distractor_ids`` is non-empty.
    """
    if not instance.distractor_ids:
        return json.loads(json.dumps(instance.initial_state))
    wrong_id = instance.distractor_ids[0]
    wrong = []
    for call in trace:
        c = dict(call)
        if "id" in c:
            c["id"] = wrong_id
        wrong.append(c)
    return apply_trace(instance, wrong)


# --------------------------------------------------------------------------
# the task
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvTask:
    """A synthesised environment task: instruction, environment, reward, reference.

    The four artifacts SPADE requires, plus the graph so the task can be
    re-derived from its own serialisation.
    """

    instance: envgen.EnvInstance
    instruction: str
    checkpoints: tuple[Checkpoint, ...]
    trace: tuple[dict, ...]
    graph: dict = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        blob = self.instance.instance_id + "|" + json.dumps(
            [c.as_dict() for c in self.checkpoints], sort_keys=True
        )
        return f"env-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

    def as_dict(self) -> dict:
        return {
            "id": self.task_id,
            "flavour": "synthesised_env",
            "env_id": self.instance.spec.env_id,
            "instance_id": self.instance.instance_id,
            "domain": self.instance.spec.domain,
            "instruction": self.instruction,
            "checkpoints": [c.as_dict() for c in self.checkpoints],
            "trace": list(self.trace),
            "initial_state": self.instance.initial_state,
            "chain": list(self.instance.chain),
            "spec": self.instance.spec.as_dict(),
            "graph": self.graph,
            "difficulty": round(self.instance.spec.difficulty(), 3),
        }

    @property
    def checkpoint_count(self) -> int:
        return len(self.checkpoints)


_INSTRUCTIONS = {
    "inventory": (
        "You are operating a warehouse inventory system through its tools. "
        "Find the {unit} that satisfies {selector}, and set its {changes}. Leave "
        "every other {unit} untouched, and make sure the change is recorded so the "
        "audit log and the summary counter stay consistent."
    ),
    "tickets": (
        "You are operating an issue tracker through its tools. Find the {unit} "
        "matching {selector}, and set its {changes}. Do not modify any other "
        "{unit}, and keep the audit log and summary counter consistent with the "
        "change."
    ),
    "accounts": (
        "You are operating a customer account ledger through its tools. Locate the "
        "{unit} where {selector}, and set its {changes}. Leave the other {unit}s "
        "alone and keep the log and counter consistent."
    ),
    "sensors": (
        "You are operating a sensor registry through its tools. Locate the {unit} "
        "where {selector}, and set its {changes}. No other {unit} may be modified, "
        "and the audit log and summary counter must remain consistent."
    ),
}


def _selector_for(instance: envgen.EnvInstance) -> str:
    """A condition that uniquely identifies the target, expressed in domain terms.

    **This is the difference between an environment task and a prompt.** Naming
    the target's id outright makes the task a one-line command: the agent is told
    which record and which value, so it never has to read the state, the tools
    are decoration, and the distractors are unreachable. Measured on the first
    version of this function, which did exactly that: the reference solution was
    `envtool set_bin west-02 closed` and an agent that emitted that single call
    without calling a single query scored 1.0.

    So the target is identified by a *property*, and the property is chosen to be
    unique **by construction** — generation searches for a condition satisfied by
    exactly one record and rejects the instance when none exists. A condition
    that matched two records would make the task ambiguous and therefore
    unsolvable, and it would look merely hard.

    Single fields are tried before pairs, so the common case yields a short
    selector. The pair case exists because a small state has few distinct values
    per field: with 7 records and `archived` boolean, single-field uniqueness
    fails often enough that rejecting those instances would throw away most of
    the draw.

    The id field itself is never used as a selector — it is unique trivially,
    which is the leak this function exists to close.
    """
    tid = instance.initial_state["_target"]["id"]
    records = instance.initial_state["records"]
    spec_fields = envgen._DOMAINS[instance.spec.domain]["fields"]
    # Only non-identity fields. `f0` is the domain's own id field and equals
    # `_id`, so selecting on it would be selecting on the identity.
    candidates = [f for f in spec_fields if f != spec_fields[0]] + ["archived"]

    def matches(rec: dict, cond: dict) -> bool:
        return all(str(rec.get(k)) == str(v) for k, v in cond.items())

    target = next(r for r in records if r["_id"] == tid)

    for f in candidates:
        cond = {f: target.get(f)}
        if sum(1 for r in records if matches(r, cond)) == 1:
            return f"`{f}` is `{target.get(f)}`"

    for i, f1 in enumerate(candidates):
        for f2 in candidates[i + 1:]:
            cond = {f1: target.get(f1), f2: target.get(f2)}
            if sum(1 for r in records if matches(r, cond)) == 1:
                return f"`{f1}` is `{target.get(f1)}` and `{f2}` is `{target.get(f2)}`"

    raise ValueError(
        f"no condition uniquely identifies record {tid!r} among "
        f"{len(records)} records; the task would be ambiguous"
    )


def _instruction_for(instance: envgen.EnvInstance, trace: list[dict]) -> str:
    """The instruction, written from the trace so it cannot contradict it.

    Two things are derived from the trace rather than assumed, and both were
    bugs when they were assumed.

    **The field and value come from the trace.** An instruction naming a field
    the reference does not touch is a task whose stated goal and graded goal
    differ; an agent that follows the instruction fails, and it looks like a
    model failure rather than a generator bug.

    **Every mutation is named, not just the last one.** The checkpoints grade
    *all* of them — ``order_observable`` compares the sequence — so an
    instruction that described only the final mutation would be asking for less
    than it grades. Measured on the first version: the instruction said "set
    `archived` to `True`" while the trace also set `bin`, so a compliant agent
    would score 0.8 and be told it was wrong.

    The target is named by :func:`_selector_for`, not by id. See that function
    for why, and for the measurement that forced it.
    """
    unit = envgen._DOMAINS[instance.spec.domain]["unit"]
    selector = _selector_for(instance)
    muts = [c for c in trace if c.get("kind") == "mutate"]

    # Collapse the trace into the set of field changes it makes, in order,
    # so a chain that sets the same field twice reads as one instruction.
    seen: dict[str, object] = {}
    for call in muts:
        if call.get("field"):
            seen[call["field"]] = call["value"]
    if not seen:
        raise ValueError("trace has no mutation naming a field; cannot write an instruction")

    changes = "; ".join(f"`{f}` to `{v}`" for f, v in seen.items())
    tmpl = _INSTRUCTIONS[instance.spec.domain]
    return tmpl.format(unit=unit, selector=selector, changes=changes)


def generate_env_task(
    *,
    domain: str,
    n_records: int = 4,
    n_distractors: int = 2,
    n_steps: int = 3,
    seed: int = 0,
) -> EnvTask:
    """Synthesise one complete environment task.

    Raises ``ValueError`` in two cases, both of which must fail loudly rather
    than produce a task:

    **The chain contains no mutation.** Such a task has zero checkpoints, grades
    0.0 forever, and looks like the hardest task in the suite.

    **No condition uniquely identifies the target.** The instruction names the
    target by property rather than by id (see :func:`_selector_for`), so if no
    property is unique the instruction is ambiguous and the task unsolvable —
    and again it would look merely hard. Rejecting is what keeps the
    environment-task property ("the agent must read the state") from silently
    degrading back into a prompt task.
    """
    spec = envgen.synthesise_env(
        domain=domain, n_records=n_records, n_distractors=n_distractors, n_steps=n_steps, seed=seed
    )
    instance = envgen.materialise(spec)
    trace = reference_trace(instance)
    if not any(c.get("kind") == "mutate" for c in trace):
        raise ValueError(
            f"chain {instance.chain} contains no mutation, so the task would have "
            f"no checkpoints; increase n_steps or change the seed"
        )

    # The chain's mutation count must equal the reference's. A shortfall means a
    # chain step was a **no-op on this particular target**, so the reference
    # silently dropped it — and `spec.difficulty()` would then report a longer
    # task than the one being graded. This is the *third* form of the same bug,
    # and the first two were each fixed by chasing a specific cause; this check
    # closes the class rather than an instance.
    #
    # The case that motivated it: `restore` is a no-op when the target starts
    # un-archived, and whether it starts un-archived is a property of the
    # *materialised instance*, not of the spec — `expand_chain` runs before the
    # records exist, so it cannot know. Measured: 33 of 60 tasks at `n_steps=2`
    # had a 3-mutation chain and a 1-step reference.
    #
    # Rejecting is right rather than repairing, because there is no honest
    # repair: shortening the chain would misreport difficulty (the bug), and
    # lengthening the reference would invent a step the chain did not ask for.
    # `generate_env_batch` skips a rejected draw, so the batch stays a pure
    # function of its seed — and the rejection rate is now *visible* in
    # `summarise_env_batch` instead of being absorbed into a wrong number.
    # Counted from the graph, not from name prefixes: `archive` and `restore`
    # are mutations whose names say nothing about it, and inferring the kind
    # from a prefix is how the reference and the chain came to disagree in the
    # first place.
    kinds = instance.spec and envgen.graph_of(instance.spec).get("kinds", {})
    chain_mutations = sum(
        1 for n in instance.chain
        if n not in envgen.NOT_REFERENCE_SAFE and kinds.get(n) == "mutate"
    )
    if len(trace) != chain_mutations:
        raise ValueError(
            f"chain has {chain_mutations} mutation(s) but the reference has "
            f"{len(trace)}: a step is a no-op on this target (chain "
            f"{instance.chain}); rejecting rather than reporting difficulty "
            f"{len(instance.chain)} for a {len(trace)}-step task"
        )

    instruction = _instruction_for(instance, trace)
    checkpoints = build_checkpoints(instance, trace=trace)
    if not checkpoints:
        raise ValueError("checkpoint construction produced an empty set")
    return EnvTask(
        instance=instance,
        instruction=instruction,
        checkpoints=tuple(checkpoints),
        trace=tuple(trace),
        graph=envgen.graph_of(spec),
    )


def generate_env_batch(n: int, *, seed: int = 0, **pins) -> list[EnvTask]:
    """``n`` distinct environment tasks. Ids are deduplicated, not overwritten.

    Same contract as ``task_gen.generate_batch``, and for the same reason: a
    collision that silently shrinks the batch is a curriculum bug wearing the
    costume of a scheduling detail. A draw that raises (no mutation in the
    chain) is skipped rather than retried with a different seed, so the batch
    stays a pure function of the requested seed.
    """
    rng = random.Random(seed)
    domains = sorted(envgen._DOMAINS)
    out: list[EnvTask] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        params = dict(pins)
        params.setdefault("domain", rng.choice(domains))
        params.setdefault("seed", rng.randrange(2**31))
        try:
            task = generate_env_task(**params)
        except ValueError:
            continue
        if task.task_id in seen:
            continue
        seen.add(task.task_id)
        out.append(task)
    return out


def summarise_env_batch(tasks: list[EnvTask]) -> dict:
    """Coverage report for a synthesised environment batch."""
    by_domain: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    cp_counts: list[int] = []
    for t in tasks:
        d = t.instance.spec.domain
        by_domain[d] = by_domain.get(d, 0) + 1
        for e in t.graph.get("edges", []):
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
        cp_counts.append(t.checkpoint_count)
    return {
        "total": len(tasks),
        "by_domain": by_domain,
        "edges_by_reason": by_reason,
        "distinct_ids": len({t.task_id for t in tasks}),
        "checkpoints_min": min(cp_counts) if cp_counts else 0,
        "checkpoints_max": max(cp_counts) if cp_counts else 0,
        "checkpoints_mean": round(sum(cp_counts) / len(cp_counts), 2) if cp_counts else 0.0,
        "with_distractors": sum(1 for t in tasks if t.instance.distractor_ids),
    }
