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

"""Synthesising *stateful environments*, not task strings.

Why this module exists
----------------------
``task_gen.py`` produces tasks by sampling a parameter vector and emitting a
prompt. That is a *task* generator. It has a property that turns out to be a
limitation the moment you try to validate a claim about generality: every task
it emits is solvable in **one shell command**, and the only thing the agent's
action changes is the contents of one file.

That is enough to measure "can the model quote a string across a harness
boundary". It is not enough to measure anything about *environments*, because
there is no environment — there is a string and a file.

This module adds the missing layer. An environment here is a **stateful system**
the agent acts on through named tools, following EnvScaler's decomposition
(arXiv:2601.05808, §2):

    E = { F_exec, E_doc, Σ_tool }

    F_exec    the executable program: state attributes + tool methods
    E_doc     the documentation the agent reads
    Σ_tool    the tool interface set: names, parameters, descriptions

and following ScaleEnv's insight (arXiv:2602.06820, §4.1.3) that the tools are
not independent — they are nodes in a **tool dependency graph**, whose edges are
derived from three concrete relations:

    data flow        an earlier tool's output is a later tool's argument
    pre/post-condition  one tool requires the state another tool establishes
    state dependency    two tools write the same state key

A task is then a *path through that graph*, not a parameter vector. That is the
difference that matters for this repository: a graph path has a natural notion
of "what the agent must have done", and the state it leaves behind is something
a verifier can inspect independently of how the agent got there.

Why the state is the thing to grade
-----------------------------------
EnvScaler's second component, ScenGenerator, decomposes a task into K
checkpoints and grades on the **final state**, with reward = the fraction of
checkpoints satisfied. Two properties make this the right choice here:

*Process-agnostic.* A final-state check accepts every path that reaches the
goal. Grading the tool-call sequence instead would fail a correct agent that
took a different route, and would do so invisibly — the task would just look
harder. This repository already holds that the *verifier must be
harness-agnostic*; grading a state is the same commitment applied one level up.

*Partial credit is real.* A boolean gives GRPO a group of identical zeros
whenever the model is close but not finished, which is exactly the regime a 0.5B
model lives in. K checkpoints give the group something to differentiate on.
``stats.py`` computes the group signal as ``1 - p^G - (1-p)^G``; a reward that
can only take 0 or 1 pushes ``p`` toward the ends of that curve where the signal
vanishes.

The failure mode this must avoid
--------------------------------
The GEF survey (arXiv:2511.09586, §5.2) names the hazard: **generator-verifier
asymmetry**. If the generator is weaker than the verifier, it emits tasks that
look solvable and are not. Here the generator *is* the verifier — both are this
module — so the asymmetry takes a specific, checkable form: the reference
solution and the checkpoint functions are two independent derivations of the
same final state, and if they disagree the environment is inconsistent.

That is what gate V1 becomes in this module, and it is stronger than the string
version: a string task can only check that two derivations of one value agree,
whereas a stateful task checks that the *entire state* the reference produces
satisfies the checkpoints. ``tests/test_rsi_envgen.py`` asserts the disagreement
is caught rather than papered over.

Determinism and the dependency-free rule
----------------------------------------
Every function is a pure function of a seed. The core is stdlib-only, enforced
by the ``core`` job in CI, so this module imports nothing outside it — no
numpy, no network, no model. Synthesis is therefore reproducible in the same
sense ``task_gen`` is, and CI can re-derive an environment and compare it
against the one that was exported.

What is *not* claimed
---------------------
The environments here are small and their tools are shell-level. They are not
AppWorld, and the tool count (3-6) is far below the 18.58 average EnvScaler
reports. The claim is narrower and is the one this repository needs: that a
generated task can carry **real state with a real dependency structure**, so
that a cross-harness gap measured on it is a statement about the environment
rather than about string quoting.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, replace

__all__ = [
    "TOOL_KINDS",
    "EnvSpec",
    "ToolSpec",
    "EnvInstance",
    "synthesise_env",
    "materialise",
    "reparameterise_env",
    "dependency_edges",
    "build_dependency_graph",
    "graph_of",
    "expand_chain",
    "summarise_env",
    "DEFAULT_MIN_SUBGRAPH",
]

#: Query tools read state; mutation tools change it. The split is not cosmetic:
#: a task that only queries has no final state to grade, and a task that only
#: mutates has no way for the agent to discover what to mutate. Every generated
#: environment carries both, and ``synthesise_env`` asserts it.
TOOL_KINDS = ("query", "mutate")

#: ScaleEnv's minimum subgraph size (``|H_n| >= 20``, §4.2.2), which exists so
#: that the agent's exploration space is not trivially exhaustible. That number
#: assumes a 50-tool environment; here the tools are shell operations on a small
#: state, so the analogue is a minimum number of *reachable states*, not tools.
#: Kept as a named constant so the difference from ScaleEnv is visible rather
#: than being an unexplained smaller number.
DEFAULT_MIN_SUBGRAPH = 8

#: State keys every generated environment has. ``records`` is the collection
#: being operated on, ``log`` is an append-only audit trail, and ``meta`` holds
#: scalars the rules constrain. Named here because the checkpoints reference
#: them and a typo would silently produce an unsatisfiable checkpoint, and
#: because the chain expansion needs to know which keys exist from the start.
#:
#: These are the *top-level* keys. The per-record fields live under ``records``
#: and are not addressable as state keys, which is why ``_tools_for`` declares
#: only top-level reads and writes: a tool that reads "one record's qty" is
#: reading ``records``, and pretending otherwise would put edges in the graph
#: that no traversal could ever satisfy.
_STATE_KEYS = ("records", "log", "meta")

#: Domain vocabulary. The words are deliberately neutral: an environment about
#: "inventory" and one about "tickets" have the same structure, and pretending
#: otherwise would be decoration. What differs between two environments is the
#: *graph*, not the nouns.
_DOMAINS = {
    "inventory": {
        "unit": "item",
        "fields": ("sku", "qty", "bin"),
        "doc": "A warehouse inventory system. Items are tracked by SKU with a quantity and a bin location.",
    },
    "tickets": {
        "unit": "ticket",
        "fields": ("tid", "state", "owner"),
        "doc": "An issue tracker. Tickets have an id, a state, and an owner.",
    },
    "accounts": {
        "unit": "account",
        "fields": ("aid", "balance", "tier"),
        "doc": "A customer account ledger. Accounts have an id, a balance, and a tier.",
    },
    "sensors": {
        "unit": "sensor",
        "fields": ("sid", "reading", "zone"),
        "doc": "A sensor registry. Sensors have an id, a reading, and a zone.",
    },
}

_WORDS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "north", "south", "east", "west", "upper", "lower", "inner", "outer",
)


# --------------------------------------------------------------------------
# the specification: what a synthesised environment is
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One tool in the interface set ``Σ_tool``.

    ``reads`` and ``writes`` are the state keys the tool touches. They are
    declared rather than inferred because the dependency graph is *built* from
    them: inferring them by parsing the implementation would make the graph a
    function of the generator's formatting rather than of its design, and a
    refactor that changed nothing would silently rewire the graph.
    """

    name: str
    kind: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    doc: str
    #: Parameters the tool takes, in order. A tool with none is a pure reader.
    params: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "params": list(self.params),
            "doc": self.doc,
        }


@dataclass(frozen=True)
class EnvSpec:
    """A synthesised environment: ``E = {F_exec, E_doc, Σ_tool}``.

    Frozen for the same reason ``TaskParams`` is: the environment recorded in a
    task must be the one that produced it, or a re-derivation cannot be compared
    against the artifact.
    """

    domain: str
    #: Difficulty knobs, and the reason this is a *search* space rather than a
    #: fixed template. See :func:`synthesise_env` for what each one does.
    n_records: int
    n_distractors: int
    n_steps: int
    seed: int
    tools: tuple[ToolSpec, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "n_records": self.n_records,
            "n_distractors": self.n_distractors,
            "n_steps": self.n_steps,
            "seed": self.seed,
            "tools": [t.as_dict() for t in self.tools],
        }

    @property
    def env_id(self) -> str:
        """Stable id from the spec, so a re-run reproduces the same environment."""
        blob = "|".join(
            f"{k}={v}"
            for k, v in sorted(self.as_dict().items())
            if k != "tools"
        ) + "|tools=" + ",".join(t.name for t in self.tools)
        return f"env-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"

    def difficulty(self) -> float:
        """A display coordinate, not a claim that the knobs are commensurable.

        Same caveat as ``TaskParams.difficulty``: the figures plot pass rate
        against this to show a trend, and the trend is what matters.
        """
        parts = [
            min(1.0, (self.n_records - 2) / 10.0),
            min(1.0, self.n_distractors / 8.0),
            min(1.0, (self.n_steps - 1) / 3.0),
            min(1.0, len(self.tools) / 6.0),
        ]
        return sum(parts) / len(parts)


# --------------------------------------------------------------------------
# the dependency graph
# --------------------------------------------------------------------------


def dependency_edges(tools: tuple[ToolSpec, ...]) -> list[dict]:
    """Directed edges between tools, with the *reason* for each edge recorded.

    ScaleEnv derives edges from three relations (§4.1.3). All three are
    implementable over the declared read/write sets, and keeping them distinct
    rather than collapsing them into "they are related" is what makes the graph
    explainable: a task built along a ``data_flow`` edge means something
    different from one built along a ``state`` edge, and a reader can tell which
    by looking at the edge.

    ``data_flow``
        ``a`` writes a key that ``b`` reads **and** ``b`` writes a key that
        ``a`` reads. The pair feeds each other, so neither order is privileged:
        ``a`` then ``b`` and ``b`` then ``a`` are both executable, and the two
        produce different states. This is the mutual case, and it is the one
        that makes ordering a real choice rather than a forced one.
    ``state``
        Neither reads what the other writes, but both write the same key. Pure
        *contention*: the tools do not feed each other, they compete over one
        field, so order is observable in the final state even though no data
        passes.
    ``precondition``
        ``a`` reads what ``b`` writes, and ``b`` does **not** read what ``a``
        writes. A one-way dependency, so exactly one order is executable. This
        is the edge type that makes a chain unsatisfiable if reversed, and it is
        the strongest structural statement of the three.

    Precedence between them is load-bearing, and both earlier versions of this
    function got it wrong in opposite directions:

    * Testing ``a.writes & b.reads`` first made ``state`` **unreachable**: every
      base mutator reads and writes ``records``, so any two of them matched the
      flow test and contention was never reached. Measured: 17 edges, 0 of them
      ``state``.
    * Testing flow *in either direction* first then made ``precondition``
      **unreachable**, for the mirror reason: the one-way cases also satisfy the
      either-direction test, so they were all absorbed. Measured: 24 edges, 0 of
      them ``precondition``.

    The rule below is therefore a *partition*, not an if-chain over overlapping
    predicates: mutual flow, else one-way flow, else shared-write contention.
    ``tests/test_rsi_envgen.py`` asserts all three reasons occur on a real
    environment, which is the assertion that would have caught both bugs.

    Self-edges are excluded: a tool that reads and writes the same key is
    internally consistent, not dependent on itself.
    """
    edges: list[dict] = []
    for a in tools:
        for b in tools:
            if a.name == b.name:
                continue
            a_to_b = bool(set(a.writes) & set(b.reads))
            b_to_a = bool(set(b.writes) & set(a.reads))
            if a_to_b and b_to_a:
                reason = "data_flow"
            elif a_to_b or b_to_a:
                reason = "precondition"
            elif set(a.writes) & set(b.writes):
                reason = "state"
            else:
                reason = None
            if reason:
                edges.append({"from": a.name, "to": b.name, "reason": reason})
    return edges


def build_dependency_graph(spec: EnvSpec) -> dict:
    """The tool dependency graph ``G`` as an adjacency structure plus its edges.

    Adjacency is materialised rather than recomputed at each traversal because
    the expansion walks it repeatedly; more importantly, materialising it means
    the graph is an *artifact* that can be dumped and diffed, which is what lets
    a test assert the graph did not change when only the formatting did.
    """
    edges = dependency_edges(spec.tools)
    adj: dict[str, list[str]] = {t.name: [] for t in spec.tools}
    for e in edges:
        adj[e["from"]].append(e["to"])
    for k in adj:
        adj[k] = sorted(set(adj[k]))
    return {"nodes": [t.name for t in spec.tools], "edges": edges, "adj": adj}


# --------------------------------------------------------------------------
# expansion: turning a seed chain into a task-sized subgraph
# --------------------------------------------------------------------------


#: Tools that may be *called* but can never appear in a reference solution.
#:
#: Declared here rather than in ``envtask`` because ``expand_chain`` needs the
#: same list and the import arrow points ``envtask -> envgen``, not back. Two
#: copies would be worse than an import: the chain would be built from tools the
#: reference then refuses to use, which is a silent difficulty bug rather than a
#: crash — see :func:`expand_chain`.
#:
#: ``bulk_update`` and ``archive_all`` touch every record, so they violate the
#: ``target_only`` checkpoint by construction; a reference that fails its own
#: checkpoints is worse than no reference. ``reset_log`` destroys the evidence
#: the other checkpoints read, so a chain ending in it scored the reference 0.4.
#:
#: All three stay in ``Σ_tool``: calling them is a *mistake the checkpoints
#: catch*, which is the useful place for a distractor action.
NOT_REFERENCE_SAFE = {
    "bulk_update": "touches every record, violating `target_only` by construction",
    "archive_all": "touches every record, violating `target_only` by construction",
    "reset_log": "destroys the log the other checkpoints read",
}

#: How many mutation steps a task can ask for, and why it is finite.
#:
#: ``expand_chain`` never repeats a tool and forbids two steps from writing the
#: same field, so the longest chain equals the number of reference-safe mutators
#: that reach **distinct** fields. The set has five such mutators — ``set_<f1>``,
#: ``set_<f2>``, ``archive``, ``restore``, ``pin`` — but ``archive`` and
#: ``restore`` both write ``archived`` and are mutually cancelling, so the
#: reachable maximum is **four**.
#:
#: The number is a *measurement*, not a design choice, and it moved twice while
#: the rules above were being fixed:
#:
#: * with three usable mutators, ``n_steps=3`` and ``n_steps=4`` produced
#:   byte-identical distributions (60/60 at chain length 3), which is how the
#:   discrepancy showed up;
#: * after the same-field rule was added, a ``reset_<f2>`` extra could not raise
#:   the ceiling at all — measured: ``n_steps=4`` succeeded on 8 of 60 seeds and
#:   its successes still had a 3-mutation chain. It was replaced by ``pin``,
#:   which writes a field nothing else touches.
#:
#: This is a real ceiling rather than a tuning choice: allowing repeats or
#: same-field pairs would let a chain set one field three times, which is one
#: task with two redundant steps, not a three-step task.
MAX_BASE_STEPS = 4


#: Mutators that are inverses of each other, so a chain containing both would
#: have a step that cancels the previous one.
#:
#: Not the same thing as ``NOT_REFERENCE_SAFE``: both tools are perfectly
#: reference-safe on their own, and a chain of exactly one of them is a valid
#: task. The problem is the *pair*. ``restore`` is a no-op when the target is
#: not archived, and whether it is archived depends on the materialised records
#: — which do not exist when ``expand_chain`` runs, since expansion happens on
#: the spec. So a chain like ``('get_item', 'archive', 'restore')`` is not
#: statically detectable as a no-op; the only place that can tell is
#: ``envtask.reference_trace``, which raises, and
#: ``envtask.generate_env_task`` then rejects the draw.
#:
#: Excluding the pair here turns a ~45% rejection rate into a small one, and it
#: does so for a structural reason rather than by tuning: a task whose solution
#: is "archive it, then un-archive it" is one task with a redundant step, which
#: is the same thing ``MAX_BASE_STEPS`` exists to prevent.
MUTUALLY_CANCELLING = (
    frozenset({"archive", "restore"}),
)


def expand_chain(
    graph: dict,
    seed: tuple[str, ...],
    *,
    max_steps: int,
    rng: random.Random,
    present: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Grow a seed tool chain into a longer one, dependency-aware.

    This is ScaleEnv's Dependency-Aware BFS (§4.2.2) in the small. The rule that
    matters is the *admissibility* condition: a tool may be appended only when
    every key it reads is available — either written by something already in the
    chain, or **already present in the environment's initial state**.

    That second clause is not a detail, and leaving it out was a bug that
    silently ignored the difficulty knob. The initial state always has
    ``records``, ``log`` and ``meta`` (see :func:`materialise`); a chain that
    tracks only what *it* has written therefore sees an empty available-set
    after a query, refuses every mutator, and dead-ends at length 2. Measured:
    ``n_steps=4`` produced a 2-step chain, with nothing in the output saying the
    parameter had been dropped. Passing ``present`` is what makes the expansion
    able to use state the environment starts with, which is the whole point of
    an environment having an initial state.

    **Only reference-safe tools are admissible.** This is the second bug in the
    same function, and it was the one that made ``n_steps`` a *fake* knob. A
    chain could be filled with ``reset_log`` and ``bulk_update`` — both of which
    :func:`envtask.reference_trace` refuses to use — so the chain grew to the
    requested length while the reference solution stayed at one or two steps.
    Measured on 60 tasks at ``n_steps=4``: 26 (43%) had a 4-step chain and a
    **1-step** reference; mean reference length was 1.58. Every downstream
    number inherited the lie, because ``spec.difficulty()`` is computed from the
    chain and ``checkpoints`` are built from the trace — the difficulty report
    said 4 and the task was 1.

    Returns the chain unchanged when no admissible extension exists. That is a
    real outcome rather than a failure — it means the seed is maximal — and the
    caller distinguishes it by comparing lengths.

    **Only mutators are appended, and ``max_steps`` counts mutations.** This is
    the third and last bug in this function, and it is the one that made the
    difficulty knob wrong *twice over*. Appending queries was not obviously
    wrong — a real agent does query — but a query contributes no reference step
    and no checkpoint, so a chain of ``Q M Q M`` was "4 steps" that graded a
    1-step solution. Measured on 60 tasks at ``n_steps=4``, after the
    ``NOT_REFERENCE_SAFE`` fix had already been applied: 32% still had a
    **1-step** reference, and the dominant shapes were ``QMQM`` (23) and
    ``QMQQ`` (19) — chains padded with readers.

    Queries are not removed from the *environment*: they stay in ``Σ_tool``,
    and they are what the agent must call to find the target. They are removed
    from the *chain*, because the chain describes the solution path, and the
    reference is allowed to know where it is going. A query in the chain was
    therefore never a step the reference took — it was bookkeeping that inflated
    the step count.

    The seed is **kept in the chain even when it is a query**, because the chain
    also records how the environment is entered — but ``reference_trace``
    handles that case by skipping queries, not by raising. See below for why
    those are different.
    """
    chain = list(seed)
    written = set(_writes_of(graph, chain)) | set(present)
    kinds = graph.get("kinds", {})

    def n_mutations(names: list[str]) -> int:
        """How many chain entries are mutations. This is the step count."""
        return sum(1 for n in names if kinds.get(n) == "mutate")

    guard = 0
    while n_mutations(chain) < max_steps and guard < max_steps * 8:
        guard += 1
        candidates = []
        for name in graph["nodes"]:
            if name in chain:
                continue
            if name in NOT_REFERENCE_SAFE:
                # Excluded from the chain, not merely from the reference. See
                # the docstring: admitting them inflates the chain length while
                # leaving the reference short, which is a difficulty report that
                # does not describe the task.
                continue
            if kinds.get(name) != "mutate":
                # Only mutations count as steps. A query would occupy a slot in
                # the chain without contributing a reference call or a
                # checkpoint, which is how `n_steps` came to be reported as 4
                # while grading a 1-step solution.
                continue
            if any(name in pair and (pair & set(chain)) for pair in MUTUALLY_CANCELLING):
                # Its inverse is already in the chain, so this step would undo
                # the previous one. See `MUTUALLY_CANCELLING`.
                continue
            # No two steps may write the same field. Two tools that reach the
            # same field are legitimate *alternatives* — the agent may use
            # either — but a chain containing both has a second step that
            # overwrites the first, which is a redundant step wearing the
            # costume of a longer task. This is the general form of the
            # `archive`/`restore` rule above: that pair is just the one case
            # where the shared field is obvious from the names.
            #
            # The field is taken from the tool's writes *minus* the structural
            # keys every mutator touches (`records`, `log`, `meta`), because
            # those are the state's containers rather than a field of a record.
            if _touched_fields(graph, name) & _touched_fields(graph, chain):
                continue
            if set(_reads_of(graph, [name])) <= written:
                candidates.append(name)
        if not candidates:
            break

        # Draw freely among admissible mutators. The old "prefer a mutator while
        # there is none" rule existed because uniform choice over *all* tools
        # filled the chain with readers; with readers no longer admissible the
        # rule has nothing left to do, and keeping it would bias the first
        # mutation without affecting any later one.
        pick = rng.choice(sorted(candidates))
        chain.append(pick)
        written |= set(_writes_of(graph, [pick]))
    return tuple(chain)


#: The graph artifact carries names, not full ToolSpecs, so the read/write sets
#: have to be recoverable from it. They are attached at build time under these
#: keys rather than looked up in a side table, because a side table is one more
#: thing that can get out of sync with the graph it describes.
def _reads_of(graph: dict, names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        out.extend(graph.get("reads", {}).get(n, ()))
    return out


def _writes_of(graph: dict, names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        out.extend(graph.get("writes", {}).get(n, ()))
    return out


#: The keys that are the state's *containers* rather than a field of a record.
#: Every mutator here writes at least one of them, so comparing raw ``writes``
#: sets would say every pair of mutators collides and the rule below would
#: exclude everything.
_STRUCTURAL_KEYS = frozenset({"records", "log", "meta"})


def _touched_fields(graph: dict, names: str | list[str] | tuple[str, ...]) -> set[str]:
    """The *record fields* a tool (or a chain of tools) writes.

    Declared read/write sets are top-level state keys — see ``_STATE_KEYS`` for
    why per-record fields are not addressable — so ``writes`` alone cannot tell
    ``set_qty`` from ``archive``: both write ``{"records", "log"}``. What
    distinguishes them is *which record field* they reach, and that is only
    recoverable from the tool name, which is why this function exists rather
    than being folded into the read/write sets.

    The structural keys are subtracted so that two tools touching disjoint
    fields do not look like they collide. Returns an empty set for a tool whose
    name matches no known field pattern, which is the safe direction: an
    unrecognised tool is never excluded from a chain on this rule's account.

    Used by :func:`expand_chain` to keep two steps from writing the same field,
    and by ``envtask`` to decide which field a chain's checkpoints grade.
    """
    if isinstance(names, str):
        names = [names]
    fields: set[str] = set()
    for n in names:
        for key in graph.get("writes", {}).get(n, ()):
            if key in _STRUCTURAL_KEYS:
                continue
            fields.add(key)
        # `writes` for these tools is `{"records", "log"}`, so the record field
        # has to come from the name. The patterns are the ones `_tools_for`
        # emits; a new tool must be added here or it will read as field-less.
        if n in ("archive", "restore"):
            fields.add("archived")
        elif n == "bulk_update":
            fields.add("*")
        elif n in ("pin", "unpin"):
            fields.add("_pin")
        elif n.startswith("set_") or n.startswith("reset_"):
            fields.add(n.split("_", 1)[1])
    return fields


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def _tools_for(domain: str) -> tuple[ToolSpec, ...]:
    """Build ``Σ_tool`` for a domain.

    The set is **fixed per domain**, not drawn. Drawing it was a bug that took a
    probe to see: ``extras`` were ``rng.shuffle``-d and sliced, so the *interface*
    varied with the seed — and since the chain-length ceiling is a property of the
    interface, ``synthesise_env``'s bound check accepted ``n_steps=4`` for some
    seeds and refused it for others. Measured: ``n_steps=5`` was refused with
    "3 usable mutators" 27 times and "4 usable mutators" 33 times out of 60.
    A difficulty knob whose *legality* depends on an unrecorded draw is the same
    defect as one whose *value* does, which is the bug this module keeps meeting.

    Fixing the set also buys comparability: every environment in a domain has the
    same shape of interface, so a difference in reward is a difference in the
    state and the chain rather than in which tools happened to exist.
    """
    d = _DOMAINS[domain]
    unit, fields = d["unit"], d["fields"]
    f0, f1, f2 = fields

    base = [
        ToolSpec(
            name=f"list_{unit}s",
            kind="query",
            reads=(),
            writes=(),
            params=(),
            doc=f"List every {unit} currently in the system.",
        ),
        ToolSpec(
            name=f"get_{unit}",
            kind="query",
            reads=("records",),
            writes=(),
            params=("id",),
            doc=f"Read one {unit} by id.",
        ),
        ToolSpec(
            name=f"set_{f1}",
            kind="mutate",
            reads=("records",),
            writes=("records", "log"),
            params=("id", "value"),
            doc=f"Set the {f1} field of one {unit}, recording the change.",
        ),
        ToolSpec(
            name="audit",
            kind="query",
            reads=("records", "log"),
            writes=(),
            params=(),
            doc="Return the full change log alongside the current records.",
        ),
        ToolSpec(
            name=f"set_{f2}",
            kind="mutate",
            reads=("records", "meta"),
            writes=("records", "meta"),
            params=("id", "value"),
            doc=f"Set the {f2} field of one {unit} and update the summary counter.",
        ),
        # A pure writer: reads nothing, writes `log`. Two things depend on it.
        #
        # It makes the `state` edge reason reachable — every other mutator here
        # both reads and writes `records`, so pairs of them are always
        # `data_flow` and pure contention over one field never gets tested. See
        # `dependency_edges`.
        #
        # It is also the tool that keeps chain expansion alive. Expansion
        # admits a tool only when everything it reads has already been written,
        # and the base set starts from a *query* (which writes nothing), so the
        # first admissible extension has to be something that needs no prior
        # state. Without such a tool the chain dead-ends at length 2 no matter
        # what `n_steps` asked for — measured before this tool existed: a
        # request for `n_steps=4` produced a 2-step chain, silently.
        ToolSpec(
            name="reset_log",
            kind="mutate",
            reads=(),
            writes=("log",),
            params=(),
            doc="Clear the change log. Requires no prior state.",
        ),
        # A third reference-safe mutator, and the reason it is needed is a
        # *measured* ceiling rather than symmetry. `expand_chain` admits only
        # mutators and never repeats one, so the longest possible chain equals
        # the number of reference-safe mutators — with two of them, `n_steps=2`
        # and `n_steps=3` produced byte-identical distributions (60/60 at chain
        # length 3) and the knob looked broken above 3.
        #
        # `archive` touches only the target record, so unlike `bulk_update` it
        # does not violate `target_only` and can appear in a reference. It
        # writes `records` and `log`, which makes it participate in the graph
        # rather than sitting outside it.
        ToolSpec(
            name="archive",
            kind="mutate",
            reads=("records",),
            writes=("records", "log"),
            params=("id",),
            doc=f"Mark one {unit} as archived, recording the change.",
        ),
        # The mirror of `archive`. It is the fourth reference-safe mutator, and
        # it is only *ever* usable because `archived` starts randomised — with a
        # constant `False` this tool could not change anything, so it would be a
        # chain slot that produces no reference call, which is the failure mode
        # `envtask.reference_trace` now raises on rather than absorbs.
        ToolSpec(
            name="restore",
            kind="mutate",
            reads=("records",),
            writes=("records", "log"),
            params=("id",),
            doc=f"Clear the archived flag on one {unit}, recording the change.",
        ),
    ]
    # `archive` moved into `base`: it is reference-safe, and the chain length
    # ceiling equals the number of reference-safe mutators. Leaving it here as
    # an optional extra made the ceiling depend on the draw, so the same
    # `n_steps` produced different maximum difficulties for different seeds.
    extras = [
        ToolSpec(
            name="bulk_update",
            kind="mutate",
            reads=("records",),
            writes=("records", "log", "meta"),
            params=("field", "value"),
            doc="Apply one field change to every record.",
        ),
        ToolSpec(
            name=f"count_by_{f0}",
            kind="query",
            reads=("records",),
            writes=(),
            params=(),
            doc=f"Count records grouped by {f0}.",
        ),
        ToolSpec(
            name="archive_all",
            kind="mutate",
            reads=("records", "meta"),
            writes=("records", "log", "meta"),
            params=(),
            doc="Archive every record that is still open, logging each one.",
        ),
        # A reference-safe extra, and the only one here that is. It exists to
        # raise the chain-length ceiling: the base set tops out at three usable
        # mutators (see `MAX_BASE_STEPS`), so without it a request for
        # `n_steps=4` is refused.
        #
        # It writes `_pin`, a field **no other tool reaches**, and that is the
        # point rather than a detail. The obvious alternative was a
        # `reset_<f2>` that writes the same field `set_<f2>` does — and that
        # tool cannot raise the ceiling at all, because `expand_chain` forbids
        # two steps writing one field (the second would overwrite the first, so
        # it is a redundant step wearing the costume of a longer task).
        # Measured with it in place: `n_steps=4` succeeded on 8 of 60 seeds and
        # the successes still had a 3-mutation chain.
        #
        # A dedicated field is what makes the fourth step *real*: `_pin` is in
        # the initial state, is readable by `audit`, and changes exactly once.
        ToolSpec(
            name="pin",
            kind="mutate",
            reads=("records", "meta"),
            writes=("records", "meta"),
            params=("id", "value"),
            doc=f"Attach a pin note to one {unit}, recording it against the counter.",
        ),
    ]
    return tuple(base + extras)


def synthesise_env(
    *,
    domain: str,
    n_records: int = 4,
    n_distractors: int = 0,
    n_steps: int = 3,
    seed: int = 0,
) -> EnvSpec:
    """Draw one environment specification.

    The knobs, and why these three:

    ``n_records``
        How many entities the state holds. More records means more to read and
        more places for a change to be applied to the wrong one — the analogue
        of EnvScaler's state-category count, which averages 21.38.
    ``n_distractors``
        Records that are present, valid, and **irrelevant to the task**. This is
        ScaleEnv's distractor injection (§4.2.1), and it is the single most
        important knob for this repository. Without distractors, "find the right
        record" is "the only record", and a policy can succeed by acting on
        everything. With them, an agent that cannot read the state has to guess,
        and guessing is exactly the behaviour a cross-harness gap should detect.
        The density is scaled by the caller, not fixed, which is the paper's
        "dynamically scaled according to predefined task complexity".
    ``n_steps``
        How long the chain is, counted in **mutations** — a query contributes
        no reference step and no checkpoint, so it does not count. Controls the
        depth of the tool dependency path the task traverses.

        Bounded by the number of reference-safe mutators, which is why the
        request is checked rather than clamped: see :data:`MAX_BASE_STEPS`.
    """
    if domain not in _DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; known: {sorted(_DOMAINS)}")
    if n_records < 1:
        raise ValueError("n_records must be >= 1")
    if n_distractors < 0:
        raise ValueError("n_distractors must be >= 0")

    tools = _tools_for(domain)
    # The bound is computed from the tools themselves rather than read off a
    # constant, and it is now a *pure function of the domain* — the tool set is
    # fixed per domain, so the same `n_steps` is legal for every seed. That was
    # not true before: `extras` were shuffled and sliced, so the ceiling varied
    # with the draw and `n_steps=5` was refused for "3 usable mutators" on 27
    # seeds and "4 usable mutators" on 33. A knob whose legality depends on an
    # unrecorded draw is the same defect as one whose value does.
    #
    # The mutually-cancelling pairs are subtracted, since a chain never contains
    # both members: counting them as two available steps overstates the ceiling
    # by exactly one.
    n_safe = sum(1 for t in tools if t.kind == "mutate" and t.name not in NOT_REFERENCE_SAFE)
    for pair in MUTUALLY_CANCELLING:
        if all(any(t.name == name for t in tools) for name in pair):
            n_safe -= len(pair) - 1
    if n_steps > n_safe:
        # Refused, not clamped. Clamping is what produced the original bug in
        # three different forms: `n_steps=9` silently became a 4-step chain, and
        # `difficulty()` reported 9. A caller asking for more steps than the
        # interface can express has a question about the interface, not about
        # this draw.
        raise ValueError(
            f"n_steps={n_steps} exceeds this interface's {n_safe} usable "
            f"reference-safe mutator(s); a chain cannot repeat a tool or write "
            f"one field twice without becoming one task with redundant steps, "
            f"and mutually-cancelling pairs cannot both appear. Raise n_steps at "
            f"most to {n_safe}, or widen the tool set."
        )
    return EnvSpec(
        domain=domain,
        n_records=n_records,
        n_distractors=n_distractors,
        n_steps=n_steps,
        seed=seed,
        tools=tools,
    )


def graph_of(spec: EnvSpec) -> dict:
    """The graph, with read/write sets attached so traversal needs no side table.

    ``build_dependency_graph`` returns the topology; this adds the two lookup
    maps that :func:`expand_chain` uses. They live on the artifact rather than
    in a closure so that a dumped graph is sufficient to re-derive a chain,
    which is what makes the exported benchmark reproducible from its JSON.
    """
    g = build_dependency_graph(spec)
    g["reads"] = {t.name: list(t.reads) for t in spec.tools}
    g["writes"] = {t.name: list(t.writes) for t in spec.tools}
    g["kinds"] = {t.name: t.kind for t in spec.tools}
    g["docs"] = {t.name: t.doc for t in spec.tools}
    g["params"] = {t.name: list(t.params) for t in spec.tools}
    return g


def summarise_env(spec: EnvSpec) -> dict:
    """Coverage report for an environment, in the style of ``summarise_batch``."""
    g = graph_of(spec)
    by_reason: dict[str, int] = {}
    for e in g["edges"]:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    kinds: dict[str, int] = {}
    for t in spec.tools:
        kinds[t.kind] = kinds.get(t.kind, 0) + 1
    return {
        "env_id": spec.env_id,
        "domain": spec.domain,
        "n_tools": len(spec.tools),
        "by_kind": kinds,
        "n_edges": len(g["edges"]),
        "edges_by_reason": by_reason,
        "n_records": spec.n_records,
        "n_distractors": spec.n_distractors,
        "n_steps": spec.n_steps,
        "difficulty": round(spec.difficulty(), 3),
    }


# --------------------------------------------------------------------------
# instances: a spec plus the concrete state it starts from
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvInstance:
    """One materialised environment: the spec, its initial state, and its graph.

    The initial state is generated here rather than by the environment program
    at load time, so that the same instance can be replayed against a reference
    solution and against a checkpoint function with no possibility of the two
    seeing different starting conditions. A generator that seeded its own state
    would make gate V1's comparison meaningless.
    """

    spec: EnvSpec
    initial_state: dict
    #: The chain the task is built along. Not shipped to the agent — it is the
    #: reference path, and ``solution`` below is its executable form.
    chain: tuple[str, ...]
    #: Ids of records that are relevant to the task, and of those that are not.
    relevant_ids: tuple[str, ...] = ()
    distractor_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "spec": self.spec.as_dict(),
            "env_id": self.spec.env_id,
            "initial_state": self.initial_state,
            "chain": list(self.chain),
            "relevant_ids": list(self.relevant_ids),
            "distractor_ids": list(self.distractor_ids),
        }

    @property
    def instance_id(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True)
        return f"{self.spec.env_id}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def materialise(spec: EnvSpec, *, seed: int | None = None) -> EnvInstance:
    """Build the initial state and pick the chain the task will follow.

    The chain is sampled from the graph, not hand-written, and it is sampled so
    that it *starts with a query*. A chain that starts with a mutation asks the
    agent to change something it has not been told about; the instruction would
    then have to name the target explicitly, which collapses the environment
    back into a prompt. Starting with a read is what makes the environment load
    bearing.
    """
    rng = random.Random(spec.seed if seed is None else seed)
    g = graph_of(spec)

    queries = [t.name for t in spec.tools if t.kind == "query"]
    mutators = [t.name for t in spec.tools if t.kind == "mutate"]
    if not queries or not mutators:
        raise ValueError("an environment needs at least one query and one mutate tool")

    seed_tool = rng.choice(sorted(queries))
    # `present` is the set of keys the initial state already carries. Passing it
    # is what lets the expansion use state the environment starts with rather
    # than only state the chain itself produced; see `expand_chain` for the bug
    # that omitting it caused.
    chain = expand_chain(
        g,
        (seed_tool,),
        max_steps=max(2, spec.n_steps),
        rng=rng,
        present=_STATE_KEYS,
    )

    # Records: `n_records` relevant ones and `n_distractors` irrelevant ones,
    # interleaved so that position carries no information. A distractor placed
    # only at the end would be defeatable by "take the first match"; ScaleEnv
    # requires distractors to be functionally orthogonal to the real trajectory,
    # which interleaving is what implements.
    f0, f1, f2 = _DOMAINS[spec.domain]["fields"]
    total = spec.n_records + spec.n_distractors
    labels = [f"{rng.choice(_WORDS)}-{i:02d}" for i in range(total)]
    rng.shuffle(labels)

    relevant = labels[: spec.n_records]
    distractors = labels[spec.n_records :]

    records = []
    # Every record gets a distinct (f1, f2) signature. Without this the state
    # collapses: `f1` has 2 values and `f2` has 4, so 8 signatures for up to
    # `n_records + n_distractors` records — with 7 records a collision is likely
    # and with more it is certain. A collision is not merely cosmetic: the
    # instruction identifies the target by a *property* (see
    # `envtask._selector_for`), so two records sharing every field means no
    # property is unique and the task is ambiguous. Measured: 48 of 100
    # generated tasks were rejected as ambiguous, entirely from collisions.
    #
    # Signatures are enumerated rather than drawn, and the target is assigned
    # one before the rest, so the target always has a signature no other record
    # can hold. Drawing would leave the collision probability nonzero and the
    # rejection would come back.
    signatures = [(a, b) for a in ("open", "closed") for b in ("north", "south", "east", "west")]
    if total > len(signatures):
        raise ValueError(
            f"state of {total} records cannot have distinct signatures: only "
            f"{len(signatures)} (f1, f2) combinations exist; reduce n_records or "
            f"n_distractors, or widen the domain's field vocabulary"
        )
    rng.shuffle(signatures)
    target_sig = signatures[0]
    other_sigs = signatures[1:]

    target_id = relevant[0] if relevant else labels[0]

    # The target's `archived` value is **derived from the chain**, not drawn.
    #
    # Drawing it was a bug with a ~50% failure rate, and the failure was silent
    # in the way this module keeps being bitten by: `archive` is a no-op when the
    # target is already archived, and `restore` is a no-op when it is not, so
    # half of all draws produced a chain step the reference had to skip — and the
    # reference then graded a shorter task than `n_steps` reported. Measured: 54
    # of 120 tasks at `n_steps=2` were rejected for exactly this, and their chain
    # shapes were indistinguishable from the ones that survived.
    #
    # Setting the seed *from* the chain removes the coupling rather than
    # detecting it. A chain containing `archive` needs a target that is not yet
    # archived; one containing `restore` needs the opposite; a chain with
    # neither is free, and keeps the random draw so the field still carries
    # information for the agent to read.
    chain_set = set(chain)
    if "archive" in chain_set:
        target_archived = False
    elif "restore" in chain_set:
        target_archived = True
    else:
        target_archived = rng.random() < 0.5

    other_idx = 0
    for rid in labels:
        if rid == target_id:
            f1_v, f2_v = target_sig
        else:
            # Strictly sequential, not modulo: wrapping would hand two records
            # the same signature once the state is large enough, which is the
            # collision this block exists to remove.
            f1_v, f2_v = other_sigs[other_idx]
            other_idx += 1
        records.append(
            {
                # `_id` is the record's identity, separate from the domain's
                # own id field (`sku`, `tid`, ...). The two are kept distinct
                # because the verifier addresses records by identity while the
                # *agent* sees the domain field — and a task where the agent
                # must map one to the other is exactly the "read the state"
                # behaviour this environment is built to require. Using the
                # domain field as the identity would let the agent address a
                # record without ever reading it.
                "_id": rid,
                f0: rid,
                f1: f1_v,
                f2: f2_v,
                # `_pin` exists so `pin` is a *real* fourth step rather than a
                # tool that rewrites a field another tool already owns. It is
                # `None` for every record, which makes `pin`'s write observable
                # from any starting state — the property `reset_<f2>` could not
                # have. The name is prefixed to keep it out of the selector's
                # candidate list (see `envtask._selector_for`), since a field the
                # instruction does not mention should not be how the target is
                # identified.
                "_pin": None,
                # The target's value is the chain's business; every other
                # record is drawn, so `archived` still varies across the state
                # and the agent cannot infer the target's value from a
                # constant. See the note above for why the target is special.
                "archived": target_archived if rid == target_id else rng.random() < 0.5,
            }
        )

    state = {
        "records": records,
        "log": [],
        "meta": {"total": len(records), "changed": 0},
        # The task's target is carried in the state under a reserved key so the
        # checkpoint functions can be generated from the *state* rather than
        # from the instruction text. Grading the instruction would mean a
        # reworded prompt changed the reward, which is the opposite of what a
        # verifier should depend on.
        #
        # Only the **identity** is fixed here. The field is not, and fixing it
        # was a bug: `materialise` picks the chain, and which field the chain
        # ends up mutating depends on which mutator the expansion chose
        # (`set_qty` touches f1, `set_bin` touches f2). A `field` recorded here
        # from the spec therefore disagreed with the reference solution roughly
        # half the time, and the disagreement was silent — the checkpoints
        # graded a field the reference never touched, so the reference scored
        # 0.8 instead of 1.0 and the *initial* state scored 0.4 instead of 0.0.
        # The field is derived from the trace instead; see
        # `envtask.build_checkpoints`.
        "_target": {"id": relevant[0] if relevant else labels[0]},
    }

    return EnvInstance(
        spec=spec,
        initial_state=state,
        chain=chain,
        relevant_ids=tuple(relevant),
        distractor_ids=tuple(distractors),
    )


def reparameterise_env(spec: EnvSpec, **changes) -> EnvSpec:
    """Derive a neighbouring environment by moving one knob.

    The mirror of ``task_gen.reparameterise`` on the environment axis, and the
    call site the curriculum needs once it is steering environments rather than
    strings.
    """
    return replace(spec, seed=spec.seed + 1, **changes)
