"""Regression test: synthesised environments must discriminate, and must export.

The bugs this locks down
------------------------
Seven defects were found while building the environment axis, every one of them
by *running* the generator rather than by reading it. They are pinned here
individually, because each has a distinct symptom and each would otherwise
reappear silently — the generator would keep producing tasks, and the tasks
would keep looking reasonable.

**One: the `state` edge reason was unreachable.** ``dependency_edges`` tested
``a.writes & b.reads`` before contention, and every base mutator both reads and
writes ``records`` — so any two mutators matched the flow test and the
contention case was never reached. Measured: 17 edges, 0 of them ``state``.

**Two: the `precondition` reason was then made unreachable instead.** Testing
flow in *either* direction absorbed the one-way cases, which are exactly the
``precondition`` cases. Measured: 24 edges, 0 of them ``precondition``. The fix
is a partition (mutual flow / one-way flow / shared-write contention), and the
assertion below is that **all three occur** — which is the check that would have
caught both.

**Three: chain expansion dead-ended at length 2.** ``expand_chain`` tracked only
what the chain itself had written, and the chain starts from a *query*, which
writes nothing — so no mutator was ever admissible. ``n_steps=4`` silently
produced a 2-step chain. The fix is to seed the available-set with the keys the
initial state already carries.

**Four: the checkpoints graded a field the reference never touched.** The target
field was recorded on the spec, but which field the chain mutates depends on
which mutator expansion picked. Measured: the reference scored **0.8** instead of
1.0, and the initial state scored **0.4** instead of 0.0. The field is now
derived from the trace.

**Five: a checkpoint passed vacuously.** ``meta_consistent`` asks "the counter
equals the number of log entries", which on an untouched state is ``0 == 0``.
That single predicate was the entire reason the initial state scored 0.2-0.25
rather than 0.0. Both it and ``order_observable`` now require a non-empty
evidence set.

**Six: the instruction leaked the target's identity.** It named the record id,
so the reference solution was one command and an agent could score 1.0 without
calling a single query — the environment was decoration. The target is now
identified by a *property* that generation guarantees is unique.

**Seven: the instruction named only the last mutation.** The checkpoints grade
every mutation, so a compliant agent scored 0.8 and was told it was wrong.

**Eight: the guidance described the environment's commands as if they were
harness tools.** It rendered a bare list —

.. code-block:: text

    envtool list_tickets
    envtool set_state <id> <value>

— and the prompt already contains a tool-schema block, because the chat
template renders the harness's own tools above the instruction. Given two lists
of that shape the model merged them and called ``envtool list_tickets`` **as a
tool name**::

    envtool list_tickets({'query': 'state=open'})
    -> Tool envtool list_tickets not found. Available: ['bash']

Measured across every harness in the pool: 70% of all 96 rollouts were this one
shape, and the model invented ``list_accounts`` and ``list_items`` on
environments that have no such command — it was answering the shape of the
prompt, not its content.

This is the defect worth remembering, because **the metric moved the wrong way
and looked like progress**. The tool-call rate rose from 35.4% to 81.2% while
the pass rate stayed at exactly 0.00: a call to a non-existent tool is still a
tool call, so the gate that was supposed to detect "the model can engage" was
satisfied by the failure mode itself. The gate is now read alongside
``scripts/env_scan_report.py``, which classifies the *shape* of each call, and
section 10 asserts the classifier still separates a bogus tool name from a real
one.

What this test asserts
----------------------
1. All three edge reasons occur on a generated environment — bugs one and two.
2. Chain length follows ``n_steps`` — bug three.
3. The reference scores exactly 1.0, the initial state exactly 0.0, and a
   distractor-applied trace strictly less — bugs four and five, across a batch
   rather than on one example.
4. The instruction never contains the target id, and names every field the
   checkpoints grade — bugs six and seven.
5. No condition-collision reaches the instruction: every generated task's
   selector is unique by construction.
6. The exported Harbor package passes its structural and trust-boundary audit,
   and the *inlined* verifier agrees with the in-process grader on the same
   state. That last one is the drift check: two implementations of one
   semantics, and a disagreement would invalidate every exported number.
7. The agent side of the package contains no solution — checked operationally,
   by rendering the package with only the agent-side files present.
8. The guidance names the *shell tool* the commands are reached through, shows
   the wrong shape explicitly, and does not present the environment's parameters
   as ``bash`` keyword arguments — bug eight, and the reason the check is on the
   shape rather than on the presence of the tool names.

Needs no model, no GPU and no Docker, so it runs in CI in well under a second.

Run:
    ./run.sh tests/test_rsi_envgen.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in str(sys.path):
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import (  # noqa: E402
    TRAIN_HARNESSES,
    core,
)
from multiharness.rsi import env_adapter, envgen, envtask, harbor_export  # noqa: E402

#: The scan classifier lives in a script, which is where the artifact-reading
#: tools belong. Imported rather than reimplemented so the test cannot pass
#: against a *different* classifier than the one that produces the report.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from env_scan_report import classify as _classify  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    # ------------------------------------------------------------------
    print("=" * 74)
    print("1. every edge reason is reachable (bugs one and two)")
    print("=" * 74)
    # The assertion is not "the graph has edges" but "all three *kinds* occur".
    # A partition bug makes one kind unreachable while the total edge count
    # still looks healthy — 17 edges and 24 edges were both wrong.
    seen: set[str] = set()
    for seed in range(6):
        spec = envgen.synthesise_env(
            domain="inventory", n_records=4, n_distractors=3, n_steps=4, seed=seed
        )
        summary = envgen.summarise_env(spec)
        seen |= set(summary["edges_by_reason"])
    for reason in ("data_flow", "precondition", "state"):
        check(f"edge reason {reason!r} occurs", reason in seen, f"only saw {sorted(seen)}")
    check(
        "the three reasons are a partition, not overlapping predicates",
        len(seen) == 3,
        f"saw {sorted(seen)}",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("2. chain length follows n_steps (bug three)")
    print("=" * 74)
    # Before the fix every chain was length 2 regardless of n_steps, because
    # expansion dead-ended after the seed query.
    #
    # The assertion is on the **mutation count**, not on `len(chain)`. The chain
    # starts with a seed query — it records how the environment is entered — so
    # its length is `n_steps + 1`. Asserting on the length is what let the knob
    # be wrong twice: a chain of `Q M Q M` has length 4 and grades a 1-step
    # solution, and a length-based test passes. `n_mutations` is the quantity
    # `n_steps` names, so it is the quantity to check.
    def n_mutations(spec):
        kinds = envgen.graph_of(spec)["kinds"]
        return sum(1 for n in envgen.materialise(spec).chain if kinds.get(n) == "mutate")

    lengths = {}
    for n_steps in (2, 3, 4):
        spec = envgen.synthesise_env(
            domain="inventory", n_records=4, n_distractors=3, n_steps=n_steps, seed=7
        )
        lengths[n_steps] = n_mutations(spec)
    check(
        "mutation count tracks n_steps",
        lengths == {2: 2, 3: 3, 4: 4},
        f"got {lengths}",
    )
    # The seed entry is a query, so the chain is one longer than the step count.
    # Asserted because the two were conflated: the reference skips queries, so a
    # chain whose length came from queries reported difficulty it did not have.
    seed_chain = envgen.materialise(
        envgen.synthesise_env(
            domain="inventory", n_records=4, n_distractors=3, n_steps=3, seed=7
        )
    ).chain
    kinds = envgen.graph_of(
        envgen.synthesise_env(
            domain="inventory", n_records=4, n_distractors=3, n_steps=3, seed=7
        )
    )["kinds"]
    check(
        "the chain starts with a query and is one entry longer than n_steps",
        len(seed_chain) == 4 and kinds[seed_chain[0]] == "query",
        f"chain {seed_chain}, kinds {[kinds[n] for n in seed_chain]}",
    )
    # Beyond the interface's reach the request must be **refused**, and refused
    # for every seed. Measured before the tool set was fixed: `n_steps=5` was
    # refused for "3 usable mutators" on 27 of 60 seeds and "4 usable mutators"
    # on 33 — the *legality* of the knob varied with an unrecorded draw, which
    # is the same defect as a knob whose value does.
    refusals = 0
    for s in range(20):
        try:
            envgen.synthesise_env(
                domain="inventory", n_records=4, n_distractors=3, n_steps=5, seed=s
            )
        except ValueError:
            refusals += 1
    check(
        "n_steps above the ceiling is refused for every seed",
        refusals == 20,
        f"only {refusals}/20 raised",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("3. discriminability across a batch (bugs four and five)")
    print("=" * 74)
    # On a batch, not on a fixture: the property has to hold for the
    # distribution. The specific seeds that failed before were 0, 1, 2, 3 and
    # 5-13; a single-example test would have passed on the ones that worked.
    tasks = envtask.generate_env_batch(
        24, seed=3, n_records=4, n_distractors=3, n_steps=4
    )
    n_ref = n_init = n_wrong = 0
    worst_wrong = 0.0
    for t in tasks:
        cps = list(t.checkpoints)
        ref = envtask.grade_state(t.instance, envtask.apply_trace(t.instance, list(t.trace)), cps)
        init = envtask.grade_state(t.instance, t.instance.initial_state, cps)
        wrong = envtask.grade_state(
            t.instance, envtask.plausible_wrong_state(t.instance, list(t.trace)), cps
        )
        n_ref += int(ref == 1.0)
        n_init += int(init == 0.0)
        n_wrong += int(wrong < ref)
        worst_wrong = max(worst_wrong, wrong)
    check(f"reference scores exactly 1.0 on all {len(tasks)}", n_ref == len(tasks), f"{n_ref}/{len(tasks)}")
    check(
        f"initial state scores exactly 0.0 on all {len(tasks)}",
        n_init == len(tasks),
        f"{n_init}/{len(tasks)} — a vacuous checkpoint makes this fail",
    )
    check(
        f"distractor trace scores strictly less on all {len(tasks)}",
        n_wrong == len(tasks),
        f"{n_wrong}/{len(tasks)}",
    )
    # The distractor must be *substantially* wrong, not wrong by rounding: a
    # trace applied to the wrong record should fail at least the field and the
    # log checkpoints. A near-1.0 "wrong" score would mean the checkpoints
    # cannot see the difference, which is the property that matters.
    check(
        "the distractor's score is not merely lower but clearly lower",
        worst_wrong <= 0.6,
        f"worst distractor score {worst_wrong:.2f}",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("4. the instruction is honest (bugs six and seven)")
    print("=" * 74)
    leaks = 0
    missing_fields = 0
    for t in tasks:
        tid = t.instance.initial_state["_target"]["id"]
        if tid in t.instruction:
            leaks += 1
        # Every field the checkpoints grade must be named in the instruction,
        # or a compliant agent scores below 1.0.
        graded = set()
        for cp in t.checkpoints:
            if cp.kind == "target_field":
                graded |= set(cp.args["fields"])
        if not all(f"`{f}`" in t.instruction for f in graded):
            missing_fields += 1
    check("no instruction contains the target record id", leaks == 0, f"{leaks} leak(s)")
    check(
        "every graded field is named in the instruction",
        missing_fields == 0,
        f"{missing_fields} task(s) grade a field they do not ask for",
    )
    # The selector has to be unique, or the task is ambiguous and unsolvable.
    # Generation guarantees it; this asserts the guarantee held for the batch.
    ambiguous = 0
    for t in tasks:
        tid = t.instance.initial_state["_target"]["id"]
        recs = t.instance.initial_state["records"]
        target = next(r for r in recs if r["_id"] == tid)
        fields = envgen._DOMAINS[t.instance.spec.domain]["fields"]
        sig = (target.get(fields[1]), target.get(fields[2]))
        if sum(1 for r in recs if (r.get(fields[1]), r.get(fields[2])) == sig) != 1:
            ambiguous += 1
    check(
        "every target is uniquely identifiable by its field signature",
        ambiguous == 0,
        f"{ambiguous} ambiguous",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("5. the Harbor package is structurally sound and does not leak")
    print("=" * 74)
    audit = harbor_export.audit_batch(tasks)
    check(f"all {audit['total']} packages audit clean", audit["ok"] == audit["total"],
          f"{audit['ok']}/{audit['total']}, problems={audit['problems_by_kind']}")

    # The tool table must be non-empty. This was a real defect: the instruction
    # rendered a header-only table because it read a `tools` key the graph does
    # not have, so the agent was never told the name of a single tool.
    files = harbor_export.package_files(tasks[0])
    instr = files[f"{tasks[0].task_id}/instruction.md"]
    tool_rows = [ln for ln in instr.splitlines() if ln.startswith("| `") and ln.count("|") >= 4]
    check(
        "the instruction lists every tool in the environment",
        len(tool_rows) == len(tasks[0].graph["nodes"]),
        f"{len(tool_rows)} rows vs {len(tasks[0].graph['nodes'])} tools",
    )
    check("the instruction says where to write the state", "/app/state.json" in instr)

    # Every tool the graph declares must be **dispatchable** by the exported
    # program, not merely listed in the instruction.
    #
    # This check exists because section 5 passed 24/24 while `harbor_local_run`
    # failed 30/30 packages with `unknown tool: restore` and `unknown tool: pin`.
    # A structural audit cannot see that: the file exists, it is well-formed, and
    # it lists the right tools — it simply does not implement two of them. The
    # assertion is on the generated source's dispatch table, which is the closest
    # a unit test can get to execution without spawning a process per tool.
    tools_src = files[f"{tasks[0].task_id}/environment/assets/tools.py"]
    declared = tasks[0].graph["nodes"]
    undispatchable = []
    for name in declared:
        if name in tools_src:
            continue
        # Prefix-dispatched families (`set_*`, `get_*`, `list_*`, `count_by_*`)
        # are handled by a `startswith` branch rather than by their own name.
        #
        # The stem is everything up to the **last** underscore, not the first:
        # `count_by_sku` is dispatched by `count_by_`, so splitting on the first
        # underscore yields `count` and looks for `startswith("count_")`, which
        # is not there. Measured: the first version of this check reported
        # `count_by_sku` as unimplemented on a package that implements it.
        stem = name.rsplit("_", 1)[0]
        if f'startswith("{stem}_")' in tools_src:
            continue
        undispatchable.append(name)
    check(
        "every declared tool is dispatchable by the exported program",
        not undispatchable,
        f"not implemented: {undispatchable}",
    )

    # The four reference-safe mutators are named individually because a typo in
    # one of them is silent: `archive` would still work and `restore` would still
    # return "unknown tool" only at *execution* time, on whichever packages
    # happened to draw it. Measured: 30 of 30 packages failed this way.
    missing_mutators = [
        m for m in ("archive", "restore", "pin")
        if f'tool == "{m}"' not in tools_src and f'"{m}"' not in tools_src
    ]
    check(
        "the reference-safe mutators have exported implementations",
        not missing_mutators,
        f"missing: {missing_mutators}",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("6. the exported verifier agrees with the in-process grader")
    print("=" * 74)
    # This is the drift check, and it is the one that would silently invalidate
    # every exported number: two implementations of one semantics. They are
    # compared by evaluating the *same* literal checkpoint specs against the
    # same state, through the inlined code path the package actually runs.
    t0 = tasks[0]
    cps = [c.as_dict() for c in t0.checkpoints]
    states = {
        "reference": envtask.apply_trace(t0.instance, list(t0.trace)),
        "initial": t0.instance.initial_state,
        "distractor": envtask.plausible_wrong_state(t0.instance, list(t0.trace)),
    }
    agree = 0
    for name, state in states.items():
        inproc = envtask.grade_state(t0.instance, state, list(t0.checkpoints))
        # Evaluate the inlined evaluator by exec'ing just the part of the
        # generated verifier above `_load_state`, which is what
        # `tests/test_state.py` runs.
        #
        # The symbol looked up is `evaluate_checkpoint`, not the `_evaluate`
        # that used to be hand-copied in. The rename is the point: the exported
        # function is now `inspect.getsource(core.evaluate_checkpoint)` verbatim,
        # so the name in the package is the name in `core`, and a test that looks
        # for the old one fails loudly instead of quietly comparing nothing.
        verifier_src = files[f"{t0.task_id}/tests/test_state.py"]
        ns: dict = {}
        helpers = verifier_src.split("def _load_state()")[0]
        exec(helpers, ns)  # noqa: S102 - test-local, on our own generated source
        passed = sum(1 for cp in cps if ns["evaluate_checkpoint"](cp, state))
        exported = harbor_export.reward_from_checkpoints(passed, len(cps))
        if abs(inproc - exported) < 1e-12:
            agree += 1
        else:
            print(f"         {name}: in-process {inproc} vs exported {exported}")
    check(
        "in-process and exported graders agree on all three states",
        agree == len(states),
        f"{agree}/{len(states)}",
    )
    check(
        "0/0 is 0.0, not 1.0",
        harbor_export.reward_from_checkpoints(0, 0) == 0.0,
        "an empty checkpoint set must not read as a perfect score",
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("7. the agent side carries no solution")
    print("=" * 74)
    # Checked operationally as well as textually: the files that exist on the
    # agent side are enumerated, and the oracle's distinguishing content is
    # confirmed absent. `audit_package` does the textual half.
    agent_files = {k: v for k, v in files.items() if "/environment/" in k}
    oracle_calls = [
        f"{c['tool']} {c.get('id', '')}".strip()
        for c in t0.trace
        if c.get("kind") == "mutate"
    ]
    leaked = [
        rel for rel, content in agent_files.items()
        if any(call and call in content for call in oracle_calls)
    ]
    check("no agent-side file contains the oracle's mutation sequence", not leaked, f"{leaked}")
    check("no agent-side file contains a CHECKPOINTS literal",
          not any("CHECKPOINTS" in c for c in agent_files.values()))
    agent_docker = files[f"{t0.task_id}/environment/Dockerfile"]
    directives = [ln.strip() for ln in agent_docker.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    check(
        "no agent Dockerfile directive references a verifier-side path",
        not any(bad in ln for ln in directives for bad in ("solution/", "tests/", "solve.sh")),
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("8. export is idempotent and refuses to merge")
    print("=" * 74)
    dest = Path(tempfile.mkdtemp(prefix="mh_harbor_test_"))
    try:
        root = harbor_export.write_harbor_task(t0, dest)
        check("the package directory is created", root.is_dir())
        check("task.toml is present", (root / "task.toml").is_file())
        # The executable bit is set by `write_harbor_task` so Harbor can run
        # `solution/solve.sh` directly, but Windows filesystems do not carry a
        # POSIX mode — `chmod(0o755)` there is accepted and discarded, and
        # `stat().st_mode` reports a read/write file. Asserting the bit on
        # Windows would therefore fail for a reason that has nothing to do with
        # the exporter, so the assertion is conditional on being able to observe
        # the mode at all. The image builds on Linux, where the bit is real.
        solve = root / "solution/solve.sh"
        if sys.platform == "win32":
            check(
                "solve.sh exists and has a shebang (mode is not observable on Windows)",
                solve.is_file() and solve.read_text(encoding="utf-8").startswith("#!"),
            )
        else:
            check("solve.sh is executable", solve.stat().st_mode & 0o111 != 0)
        try:
            harbor_export.write_harbor_task(t0, dest)
            check("a second export into the same dir is refused", False, "it was allowed")
        except FileExistsError:
            check("a second export into the same dir is refused", True)
    finally:
        harbor_export.clean_package_dir(dest)

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("9. the artifact is JSON-serialisable")
    print("=" * 74)
    d = t0.as_dict()
    for key in ("id", "instruction", "checkpoints", "trace", "initial_state", "graph"):
        check(f"the artifact carries {key!r}", key in d)
    check("the artifact is JSON-serialisable", isinstance(json.dumps(d), str))

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("10. the harness-pool batch survives a JSON round trip")
    print("=" * 74)
    # This is the check whose absence let a callable into the task dict, and the
    # failure it caused was not a crash at export time — it was that the
    # environment batch could not be *written at all*, so `probe.py --from-batch`
    # had nothing to scan and the cross-harness gap could only ever be measured
    # on string tasks. Measured: `TypeError: Object of type function is not JSON
    # serializable` on `env_batch.json`.
    #
    # The round trip is asserted rather than the dump alone, because a batch that
    # serialises and cannot be re-registered is no better: `reset` would resolve
    # `env` to nothing, every rollout would read the same default state path, and
    # the scan would report a plausible number for an experiment that was not run.
    harness_batch = env_adapter.as_harness_batch(tasks)
    try:
        blob = json.dumps(harness_batch)
        reloaded = json.loads(blob)
        serialisable = True
    except TypeError as exc:
        blob, reloaded, serialisable = "", [], False
        check("the harness batch is JSON-serialisable", False, str(exc))
    if serialisable:
        check("the harness batch is JSON-serialisable", True)
        t0h = reloaded[0]
        check(
            "the reloaded task keeps its verify mode and checkpoints",
            t0h.get("verify") == "state_checkpoints" and t0h.get("checkpoints"),
        )
        env_spec = t0h.get("env")
        check(
            "the reloaded env is a declarative template, not a callable",
            isinstance(env_spec, dict) and not callable(env_spec),
            f"{type(env_spec).__name__}",
        )
        # The template has to name the *rollout's* directory symbolically. A
        # resolved path here would be the export-time directory, shared by every
        # rollout — an environment that appears to work while coupling them.
        check(
            "the env template defers the workdir to reset time",
            isinstance(env_spec, dict) and any(
                "{workdir}" in str(v) for v in env_spec.values()
            ) or "{workdir}" in str(env_spec.get("_path_prefix", [])),
            f"{env_spec}",
        )
        check(
            "the setup mapping names the files the environment needs",
            set(t0h.get("setup", {})) >= {"tools.py", "initial_state.json"},
            f"{sorted(t0h.get('setup', {}))}",
        )
        # The guidance override, which is the check whose absence cost 96
        # rollouts. Each harness appends a static block written for string tasks;
        # for a stateful task it says to submit `answer.txt` (the verifier reads
        # `state.json`) and never mentions `envtool`. Measured: 0.00 in every cell
        # of every harness, with the model inventing tool names in the
        # transcripts. The assertions below are the two sentences that were
        # wrong, plus the tool table that was missing.
        guidance = t0h.get("guidance") or ""
        check(
            "the task carries a harness guidance override",
            bool(guidance),
            "without it each harness appends guidance written for string tasks",
        )
        check(
            "the guidance names the environment's tools",
            "envtool" in guidance and all(n in guidance for n in t0.graph["nodes"][:3]),
            "the agent cannot call a tool it was not told about",
        )
        check(
            "the guidance does not tell the agent to write an answer file",
            "answer.txt" not in guidance,
            "the verifier reads state.json; answer.txt would score 0.0",
        )
        check(
            "the guidance says the state is what is graded",
            "state" in guidance.lower() and "graded" in guidance.lower(),
        )
        # The second guidance defect, and the reason these assertions exist at
        # all: the first version rendered the environment's commands as a bare
        # list, the prompt already contains a tool-schema block, and the model
        # merged the two — calling `envtool list_tickets` *as a tool name* in
        # every harness ("Tool envtool list_tickets not found"). The tool-call
        # rate went UP (35.4% → 81.2%) while the pass rate stayed at exactly
        # 0.00, because a call to a non-existent tool still counts as a call.
        #
        # So the shape has to be asserted, not just the presence of the names.
        check(
            "the guidance wraps commands in the harness's shell tool",
            f"{env_adapter.SHELL_TOOL}(command=" in guidance,
            "the environment's commands are reached through bash, not as tools",
        )
        check(
            "the guidance shows the wrong shape explicitly",
            "no such tool" in guidance and "wrong" in guidance,
            "a small model repeats a shape it saw; a negative example is the fix",
        )
        check(
            "the guidance forbids calling envtool as a tool name",
            "not" in guidance and "tools of this harness" in guidance,
            "measured failure: 'Tool envtool list_tickets not found'",
        )
        # The tool name in the guidance must be one the pool actually exposes.
        # `SHELL_TOOL` is a claim about every harness, so it is checked against
        # every harness rather than against the one being used here.
        missing = [
            h for h, cls in TRAIN_HARNESSES.items()
            if env_adapter.SHELL_TOOL not in core.tool_names(cls())
        ]
        check(
            "every harness in the pool exposes the shell tool the guidance names",
            not missing,
            f"harnesses without {env_adapter.SHELL_TOOL!r}: {missing}",
        )

        # The placeholders must stay inside the command string. `bash` takes one
        # argument named `command`; the model was passing the environment's
        # parameters alongside it:
        #   bash({'command': 'envtool list_tickets', 'owner': 'east'})
        #   -> bad arguments for bash: unexpected keyword argument 'owner'
        check(
            "the guidance does not present the environment's parameters as bash kwargs",
            "owner=" not in guidance and "task_id=" not in guidance,
            "parameters belong inside the command string, not beside it",
        )

        # The stage classifier in env_scan_report is the tool that found this;
        # it has to keep classifying a known-bad transcript as a bad shape.
        check(
            "the scan classifier separates a bogus tool name from a real call",
            _classify({
                "tool_calls": 1, "tool_errors": 1, "reward": 0.0,
                "raw": ["envtool list_tickets({'query': 'state=open'}) -> "
                        "{'error': \"Tool envtool list_tickets not found. "
                        "Available: ['bash']\"}"],
            }) == "unknown_tool",
            "a gate a wrong call satisfies is not a gate",
        )

    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — synthesised environments discriminate and export")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
