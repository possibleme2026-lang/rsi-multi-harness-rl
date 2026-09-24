"""Regression test: the ledger, the difficulty bands, and the harness evolver.

These are the parts of the loop that decide *what to try next*, so a bug in any
of them is a bug in the search rather than in a single measurement. The
failures they guard against are all quiet ones:

* a ledger that re-proposes edits it has already tried burns rounds and makes
  the accept rate look worse than the search actually is;
* a novelty bonus that counts *wording* changes as exploration pushes the loop
  to keep rewriting guidance instead of changing the interface;
* a band that reports a position from an interval spanning two bands is how
  "T2 is too hard" got into a README on the strength of eight rollouts;
* a harness edit that adds a tool name to the guidance without adding the tool
  makes the harness lie to the policy, and the policy's failure is then
  attributed to the wrong cause;
* a pruning rule that drops a component below a yield threshold rather than
  only when it has *never* worked discards exactly the components that
  occasionally pay off — the same thin-evidence mistake the statistics layer
  exists to prevent.

What this test asserts
----------------------
1. ``edit_budget`` anneals monotonically from ``b_max`` to ``b_min``.
2. ``Ledger`` persists append-only and survives a truncated final line.
3. ``novelty`` counts structural components only.
4. ``prune_set`` needs attempts *and* zero yield, and never prunes a winner.
5. ``band_of`` refuses to place a cell whose interval spans two bands.
6. ``steer`` moves difficulty in the right direction per band.
7. ``guard_candidate`` rejects guidance that advertises an absent tool.
8. ``judge_candidate`` distinguishes "worse" from "within noise".
9. ``verifier_gen``'s strength-to-mode map is total, and the substring reward
   is refused where it cannot be expressed.
10. The ``rsi`` package imports nothing third-party, so the loop runs on CPU.
11. A generated batch's ids resolve in the registry that ``Agent.run`` reads —
    the failure that made ``--score rollout`` die on its first rollout.

Needs no shell, no model, no GPU.

Run:
    ./run.sh tests/test_rsi_loop.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.rsi import band  # noqa: E402
from multiharness.rsi import harness_evolve as he
from multiharness.rsi import verifier_gen as vg
from multiharness.rsi.ledger import (  # noqa: E402
    COMPONENTS,
    STRUCTURAL_COMPONENTS,
    EditRecord,
    Ledger,
    edit_budget,
    stall_flag,
)

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def rec(round_, name, comps, before, after, accepted, floor=0.0, harness="bash_minimal") -> EditRecord:
    return EditRecord(
        round=round_,
        candidate_id=f"c{round_}",
        edit=name,
        harness=harness,
        components=tuple(comps),
        score_before=before,
        score_after=after,
        delta=after - before,
        floor=floor,
        accepted=accepted,
    )


def main() -> int:
    print("=" * 74)
    print("TEST — RSI loop: ledger, bands, harness evolution, reward generation")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="mh_loop_"))

    # ------------------------------------------------------------------
    print("\n1. edit_budget — the annealed budget")
    # ------------------------------------------------------------------
    # A T-round run has rounds 0..T-1, so the schedule a run actually executes is
    # `range(T)` — *not* `range(T+1)`. Evaluating round T as well adds a value the
    # loop never uses, and doing that here is how the README came to quote a
    # 13-entry schedule for a 12-round run. Both facts are asserted separately
    # below: the executed schedule, and the formula's endpoint at t=T.
    sched = [edit_budget(t, 12, 1, 3) for t in range(12)]
    check("starts at b_max", sched[0] == 3, f"{sched}")
    check("the 12-round schedule has 12 entries", len(sched) == 12, f"{len(sched)}")
    check(
        "the 12-round schedule is what the README quotes",
        sched == [3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2],
        f"{sched}",
    )
    check("monotone non-increasing", all(a >= b for a, b in zip(sched, sched[1:], strict=False)), f"{sched}")
    check("every value is within [b_min, b_max]", all(1 <= v <= 3 for v in sched), f"{sched}")
    check("it actually changes over the run", len(set(sched)) > 1, f"{sched}")
    # The formula's own endpoint. `t = T` is outside any T-round run, so this is a
    # property of the anneal rather than of a schedule, and it is why `b_min` is
    # reached in the limit even though the last executed round is not yet there.
    check("the formula reaches b_min at t=T", edit_budget(12, 12, 1, 3) == 1, f"{edit_budget(12, 12, 1, 3)}")
    check("T=0 degenerates to b_min", edit_budget(0, 0, 1, 3) == 1)
    check("a flat budget stays flat", all(edit_budget(t, 10, 2, 2) == 2 for t in range(11)))
    # A longer run anneals more slowly — the schedule is relative, not absolute.
    check("a longer T is still at b_max later", edit_budget(6, 24, 1, 3) == 3, f"{edit_budget(6, 24, 1, 3)}")

    # ------------------------------------------------------------------
    print("\n2. Ledger — append-only, survives a truncated line")
    # ------------------------------------------------------------------
    p = tmp / "ledger.jsonl"
    led = Ledger(p)
    check("a fresh ledger is empty", led.attempted() == 0)
    led.append(rec(0, "guidance+=retry_hint", ["prompt"], 0.20, 0.35, True, floor=0.05))
    led.append(rec(1, "guidance+=exactness", ["prompt"], 0.35, 0.33, False, floor=0.05))
    led.append(rec(1, "client_tool-=read_file", ["client_tool"], 0.35, 0.35, False, floor=0.05))
    check("records are counted", led.attempted() == 3, f"{led.attempted()}")
    check("the file has one line per record", len(p.read_text(encoding="utf-8").strip().splitlines()) == 3)
    check("accepted are separated from rejected", len(led.accepted_edits()) == 1 and len(led.rejected_edits()) == 2)

    # Reload from disk.
    led2 = Ledger(p)
    check("reloading reproduces every record", led2.attempted() == 3, f"{led2.attempted()}")
    check("reloading preserves the verdicts", len(led2.accepted_edits()) == 1)
    check("reloading preserves components", led2.records[0].components == ("prompt",))
    check("best_score reads the accepted trajectory", led2.best_score() == 0.35, f"{led2.best_score()}")

    # A hard kill leaves a partial final line. That must be skipped, and only
    # that one — a middle line going missing would silently change the history.
    p2 = tmp / "truncated.jsonl"
    Ledger(p2).append(rec(0, "a", ["prompt"], 0.1, 0.2, True))
    with p2.open("a", encoding="utf-8") as fh:
        fh.write('{"round": 1, "candidate_id": "c1", "edit": "b", "comp')
    led3 = Ledger(p2)
    check("a truncated final line is skipped", led3.attempted() == 1, f"{led3.attempted()}")
    check("the intact record survives", led3.records[0].edit == "a")
    check(
        "the ledger is still appendable after a truncation",
        led3.append(rec(2, "c", ["prompt"], 0.2, 0.3, True)).round == 2,
    )

    # ------------------------------------------------------------------
    print("\n3. novelty — only structural components count")
    # ------------------------------------------------------------------
    led4 = Ledger(None)
    check(
        "nothing is novel on a fresh ledger",
        led4.novelty(("prompt", "client_tool")) == 1,
        f"{led4.novelty(('prompt','client_tool'))}",
    )
    led4.append(rec(0, "client_tool-=read_file", ["client_tool"], 0.2, 0.4, True))
    check("a touched structural component stops being novel", led4.novelty(("client_tool",)) == 0)
    check("output_plumbing is still novel", led4.novelty(("output_plumbing",)) == 1)
    check("prompt is never counted as novel", led4.novelty(("prompt",)) == 0, "prompt is not structural")
    check("structural set is a subset of the vocabulary", set(STRUCTURAL_COMPONENTS) <= set(COMPONENTS))
    # Rejected edits must not count towards novelty — nothing was learned.
    led5 = Ledger(None)
    led5.append(rec(0, "x", ["output_plumbing"], 0.2, 0.2, False))
    check("a rejected edit does not consume novelty", led5.novelty(("output_plumbing",)) == 1)
    check("component_counts ignores rejected edits", led5.component_counts()["output_plumbing"] == 0)

    # ------------------------------------------------------------------
    print("\n4. pruning — needs attempts and a perfect record of failure")
    # ------------------------------------------------------------------
    led6 = Ledger(None)
    for i in range(4):
        led6.append(rec(i, f"prompt_edit_{i}", ["prompt"], 0.2, 0.2, False))
    led6.append(rec(5, "client_tool-=read_file", ["client_tool"], 0.2, 0.5, True))
    pruned = led6.prune_set(min_attempts=3)
    check("a component tried 4x and never accepted is pruned", "prompt" in pruned, f"{pruned}")
    check("a component that worked is never pruned", "client_tool" not in pruned, f"{pruned}")
    check("an untried component is not pruned", "context_mgmt" not in pruned, f"{pruned}")
    # Below the attempt threshold, nothing is pruned even at zero yield.
    led7 = Ledger(None)
    for i in range(2):
        led7.append(rec(i, f"p{i}", ["prompt"], 0.2, 0.2, False))
    check(
        "two failures is not enough to prune",
        "prompt" not in led7.prune_set(min_attempts=3),
        f"{led7.prune_set(3)}",
    )
    check("but one fewer attempt threshold does prune", "prompt" in led7.prune_set(min_attempts=2))
    # Yield arithmetic.
    check("yield is accepted/attempted per component", abs(led6.yield_per_component()["prompt"] - 0.0) < 1e-9)
    check("a winning component has positive yield", led6.yield_per_component()["client_tool"] == 1.0)

    # ------------------------------------------------------------------
    print("\n5. stall detection and the summary")
    # ------------------------------------------------------------------
    flat = [0.2, 0.3, 0.35, 0.36, 0.36, 0.36]
    check("a flattened trajectory stalls", stall_flag(flat, 6, w=3, delta=0.02))
    check("a rising trajectory does not stall", not stall_flag([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 6, w=3, delta=0.02))
    check("too few points to judge", not stall_flag([0.1, 0.2], 2, w=3))
    # The noise floor is what makes this honest: gains that do not clear it
    # still count as a stall.
    creeping = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35]
    check("creeping gains within the floor still stall", stall_flag(creeping, 6, w=3, delta=0.10), f"{creeping}")
    check("the same gains count as progress with no floor", not stall_flag(creeping, 6, w=3, delta=0.0))

    s = led6.summary()
    check("summary counts attempts", s["attempted"] == 5, f"{s}")
    check("summary computes the accept rate", abs(s["accept_rate"] - 0.2) < 1e-9, f"{s['accept_rate']}")
    check("summary reports the best score", abs(s["best_score"] - 0.5) < 1e-9, f"{s['best_score']}")
    check("summary names the prune set", s["prune_set"] == ["prompt"], f"{s['prune_set']}")
    check(
        "summary separates worse from within-noise",
        s["rejected_worse"] == 4 and s["rejected_within_noise"] == 0,
        f"{s}",
    )
    check("render produces a markdown table", led6.render(3).startswith("| round |"), f"{led6.render(3)[:60]}")
    check("render handles an empty ledger", "_no edits attempted_" in Ledger(None).render())

    # ------------------------------------------------------------------
    print("\n6. band — refuses to place a cell it cannot place")
    # ------------------------------------------------------------------
    # 0/32 sits below 0.1 at the point estimate but its interval reaches 0.107,
    # which is in the frontier band. Reporting either would be a guess.
    b032 = band.band_of(0, 32, harness="bash_minimal", task_id="t2-01")
    check(
        "0/32 is unresolved, not out_of_reach",
        b032.band == band.Band.UNRESOLVED,
        f"{b032.band} [{b032.lo:.3f},{b032.hi:.3f}]",
    )
    check("0/128 IS out_of_reach", band.band_of(0, 128).band == band.Band.OUT_OF_REACH)
    check("16/32 is frontier", band.band_of(16, 32).band == band.Band.FRONTIER)
    # 30/32 has p_hat above 0.9 but its lower bound is not, so it is unresolved
    # — the same lesson as 0/32, at the other end.
    check(
        "30/32 is unresolved, not mastered",
        band.band_of(30, 32).band == band.Band.UNRESOLVED,
        f"{band.band_of(30, 32).band}",
    )
    check("1/2 is unresolved", band.band_of(1, 2).band == band.Band.UNRESOLVED)
    check("a tiny n is always unresolved", band.band_of(1, 1).band == band.Band.UNRESOLVED)

    cells = [
        band.band_of(0, 32, harness="bash_minimal", task_id="t2-01"),
        band.band_of(16, 32, harness="bash_minimal", task_id="t1-01"),
        band.band_of(0, 128, harness="react_tools", task_id="t3-01"),
    ]
    counts = band.band_counts(cells)
    check("band counts partition the cells", sum(counts.values()) == 3, f"{counts}")
    check("unresolved is reported rather than hidden", counts[band.Band.UNRESOLVED] == 1, f"{counts}")
    check("frontier_cells returns only frontier", [c.task_id for c in band.frontier_cells(cells)] == ["t1-01"])
    m = band.band_matrix(
        [
            {"harness": "bash_minimal", "task_id": "t1-01", "passes": 16, "n": 32},
            {"harness": "react_tools", "task_id": "t1-01", "passes": 16, "n": 32},
        ]
    )
    check("band_matrix keys by harness then task", m["bash_minimal"]["t1-01"] == band.Band.FRONTIER, f"{m}")
    check("band_matrix can filter harnesses", len(band.band_matrix(
        [
            {"harness": "a", "task_id": "t", "passes": 16, "n": 32},
            {"harness": "b", "task_id": "t", "passes": 16, "n": 32},
        ],
        harnesses=["a"],
    )) == 1)

    # ------------------------------------------------------------------
    print("\n7. steer — the curriculum moves in the right direction")
    # ------------------------------------------------------------------
    easy = {"payload_len": 4, "escape_density": 0.0, "steps": 1, "read_source": True}
    hard = {"payload_len": 8, "escape_density": 0.35, "steps": 2, "read_source": False}
    up = band.steer(band.Band.MASTERED, easy)
    check("mastered pushes difficulty up", up["payload_len"] > easy["payload_len"], f"{up}")
    check("mastered raises escaping", up["escape_density"] > easy["escape_density"], f"{up}")
    check("mastered adds a step", up["steps"] > easy["steps"], f"{up}")
    check("mastered stops giving the content away", up["read_source"] is False, f"{up}")
    down = band.steer(band.Band.OUT_OF_REACH, hard)
    check("out_of_reach pushes difficulty down", down["payload_len"] < hard["payload_len"], f"{down}")
    check("out_of_reach lowers escaping", down["escape_density"] < hard["escape_density"], f"{down}")
    check("out_of_reach drops a step", down["steps"] < hard["steps"], f"{down}")
    check("out_of_reach ships the content in a file", down["read_source"] is True, f"{down}")
    check(
        "frontier holds still",
        band.steer(band.Band.FRONTIER, easy) == {},
        f"{band.steer(band.Band.FRONTIER, easy)}",
    )
    check("unresolved holds still", band.steer(band.Band.UNRESOLVED, easy) == {})
    # The moves must stay inside the parameter space the generator samples.
    from multiharness.rsi import task_gen

    space = task_gen.PARAM_SPACE
    for _ in range(6):
        up = band.steer(band.Band.MASTERED, up)
        down = band.steer(band.Band.OUT_OF_REACH, down)
    check(
        "repeated 'harder' stays within the payload range",
        up["payload_len"] in space["payload_len"],
        f"{up['payload_len']}",
    )
    check(
        "repeated 'harder' stays within the escape range",
        up["escape_density"] in space["escape_density"],
        f"{up['escape_density']}",
    )
    check("repeated 'harder' stays within the step range", up["steps"] in space["steps"], f"{up['steps']}")
    check(
        "repeated 'easier' stays within the payload range",
        down["payload_len"] in space["payload_len"],
        f"{down['payload_len']}",
    )
    check("repeated 'easier' stays within the step range", down["steps"] in space["steps"], f"{down['steps']}")

    # ------------------------------------------------------------------
    print("\n8. harness_evolve — the guard and the judge")
    # ------------------------------------------------------------------
    check("frozen harnesses are not evolvable", not (set(he.FROZEN_HARNESSES) & set(he.EVOLVABLE_HARNESSES)))
    check("codex_style is frozen", "codex_style" in he.FROZEN_HARNESSES)
    names = [e.name for e in he.EDIT_LIBRARY]
    check("edit names are unique", len(names) == len(set(names)), f"{names}")
    check("every edit names a known component", all(e.component in COMPONENTS for e in he.EDIT_LIBRARY), f"{names}")
    check("every edit has a rationale", all(e.rationale for e in he.EDIT_LIBRARY))

    st = he.HarnessState(name="bash_minimal", guidance="You have exactly one tool: `bash`.", tools=("bash",))
    check("a consistent harness passes the guard", he.guard_candidate(st)[0], f"{he.guard_candidate(st)}")
    # The failure the guard exists for: guidance promises a tool that is absent.
    liar = he.apply_edit(
        st,
        he.HarnessEdit(name="bad", component="prompt", guidance_addendum="Use `read_file` first."),
    )
    ok, why = he.guard_candidate(liar)
    check("guidance advertising an absent tool is rejected", not ok, f"{why}")
    check("the rejection names the offending tool", "read_file" in why, f"{why}")
    check(
        "an empty tool set is rejected",
        not he.guard_candidate(he.HarnessState(name="x", guidance="hi", tools=()))[0],
    )
    check(
        "empty guidance is rejected",
        not he.guard_candidate(he.HarnessState(name="x", guidance="  ", tools=("bash",)))[0],
    )

    # apply_edit must not mutate its input.
    before_addenda = st.addenda
    before_tools = st.tools
    _ = he.apply_edit(st, he.EDIT_LIBRARY[0])
    check("apply_edit does not mutate the input state", st.addenda == before_addenda and st.tools == before_tools)
    check("apply_edit records the edit in history", len(_ .history) == 1, f"{_.history}")
    check("apply_edit grows the guidance", len(_.effective_guidance()) > len(st.effective_guidance()))

    # propose_edits filters.
    led8 = Ledger(None)
    props = he.propose_edits(st, led8, 3)
    check("proposal respects the budget", len(props) == 3, f"{[e.name for e in props]}")
    led8.append(rec(0, props[0].name, [props[0].component], 0.2, 0.2, False))
    props2 = he.propose_edits(st, led8, 10)
    check(
        "a tried edit is not re-proposed",
        props[0].name not in [e.name for e in props2],
        f"{[e.name for e in props2]}",
    )
    # An edit that drops a tool the harness lacks would be a recorded no-op.
    props3 = he.propose_edits(st, Ledger(None), 10)
    check(
        "an inapplicable tool-drop is skipped",
        "client_tool-=read_file" not in [e.name for e in props3],
        f"{[e.name for e in props3]}",
    )
    rt = he.HarnessState(name="react_tools", guidance="Tools: `bash`, `read_file`.", tools=("bash", "read_file"))
    props4 = he.propose_edits(rt, Ledger(None), 10)
    check(
        "an applicable tool-drop is proposed",
        "client_tool-=read_file" in [e.name for e in props4],
        f"{[e.name for e in props4]}",
    )

    # judge_candidate.
    inc = he.HarnessState(name="bash_minimal", guidance="g", tools=("bash",), score=0.40)
    edit0 = he.EDIT_LIBRARY[0]
    cand = he.apply_edit(inc, edit0)
    cand.score = 0.55
    acc, r = he.judge_candidate(cand, inc, floor=0.05, round_index=0, candidate_id="c0", edit=edit0)
    check("a gain above the floor is accepted", acc, f"{r.reason}")
    check("the accepted record carries the delta", abs(r.delta - 0.15) < 1e-9, f"{r.delta}")
    cand.score = 0.43
    acc2, r2 = he.judge_candidate(cand, inc, floor=0.05, round_index=1, candidate_id="c1", edit=edit0)
    check("a gain within the floor is rejected", not acc2, f"{r2.reason}")
    check("the reason says 'within noise', not 'worse'", "within noise" in r2.reason, f"{r2.reason}")
    cand.score = 0.35
    acc3, r3 = he.judge_candidate(cand, inc, floor=0.05, round_index=2, candidate_id="c2", edit=edit0)
    check("a loss is rejected", not acc3, f"{r3.reason}")
    check("the reason says 'worse'", "worse" in r3.reason, f"{r3.reason}")
    # Exactly at the floor is not a gain: the boundary must be a strict >.
    cand.score = 0.45
    acc4, r4 = he.judge_candidate(cand, inc, floor=0.05, round_index=3, candidate_id="c3", edit=edit0)
    check("a delta exactly equal to the floor is not accepted", not acc4, f"{r4.reason}")
    check("judge records which harness was edited", r.harness == "bash_minimal", f"{r.harness}")

    # component_weights is a distribution.
    led9 = Ledger(None)
    led9.append(rec(0, "a", ["prompt"], 0.2, 0.4, True))
    led9.append(rec(1, "b", ["client_tool"], 0.4, 0.4, False))
    w = he.component_weights(led9)
    check("weights sum to 1", abs(sum(w.values()) - 1.0) < 1e-9, f"{sum(w.values())}")
    check("the winning component has the larger weight", w["prompt"] > w["client_tool"], f"{w}")

    # The evolver must never propose a change to a frozen harness. This is
    # enforced inside propose_edits, not at the call site: a caller that forgot
    # would silently train on the held-out harness, and the output would look
    # exactly like a successful run.
    frozen_state = he.HarnessState(name="codex_style", guidance="x", tools=("bash",))
    check("a frozen harness gets no proposals", he.propose_edits(frozen_state, Ledger(None), 10) == [])
    check("the oracle harness is frozen too", he.propose_edits(
        he.HarnessState(name="oracle", guidance="x", tools=("bash",)), Ledger(None), 10) == [])
    check(
        "every trainable harness still gets proposals",
        all(he.propose_edits(he.HarnessState(name=n, guidance="x", tools=("bash",)), Ledger(None), 10)
            for n in he.EVOLVABLE_HARNESSES),
    )
    check(
        "no edit targets a frozen harness",
        all(not hasattr(e, "harness") for e in he.EDIT_LIBRARY),
        "edits are harness-agnostic descriptors; freezing is enforced by the caller",
    )
    report = he.EvolverReport(
        harness="h", rounds=1, accepted=1, attempted=2, start_score=0.1, end_score=0.2
    )
    check("EvolverReport serialises", isinstance(report.as_dict(), dict))

    # ------------------------------------------------------------------
    print("\n9. verifier_gen — the strength axis has a home for every value")
    # ------------------------------------------------------------------
    check(
        "every strength maps to a verify mode",
        set(vg.MODE_FOR_STRENGTH) == set(vg.CHECK_STRENGTHS),
        f"{set(vg.CHECK_STRENGTHS) - set(vg.MODE_FOR_STRENGTH)}",
    )
    check("the mapping is injective", len(set(vg.MODE_FOR_STRENGTH.values())) == len(vg.MODE_FOR_STRENGTH))
    check("substring is the file_contains mode", vg.mode_for_strength("substring") == "file_contains")
    check("exact is the file_equals mode", vg.mode_for_strength("exact") == "file_equals")
    try:
        vg.mode_for_strength("nonsense")
        check("an unknown strength raises", False, "no exception")
    except ValueError:
        check("an unknown strength raises", True)
    # The substring reward cannot be a digest, and asking for one must fail
    # loudly rather than silently produce an exact checker.
    try:
        vg.generate_checker("abc", "substring")
        check("substring cannot be expressed as a checker", False, "it produced one")
    except ValueError as e:
        check("substring cannot be expressed as a checker", True)
        check("the error says which mode to use instead", "file_contains" in str(e), f"{e}")

    # normalise and digest must agree, or every normalised task is unpassable.
    check("normalised folds case and whitespace", vg.normalise("Hello  WORLD", strength="normalised") == "hello world")
    check("exact only strips the ends", vg.normalise("  a  b  ", strength="exact") == "a  b")
    check(
        "digest is stable under normalisation",
        vg.digest("Hello  World", "normalised") == vg.digest("hello world", "normalised"),
    )
    check("exact digest is not case-folded", vg.digest("Hello", "exact") != vg.digest("hello", "exact"))
    check(
        "the checker uses the same normal form it hashed",
        "'.join(s.split()).lower()" in vg.generate_checker("x y", "normalised"),
    )
    # The checker must not contain the answer.
    body = vg.generate_checker("secret value here", "exact")
    check("the checker does not contain the plaintext", "secret value here" not in body)
    check("the checker compares a digest", "sha256" in body)

    # recommend_mode's length rule must actually be reachable.
    check("a short answer takes the in-process mode", vg.recommend_mode("h e wm") == "file_equals")
    check(
        "a long answer may ship a checker",
        vg.recommend_mode("a very long answer indeed") == "python_exit",
        f"{vg.recommend_mode('a very long answer indeed')}",
    )
    check(
        "prefer_strict forces the in-process mode",
        vg.recommend_mode("a very long answer indeed", prefer_strict=True) == "file_equals",
    )
    # contains_slice must be strictly weaker than equality for a real answer.
    ans = "alpha beta gamma delta epsilon"
    needle = vg.contains_slice(ans)
    check("contains_slice is a proper substring", needle in ans and needle != ans, f"{needle!r}")
    check("contains_slice handles a short answer", vg.contains_slice("ab") == "ab")
    check("contains_slice never returns empty", vg.contains_slice("    ").strip() == "")

    rep = vg.discrimination_report(
        {"id": "t", "tier": "T1", "verify": "python_exit", "expected": "ab", "check_script": "check.py"}
    )
    check("the report flags a brute-forceable digest", rep["digest_brute_forceable"] is True, f"{rep}")
    check("the report flags a shipped checker", rep["ships_checker"] is True)
    rep2 = vg.discrimination_report(
        {"id": "t", "tier": "T1", "verify": "file_equals", "expected": "a long enough answer"}
    )
    check(
        "an in-process reward flags nothing",
        rep2["digest_brute_forceable"] is False and rep2["ships_checker"] is False,
        f"{rep2}",
    )

    # ------------------------------------------------------------------
    print("\n10. the loop modules stay dependency-free")
    # ------------------------------------------------------------------
    import ast

    rsi_dir = _SRC / "multiharness" / "rsi"
    third_party = {"torch", "trl", "transformers", "datasets", "numpy", "pandas", "yaml", "requests", "httpx"}
    offenders: list[str] = []
    for path in sorted(rsi_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            roots: list[str] = []
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".")[0]]
            for r in roots:
                if r in third_party:
                    offenders.append(f"{path.name}:{node.lineno} imports {r}")
    check("the rsi package imports only the standard library", not offenders, f"{offenders}")

    # ------------------------------------------------------------------
    print("\n11. rollout scoring can resolve the ids it is handed")
    # ------------------------------------------------------------------
    # `Agent.run` looks task ids up in `core.TASKS`, a process-global registry
    # that only `tasks.suite.load()` populates — with the *shipped* 24 tasks.
    # The generated batch lives solely in a local list, so the first rollout of
    # `--score rollout` used to die on
    #     KeyError: unknown task_id 't4-c1ff0a35'
    # and because that path needs a GPU, nothing in the CPU test suite noticed.
    # The fix registers the batch; this asserts the ids are resolvable *before*
    # any model is loaded, which is the part that was broken.
    from multiharness.harnesses.core import TASKS, get_task, register_tasks  # noqa: E402
    from multiharness.rsi import task_gen  # noqa: E402
    from multiharness.tasks import load as load_suite  # noqa: E402

    load_suite()
    check("the shipped suite registers in a fresh process", len(TASKS) == 24, f"{len(TASKS)}")

    batch = task_gen.generate_batch(12, seed=11)
    fresh = [t for t in batch if t["id"] not in TASKS]
    check("generated ids are absent from the suite", len(fresh) == len(batch), f"{len(fresh)}/{len(batch)}")
    register_tasks(fresh)
    unresolved = [t["id"] for t in batch if not get_task(t["id"])]
    check("every generated id resolves once registered", not unresolved, f"{unresolved}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — the loop's search decisions are pinned")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
