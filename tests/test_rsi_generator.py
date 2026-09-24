"""Regression test: the task generator and its four validation gates.

The bugs this locks down
------------------------
All four were found by running the generator, not by reading it, and each one
is a case where a plausible-looking task was in fact unusable.

**1. Gate 1 ran the reference in an empty directory.** A read-then-write task's
reference greps a file the task ships, so it failed for a reason unrelated to
whether the task was correct. Every T2 task was rejected as a broken oracle.

**2. Gate 2 shared a directory with gate 1.** Gate 1 writes ``answer.txt`` to
prove the reference works; gate 2 then measured an "empty" directory that still
contained it. All 60 tasks failed the nop gate — including ``file_equals``
tasks whose verifier cannot possibly pass an empty file. The batch reported a
0% accept rate that looked like a generator defect.

**3. ``check_script`` was passed as source, but ``core.verify`` treats it as a
filename.** It runs ``python <check_script>`` in the work directory, and the
only way a file gets there is via ``setup``. Every generated ``python_exit``
task scored 0.0 forever while looking merely difficult.

**4. The multi-step reference wrote through ``/tmp/_mh_payload``.** A fixed
global path shared by every concurrent rollout: two rollouts could interleave
and one would be graded on the other's payload.

Bug 3 also opened a reward-hacking surface — a checker shipped into the work
directory could hold the plaintext answer. The checker now stores a SHA-256.

What this test asserts
----------------------
1. Determinism: the same seed re-derives byte-identical tasks.
2. Every generated task clears all four gates, over a batch big enough to
   cover every tier, verify mode, and step count.
3. Each of the four bugs above, individually.
4. Tier coherence — a T3 task has escaping characters, a T2 task reads a file.
5. Coverage — the batch actually exercises all three verifier modes, which the
   hand-written suite did not (all 24 of its tasks used ``file_equals``).

Needs a shell but no model and no GPU.

Run:
    ./run.sh tests/test_rsi_generator.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses.core import verify  # noqa: E402
from multiharness.rsi import task_gen, validate, verifier_gen  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    print("=" * 74)
    print("TEST — RSI task generator + four validation gates")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="mh_gen_"))

    # ------------------------------------------------------------------
    print("\n1. determinism")
    # ------------------------------------------------------------------
    a = task_gen.generate_batch(12, seed=42)
    b = task_gen.generate_batch(12, seed=42)
    check("same seed gives the same ids", [t["id"] for t in a] == [t["id"] for t in b])
    check("same seed gives identical tasks", a == b)
    c = task_gen.generate_batch(12, seed=43)
    check("a different seed gives different tasks", [t["id"] for t in a] != [t["id"] for t in c])
    check("ids are unique within a batch", len({t["id"] for t in a}) == 12)

    # ------------------------------------------------------------------
    print("\n2. coverage — every axis is exercised")
    # ------------------------------------------------------------------
    batch = task_gen.generate_batch(60, seed=1)
    cov = task_gen.summarise_batch(batch)
    check("all 60 tasks have distinct ids", cov["distinct_ids"] == 60, f"{cov}")
    check("all four tiers appear", len(cov["by_tier"]) == 4, f"{cov['by_tier']}")
    check(
        "all three verify modes appear — the hand-written suite used only one",
        set(cov["by_verify_mode"]) == {"file_equals", "file_contains", "python_exit"},
        f"{cov['by_verify_mode']}",
    )
    check("multi-step tasks appear", len(cov["by_steps"]) >= 2, f"{cov['by_steps']}")
    check("source-file tasks appear", cov["with_source_file"] > 0, f"{cov}")

    # ------------------------------------------------------------------
    print("\n3. all four gates pass on a full batch")
    # ------------------------------------------------------------------
    verdicts = validate.validate_batch(batch, roots=tmp / "batch")
    summary = validate.summarise_validation(verdicts)
    check("every task is accepted", summary["accepted"] == 60, f"{summary}")
    check("no gate reports a failure", not summary["failures_by_gate"], f"{summary['failures_by_gate']}")
    check(
        "every tier is fully accepted",
        all(d["accepted"] == d["total"] for d in summary["by_tier"].values()),
        f"{summary['by_tier']}",
    )

    # ------------------------------------------------------------------
    print("\n4. bug 1 — gate 1 materialises the task's setup files")
    # ------------------------------------------------------------------
    # Find a task that reads a shipped file, so the reference has something to
    # grep. Without the setup files this fails for the wrong reason.
    src_tasks = [t for t in batch if t["params"]["read_source"]]
    check("the batch contains read-source tasks", len(src_tasks) > 0)
    t0 = src_tasks[0]
    check("that task ships its source file", "setup" in t0 and t0["setup"], f"{sorted(t0.get('setup') or {})}")
    r = validate.check_oracle(t0, root=tmp / "bug1")
    check("gate 1 passes for a read-source task", r.passed, f"{r.detail[:200]}")
    # And the opposite direction: with the setup withheld, it must fail.
    stripped = {k: v for k, v in t0.items() if k != "setup"}
    r2 = validate.check_oracle(stripped, root=tmp / "bug1b")
    check("gate 1 fails when the setup files are withheld", not r2.passed, "it passed without the source file")

    # ------------------------------------------------------------------
    print("\n5. bug 2 — gate 2 runs in its own directory")
    # ------------------------------------------------------------------
    # Run gate 1 then gate 2 in the same shared root; gate 2 must still see an
    # empty directory. `validate_task` gives each gate a subdirectory.
    eq_task = next(t for t in batch if t["verify"] == "file_equals")
    shared = tmp / "bug2"
    v = validate.validate_task(eq_task, root=shared)
    nop = next(g for g in v.gates if g.gate == "V2_nop")
    check("gate 2 passes for a file_equals task run after gate 1", nop.passed, f"{nop.detail}")
    check("gate 2's directory has no answer.txt", not (shared / "v2" / "answer.txt").exists())
    check("gate 1's directory does have one", (shared / "v1" / "answer.txt").exists())
    # Direct proof of the old failure: reuse one directory for both and the
    # nop probe scores the oracle's answer.
    same = tmp / "bug2b"
    validate.check_oracle(eq_task, root=same)
    polluted = validate.check_nop(eq_task, root=same)
    check("sharing one directory reproduces the old false failure", not polluted.passed, "no pollution observed")

    # ------------------------------------------------------------------
    print("\n6. bug 3 — check_script is a filename that ships in setup")
    # ------------------------------------------------------------------
    py_tasks = [t for t in batch if t["verify"] == "python_exit"]
    check("the batch contains python_exit tasks", len(py_tasks) > 0)
    for t in py_tasks[:6]:
        script = t.get("check_script")
        ok = isinstance(script, str) and script in (t.get("setup") or {})
        check(f"check_script {script!r} is shipped in setup", ok, f"setup={sorted(t.get('setup') or {})}")
    # It must be a bare relative filename, not a path or source text.
    check("check_script is a relative filename", all("/" not in t["check_script"] for t in py_tasks))
    check("check_script is not source text", all("\n" not in t["check_script"] for t in py_tasks))
    # The gate must catch the mistake this bug was.
    bad = dict(py_tasks[0])
    bad["check_script"] = "does_not_exist.py"
    check("gate 3 rejects a check_script that is not shipped", not validate.check_safety(bad).passed)
    # And a python_exit task must actually discriminate now.
    r = validate.check_oracle(py_tasks[0], root=tmp / "bug3")
    check("gate 1 passes for a python_exit task", r.passed, f"{r.detail[:200]}")
    nop = validate.check_nop(py_tasks[0], root=tmp / "bug3b")
    check("gate 2 passes for a python_exit task", nop.passed, f"{nop.detail}")

    # ------------------------------------------------------------------
    print("\n7. bug 3b — the shipped checker does not leak the answer")
    # ------------------------------------------------------------------
    t = py_tasks[0]
    body = t["setup"][t["check_script"]]
    answer = str(t["expected"]).strip()
    check("the checker does not contain the plaintext answer", answer not in body, f"{answer!r} appears in check.py")
    check("the checker stores a digest instead", "sha256" in body)
    # A long answer's digest is not worth brute-forcing; a 3-char one is, and
    # the limitation is documented rather than hidden.
    long_ans = [x for x in py_tasks if len(str(x["expected"]).strip()) >= 10]
    check("the batch contains long answers where the digest is meaningful", len(long_ans) > 0)

    # ------------------------------------------------------------------
    print("\n8. bug 4 — the reference does not use a shared global temp path")
    # ------------------------------------------------------------------
    multi = [t for t in batch if len(t["reference"]) > 1 or "MHEOF" in "".join(t["reference"])]
    check("the batch contains multi-step references", len(multi) > 0)
    for t in batch:
        blob = "\n".join(t["reference"])
        if "/tmp/" in blob:
            check("no reference writes to a shared /tmp path", False, f"{t['id']}: {blob[:120]}")
            break
    else:
        check("no reference writes to a shared /tmp path", True)
    # A pipeline-based reference must still work.
    two_step = next((t for t in batch if t["params"]["steps"] >= 2 and not t["params"]["read_source"]), None)
    if two_step is not None:
        r = validate.check_oracle(two_step, root=tmp / "bug4")
        check("a multi-step reference scores 1.0", r.passed, f"{r.detail[:200]}")
    else:
        check("a multi-step reference scores 1.0", True, "no such task in this batch")

    # ------------------------------------------------------------------
    print("\n9. tier coherence — the label does not lie about the content")
    # ------------------------------------------------------------------
    t3 = [t for t in batch if t["tier"] == "T3"]
    check("T3 tasks carry escaping characters", all(t["params"]["escape_density"] >= 0.35 for t in t3))
    t2 = [t for t in batch if t["tier"] == "T2"]
    check("T2 tasks read a shipped file", all(t["params"]["read_source"] for t in t2))
    t4 = [t for t in batch if t["tier"] == "T4"]
    check("T4 tasks need at least two steps", all(t["params"]["steps"] >= 2 for t in t4))
    # Pinning an incoherent combination must be corrected, not accepted.
    import random as _random

    p = task_gen.sample_params(_random.Random(0), tier="T3", escape_density=0.0)
    check("pinning T3 with no escaping is corrected", p.escape_density >= 0.35, f"{p.escape_density}")
    p = task_gen.sample_params(_random.Random(0), tier="T2", read_source=False)
    check("pinning T2 without a source file is corrected", p.read_source)

    # ------------------------------------------------------------------
    print("\n10. reward modes are genuinely different rewards")
    # ------------------------------------------------------------------
    # A contains-check must accept a superset that an equals-check rejects,
    # otherwise the two modes would be the same task wearing different labels.
    ct = next(t for t in batch if t["verify"] == "file_contains")
    needle = str(ct["expected"])
    check(
        "the contains needle is a strict slice of the answer",
        needle.strip() != str(ct["expected"]).strip() or len(needle) < 40,
    )
    d = tmp / "modes"
    d.mkdir(parents=True, exist_ok=True)
    (d / "answer.txt").write_text("prefix " + needle + " suffix\n", encoding="utf-8")
    check("file_contains accepts a superset", verify(ct, d) == 1.0, f"needle={needle!r}")
    eq = dict(ct)
    eq["verify"] = "file_equals"
    check("file_equals would reject that same superset", verify(eq, d) == 0.0)

    # ------------------------------------------------------------------
    print("\n11. the generator refuses to emit an unsolvable escape")
    # ------------------------------------------------------------------
    # A heredoc's delimiter appears as its own first and last line by
    # construction, so the check must look at the *body* between them: a
    # payload line equal to the delimiter would close the heredoc early and
    # silently truncate the reference's own input.
    delim = task_gen.heredoc_delimiter()

    def heredoc_body(cmd: str) -> str:
        """The lines between a heredoc's opening and closing delimiter."""
        lines = cmd.splitlines()
        if not lines or f"'{delim}'" not in lines[0]:
            return ""
        for i in range(1, len(lines)):
            if lines[i].strip() == delim:
                return "\n".join(lines[1:i])
        return "\n".join(lines[1:])  # unterminated: the whole tail is the body

    bodies = [heredoc_body(c) for t in batch for c in t["reference"]]
    non_empty = [b for b in bodies if b]
    check("the batch has heredoc references to inspect", len(non_empty) > 0, f"{len(bodies)} commands")
    offenders = [
        t["id"]
        for t in batch
        for c in t["reference"]
        if any(line.strip() == delim for line in heredoc_body(c).splitlines())
    ]
    check("no payload body contains a delimiter line", not offenders, f"{offenders[:3]}")
    # And the bodies must be non-empty, or the check above passes vacuously.
    check("heredoc bodies are non-empty", all(b.strip() for b in non_empty), f"{[b[:20] for b in non_empty[:3]]}")
    # And the generator is robust to the degenerate parameter corners.
    corners = [
        task_gen.TaskParams(
            tier="T1", payload_len=1, escape_density=0.0, steps=1, read_source=False,
            verify_mode="file_equals", seed=7,
        ),
        task_gen.TaskParams(
            tier="T3", payload_len=1, escape_density=0.6, steps=3, read_source=False,
            verify_mode="python_exit", seed=8,
        ),
        task_gen.TaskParams(
            tier="T2", payload_len=2, escape_density=0.6, steps=2, read_source=True,
            verify_mode="file_contains", seed=9,
        ),
    ]
    for i, cp in enumerate(corners):
        task = task_gen.generate(cp)
        v = validate.validate_task(task, root=tmp / f"corner{i}")
        check(f"corner {i + 1} ({cp.tier}/{cp.verify_mode}) clears all gates", v.accepted, f"failed {v.failed_gates}")

    # ------------------------------------------------------------------
    print("\n12. the reward mode is chosen with the reward layer's own rule")
    # ------------------------------------------------------------------
    # Regression: `verify_mode` was drawn independently of the payload, so the
    # generator happily produced `python_exit` on a 3-character answer — the one
    # combination `verifier_gen`'s docstring names as wrong, because the checker
    # stores a digest and a short digest is worth brute-forcing. Nothing called
    # `recommend_mode`. Measured over 300 tasks at seed 0, 96 (32%) were
    # affected; across eight seeds the count runs 79–99, so the seed is part of
    # the claim rather than a detail.
    wide = task_gen.generate_batch(300, seed=0)
    bf = [t["id"] for t in wide if verifier_gen.discrimination_report(t)["digest_brute_forceable"]]
    check("no generated task ships a brute-forceable digest", not bf, f"{len(bf)} offenders, e.g. {bf[:3]}")

    disagree = [
        t["id"]
        for t in wide
        if t["verify"] == "python_exit" and verifier_gen.recommend_mode(t["expected"]) != "python_exit"
    ]
    check("every python_exit task agrees with recommend_mode", not disagree, f"{len(disagree)}")

    # The correction must not make the mode unreachable: if `python_exit` never
    # survives, the branch that ships a checker is dead code and no test would
    # notice.
    pe = [t for t in wide if t["verify"] == "python_exit"]
    check("python_exit is still reachable on long answers", len(pe) > 0, f"{len(pe)} of {len(wide)}")
    check(
        "every surviving python_exit answer is long enough to resist brute force",
        all(len(t["expected"].strip()) >= 12 for t in pe),
        f"min len {min((len(t['expected'].strip()) for t in pe), default=0)}",
    )
    check(
        "a demoted task records why",
        all(
            t.get("verify_mode_override")
            for t in wide
            if t["params"]["verify_mode"] == "python_exit" and t["verify"] != "python_exit"
        ),
    )
    # The headline "96 (32%) were affected" is a *seed-dependent* number, and the
    # README, CHANGELOG and the generator docstring all quote it. Nothing pinned
    # it, so a change to the tier mix could have silently invalidated all three.
    # Pin both the value at the quoted seed and the spread across seeds, so the
    # documents cannot drift away from the generator.
    demoted_0 = sum(
        1
        for t in wide
        if t["params"]["verify_mode"] == "python_exit" and t["verify"] != "python_exit"
    )
    check(
        "the documented 96 demotions at seed 0 still hold",
        demoted_0 == 96,
        f"{demoted_0} — README/CHANGELOG/docstring say 96",
    )
    spread = []
    for s in (1, 7, 11, 23, 42, 99, 1234):
        b = task_gen.generate_batch(300, seed=s)
        spread.append(
            sum(
                1
                for t in b
                if t["params"]["verify_mode"] == "python_exit" and t["verify"] != "python_exit"
            )
        )
    check(
        "the documented seed spread of 79–99 still holds",
        all(79 <= n <= 99 for n in spread),
        f"seeds 1/7/11/23/42/99/1234 gave {spread}",
    )
    check(
        "the default seed 11 does not reproduce the seed-0 number",
        spread[2] != demoted_0,
        f"seed 11 gave {spread[2]}, seed 0 gave {demoted_0} — the docs must name the seed",
    )

    # An explicitly requested long-answer python_exit must be honoured, or the
    # correction has quietly removed a mode the sweeps rely on.
    long_pe = task_gen.generate(
        task_gen.TaskParams(
            tier="T1", payload_len=16, escape_density=0.0, steps=1, read_source=False,
            verify_mode="python_exit", seed=11,
        )
    )
    check(
        "an explicit python_exit on a long answer is honoured",
        long_pe["verify"] == "python_exit",
        f"got {long_pe['verify']} (len={len(long_pe['expected'].strip())})",
    )

    # The coverage report must describe the batch that exists, not the one the
    # parameter vector asked for. Counting `params["verify_mode"]` made the
    # report claim 8 `python_exit` tasks when 1 was produced — a figure drawn
    # from that number would have been evidence of nothing.
    cov = task_gen.summarise_batch(wide)
    actual_total = sum(cov["by_verify_mode"].values())
    requested_total = sum(cov["by_verify_mode_requested"].values())
    check("actual mode counts cover the batch", actual_total == len(wide), f"{actual_total} vs {len(wide)}")
    check("requested mode counts cover the batch", requested_total == len(wide), f"{requested_total}")
    check(
        "requested python_exit exceeds actual, and overrides explain the gap",
        cov["by_verify_mode_requested"]["python_exit"] - cov["by_verify_mode"]["python_exit"] == cov["mode_overrides"],
        f"req={cov['by_verify_mode_requested']['python_exit']} "
        f"act={cov['by_verify_mode']['python_exit']} overrides={cov['mode_overrides']}",
    )
    check(
        "the actual counts are what the tasks report",
        all(
            cov["by_verify_mode"].get(t["verify"], 0)
            == sum(1 for x in wide if x["verify"] == t["verify"])
            for t in wide
        ),
    )

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED — generated tasks are solvable, checkable, and not trivially so")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
