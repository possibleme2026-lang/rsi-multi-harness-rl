"""Regression test: the verifier's three modes must actually discriminate.

The bug this locks down
-----------------------
``verify(mode="python_exit")`` decided pass/fail by reading the *text* the
shell returned::

    ok = "[error]" not in out and "Traceback" not in out and "Error" not in out
    return 1.0 if ok else 0.0

``_run_shell`` renders a *silent* command as ``"(no output, exit=N)"``, so
``exit=0`` and ``exit=1`` produce the same string modulo one digit. Neither
contains ``[error]``, a traceback, or the word ``Error``. The check therefore
returned 1.0 for **both**. A self-generated Python verifier that printed
nothing and exited non-zero — the single most common way for a check to fail —
was indistinguishable from one that passed, so every generated verifier was a
constant-1.0 oracle.

That is a false *positive*, which is the direction that silently corrupts a
result: a task whose verifier cannot fail looks like a task the model solved,
and it inflates every pass rate computed from it.

What this test asserts
----------------------
1. ``_run_shell_rc`` reports the true exit status, including for silent
   commands and including non-zero ones.
2. A silent ``exit 1`` scores 0.0 and a silent ``exit 0`` scores 1.0 — the
   exact pair the old code could not tell apart.
3. A verifier that fails *loudly* still fails: printing a reason cannot buy a
   pass.
4. ``file_equals`` and ``file_contains`` discriminate in both directions.
5. A task that passes its own reference answer scores 1.0 and one that does
   not scores 0.0 — the property the whole difficulty scan rests on.
6. ``apply_patch``'s shell fallback is a *legal* command: a valid unified diff
   is applied, the file appears, and the patched answer scores 1.0. The
   fallback previously appended ``|| echo ...`` after a heredoc terminator,
   making the whole command a syntax error and silently rejecting every valid
   patch — which is what made the held-out harness look unsolvable.

Needs no model and no GPU, so it runs in well under a second.

Run:
    ./run.sh tests/test_verifier_modes.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses.core import (  # noqa: E402
    ANSWER_NAME,
    _run_shell_rc,
    verify,
)

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    print("=" * 74)
    print("TEST — verifier modes discriminate (the python_exit false-positive)")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="mh-verifier-modes-"))

    # -- 1. the exit status is carried out of band -------------------------
    print("\n-- 1. _run_shell_rc reports the true exit status --")
    for command, want in (
        ("exit 0", 0),
        ("exit 1", 1),
        ("exit 7", 7),
        ("true", 0),
        ("false", 1),
        ("echo hello", 0),
    ):
        _out, rc = _run_shell_rc(command, cwd=tmp, timeout=10)
        check(f"`{command}` -> rc={want}", rc == want, f"got rc={rc!r}")

    # A silent non-zero exit and a silent zero exit must be distinguishable.
    out0, rc0 = _run_shell_rc("exit 0", cwd=tmp, timeout=10)
    out1, rc1 = _run_shell_rc("exit 1", cwd=tmp, timeout=10)
    check("silent exit 0 and silent exit 1 differ in rc", rc0 != rc1, f"{rc0!r} vs {rc1!r}")
    check("the text alone cannot tell them apart (why rc exists)",
          out0.replace("0", "N") == out1.replace("1", "N"), f"{out0!r} vs {out1!r}")

    # A timeout is not a verdict on the command: no status to report.
    _out, rc = _run_shell_rc("sleep 30", cwd=tmp, timeout=2)
    check("a timed-out command reports rc=None, not 0", rc is None, f"got rc={rc!r}")

    # -- 2. the exact pair the old code conflated --------------------------
    print("\n-- 2. python_exit: silent success vs silent failure --")

    def py_exit(script_body: str, name: str) -> float:
        """Write a check script and grade it through verify()."""
        (tmp / name).write_text(script_body, encoding="utf-8")
        task = {"id": "gen", "verify": "python_exit", "check_script": name, "expected": ""}
        return verify(task, tmp)

    # Fails silently: writes nothing, exits non-zero. The old code scored 1.0.
    got = py_exit("import sys\nsys.exit(1)\n", "silent_fail.py")
    check("silent non-zero exit scores 0.0", got == 0.0, f"got {got}")

    # Passes silently. Must score 1.0 — the fix must not invert the happy path.
    got = py_exit("import sys\nsys.exit(0)\n", "silent_pass.py")
    check("silent zero exit scores 1.0", got == 1.0, f"got {got}")

    # A real check: reads answer.txt and branches on it.
    (tmp / ANSWER_NAME).write_text("42", encoding="utf-8")
    got = py_exit(
        f"import pathlib, sys\n"
        f"got = pathlib.Path({ANSWER_NAME!r}).read_text().strip()\n"
        f"sys.exit(0 if got == '42' else 1)\n",
        "real_check_pass.py",
    )
    check("a check that reads the answer and agrees scores 1.0", got == 1.0, f"got {got}")

    (tmp / ANSWER_NAME).write_text("43", encoding="utf-8")
    got = py_exit(
        f"import pathlib, sys\n"
        f"got = pathlib.Path({ANSWER_NAME!r}).read_text().strip()\n"
        f"sys.exit(0 if got == '42' else 1)\n",
        "real_check_fail.py",
    )
    check("the same check on a wrong answer scores 0.0", got == 0.0, f"got {got}")

    # -- 3. a loud failure must still fail --------------------------------
    print("\n-- 3. a loud failure cannot buy a pass by printing --")
    got = py_exit("import sys\nprint('Error: this looks like a failure')\nsys.exit(1)\n", "loud_fail.py")
    check("a loud non-zero exit scores 0.0", got == 0.0, f"got {got}")
    got = py_exit("import sys\nprint('all good')\nsys.exit(0)\n", "loud_pass.py")
    check("a loud zero exit scores 1.0", got == 1.0, f"got {got}")

    # A crashing script has no clean status; it must not score 1.0.
    got = py_exit("raise RuntimeError('boom')\n", "crash.py")
    check("a crashing check scores 0.0", got == 0.0, f"got {got}")

    # Missing script is a failure, not a free pass.
    got = verify({"id": "gen", "verify": "python_exit", "expected": ""}, tmp)
    check("a python_exit task with no check_script scores 0.0", got == 0.0, f"got {got}")

    # -- 4. file_equals / file_contains discriminate both ways -------------
    print("\n-- 4. file_equals and file_contains discriminate --")
    (tmp / ANSWER_NAME).write_text("hello world", encoding="utf-8")
    eq = {"id": "e", "verify": "file_equals", "expected": "hello world"}
    check("file_equals: exact match scores 1.0", verify(eq, tmp) == 1.0)
    check("file_equals: trailing newline is tolerated",
          verify(eq, tmp) == 1.0)

    (tmp / ANSWER_NAME).write_text("hello world\n", encoding="utf-8")
    check("file_equals: a trailing newline still matches", verify(eq, tmp) == 1.0)

    (tmp / ANSWER_NAME).write_text("hello worlds", encoding="utf-8")
    check("file_equals: a superset does NOT match", verify(eq, tmp) == 0.0)

    (tmp / ANSWER_NAME).write_text("HELLO WORLD", encoding="utf-8")
    check("file_equals is case-sensitive", verify(eq, tmp) == 0.0)

    (tmp / ANSWER_NAME).unlink()
    check("file_equals: no file scores 0.0", verify(eq, tmp) == 0.0)

    (tmp / ANSWER_NAME).write_text("prefix hello world suffix", encoding="utf-8")
    con = {"id": "c", "verify": "file_contains", "expected": "hello world"}
    check("file_contains: substring present scores 1.0", verify(con, tmp) == 1.0)
    (tmp / ANSWER_NAME).write_text("nothing here", encoding="utf-8")
    check("file_contains: substring absent scores 0.0", verify(con, tmp) == 0.0)
    (tmp / ANSWER_NAME).unlink()
    check("file_contains: no file scores 0.0", verify(con, tmp) == 0.0)

    # -- 5. unknown mode is an error, not a silent pass -------------------
    print("\n-- 5. an unknown verify mode raises rather than passing --")
    try:
        verify({"id": "x", "verify": "no_such_mode", "expected": ""}, tmp)
        check("unknown verify mode raises", False, "it returned a value instead")
    except ValueError:
        check("unknown verify mode raises ValueError", True)

    # -- 6. the default mode is file_equals ------------------------------
    # A task dict with no `verify` key must not silently fall through to
    # "always pass" — the default has to be the strict comparison.
    print("\n-- 6. a task with no verify key defaults to the strict mode --")
    (tmp / ANSWER_NAME).write_text("right", encoding="utf-8")
    check("default mode scores 1.0 on the right answer",
          verify({"id": "d", "expected": "right"}, tmp) == 1.0)
    (tmp / ANSWER_NAME).write_text("wrong", encoding="utf-8")
    check("default mode scores 0.0 on the wrong answer",
          verify({"id": "d", "expected": "right"}, tmp) == 0.0)

    # -- 7. the apply_patch fallback is a legal shell command --------------
    # The fallback used to append `|| echo ...` *after* the heredoc
    # terminator. A heredoc ends at its terminator, so the `||` became a line
    # of its own and the whole command was a bash syntax error — every valid
    # unified diff came back as `syntax error near unexpected token '||'` and
    # was never applied. That is a harness defect that reads as a model
    # failure, and it is why the held-out harness scored 0/8 on every eval
    # task in every arm.
    print("\n-- 7. apply_patch applies a valid unified diff --")
    import shutil as _shutil

    from multiharness.harnesses.pool import CodexStyleEnv  # noqa: E402
    from multiharness.tasks import load as _load_tasks  # noqa: E402

    _load_tasks()
    env = CodexStyleEnv()
    # `reset` is what binds both the workdir *and* the task, and `get_reward`
    # needs the task — setting `_workdir` alone leaves the reward unable to
    # decide anything, which would make the last check below pass for the
    # wrong reason (or, as it first did here, fail for one).
    env.reset(task_id="t1-09")
    patch_dir = Path(env._workdir)
    assert patch_dir is not None and patch_dir.is_dir()

    diff = f"--- {ANSWER_NAME}\n+++ {ANSWER_NAME}\n@@ -0,0 +1 @@\n+reward is one"
    out = env.apply_patch(diff)
    check("a valid unified diff does not produce a shell syntax error",
          "syntax error" not in out.lower(), f"got: {out!r}")
    check("the file is actually created by the patch",
          (patch_dir / ANSWER_NAME).is_file(), f"dir holds: {sorted(p.name for p in patch_dir.iterdir())}")
    check("the patched content is exactly the expected line",
          (patch_dir / ANSWER_NAME).read_text(encoding="utf-8").strip() == "reward is one")

    # The end-to-end property: patching in a correct answer has to score 1.0.
    # Without this the fix could pass the three checks above and still not
    # move a reward, which is the only thing the eval reads.
    check("a patched correct answer scores 1.0",
          env.get_reward() == 1.0, f"got {env.get_reward()!r}")

    _shutil.rmtree(patch_dir, ignore_errors=True)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 74)
    if FAILS:
        print(f"TEST: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("TEST: OK — every verifier mode discriminates pass from fail")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
