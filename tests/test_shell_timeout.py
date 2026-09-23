"""Regression test for the shell-timeout deadlock.

The bug this guards against, observed live and worth 84 minutes of idle GPU:

    ``subprocess.run(capture_output=True, timeout=30)`` kills only the direct
    child when the timeout fires, then drains the pipes. A grandchild that
    inherited the write end keeps them open, so the drain never sees EOF and
    blocks forever. The timeout fires and then the cleanup itself hangs — the
    safety net becomes the trap.

    The command that triggered it in practice spawns a child which outlives
    the shell, e.g. ``bash -c 'sleep 300 &'`` or a tool that daemonises.

Four cases are checked, each a distinct way the old code could hang:

  1. a command whose grandchild outlives the shell must return at the timeout
  2. a command reading stdin must not block (stdin is DEVNULL)
  3. a command producing more output than the buffer must not deadlock on a
     full pipe
  4. the normal path still returns correct output and exit status

Run:
    ./run.sh tests/test_shell_timeout.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses.core import _run_shell  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def timed(fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    return out, time.time() - t


def _kill_tree_is_guarded() -> bool:
    """Does ``_kill_tree`` refuse to signal the caller's own process group?

    Read from the source rather than exercised, because the failure mode is
    "the caller dies" — there is no way to assert on that from inside the
    caller. A source check is weak in general; here it is the only option, and
    the property it guards (never SIGKILL your own process group) is not one a
    refactor is likely to reproduce by accident.
    """
    import inspect

    from multiharness.harnesses.core import _kill_tree

    src = inspect.getsource(_kill_tree)
    return "getpgid(0)" in src and "killpg" in src


def main() -> int:
    print("=" * 74)
    print("SHELL TIMEOUT / DEADLOCK REGRESSION")
    print("=" * 74)
    # A throwaway directory, not `outputs/`. This test writes stray files
    # (`out.txt`) and leaves a `sleep 300` running; keeping that out of the
    # artifact tree means a failed run cannot be mistaken for a real result.
    tmp = Path(tempfile.mkdtemp(prefix="mh-shell-timeout-"))
    print(f"scratch: {tmp}")

    # -- 1. grandchild outliving the shell --------------------------------
    # This is the exact shape that hung for 84 minutes.
    #
    # Note what the *old* implementation actually did here: bash does not wait
    # for a backgrounded job, so the shell exits immediately and the direct
    # child dies long before the timeout. Nothing times out at all — the hang
    # is entirely in `communicate()` afterwards, which drains the pipes and
    # never sees EOF because the surviving grandchild still holds the write
    # end. That is why the symptom was "no output for 84 minutes" rather than
    # "timed out after 30s", and why a timeout-only fix would not have helped.
    #
    # So this case asserts *prompt return with the pre-timeout output intact*,
    # not a timeout message: there is no timeout to report.
    print("\n-- 1. grandchild outlives the shell (the original hang) --")
    out, dt = timed(_run_shell, "sleep 300 & echo started", cwd=tmp, timeout=3)
    check("returned instead of hanging", dt < 20, f"{dt:.1f}s")
    check("kept the output printed before the shell exited", "started" in out, out.strip()[:80])

    # A command that blocks in the foreground *does* hit the timeout, and that
    # path must also return rather than hanging in cleanup.
    print("\n-- 1b. foreground blocking command hits the timeout --")
    out, dt = timed(_run_shell, "sleep 300", cwd=tmp, timeout=3)
    check("returned instead of hanging", dt < 20, f"{dt:.1f}s")
    check("reports the timeout", "timed out" in out, out.strip()[:80])

    # -- 2. command that reads stdin --------------------------------------
    print("\n-- 2. command that reads stdin --")
    out, dt = timed(_run_shell, "cat", cwd=tmp, timeout=5)
    check("bare `cat` did not block on stdin", dt < 10, f"{dt:.1f}s")

    # -- 3. large output (full-pipe deadlock) -----------------------------
    print("\n-- 3. output far larger than a pipe buffer --")
    out, dt = timed(_run_shell, "yes A | head -c 2000000", cwd=tmp, timeout=30)
    check("2 MB of output returned", dt < 30 and len(out) > 0, f"{dt:.1f}s, {len(out)} chars")
    check("output was truncated, not lost", "truncated" in out)

    # -- 4. normal path ---------------------------------------------------
    print("\n-- 4. normal command still works --")
    out, dt = timed(_run_shell, "echo -n hello && echo world >&2", cwd=tmp, timeout=10)
    check("stdout captured", "hello" in out, out.strip()[:60])
    check("stderr merged into output", "world" in out, out.strip()[:60])
    check("fast path is fast", dt < 5, f"{dt:.1f}s")

    out, _ = timed(_run_shell, "exit 7", cwd=tmp, timeout=10)
    check("silent failure reports exit code", "exit=7" in out, out.strip()[:60])

    out, _ = timed(_run_shell, "echo -n abc > out.txt && cat out.txt", cwd=tmp, timeout=10)
    check("file redirection works", out.strip() == "abc", repr(out.strip()[:40]))

    # -- 5. the child must NOT share our process group --------------------
    # This is the difference between a timeout that kills the runaway command
    # and a timeout that kills the caller. `_kill_tree` signals the child's
    # process group; if the child inherited ours, that group is the whole
    # harness process -- and in CI, the runner's own step shell.
    #
    # Observed live: the first CI run of this repository sat `in_progress` for
    # half an hour instead of failing. The step's shell had been killed by its
    # own timeout cleanup, so the job never reported a result. A regression here
    # is therefore silent, which is exactly why it needs its own assertion.
    print("\n-- 5. child runs in its own process group (killpg cannot hit us) --")
    if os.name == "nt":
        print("  [skip] process groups are a POSIX concept; Windows uses taskkill /T")
    else:
        import subprocess as _sp

        from multiharness.harnesses.core import BASH

        child_pgid = _sp.run(
            [BASH, "-c", "ps -o pgid= -p $$"],
            capture_output=True, text=True, start_new_session=True,
        ).stdout.strip()
        our_pgid = str(os.getpgid(0))
        check("a session-leader child gets a fresh pgid",
              child_pgid != our_pgid, f"child={child_pgid} ours={our_pgid}")
        # And the real thing: run through the harness and compare groups.
        marker = tmp / "pgid.txt"
        _run_shell(f"ps -o pgid= -p $$ | tr -d ' ' > {marker.name}", cwd=tmp, timeout=10)
        if marker.is_file():
            observed = marker.read_text(encoding="utf-8").strip()
            check("_run_shell's child is in its own process group",
                  observed != our_pgid, f"child={observed} ours={our_pgid}")
            check("_kill_tree refuses to signal the caller's own group",
                  _kill_tree_is_guarded(), "guard present in _kill_tree")
        else:
            check("pgid probe produced output", False, "no pgid.txt written")

    print("\n" + "=" * 74)
    # Best-effort: case 1 deliberately leaves a `sleep 300` alive, and on
    # Windows a live process holding the directory as its cwd blocks deletion.
    # A leftover temp directory is harmless; a failed test that reports a
    # spurious cleanup error is not.
    shutil.rmtree(tmp, ignore_errors=True)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
