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
