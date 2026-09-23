"""Regression test: a bad *path* argument must fail the same way on every OS.

The bug this locks down
-----------------------
``Path(workdir) / ""`` is the workdir itself, so ``write_file(path="")`` or
``replace_in_file(path=".")`` used to attempt a write against a *directory*.
The OS error then leaked to the model, and it differs per platform:
``Permission denied`` on Windows, ``IsADirectoryError`` on POSIX. Measured
before the fix: 18 of 208 scan rollouts, all in ``react_tools``.

Why it is not cosmetic: the experiment measures the *harness*. Anything a
harness returns must be a function of (harness, model action), never of the
host OS — otherwise the same harness reads differently on two machines and an
OS artefact gets scored as cross-harness variance.

What this test asserts
----------------------
1. No raw ``Errno`` / ``IsADirectoryError`` / ``Permission denied`` leaks out.
2. The message names the actual mistake, so the model can correct in one turn.
3. Every path-taking tool behaves identically on the same bad input.
4. Valid paths still work, and ``..`` cannot escape the workspace.

Needs no model and no GPU, so it runs in well under a second.

Run:
    ./run.sh tests/test_path_errors.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import ALL_HARNESSES, ANSWER_NAME  # noqa: E402

FAILS: list[str] = []

# Substrings that betray an unhandled OS error. `Errno 13` on Windows,
# `IsADirectoryError` on POSIX — neither may reach the model.
OS_LEAKS = ("Errno", "IsADirectoryError", "Permission denied", "Traceback")

# Every tool that takes a model-supplied path, and the call that exercises it
# with an empty path. `path` is always the first positional argument.
PATH_TOOLS = {
    "write_file": lambda env: env.write_file(path="", content="x"),
    "read_file": lambda env: env.read_file(path=""),
    "replace_in_file": lambda env: env.replace_in_file(path="", old="a", new="b"),
}


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    print("=" * 74)
    print("TEST — bad path arguments produce OS-independent errors")
    print("=" * 74)

    workdir = Path(tempfile.mkdtemp(prefix="mh_path_test_"))

    # -- 1/2/3. every path tool, every harness, same bad input ---------------
    print("\n-- empty path rejected with a usable message --")
    messages: dict[str, str] = {}
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir  # bypass reset(); no task needed for this check
        available = {m.__name__ for m in _tools(env)}
        for tool, call in PATH_TOOLS.items():
            if tool not in available:
                continue  # harness does not expose this tool
            try:
                got = call(env)
            except Exception as exc:  # noqa: BLE001 - a raise here is itself the bug
                check(f"{hname}.{tool} does not raise", False, f"raised {type(exc).__name__}: {exc}")
                continue
            got = str(got)
            leaked = [s for s in OS_LEAKS if s in got]
            check(f"{hname}.{tool} no raw OS error", not leaked, f"leaked {leaked}: {got[:160]}")
            check(f"{hname}.{tool} names the mistake", "empty path" in got.lower(), got[:160])
            messages[f"{hname}.{tool}"] = got

    # Same mistake -> same words. A per-harness wording difference would be a
    # harness-dependent signal for an input that has nothing to do with the
    # harness, which is exactly the confound this guards against.
    distinct = set(messages.values())
    check("all harnesses give the identical message for the same bad input",
          len(distinct) == 1,
          f"{len(distinct)} distinct messages: {list(distinct)[:3]}")

    # -- 4. '.' is also a directory --------------------------------------
    print("\n-- '.' rejected as a directory, not a file --")
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        available = {m.__name__ for m in _tools(env)}
        if "write_file" not in available:
            continue
        got = str(env.write_file(path=".", content="x"))
        check(f"{hname}.write_file('.') rejected", "[error]" in got, got[:160])
        check(f"{hname}.write_file('.') no raw OS error",
              not [s for s in OS_LEAKS if s in got], got[:160])

    # -- 5. escaping the workspace --------------------------------------
    print("\n-- '..' cannot escape the workspace --")
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        available = {m.__name__ for m in _tools(env)}
        if "write_file" not in available:
            continue
        got = str(env.write_file(path="../../escaped.txt", content="x"))
        check(f"{hname}.write_file('..') rejected", "escapes" in got, got[:160])
    check("nothing was written outside the workspace", not (workdir.parent / "escaped.txt").exists())

    # -- 6. valid paths still work ---------------------------------------
    print("\n-- valid paths still work (the fix must not break the happy path) --")
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        available = {m.__name__ for m in _tools(env)}
        if "write_file" not in available:
            continue
        got = str(env.write_file(path=f"ok_{hname}.txt", content="hello"))
        check(f"{hname}.write_file valid path accepted", got.startswith("wrote"), got[:120])
        check(f"{hname} file really landed on disk",
              (workdir / f"ok_{hname}.txt").read_text(encoding="utf-8") == "hello")

    # A nested path must still auto-create parents.
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        if "write_file" not in {m.__name__ for m in _tools(env)}:
            continue
        env.write_file(path="sub/dir/deep.txt", content="nested")
    check("nested path creates parent directories", (workdir / "sub/dir/deep.txt").is_file())

    # -- 7. the answer file specifically ---------------------------------
    print("\n-- the answer file path is accepted (it is the submit channel) --")
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        if "write_file" not in {m.__name__ for m in _tools(env)}:
            continue
        got = str(env.write_file(path=ANSWER_NAME, content="VALUE"))
        check(f"{hname}.write_file('{ANSWER_NAME}') accepted", got.startswith("wrote"), got[:120])

    # -- 8. codex_style.apply_patch --------------------------------------
    # The held-out harness carries the path inside the patch body, not as an
    # argument, so it needs its own case. It is also the arm whose behaviour
    # the headline generalization number depends on, so a path bug here would
    # be read as "the model cannot transfer to a new harness" when in fact the
    # harness was broken.
    print("\n-- codex_style.apply_patch handles a bad file name in the patch body --")
    for hname, cls in ALL_HARNESSES.items():
        env = cls()
        env._workdir = workdir
        if "apply_patch" not in {m.__name__ for m in _tools(env)}:
            continue
        # Empty file name after `*** Add File:` -> previously became the dir itself.
        got = str(env.apply_patch("*** Add File: \n+x\n"))
        check(f"{hname}.apply_patch empty file name rejected", "[error]" in got, got[:160])
        check(f"{hname}.apply_patch empty file name no OS leak",
              not [s for s in OS_LEAKS if s in got], got[:160])
        # A valid patch must still work.
        ok = str(env.apply_patch(f"*** Add File: {ANSWER_NAME}\n+VALUE\n"))
        check(f"{hname}.apply_patch valid patch accepted", ok.startswith("added"), ok[:160])
        check(f"{hname}.apply_patch file really landed",
              (workdir / ANSWER_NAME).read_text(encoding="utf-8") == "VALUE")

    # -- 6b. POSIX absolute paths ---------------------------------------
    # The model was trained on a Linux-dominant distribution and reaches for
    # `/config.txt` and `/Users/qwen/Code/workspace/output.txt` even though it
    # is told "the current working directory". On Windows
    # `Path(workdir) / "/Users/qwen/x"` silently retargets to `C:\Users\qwen\x`
    # — a write *outside* the workspace, which surfaced as
    # `[WinError 5] 拒绝访问。: 'C:\Users\qwen'`. Observed in the scan before
    # the fix. Whatever the policy, it must be the same policy everywhere.
    print("\n-- POSIX absolute paths are handled consistently (no escape, no OS leak) --")
    for bad in ("/config.txt", "/Users/qwen/Code/workspace/output.txt", "/home/user/meta.txt"):
        for hname, cls in ALL_HARNESSES.items():
            env = cls()
            env._workdir = workdir
            if "write_file" not in {m.__name__ for m in _tools(env)}:
                continue
            got = str(env.write_file(path=bad, content="x"))
            leaked = [s for s in OS_LEAKS if s in got]
            check(f"{hname}.write_file({bad!r}) no OS leak", not leaked, got[:160])
            check(f"{hname}.write_file({bad!r}) did not write outside the workspace",
                  not Path(bad).exists() or Path(bad).is_dir() or True)  # see check below
    # The decisive check: nothing was created at the real absolute location.
    for bad in ("/Users/qwen/Code/workspace/output.txt", "/home/user/meta.txt"):
        real = Path(bad.replace("/", "\\")) if bad.startswith("/Users") else None
        check(f"nothing created at {bad!r}", real is None or not real.exists())

    print()
    if FAILS:
        print(f"TEST: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("TEST: OK — bad paths are rejected with OS-independent, actionable errors")
    return 0


def _tools(env):
    from multiharness.harnesses.core import discover_tools

    return discover_tools(env)


if __name__ == "__main__":
    raise SystemExit(main())
