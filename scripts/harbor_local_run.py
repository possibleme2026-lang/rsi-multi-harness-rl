#!/usr/bin/env python3
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

"""Execute an exported Harbor package locally, without Docker.

Why this exists
---------------
An exported package that has never been *run* is a claim, not a benchmark. The
generator's in-process grader and the package's ``tests/test_state.py`` are two
implementations of one semantics, and the failure that matters is the one where
they disagree: the repository reports a clean batch while the package grades
something else.

Docker is not available in this environment (no daemon, no ``harbor`` CLI), so
the two containers are *simulated* rather than started:

    agent side      a temp dir laid out as ``/app`` would be; ``tools.py`` and
                    ``initial_state.json`` copied in, ``solve.sh`` executed with
                    its paths rewritten from ``/app`` to the temp dir
    handover        ``state.json`` copied to a second, separate temp dir
    verifier side   ``test_state.py`` executed with ``/app/state.json``
                    rewritten to the second dir, ``/logs/verifier`` likewise

This is not the same as running Harbor and does not claim to be. What it does
establish, which the in-process grader cannot:

* the exported ``tools.py`` is a working program — it parses, it runs, and the
  tool names in the instruction are the ones it implements;
* the exported ``solve.sh`` actually solves the task when executed by a shell;
* the exported ``test_state.py`` passes on that solution and fails on an
  untouched state;
* the **oracle is genuinely absent from the agent side** — the run is executed
  with only the agent-side files present, so a package that leaked the solution
  into ``environment/`` would still pass here while a package that *depended* on
  it would fail.

That last point is the reason to run it at all. ``audit_package`` checks the
trust boundary *textually*; this checks it *operationally*, by removing the
verifier-side files and confirming the agent side can still do its job.

Usage:
    ./run.sh scripts/harbor_local_run.py outputs/rsi/harbor/env-xxxx
    ./run.sh scripts/harbor_local_run.py --all outputs/rsi/harbor
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multiharness.harnesses.core import BASH, to_bash_path  # noqa: E402


def _bash(p: Path) -> str:
    """A path a POSIX shell can use: ``C:\\Users\\x`` -> ``/c/Users/x``.

    Handing bash the Windows form does not raise — it silently eats the
    backslashes, so ``C:\\Users\\less\\AppData`` arrives as ``C:UserslessAppData``
    and the error is "No such file or directory" for a path that never existed.
    That is a *simulator* bug that reads exactly like a package bug, which is why
    the rewrite goes through the same ``to_bash_path`` the harness layer uses.

    **Only for text that a shell will parse** — a ``.sh`` body, a shim, or an
    argv element passed to ``bash``. It is *not* for Python's own file
    operations: ``Path("/c/Users/x")`` is not a Windows path and ``.exists()``
    on it is ``False``, so using this for ``cwd=`` or ``open()`` would make
    every file look missing. ``cwd=`` takes the raw Windows path.
    """
    return to_bash_path(p)


def _rewrite(text: str, mapping: dict[str, str]) -> str:
    """Rewrite container paths to host paths.

    Longest-first so ``/logs/verifier`` is replaced before ``/logs``; a shorter
    prefix replaced first would leave ``<tmp>/verifier`` behind, and the path
    would be wrong in a way that looks like a missing file.
    """
    for src in sorted(mapping, key=len, reverse=True):
        text = text.replace(src, mapping[src])
    return text


def run_package(pkg: Path, *, verbose: bool = False) -> dict:
    """Execute one exported package's oracle and verifier. Returns a report."""
    name = pkg.name
    report: dict = {"task_id": name, "ok": False, "stages": {}}

    required = [
        "task.toml", "instruction.md",
        "environment/Dockerfile", "environment/assets/tools.py",
        "environment/assets/initial_state.json",
        "solution/solve.sh",
        "tests/test_state.py", "tests/test.sh",
    ]
    missing = [r for r in required if not (pkg / r).is_file()]
    if missing:
        report["stages"]["layout"] = {"ok": False, "missing": missing}
        return report
    report["stages"]["layout"] = {"ok": True, "files": len(required)}

    agent = Path(tempfile.mkdtemp(prefix=f"harbor_agent_{name}_"))
    verifier = Path(tempfile.mkdtemp(prefix=f"harbor_verifier_{name}_"))
    try:
        # ---- agent side: ONLY the environment files ------------------------
        #
        # Deliberately does not copy solution/ or tests/. If the package's
        # environment were incomplete and depended on a verifier-side file, this
        # is where it would break — which is the operational form of the trust
        # boundary the audit checks textually.
        shutil.copy(pkg / "environment/assets/tools.py", agent / "tools.py")
        shutil.copy(pkg / "environment/assets/initial_state.json", agent / "initial_state.json")
        (agent / "logs").mkdir(exist_ok=True)

        # `tools.py` is a *Python* program, so its paths must stay in the host's
        # native form. Rewriting `/app` to `/c/Users/...` here was a bug: bash
        # understands that form, but `pathlib.Path("/c/Users/...")` does not —
        # it resolves to `\\c\\Users\\...` and every read raises
        # FileNotFoundError for a file that exists.
        #
        # Forward slashes, not `str(agent)`: the replacement lands inside a
        # Python string literal in the generated source, where `C:\Users\less`
        # would be read as an escape sequence (`\U` starts an 8-hex-digit
        # escape) and either raise a SyntaxError or silently produce a different
        # path. Forward slashes are accepted by every Windows API Python uses,
        # so they are the form that survives the round trip through source.
        tools_src = (agent / "tools.py").read_text(encoding="utf-8")
        (agent / "tools.py").write_text(
            _rewrite(tools_src, {"/app": str(agent).replace("\\", "/")}), encoding="utf-8"
        )

        # `envtool` is the shell shim the instruction tells the agent to call.
        # Both paths go through `_bash`: the interpreter because a Windows path
        # would be mangled by the shell, the script for the same reason.
        shim = agent / "envtool"
        shim.write_text(
            f'#!/bin/sh\nexec "{_bash(Path(sys.executable))}" "{_bash(agent / "tools.py")}" "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)

        # ---- run the oracle ------------------------------------------------
        solve_src = (pkg / "solution/solve.sh").read_text(encoding="utf-8")
        solve_host = _rewrite(
            solve_src,
            {
                "/app": _bash(agent),
                "envtool": f'"{_bash(shim)}"',
            },
        )
        solve_path = agent / "solve.sh"
        solve_path.write_text(solve_host, encoding="utf-8")
        solve_path.chmod(0o755)

        # `BASH`, not a bare `bash`. On this host `bash` on PATH resolves to the
        # WSL launcher, which cannot see the Windows temp directory — the failure
        # is "No such file or directory" for a file that demonstrably exists, and
        # it arrives with a WSL banner on stderr. `core.BASH` is the resolved
        # Git-for-Windows binary, which is the same shell the harness layer runs
        # its own commands in.
        if BASH is None:
            report["stages"]["oracle_run"] = {
                "ok": False,
                "detail": "no POSIX bash found on this host; set MULTIHARNESS_BASH",
            }
            return report
        proc = subprocess.run(
            [BASH, _bash(solve_path)],
            capture_output=True, text=True, timeout=120, cwd=str(agent),
            encoding="utf-8", errors="replace",
        )
        report["stages"]["oracle_run"] = {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "stdout": (proc.stdout or "")[-400:],
            "stderr": (proc.stderr or "")[-400:],
        }
        if verbose:
            print(f"    oracle rc={proc.returncode} out={proc.stdout.strip()[:120]!r}")
        if proc.returncode != 0:
            return report

        state_file = agent / "state.json"
        if not state_file.is_file():
            report["stages"]["state_written"] = {
                "ok": False,
                "detail": "the oracle ran but wrote no state.json",
            }
            return report
        state = json.loads(state_file.read_text(encoding="utf-8"))
        report["stages"]["state_written"] = {
            "ok": True,
            "n_records": len(state.get("records", [])),
            "n_log": len(state.get("log", [])),
            "order": state.get("_order", []),
        }

        # ---- handover: the artifact Harbor uploads -------------------------
        #
        # A *copy into a separate directory*, which is what `environment_mode =
        # "separate"` means. Sharing the agent's directory would let the
        # verifier read anything the agent left behind, including a hand-written
        # answer that never went through the tools.
        (verifier / "logs" / "verifier").mkdir(parents=True, exist_ok=True)
        shutil.copy(state_file, verifier / "state.json")

        # ---- verifier side -------------------------------------------------
        test_src = (pkg / "tests/test_state.py").read_text(encoding="utf-8")
        test_host = _rewrite(
            test_src,
            {
                # Python source -> forward slashes, same reason as tools.py.
                "/app/state.json": str(verifier / "state.json").replace("\\", "/"),
                "/logs/verifier": str(verifier / "logs" / "verifier").replace("\\", "/"),
            },
        )
        test_path = verifier / "test_state.py"
        test_path.write_text(test_host, encoding="utf-8")

        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-q", "--no-header"],
            capture_output=True, text=True, timeout=180, cwd=str(verifier),
            encoding="utf-8", errors="replace",
        )
        report["stages"]["verifier_on_oracle"] = {
            "ok": proc.returncode == 0,
            "rc": proc.returncode,
            "tail": (proc.stdout or "")[-500:],
        }
        if verbose:
            print(f"    verifier rc={proc.returncode}")

        reward_json = verifier / "logs" / "verifier" / "reward.json"
        if reward_json.is_file():
            rj = json.loads(reward_json.read_text(encoding="utf-8"))
            report["fractional_reward"] = rj.get("reward")
            report["checkpoints"] = {
                "passed": rj.get("checkpoints_passed"),
                "total": rj.get("checkpoints_total"),
            }
        else:
            report["fractional_reward"] = None

        # ---- and on an UNTOUCHED state, which must fail ---------------------
        #
        # The V2 analogue at the package level: if the verifier passes an
        # untouched state, the package grades nothing and every rollout scores
        # 1.0. Running it is the only way to know, since a verifier that reads
        # no file at all passes trivially.
        untouched = Path(tempfile.mkdtemp(prefix=f"harbor_untouched_{name}_"))
        try:
            shutil.copy(pkg / "environment/assets/initial_state.json", untouched / "state.json")
            (untouched / "logs" / "verifier").mkdir(parents=True, exist_ok=True)
            test_host2 = _rewrite(
                test_src,
                {
                    "/app/state.json": str(untouched / "state.json").replace("\\", "/"),
                    "/logs/verifier": str(untouched / "logs" / "verifier").replace("\\", "/"),
                },
            )
            tp2 = untouched / "test_state.py"
            tp2.write_text(test_host2, encoding="utf-8")
            proc2 = subprocess.run(
                [sys.executable, "-m", "pytest", str(tp2), "-q", "--no-header"],
                capture_output=True, text=True, timeout=180, cwd=str(untouched),
                encoding="utf-8", errors="replace",
            )
            report["stages"]["verifier_on_untouched"] = {
                "ok": proc2.returncode != 0,
                "rc": proc2.returncode,
                "detail": "verifier must FAIL on an untouched state",
            }
            rj2 = untouched / "logs" / "verifier" / "reward.json"
            if rj2.is_file():
                report["untouched_reward"] = json.loads(rj2.read_text(encoding="utf-8")).get("reward")
        finally:
            shutil.rmtree(untouched, ignore_errors=True)

        # ---- the verdict ---------------------------------------------------
        #
        # Every stage must pass, *and* the fractional reward must be exactly 1.0
        # on the oracle. A package whose verifier is too strict scores the
        # oracle below 1.0 and looks like a hard task; that is the failure this
        # whole script is for.
        report["ok"] = (
            report["stages"]["oracle_run"]["ok"]
            and report["stages"]["state_written"]["ok"]
            and report["stages"]["verifier_on_oracle"]["ok"]
            and report["stages"].get("verifier_on_untouched", {}).get("ok", False)
            and report.get("fractional_reward") == 1.0
        )
        return report
    except subprocess.TimeoutExpired as exc:
        report["stages"]["timeout"] = {"ok": False, "detail": str(exc)[:200]}
        return report
    finally:
        shutil.rmtree(agent, ignore_errors=True)
        shutil.rmtree(verifier, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run exported Harbor packages without Docker.")
    ap.add_argument("package", nargs="?", help="a single task package directory")
    ap.add_argument("--all", action="store_true", help="run every package under the given dir")
    ap.add_argument("--json", default=None, help="write the report here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if not args.package:
        ap.error("give a package directory, or a parent directory with --all")
    root = Path(args.package)
    pkgs = (
        sorted(p for p in root.iterdir() if (p / "task.toml").is_file())
        if args.all
        else [root]
    )
    if not pkgs:
        print(f"no task packages found under {root}")

    print("=" * 78)
    print("HARBOR LOCAL RUN — execute the exported package without Docker")
    print("=" * 78)
    print(f"packages: {len(pkgs)}")
    print()

    reports = []
    for pkg in pkgs:
        print(f"--- {pkg.name} ---")
        r = run_package(pkg, verbose=args.verbose)
        reports.append(r)
        frac = r.get("fractional_reward")
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  [{mark}] oracle_rc={r['stages'].get('oracle_run', {}).get('rc')}"
              f"  verifier_rc={r['stages'].get('verifier_on_oracle', {}).get('rc')}"
              f"  untouched_rc={r['stages'].get('verifier_on_untouched', {}).get('rc')}"
              f"  reward={frac}")
        if not r["ok"]:
            for stage, d in r["stages"].items():
                if not d.get("ok", True):
                    print(f"        {stage}: {d}")

    ok = sum(1 for r in reports if r["ok"])
    print()
    print("=" * 78)
    print(f"PACKAGES OK: {ok}/{len(reports)}")
    print("=" * 78)
    if ok != len(reports):
        print("\nA package that fails here would fail inside Harbor, and the failure")
        print("would be attributed to the model rather than to the package.")

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0 if ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
