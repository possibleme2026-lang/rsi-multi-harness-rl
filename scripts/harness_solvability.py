"""Gate V1 for the *harness pool*: is every task solvable through every harness?

Why this exists
---------------
``tests/smoke_env.py`` already asserts the oracle scores 1.0 on all 24 tasks --
but it does so through ``OracleHarness``, which writes the answer directly and
never touches the harness's advertised tools. That proves the *verifier* is
correct. It does not prove the task is solvable by an agent holding that
harness's tool surface, and the two are different claims:

    a task can be perfectly verifiable and still be unsolvable in a harness
    that has no tool capable of producing the required file

The gap is not hypothetical. ``codex_style`` scored **0/32 in every arm** of
``eval_ablation.json`` -- baseline, single-harness and multi-harness alike --
which pins the held-out term of the headline metric at 0 and makes
``gap == mean(train)`` an identity. Before that zero can be reported as "a 0.5B
model cannot drive a patch-based harness", the same path has to be shown to
reach 1.0. Otherwise it is measuring the plumbing, and a broken harness and a
weak policy both produce a column of zeros.

This is gate V1 (``rsi/validate.py``) applied to the harness axis rather than
to the task generator: a task is not trusted until the reference solution
scores 1.0 **on the harness that will run it**.

How the reference is driven
---------------------------
Each shipped task carries ``expected`` and ``verify``, and the reference
behaviour is "make the answer appear in ``answer.txt``". So the oracle here is
deliberately naive and harness-agnostic in *intent*, but forced through each
harness's own tool surface in *mechanism*:

  * harnesses exposing ``bash``      -> a shell command that writes the file
  * ``json_strict``                  -> its ``submit`` tool
  * ``codex_style``                  -> ``apply_patch`` with a unified diff
                                        (and ``bash`` as the second route)

A harness that cannot produce the file by any route it advertises fails here,
and that failure is a harness defect rather than a model result.

Run:
    ./run.sh scripts/harness_solvability.py
    ./run.sh scripts/harness_solvability.py --harness codex_style --verbose
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import (  # noqa: E402
    ALL_HARNESSES,
    HELDOUT_HARNESSES,
    TASKS,  # noqa: E402
)
from multiharness.harnesses.core import discover_tools  # noqa: E402
from multiharness.tasks import load as load_tasks  # noqa: E402


def _write_via_bash(env, value: str) -> str:
    """Write the answer through the harness's shell tool.

    The value is quoted with ``shlex.quote`` rather than interpolated, because
    the T3 tier deliberately contains shell metacharacters (``$``, backticks,
    quotes) and an unquoted interpolation would silently write a *different*
    string -- a false failure that reads as "the harness cannot do this".
    """
    cmd = f"printf %s {shlex.quote(value)} > answer.txt"
    return str(env.bash(cmd))


def _write_via_patch(env, value: str) -> str:
    """Write the answer through ``apply_patch`` using a minimal Add File body."""
    body = "".join(f"+{line}\n" for line in value.split("\n"))
    patch = f"*** Add File: answer.txt\n{body}"
    return str(env.apply_patch(patch))


def _write_via_submit(env, value: str) -> str:
    """Write the answer through ``json_strict``'s submit tool."""
    return str(env.submit(value))


#: Which routes to attempt per harness, in order. Each entry is
#: ``(label, callable)``. A harness passes if *any* advertised route reaches 1.0.
#:
#: ``codex_style`` carries two routes on purpose. It is the held-out harness, so
#: it is the one whose zero has to be interpreted, and a single route would
#: conflate "apply_patch is broken" with "the patch route is the only one it has".
ROUTES: dict[str, list[tuple[str, str]]] = {
    "bash_minimal": [("bash", "_write_via_bash")],
    "react_tools": [("bash", "_write_via_bash"), ("write_file", "write_file")],
    "json_strict": [("submit", "_write_via_submit"), ("bash", "_write_via_bash")],
    "longctx_summary": [("bash", "_write_via_bash")],
    "codex_style": [("apply_patch", "_write_via_patch"), ("bash", "_write_via_bash")],
}


def _invoke(env, route: str, value: str) -> str:
    """Dispatch one route, tolerating a harness that lacks the tool.

    A missing attribute is reported as a failed route rather than raising: the
    question this script answers is "can this harness produce the file", and
    "it has no such tool" is a legitimate answer to it.
    """
    if route == "_write_via_bash":
        if not hasattr(env, "bash"):
            return "[no bash tool]"
        return _write_via_bash(env, value)
    if route == "_write_via_patch":
        if not hasattr(env, "apply_patch"):
            return "[no apply_patch tool]"
        return _write_via_patch(env, value)
    if route == "_write_via_submit":
        if not hasattr(env, "submit"):
            return "[no submit tool]"
        return _write_via_submit(env, value)
    if route == "write_file":
        if not hasattr(env, "write_file"):
            return "[no write_file tool]"
        # react_tools.write_file takes (path, content) positionally.
        return str(env.write_file("answer.txt", value))
    return f"[unknown route {route}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default=None,
                    help="restrict to one harness (default: all five, incl. held-out)")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated task ids (default: the whole suite)")
    ap.add_argument("--verbose", action="store_true",
                    help="print the tool output for every attempt")
    args = ap.parse_args()

    load_tasks()

    names = ([args.harness] if args.harness else list(ALL_HARNESSES))
    unknown = [n for n in names if n not in ALL_HARNESSES]
    if unknown:
        print(f"!! unknown harness(es): {unknown}; known: {sorted(ALL_HARNESSES)}")
        return 1

    task_ids = ([t.strip() for t in args.tasks.split(",") if t.strip()]
                if args.tasks else sorted(TASKS))
    bad_ids = [t for t in task_ids if t not in TASKS]
    if bad_ids:
        print(f"!! unknown task(s): {bad_ids}")
        return 1

    print("=" * 78)
    print("HARNESS SOLVABILITY — can the reference solution be expressed")
    print("through every harness's own tool surface?")
    print("=" * 78)
    print(f"harnesses : {names}")
    print(f"tasks     : {len(task_ids)}")
    print("claim     : for each (harness, task), SOME advertised route scores 1.0")
    print("if this fails, a 0.00 rollout rate is a harness defect, not a model result")

    # Results[harness][task] = (best_route, best_reward, detail)
    results: dict[str, dict[str, tuple[str | None, float, str]]] = {}
    failures: list[tuple[str, str, str]] = []

    for name in names:
        cls = ALL_HARNESSES[name]
        kind = "HELD-OUT" if name in HELDOUT_HARNESSES else "train"
        routes = ROUTES.get(name, [("bash", "_write_via_bash")])
        print(f"\n--- {name}  [{kind}]  routes={[r for r, _ in routes]} ---")
        results[name] = {}

        env0 = cls()
        advertised = sorted(m.__name__ for m in discover_tools(env0))
        print(f"    advertised tools: {advertised}")

        for tid in task_ids:
            task = TASKS[tid]
            expected = str(task.get("expected", ""))
            best_route: str | None = None
            best_reward = 0.0
            detail = ""

            for label, route in routes:
                env = cls()
                env.reset(task_id=tid)
                # Before any action the task must score 0.0. If it does not, the
                # reward path is broken and a later 1.0 would prove nothing.
                before = float(env.get_reward())
                out = _invoke(env, route, expected)
                after = float(env.get_reward())
                if args.verbose:
                    print(f"      [{tid}] {label}: {before:.2f} -> {after:.2f}  {out[:70]!r}")
                if after > best_reward:
                    best_route, best_reward = label, after
                    detail = f"before={before:.2f} out={out[:60]!r}"
                if after == 1.0:
                    break

            results[name][tid] = (best_route, best_reward, detail)
            if best_reward != 1.0:
                failures.append((name, tid, f"best={best_reward:.2f} via {best_route}"))

        solved = sum(1 for v in results[name].values() if v[1] == 1.0)
        print(f"    solvable: {solved}/{len(task_ids)}")
        if solved != len(task_ids):
            # Show the first few failures inline; the summary repeats them.
            shown = [(t, r) for t, (_, r, _) in results[name].items() if r != 1.0]
            for t, r in shown[:5]:
                print(f"      !! {t}: best reward {r:.2f}")

    # -- verdict -----------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    total = len(names) * len(task_ids)
    print(f"  (harness, task) pairs checked : {total}")
    print(f"  solved by at least one route  : {total - len(failures)}")
    print(f"  unsolvable                    : {len(failures)}")

    if failures:
        print("\n  UNSOLVABLE — these are harness defects, not model failures:")
        by_harness: dict[str, list[str]] = {}
        for h, t, why in failures:
            by_harness.setdefault(h, []).append(f"{t} ({why})")
        for h, items in sorted(by_harness.items()):
            print(f"    {h}: {len(items)}  e.g. {items[:3]}")

    print("\n" + "-" * 78)
    if failures:
        print("  A 0.00 rollout rate on these cells cannot be reported as a capability")
        print("  finding until the reference solution reaches 1.0 through the same")
        print("  harness. Fix the harness first, then re-measure.")
        return 1
    print("  Every task is solvable through every harness's advertised tools.")
    print("  Therefore a 0.00 rollout rate on a held-out harness is a *capability*")
    print("  floor, and the cross-harness gap may be reported as bounded by it --")
    print("  not as a measured generalization gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
