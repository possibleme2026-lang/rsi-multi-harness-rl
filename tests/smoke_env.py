"""Smoke test A — the harness layer, with no model loaded.

Checks the three things the whole experiment rests on:

  1. every harness exposes the tool set it claims, and TRL's own tool
     discovery rule (``inspect.getmembers(env, ismethod)`` minus
     ``reset`` / ``get_reward`` / ``_``-prefixed) sees exactly those tools;
  2. ``reset()`` materialises the task workspace and returns the
     instruction, and ``bash`` actually executes on this host;
  3. the reward hook is a *method* named ``get_reward`` (a ``@property``
     would be invisible to ``inspect.ismethod`` and silently never
     registered), and it grades through ``answer.txt`` only.

Run:
    ./run.sh tests/smoke_env.py
"""

from __future__ import annotations

import inspect
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import (  # noqa: E402
    ALL_HARNESSES,
    BASH,
    HELDOUT_HARNESSES,
    TRAIN_HARNESSES,
)
from multiharness.tasks import load as load_tasks  # noqa: E402

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def trl_tools(env) -> list[str]:
    """Reproduce GRPOTrainer's discovery rule verbatim (grpo_trainer.py:654)."""
    out = []
    for name, _member in inspect.getmembers(env, predicate=inspect.ismethod):
        if name in ("reset", "get_reward"):
            continue
        if name.startswith("_"):
            continue
        out.append(name)
    return sorted(out)


EXPECTED_TOOLS = {
    "bash_minimal": ["bash"],
    "react_tools": ["bash", "finish", "read_file", "write_file"],
    "json_strict": ["bash", "submit"],
    "longctx_summary": ["bash", "read_file", "replace_in_file"],
    "codex_style": ["apply_patch", "bash"],
}


def main() -> int:
    print("=" * 74)
    print("SMOKE A — harness layer (no model)")
    print("=" * 74)

    print(f"\nhost bash: {BASH}")

    load_tasks()
    from multiharness.harnesses import TASKS

    print(f"tasks registered: {len(TASKS)}")
    by_tier: dict[str, int] = {}
    for t in TASKS.values():
        by_tier[t["tier"]] = by_tier.get(t["tier"], 0) + 1
    print(f"  by tier: {by_tier}")
    check("24 tasks registered", len(TASKS) == 24, str(len(TASKS)))
    check("all three tiers present", set(by_tier) == {"T1", "T2", "T3"}, str(sorted(by_tier)))

    # -- 1. tool surface ---------------------------------------------------
    print("\n-- 1. tool surface as TRL sees it --")
    for name, cls in ALL_HARNESSES.items():
        env = cls()
        got = trl_tools(env)
        check(
            f"{name:<16} tools == {EXPECTED_TOOLS[name]}",
            got == EXPECTED_TOOLS[name],
            f"got {got}",
        )
        check(
            f"{name:<16} get_reward is a discoverable method",
            "get_reward" in inspect.getmembers(env, predicate=inspect.ismethod)
            or hasattr(env, "get_reward"),
        )
        # A property would pass hasattr() but fail ismethod().
        check(
            f"{name:<16} get_reward passes ismethod() (not a property)",
            "get_reward" in dict(inspect.getmembers(env, predicate=inspect.ismethod)),
        )
    check("held-out pool is disjoint from train pool", not (set(TRAIN_HARNESSES) & set(HELDOUT_HARNESSES)))
    check("4 train + 1 held-out", len(TRAIN_HARNESSES) == 4 and len(HELDOUT_HARNESSES) == 1)

    # -- 2. reset + bash ---------------------------------------------------
    print("\n-- 2. reset() materialises the workspace; bash runs --")
    from multiharness.harnesses import BashMinimalEnv, JsonStrictEnv

    env = BashMinimalEnv()
    obs = env.reset(task_id="t2-01")
    check("reset returns a non-empty instruction", bool(obs and obs.strip()))
    check("instruction carries the task text", "seed=" in obs, obs[:60].replace("\n", " "))
    check("harness guidance is appended", "answer.txt" in obs)
    check("workdir exists on disk", env._workdir is not None and env._workdir.is_dir(), str(env._workdir))
    check("task file materialised", (env._workdir / "config.txt").is_file())
    out = env.bash("cat config.txt")
    check("bash reads the shipped file", "seed=1234" in out, out.strip().replace("\n", " | ")[:70])
    out = env.bash("echo -n 1234 > answer.txt && cat answer.txt")
    check("bash writes answer.txt", "1234" in out, out.strip()[:70])
    check("reward == 1.0 after correct answer", env.get_reward() == 1.0, str(env.get_reward()))
    check("reward is cached, not recomputed", env.reward == 1.0)

    # -- 3. submit protocol differs per harness ---------------------------
    print("\n-- 3. submit channel is harness-specific --")
    js = JsonStrictEnv()
    js.reset(task_id="t1-03")
    check("json_strict starts with no answer.txt", js.get_reward() == 0.0, str(js.get_reward()))
    js.bash("echo -n 42 > answer.txt")
    check(
        "json_strict still scores 1.0 when the file exists (verifier is harness-agnostic)",
        js.get_reward() == 1.0,
    )

    js2 = JsonStrictEnv()
    js2.reset(task_id="t1-03")
    msg = js2.submit("42")
    check("json_strict submit() writes through", js2.get_reward() == 1.0, msg)
    check("json_strict rejects an empty answer", "error" in js2.submit("").lower())

    # -- 4. verifier correctness on the whole suite -----------------------
    print("\n-- 4. oracle path: gold answer scores 1.0 on every task --")
    from multiharness.harnesses import OracleHarness

    bad = []
    for tid, task in sorted(TASKS.items()):
        o = OracleHarness()
        o.reset(task_id=tid)
        if o.get_reward() != 1.0:
            bad.append(tid)
    check(f"oracle scores 1.0 on all {len(TASKS)} tasks", not bad, f"failures: {bad}")

    empty = []
    for tid in sorted(TASKS):
        e = BashMinimalEnv()
        e.reset(task_id=tid)
        if e.get_reward() != 0.0:
            empty.append(tid)
    check("empty workspace scores 0.0 on all tasks (no free reward)", not empty, f"failures: {empty}")

    # -- 5. T3 escaping actually reaches the file intact ------------------
    print("\n-- 5. T3 payloads survive a correct bash write --")
    import shlex

    from multiharness.harnesses import BashMinimalEnv as BME

    t3_bad = []
    for tid, task in sorted(TASKS.items()):
        if task["tier"] != "T3":
            continue
        e = BME()
        e.reset(task_id=tid)
        # POSIX single-quote escaping: ' -> '\''
        payload = task["expected"].replace("'", "'\\''")
        e.bash(f"printf '%s' '{payload}' > answer.txt")
        if e.get_reward() != 1.0:
            t3_bad.append((tid, task["expected"]))
    check("all 6 T3 payloads round-trip via printf + single quotes", not t3_bad, str(t3_bad))

    print("\n" + "=" * 74)
    if FAILS:
        print(f"SMOKE A: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("SMOKE A: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
