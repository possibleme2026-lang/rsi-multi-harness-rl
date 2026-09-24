"""Can the harness pool actually *run* a synthesised environment task?

Why this runs before any model scan
-----------------------------------
A cross-harness gap measured on tasks the harnesses cannot execute is not a
measurement. Before spending GPU time on a rollout sweep over environment tasks,
this script drives one task through the real harness classes with a **scripted
policy** — no model — and asserts four things that a model-based scan would
otherwise confound with model capability:

1. ``reset`` materialises the environment (tools on PATH, state on disk);
2. a tool call issued through the harness's own shell changes the state;
3. ``get_reward`` reads the *state* and returns the fractional checkpoint score;
4. an untouched rollout scores 0.0 while the reference scores 1.0.

Step 4 is the one that matters. It is the same discrimination the in-process
grader asserts, but reached through the harness layer — a different code path,
a different process boundary, and a real shell. If the two disagree, the
benchmark is measuring the plumbing rather than the policy, and every gap
number computed from it would be an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import TRAIN_HARNESSES  # noqa: E402
from multiharness.rsi import env_adapter, envtask  # noqa: E402


def _call_via_shell(env, command: str) -> str:
    """Run one shell command through the harness's own exec path.

    Deliberately *not* ``subprocess``: the point is to exercise the same
    mechanism a model's tool call goes through, so a break in the harness's
    shell plumbing shows up here rather than as a mysterious zero in a scan.
    """
    for attr in ("bash", "run_bash", "shell", "run_shell"):
        fn = getattr(env, attr, None)
        if callable(fn):
            try:
                return str(fn(command))
            except TypeError:
                continue
    return ""


def check_task(task: envtask.EnvTask, harness_name: str) -> dict:
    """Drive one task through one harness with a scripted policy."""
    cls = TRAIN_HARNESSES[harness_name]
    env = cls()
    observation = env.reset(task_id=task.task_id)

    # The tools have to be *visible* from the rollout's shell. A synthesised
    # environment whose tools are not on PATH is a task the agent cannot act on,
    # and it would score 0.0 in a way indistinguishable from a weak policy.
    out = _call_via_shell(env, "envtool list_items 2>&1 || envtool list_tickets 2>&1 "
                              "|| envtool list_accounts 2>&1 || envtool list_sensors 2>&1")
    listing_ok = '"' in out or "{" in out

    # Now the reference, executed one call at a time through the harness shell.
    step_outputs = []
    for call in task.trace:
        args = " ".join(str(call[k]) for k in ("id", "value") if k in call)
        cmd = f"envtool {call['tool']} {args}".strip()
        step_outputs.append(_call_via_shell(env, cmd + " 2>&1"))

    reward = env.get_reward()

    # And an untouched rollout, on a fresh instance, as the floor.
    env2 = cls()
    env2.reset(task_id=task.task_id)
    untouched = env2.get_reward()

    return {
        "task_id": task.task_id,
        "harness": harness_name,
        "observation_mentions_state": "state.json" in observation or "envtool" in observation,
        "listing_ok": listing_ok,
        "steps": len(task.trace),
        "step_outputs": step_outputs,
        "reward_reference": reward,
        "reward_untouched": untouched,
        "discriminates": reward == 1.0 and untouched == 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="how many environment tasks to check")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--harnesses", default=",".join(TRAIN_HARNESSES))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tasks = envtask.generate_env_batch(
        args.n, seed=args.seed, n_records=4, n_distractors=3, n_steps=4
    )
    names = [h.strip() for h in args.harnesses.split(",") if h.strip()]

    # Registration is not optional, and forgetting it is the first thing that
    # goes wrong here: `env.reset` resolves the id through the process-global
    # registry, and an environment task's id (`env-...`) is in no shipped suite.
    # The error is a `KeyError: unknown task_id`, which reads as "the task does
    # not exist" rather than "you did not register it" — so it is done here,
    # before the first reset, and the adapter is the thing that does it.
    from multiharness.harnesses.core import TASKS, register_tasks

    harness_tasks = env_adapter.as_harness_batch(tasks)
    fresh = [t for t in harness_tasks if t["id"] not in TASKS]
    register_tasks(fresh)
    print("=" * 78)
    print("ENVIRONMENT TASKS THROUGH THE HARNESS POOL — scripted policy, no model")
    print("=" * 78)
    print(f"tasks     : {len(tasks)}  ({len(fresh)} newly registered)")
    print(f"harnesses : {names}")

    rows: list[dict] = []
    for name in names:
        print(f"\n--- {name} ---")
        for t in tasks:
            r = check_task(t, name)
            rows.append(r)
            mark = "OK  " if r["discriminates"] else "FAIL"
            print(f"  [{mark}] {r['task_id']}  steps={r['steps']}  "
                  f"ref={r['reward_reference']}  untouched={r['reward_untouched']}  "
                  f"tools_visible={r['listing_ok']}")
            if not r["discriminates"]:
                for so in r["step_outputs"][:3]:
                    print(f"          step -> {so[:110]}")

    ok = sum(1 for r in rows if r["discriminates"])
    print("\n" + "=" * 78)
    print(f"CELLS DISCRIMINATING: {ok}/{len(rows)}")
    print("=" * 78)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
