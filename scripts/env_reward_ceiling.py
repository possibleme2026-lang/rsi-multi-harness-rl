"""The reward ceiling, reached through the harness layer with no model.

Why this exists
---------------
The environment scan returned **0.00 in all 96 cells**, and a floor is not
evidence on its own: a broken reward path and a weak policy both produce a
column of zeros. Before that number can be reported as a capability finding, the
same path has to be shown to reach 1.0 — otherwise the scan is measuring the
plumbing.

``env_harness_smoke.py`` already drives the oracle through the harnesses, and it
passes 12/12. This script is the narrower claim underneath it: for *one* task,
walk the reference call by call, print the state after each step, and show the
reward climbing to 1.0 — so a reader can see which checkpoint each step earns
rather than trusting a single terminal number.

It also asserts the intermediate case, which is the one that matters for
training: a **partially** correct attempt must score strictly between 0 and 1.
If it did not, the reward would be a boolean wearing a fraction's clothes, and
GRPO's group signal would vanish exactly where a 0.5B model lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import TRAIN_HARNESSES  # noqa: E402
from multiharness.harnesses.core import TASKS, register_tasks  # noqa: E402
from multiharness.rsi import env_adapter, envtask  # noqa: E402


def main() -> int:
    harness = sys.argv[1] if len(sys.argv) > 1 else "bash_minimal"
    tasks = envtask.generate_env_batch(1, seed=5, n_records=4, n_distractors=3, n_steps=4)
    task = tasks[0]
    harness_tasks = env_adapter.as_harness_batch(tasks)
    register_tasks([t for t in harness_tasks if t["id"] not in TASKS])

    cls = TRAIN_HARNESSES[harness]
    env = cls()
    env.reset(task_id=task.task_id)

    print("=" * 74)
    print(f"REWARD CEILING — {harness}, task {task.task_id}")
    print("=" * 74)
    print(f"chain    : {list(task.instance.chain)}")
    print(f"trace    : {[c['tool'] for c in task.trace]}")
    print(f"checkpoints: {task.checkpoint_count}")
    print(f"\nreward before any action : {env.get_reward():.2f}")

    # Walk the reference, reporting the reward after each call. The score is
    # recomputed from the state file, so this is the same path a rollout takes.
    for i, call in enumerate(task.trace, 1):
        args = " ".join(str(call[k]) for k in ("id", "value") if k in call)
        cmd = f"envtool {call['tool']} {args}".strip()
        out = env.bash(cmd)
        print(f"\n  step {i}: {cmd}")
        print(f"    -> {str(out).strip()[:90]}")
        print(f"    reward now: {env.get_reward():.2f}")

    final = env.get_reward()
    print("\n" + "-" * 74)
    print(f"final reward             : {final:.2f}")

    # The partial-credit case, on a fresh instance: do only the first mutation.
    env2 = cls()
    env2.reset(task_id=task.task_id)
    first = task.trace[0]
    args = " ".join(str(first[k]) for k in ("id", "value") if k in first)
    env2.bash(f"envtool {first['tool']} {args}".strip())
    partial = env2.get_reward()
    print(f"partial (1 of {len(task.trace)} calls): {partial:.2f}")

    ok_full = final == 1.0
    ok_partial = 0.0 < partial < 1.0
    print("\n" + "=" * 74)
    print(f"  oracle reaches 1.0          : {'YES' if ok_full else 'NO'}  ({final:.2f})")
    print(f"  partial credit is fractional: {'YES' if ok_partial else 'NO'}  ({partial:.2f})")
    print("=" * 74)
    if not (ok_full and ok_partial):
        print("A floor of 0.00 cannot be reported as a capability finding until")
        print("this path is shown to reach 1.0.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
