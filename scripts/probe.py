"""Capability probe — the Go/No-Go gate for the whole experiment.

Before spending hours on training, establish whether Qwen2.5-0.5B-Instruct
can drive these harnesses at all. If it cannot call tools and cannot ever
score, GRPO has nothing to amplify and any "gap" we later measure would be
noise, not generalization.

Four gates, all measured with the *same* rollout driver the evaluation uses:

  G1  tool-call rate        >= 50%   the model emits parseable tool calls
  G2  pass@8                >= 1/8   on at least one T1 task — it is reachable
  G3  reward variance       non-uniform across harnesses (otherwise there is
                                     no cross-harness signal to learn from)
  G4  multi-turn uptake     >= 50%   of rollouts that called a tool also
                                     consumed the result and continued

On G4: the obvious definition — "mean turns >= 2" — is wrong, and was wrong
here in a way worth recording. A capable model that solves a one-step task on
its first tool call has a low turn count *because it succeeded*. Gating on
turns punishes competence. What actually matters is whether the model engages
the loop at all: among rollouts that emitted a tool call, did it come back for
another turn after seeing the tool result? A model that fires one tool call and
then stops talking has not learned to use a harness, however many turns the
counter reports.

A failure here is a *result*, not a bug: it says "0.5B cannot do this", which
is exactly the question the probe exists to answer.

Run:
    ./run.sh scripts/probe.py
    ./run.sh scripts/probe.py --n 8 --tasks t1-01,t1-03
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness._bootstrap import outputs_root
from multiharness.harnesses import TRAIN_HARNESSES
from multiharness.rollout import Agent
from multiharness.tasks import load as load_tasks

OUT_DIR = outputs_root()


def _default_out() -> str:
    """Default scan path, computed lazily so ``--help`` writes nothing."""
    return str(OUT_DIR / "scan_all.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="rollouts per (harness, task) cell")
    ap.add_argument("--tasks", default="t1-01,t1-03,t3-01",
                    help="comma-separated task ids to probe")
    # A generated batch is a different task set from the shipped suite: the
    # suite's ids are `t1-01`, a generated task's are `t1-8f87ad9e`, and the two
    # do not intersect. Without this flag the curriculum could never scan the
    # batch it intends to steer — it would read a scan of 16 ids, find none of
    # them in the batch, and correctly but uselessly refuse to move anything.
    #
    # The batch is read from a JSON file written by `rsi_loop.py --dump-batch`,
    # not regenerated here, because regeneration is deterministic only given the
    # same seed *and* the same call order, and a second process is the easiest
    # way to get a subtly different batch while every id still looks valid.
    ap.add_argument("--from-batch", default=None,
                    help="JSON file of generated tasks to probe instead of the shipped suite")
    ap.add_argument("--harnesses", default=",".join(TRAIN_HARNESSES),
                    help="comma-separated harness names")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--out", default=None,
                    help="scan dump path (default: $MULTIHARNESS_OUT/scan_all.json)")
    args = ap.parse_args()

    if args.from_batch:
        from multiharness.harnesses.core import TASKS, register_tasks

        batch = json.loads(Path(args.from_batch).read_text(encoding="utf-8"))
        # The shipped suite still has to be loaded: `Agent.run` resolves task ids
        # through the process-global registry, and `reset` on a shipped id would
        # fail in a fresh process without it.
        load_tasks()
        fresh = [t for t in batch if t["id"] not in TASKS]
        register_tasks(fresh)
        task_ids = [t["id"] for t in batch]
        print(f"batch     : {args.from_batch}  ({len(task_ids)} generated tasks registered)")
    else:
        load_tasks()
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    harness_names = [h.strip() for h in args.harnesses.split(",") if h.strip()]

    print("=" * 78)
    print("CAPABILITY PROBE — Qwen2.5-0.5B-Instruct vs the harness pool")
    print("=" * 78)
    print(f"harnesses : {harness_names}")
    print(f"tasks     : {task_ids}")
    print(f"rollouts  : {args.n} per cell  ({len(harness_names) * len(task_ids) * args.n} total)")
    print(f"max_turns : {args.max_turns}   max_new_tokens: {args.max_new_tokens}")

    agent = Agent(max_turns=args.max_turns, max_new_tokens=args.max_new_tokens)
    print(f"model     : {agent.model_id} on {agent.device}")

    records: list[dict] = []
    matrix: dict[str, dict[str, float]] = {}
    out_path = Path(args.out) if args.out else Path(_default_out())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def flush(partial: bool) -> None:
        """Persist after every cell.

        A scan is tens of minutes of GPU time; the first version wrote only at
        the very end, so a hang at cell 30 of 64 lost everything. Flushing per
        cell costs nothing and makes an interrupted run resumable by hand.
        """
        out_path.write_text(
            json.dumps(
                {
                    "model": agent.model_id,
                    "n": args.n,
                    "max_turns": args.max_turns,
                    "max_new_tokens": args.max_new_tokens,
                    "matrix": matrix,
                    "partial": partial,
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    for hname in harness_names:
        cls = TRAIN_HARNESSES[hname]
        matrix[hname] = {}
        print(f"\n--- {hname}  (tools: {agent.tool_names(cls())}) ---", flush=True)
        for tid in task_ids:
            t0 = time.time()
            rollouts = agent.run(cls, tid, n=args.n)
            rewards = [r.reward for r in rollouts]
            pass_rate = sum(rewards) / len(rewards)
            called = sum(1 for r in rollouts if r.tool_calls > 0)
            call_rate = called / len(rollouts)
            mean_turns = sum(r.turns for r in rollouts) / len(rollouts)
            errs = sum(r.tool_errors for r in rollouts)
            matrix[hname][tid] = pass_rate

            # flush=True + elapsed stamp: without the timestamp a stall is
            # indistinguishable from a slow cell when reading the log later.
            print(
                f"  {tid:<7} pass={pass_rate:5.2f} ({int(sum(rewards))}/{len(rewards)})"
                f"  toolcall={call_rate:5.2f}  turns={mean_turns:4.1f}  errs={errs}"
                f"  [{time.time() - t0:5.1f}s, total {time.time() - t_start:6.0f}s]",
                flush=True,
            )
            for r in rollouts:
                records.append({**r.as_dict(), "raw": r.transcript[:6]})
            flush(partial=True)

    # ---- gates ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("GATE VERDICT")
    print("=" * 78)

    total = len(records)
    overall_call = sum(1 for r in records if r["tool_calls"] > 0) / total
    overall_turns = sum(r["turns"] for r in records) / total
    best_pass = max((r["reward"] for r in records), default=0.0)
    # G2 is pass@k per cell, not a single sample.
    per_cell_best: dict[tuple[str, str], float] = {}
    for r in records:
        k = (r["harness"], r["task_id"])
        per_cell_best[k] = max(per_cell_best.get(k, 0.0), r["reward"])
    cells_passed = sum(1 for v in per_cell_best.values() if v > 0)
    # G3: does the pass rate differ across harnesses on at least one task?
    spread = 0.0
    for tid in task_ids:
        col = [matrix[h][tid] for h in harness_names]
        spread = max(spread, max(col) - min(col))

    # G4: of the rollouts that engaged the loop, how many went round it again?
    engaged = [r for r in records if r["tool_calls"] > 0]
    multi = [r for r in engaged if r["turns"] >= 2]
    g4_rate = len(multi) / len(engaged) if engaged else 0.0

    g1 = overall_call >= 0.50
    g2 = best_pass > 0.0
    g3 = spread > 0.0
    g4 = g4_rate >= 0.50

    print(f"  G1 tool-call rate      {overall_call:6.1%}  >= 50%      {'PASS' if g1 else 'FAIL'}")
    print(f"  G2 reachable (pass@k)  {cells_passed:6d}   >= 1 cell   {'PASS' if g2 else 'FAIL'}")
    print(f"  G3 cross-harness spread{spread:7.2f}  > 0         {'PASS' if g3 else 'FAIL'}")
    print(f"  G4 multi-turn uptake   {g4_rate:6.1%}  >= 50%      {'PASS' if g4 else 'FAIL'}"
          f"   (of {len(engaged)} rollouts that called a tool)")
    print(f"     [info] mean turns   {overall_turns:6.2f}  (reported, not gated)")

    print("\n  pass-rate matrix (rows=harness, cols=task)")
    header = "    " + " " * 18 + "".join(f"{t:>9}" for t in task_ids)
    print(header)
    for h in harness_names:
        row = "".join(f"{matrix[h][t]:9.2f}" for t in task_ids)
        print(f"    {h:<18}{row}")

    payload = {
        "model": agent.model_id,
        "n": args.n,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "matrix": matrix,
        "gates": {
            "G1_tool_call_rate": overall_call,
            "G1_pass": g1,
            "G2_cells_reached": cells_passed,
            "G2_pass": g2,
            "G3_cross_harness_spread": spread,
            "G3_pass": g3,
            "G4_multi_turn_rate": g4_rate,
            "G4_pass": g4,
            "mean_turns": overall_turns,
        },
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")

    ok = all([g1, g2, g3, g4])
    print("\n" + ("PROBE: GO — 0.5B can drive the harness pool" if ok
                  else "PROBE: NO-GO — see failing gates above"))
    if not ok and not g2:
        print("\n  Diagnostic for G2: the model never scored. Look at the first")
        print("  rollout transcripts in the JSON — if the tool call is emitted but")
        print("  the arguments are empty/malformed, this is a 0.5B formatting")
        print("  limit, not a harness bug.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
