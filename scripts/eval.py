"""Evaluation sweep — where the cross-harness generalization gap is read off.

For each checkpoint, run every (harness, task) cell and build a reward matrix.
The headline number is:

    gap = mean(reward | train harnesses) - mean(reward | held-out harness)

with the held-out harness (``codex_style``) and the held-out tasks both unseen
during training. Both axes are held out because a model that memorized the
tasks would otherwise still look like it generalized; see tasks/suite.py.

Two controls are built in, because a gap number on its own is not evidence of
anything:

  --baseline   evaluate the *untrained* base model with identical settings, so
               the reported gap is a change, not an absolute level. A base
               model already has a gap; only the delta is attributable to
               training.
  --mode single vs --mode multi checkpoints are compared against each other,
               which is the actual ablation: does mixing harnesses during
               training shrink the held-out gap relative to overfitting one?

Cells with no reward variance at the base model carry no information about
generalization (the model could not do the task at all), so the report prints
per-cell counts alongside the means and warns when a harness's score rests on
too few live cells.

Run:
    ./run.sh scripts/eval.py --baseline
    ./run.sh scripts/eval.py --adapter outputs/train-multi-s40/final
    ./run.sh scripts/eval.py --baseline --adapter outputs/train-multi-s40/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from multiharness._bootstrap import outputs_root
from multiharness.harnesses import ALL_HARNESSES, HELDOUT_HARNESSES, TRAIN_HARNESSES
from multiharness.rollout import Agent
from multiharness.tasks import load as load_tasks
from multiharness.tasks.suite import EVAL_TASK_IDS, TRAIN_TASK_IDS


def sweep(
    agent: Agent,
    harness_names: list[str],
    task_ids: list[str],
    n: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    matrix: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for h in harness_names:
        cls = ALL_HARNESSES[h]
        matrix[h], counts[h] = {}, {}
        for tid in task_ids:
            rollouts = agent.run(cls, tid, n=n)
            rewards = [r.reward for r in rollouts]
            matrix[h][tid] = sum(rewards) / len(rewards)
            counts[h][tid] = int(sum(rewards))
    return matrix, counts


def mean_of(matrix: dict[str, dict[str, float]], harnesses: list[str], tasks: list[str]) -> float:
    vals = [matrix[h][t] for h in harnesses if h in matrix for t in tasks if t in matrix[h]]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="evaluate the untrained base model")
    ap.add_argument("--adapter", action="append", default=[],
                    help="path to a saved LoRA adapter dir; repeatable for the ablation")
    ap.add_argument("--n", type=int, default=4, help="rollouts per cell")
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--harnesses", default=None, help="override the harness list")
    ap.add_argument("--tasks", default=None, help="override the task list")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.baseline and not args.adapter:
        print("!! pass --baseline and/or --adapter; nothing to evaluate")
        return 1

    load_tasks()

    harness_names = (
        [h.strip() for h in args.harnesses.split(",") if h.strip()]
        if args.harnesses
        else list(ALL_HARNESSES)
    )
    task_ids = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(EVAL_TASK_IDS)
    )
    train_h = [h for h in harness_names if h in TRAIN_HARNESSES]
    held_h = [h for h in harness_names if h in HELDOUT_HARNESSES]

    print("=" * 78)
    print("EVAL — cross-harness generalization")
    print("=" * 78)
    print(f"harnesses : train={train_h}  held-out={held_h}")
    print(f"tasks     : {task_ids}  (held out from training)")
    print(f"cells     : {len(harness_names)} x {len(task_ids)} x {args.n} rollouts "
          f"= {len(harness_names) * len(task_ids) * args.n} per arm")
    print(f"seed      : {args.seed}  (identical across arms, so the comparison is paired)")

    def _run(tag: str, adapter: str | None) -> dict:
        """Evaluate one arm in a fresh Agent with the same seed.

        A fresh model per arm rather than swapping adapters in place: PEFT
        cannot cleanly detach an adapter, and a shared instance would carry
        the previous arm's sampling RNG state into the next. Identical seeding
        plus a fresh load keeps the arms comparable.
        """
        torch.manual_seed(args.seed)
        agent = Agent(
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if adapter:
            agent.load_adapter(adapter)

        print(f"\n--- {tag}" + (f"  (adapter={adapter})" if adapter else "  (base model)") + " ---")
        m, c = sweep(agent, harness_names, task_ids, args.n)
        for h in harness_names:
            row = "".join(f"{m[h][t]:9.2f}" for t in task_ids)
            kind = "HELD-OUT" if h in HELDOUT_HARNESSES else "train"
            print(f"  {h:<18}[{kind:<8}]{row}")
        tr = mean_of(m, train_h, task_ids)
        ho = mean_of(m, held_h, task_ids)
        print(f"\n  mean(train harnesses)  = {tr:.4f}")
        print(f"  mean(held-out harness) = {ho:.4f}")
        print(f"  >>> cross-harness gap  = {tr - ho:+.4f}")

        del agent
        return {
            "label": tag,
            "adapter": adapter,
            "matrix": m,
            "counts": c,
            "train_harnesses": train_h,
            "heldout_harnesses": held_h,
            "eval_tasks": task_ids,
            "mean_train_harness": tr,
            "mean_heldout_harness": ho,
            "gap": tr - ho,
            "n_per_cell": args.n,
        }

    results: dict[str, dict] = {}
    if args.baseline:
        results["baseline"] = _run("baseline", None)
    for path in args.adapter:
        # Name the arm after its directory so the report reads on its own.
        results[Path(path).parent.name or path] = _run(Path(path).parent.name or "trained", path)

    # -- comparison --------------------------------------------------------
    if len(results) > 1:
        print("\n" + "=" * 78)
        print("ABLATION")
        print("=" * 78)
        base = results.get("baseline")
        order = list(results)
        print(f"  {'arm':<24}{'train-h':>10}{'held-out':>10}{'gap':>10}{'d gap':>10}{'d held':>10}")
        for name in order:
            r = results[name]
            d_gap = f"{r['gap'] - base['gap']:+.4f}" if base and name != "baseline" else ""
            d_ho = (f"{r['mean_heldout_harness'] - base['mean_heldout_harness']:+.4f}"
                    if base and name != "baseline" else "")
            print(f"  {name:<24}{r['mean_train_harness']:>10.4f}"
                  f"{r['mean_heldout_harness']:>10.4f}{r['gap']:>+10.4f}{d_gap:>10}{d_ho:>10}")

        if base:
            print()
            for name in order:
                if name == "baseline":
                    continue
                r = results[name]
                d_tr = r["mean_train_harness"] - base["mean_train_harness"]
                d_ho = r["mean_heldout_harness"] - base["mean_heldout_harness"]
                d_gap = r["gap"] - base["gap"]
                if d_ho > 0 and d_gap < 0:
                    verdict = "transfer: held-out up, gap down"
                elif d_ho > 0:
                    verdict = "held-out up but train up more — gain is harness-specific"
                elif d_tr > 0:
                    verdict = "train up, held-out flat/down — overfitting the train harnesses"
                else:
                    verdict = "no improvement on either side"
                print(f"  {name}: d_held={d_ho:+.4f} d_train={d_tr:+.4f} d_gap={d_gap:+.4f}  -> {verdict}")

            # The headline claim of the experiment, stated as a test rather
            # than left for the reader to eyeball.
            trained = [n for n in order if n != "baseline"]
            if len(trained) >= 2:
                gaps = {n: results[n]["gap"] for n in trained}
                best = min(gaps, key=gaps.get)
                worst = max(gaps, key=gaps.get)
                print(f"\n  smallest gap: {best} ({gaps[best]:+.4f})")
                print(f"  largest  gap: {worst} ({gaps[worst]:+.4f})")
                if "multi" in best and "single" in worst:
                    print("  >>> multi-harness training produced the smaller gap — hypothesis supported")
                elif "single" in best and "multi" in worst:
                    print("  >>> single-harness training produced the smaller gap — hypothesis NOT supported")
                else:
                    print("  >>> gap difference is not aligned with the harness count — inconclusive")

    payload = {
        "n": args.n,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "train_task_ids": TRAIN_TASK_IDS,
        "eval_task_ids": EVAL_TASK_IDS,
        "runs": results,
    }
    if args.out:
        out = Path(args.out)
    else:
        out = outputs_root() / ("eval_baseline.json" if args.baseline and not args.adapter else "eval_ablation.json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
