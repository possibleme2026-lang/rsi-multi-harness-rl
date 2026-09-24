"""Synthesise an environment batch and dump it for scanning.

Why this is its own script
--------------------------
The environment batch is the *input* to the benchmark, and it has to be
reproducible from a command. The first version of this experiment generated it
inside ``rsi_loop.py`` as a side effect of the full loop — so regenerating the
batch after a generator fix meant re-running the task axis, the curriculum axis
and the harness axis to get at it. Worse, the first scan on record was produced
by an ad-hoc command that was never written down, so the artifact could not be
rebuilt once its producer changed. An artifact whose producer is a shell history
entry is not evidence.

The dump is deliberately *not* the same file the loop writes. ``rsi_loop``
writes the batch it measured; this writes the batch you asked for. Both go
through ``env_adapter.as_harness_batch``, so they cannot diverge in shape.

Run:
    ./run.sh scripts/env_batch_make.py --n 12
    ./run.sh scripts/env_batch_make.py --n 12 --steps 2 --out outputs/rsi/env_batch_easy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness._bootstrap import outputs_root
from multiharness.rsi import env_adapter, envtask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=12, help="environment tasks to synthesise")
    ap.add_argument("--seed", type=int, default=11,
                    help="generation seed; the batch is deterministic given it")
    ap.add_argument("--records", type=int, default=4, help="relevant records per environment")
    ap.add_argument("--distractors", type=int, default=3,
                    help="irrelevant records; the knob that makes reading state necessary")
    ap.add_argument("--steps", type=int, default=4,
                    help="tool-chain length (2-4; 5 is refused as unreachable)")
    ap.add_argument("--out", default=None,
                    help="dump path (default: $MULTIHARNESS_OUT/rsi/env_batch.json)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing batch whose task ids differ")
    args = ap.parse_args()

    out = Path(args.out) if args.out else outputs_root() / "rsi" / "env_batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = envtask.generate_env_batch(
        args.n,
        seed=args.seed,
        n_records=args.records,
        n_distractors=args.distractors,
        n_steps=args.steps,
    )
    cov = envtask.summarise_env_batch(tasks)
    batch = env_adapter.as_harness_batch(tasks)

    # Refuse to clobber a different batch without being told to. Measured
    # reason: the first scan of an environment batch was overwritten by the
    # re-scan that was meant to be compared *against* it, because both wrote
    # ``env_scan.json``. The before-picture is the only evidence that a fix
    # changed anything, and it was lost to a default path. A batch is cheap to
    # regenerate, so the cost of asking is one flag.
    if out.exists() and not args.force:
        old = json.loads(out.read_text(encoding="utf-8"))
        old_ids = [t.get("id") for t in old] if isinstance(old, list) else []
        new_ids = [t["id"] for t in batch]
        if old_ids != new_ids:
            print(f"{out} already holds a different batch "
                  f"({len(old_ids)} tasks, first {old_ids[:1]})", file=sys.stderr)
            print("pass --force to overwrite, or --out to write elsewhere",
                  file=sys.stderr)
            return 2

    out.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"synthesised {cov['total']} environment tasks")
    print(f"  seed={args.seed} records={args.records} distractors={args.distractors} "
          f"steps={args.steps}")
    print(f"  by domain        : {cov['by_domain']}")
    print(f"  edges by reason  : {cov['edges_by_reason']}")
    print(f"  checkpoints      : min {cov['checkpoints_min']}  max {cov['checkpoints_max']}  "
          f"mean {cov['checkpoints_mean']}")
    print(f"  with distractors : {cov['with_distractors']}")
    print(f"wrote {out}  ({len(batch)} harness tasks)")
    print()
    print("scan it with:")
    print(f"  ./run.sh scripts/probe.py --from-batch {out} \\")
    print(f"      --out {out.parent / 'env_scan.json'} --n 2 --max-turns 6 \\")
    print("      --max-new-tokens 256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
