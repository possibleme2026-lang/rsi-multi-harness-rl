"""Training entry point — single-harness baseline and multi-harness run.

One script for both arms of the ablation, because the comparison is only
meaningful if the two differ in exactly one thing: which environments the
trainer was given. Everything else — tasks, hyperparameters, seed, reward
source — is shared code, not copy-pasted code.

    --mode single   one harness, so the model is free to overfit its scaffold
    --mode multi    four harnesses, so it cannot

Both modes use the same 16 training tasks. The evaluation sweep (``eval.py``)
then measures each checkpoint on *held-out* tasks and the *held-out* harness,
which is where the generalization gap is read off.

Reward comes from the environment itself: every harness defines ``get_reward``,
which TRL auto-registers as a reward column (grpo_trainer.py:677). No
reward_funcs are passed — the verifier is deliberately harness-agnostic and
lives behind ``get_reward`` so the grader cannot see which scaffold produced
an answer.

Run:
    ./run.sh scripts/train.py --mode single --steps 40
    ./run.sh scripts/train.py --mode multi  --steps 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from multiharness._bootstrap import outputs_root
from multiharness.harnesses import (
    ALL_HARNESSES,
    HELDOUT_HARNESSES,
    TRAIN_HARNESSES,
)
from multiharness.tasks import load as load_tasks
from multiharness.tasks.suite import TRAIN_TASK_IDS

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_dataset(
    harness_names: list[str],
    task_ids: list[str],
    num_generations: int,
    per_step_unique: int = 2,
) -> Dataset:
    """One row per (task, harness) pair, round-robin so the split is even.

    The dataset carries no ``prompt`` column on purpose: the harness's
    ``reset()`` supplies the observation, which is what makes the harness —
    not the data file — the thing under test.

    ``per_step_unique`` must match ``per_device_batch / num_generations``,
    because GRPO draws that many distinct prompts per optimizer step. If the
    row count is not a multiple of it, the tail rows are silently dropped
    every epoch and those tasks never train — a failure that looks like
    "the model did not learn task X" rather than "task X was never shown".
    """
    rows = []
    for k, tid in enumerate(task_ids):
        # Rotate the starting harness so no harness is systematically paired
        # with the same tasks when len(harnesses) does not divide len(tasks).
        for j in range(len(harness_names)):
            h = harness_names[(k + j) % len(harness_names)]
            rows.append({"environment": h, "task_id": tid})

    if per_step_unique > 1 and len(rows) % per_step_unique:
        rows = rows[: len(rows) - (len(rows) % per_step_unique)]
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# signal filter
# --------------------------------------------------------------------------
# A binary reward gives GRPO a gradient only when the completions *within a
# group* disagree. For a task the model always passes (p=1) or never passes
# (p=0), every group of size G has zero variance with probability 1 — those
# rows are pure compute burn and, worse, they dilute the logged mean reward so
# a run can look like it is learning when nothing is moving.
#
# Measured on this machine (probe.py, n=8): T2 and most of T3 sit at p=0.00
# for every harness, so a naive training set is more than half dead rows.
#
# The filter is applied at *dataset construction* using pass rates measured
# beforehand, never during training, so it cannot leak eval information.
# Thresholds are deliberately wide: 0.05 < p < 0.95 keeps everything with
# any chance of signal while excluding the certain-dead rows.

SIGNAL_LO = 0.05
SIGNAL_HI = 0.95


def filter_live_rows(
    rows: list[dict], pass_rates: dict[tuple[str, str], float]
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Drop (harness, task) pairs whose measured pass rate is certainly dead."""
    live, dead = [], []
    for r in rows:
        key = (r["environment"], r["task_id"])
        p = pass_rates.get(key)
        if p is None:
            live.append(r)  # unmeasured: keep, do not silently drop data
        elif SIGNAL_LO < p < SIGNAL_HI:
            live.append(r)
        else:
            dead.append(key)
    return live, dead


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi"], required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--per-device-batch", type=int, default=8,
                    help="counts COMPLETIONS; unique prompts = this / num-generations")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--max-tool-iterations", type=int, default=3)
    ap.add_argument("--tag", default=None, help="output subdirectory name")
    ap.add_argument("--scan", default=None,
                    help="probe.json from a difficulty scan; enables the signal filter")
    ap.add_argument("--no-filter", action="store_true",
                    help="train on every row, including provably dead ones")
    args = ap.parse_args()

    load_tasks()

    if args.mode == "single":
        # bash_minimal is the plainest scaffold, so the baseline is not
        # handicapped by a harness the model would struggle with anyway.
        harnesses = {"bash_minimal": TRAIN_HARNESSES["bash_minimal"]}
    else:
        harnesses = dict(TRAIN_HARNESSES)

    assert not (set(harnesses) & set(HELDOUT_HARNESSES)), "held-out harness leaked into training"

    tag = args.tag or f"{args.mode}-s{args.steps}"
    out_dir = outputs_root() / f"train-{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_step_unique = max(1, args.per_device_batch // args.num_generations)

    # -- rows, then the signal filter -------------------------------------
    raw_rows = []
    for k, tid in enumerate(TRAIN_TASK_IDS):
        for j in range(len(harnesses)):
            h = list(harnesses)[(k + j) % len(harnesses)]
            raw_rows.append({"environment": h, "task_id": tid})

    pass_rates: dict[tuple[str, str], float] = {}
    dead: list[tuple[str, str]] = []
    if args.scan and not args.no_filter:
        scan = json.loads(Path(args.scan).read_text(encoding="utf-8"))
        for h, per_task in scan.get("matrix", {}).items():
            for tid, p in per_task.items():
                pass_rates[(h, tid)] = float(p)
        raw_rows, dead = filter_live_rows(raw_rows, pass_rates)

    # Keep only rows whose harness is actually in this arm (single mode).
    raw_rows = [r for r in raw_rows if r["environment"] in harnesses]

    if per_step_unique > 1 and len(raw_rows) % per_step_unique:
        raw_rows = raw_rows[: len(raw_rows) - (len(raw_rows) % per_step_unique)]
    if not raw_rows:
        print("!! no live rows left after filtering — nothing to train on")
        return 1
    dataset = Dataset.from_list(raw_rows)

    print("=" * 78)
    print(f"TRAIN  mode={args.mode}  harnesses={list(harnesses)}")
    print("=" * 78)
    print(f"dataset rows       : {len(dataset)} of {len(TRAIN_TASK_IDS) * len(harnesses)} possible")
    if dead:
        by_h: dict[str, int] = {}
        for h, _ in dead:
            by_h[h] = by_h.get(h, 0) + 1
        print(f"dropped as dead    : {len(dead)} rows  (p<=0.05 or p>=0.95) {by_h}")
    elif not args.scan:
        print("signal filter      : OFF (no --scan given); expect many zero-gradient steps")
    print(f"unique prompts/step: {per_step_unique}   completions/step: {args.per_device_batch}")
    print(f"steps              : {args.steps}   lr={args.lr}   lora_r={args.lora_r}")
    print(f"output             : {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    cfg = GRPOConfig(
        output_dir=str(out_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_iterations,
        max_steps=args.steps,
        logging_steps=1,
        save_steps=max(args.steps, 1),
        save_strategy="steps",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=42,
        use_vllm=False,
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        # PeftConfig -> LoRA. Full fine-tuning of even 0.5B does not fit
        # alongside the rollout activations on an 8 GB card.
        peft_config=_lora_config(args.lora_r),
        environment_factory=harnesses,
    )

    print(f"\nreward columns from env.get_reward: {trainer.reward_func_names}")
    assert trainer.reward_func_names, "no reward source — the run would be meaningless"

    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    # -- did anything actually happen? -------------------------------------
    # `frac_reward_zero_std` is the fraction of groups with no within-group
    # reward variance. Those groups have advantage exactly 0, so they cannot
    # move the policy. A run where this sits at 1.0 is a very expensive way to
    # do nothing, and `loss=0 / grad_norm=0` alone does not make that obvious
    # in the progress bar. This is the number to read before interpreting any
    # downstream eval.
    steps = [h for h in trainer.state.log_history if "frac_reward_zero_std" in h]
    zero_frac = [h["frac_reward_zero_std"] for h in steps]
    rewards = [h["reward"] for h in steps if "reward" in h]
    grad_norms = [h["grad_norm"] for h in steps if "grad_norm" in h]

    print("\n" + "-" * 78)
    print("LEARNING SIGNAL")
    print("-" * 78)
    if zero_frac:
        dead_steps = sum(1 for z in zero_frac if z >= 1.0)
        print(f"  steps logged              : {len(zero_frac)}")
        print(f"  steps with ZERO gradient  : {dead_steps} / {len(zero_frac)}"
              f"  ({dead_steps / len(zero_frac):.0%})")
        print(f"  mean frac_reward_zero_std : {sum(zero_frac) / len(zero_frac):.3f}")
    if rewards:
        print(f"  reward first -> last      : {rewards[0]:.3f} -> {rewards[-1]:.3f}")
    if grad_norms:
        nz = [g for g in grad_norms if g and g > 0]
        print(f"  nonzero grad_norm steps   : {len(nz)} / {len(grad_norms)}")
        if nz:
            print(f"  grad_norm max             : {max(nz):.4f}")

    summary = {
        "mode": args.mode,
        "harnesses": list(harnesses),
        "train_tasks": TRAIN_TASK_IDS,
        "rows_used": len(raw_rows),
        "rows_dropped_dead": [[h, t] for h, t in dead],
        "scan": args.scan,
        "steps": args.steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "num_generations": args.num_generations,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "max_tool_iterations": args.max_tool_iterations,
        "final_loss": result.training_loss,
        "output_dir": str(out_dir),
        "zero_gradient_steps": sum(1 for z in zero_frac if z >= 1.0),
        "steps_logged": len(zero_frac),
        "log_history": trainer.state.log_history,
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nfinal loss: {result.training_loss}")
    print(f"wrote {out_dir / 'train_summary.json'}")
    return 0


def _lora_config(r: int):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
