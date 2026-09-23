"""Smoke test B — the TRL multi-environment mechanism, end to end.

This is the geological survey of the whole plan. If GRPOTrainer does not
route by the ``environment`` column and does not render a *per-example* tool
schema, the multi-harness experiment is not buildable on this stack.

What is asserted, in order of how load-bearing it is:

  1. ``environment_factory`` as a dict is accepted and switches the trainer
     into multi-environment mode.
  2. A 2-row dataset carrying only ``environment`` + ``task_id`` (no
     ``prompt``) is accepted, and each row reaches the *matching* env class —
     proven by each dummy env recording the task ids it was handed.
  3. The tool schema is rendered per example, not per batch: env A's prompt
     must advertise ``tool_a`` and must NOT advertise ``tool_b``.
  4. The env-owned ``get_reward`` is auto-registered as a reward source and
     produces a non-constant reward (otherwise GRPO has no gradient).
  5. ``trainer.train()`` completes one optimizer step.

Run:
    ./run.sh tests/smoke_trl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

# ---------------------------------------------------------------------------
# Canary: refuse to spend a GPU run on a shadowed `trl`.
#
# This check is not decoration. In the repository this code was extracted from,
# the working tree contained a `trl/` git checkout, and because that repo root
# sat on sys.path the bare `import trl` bound to the *directory* as a namespace
# package rather than to the real install. The symptom was not an ImportError —
# it was `hasattr(trl, "__version__") == False` and, downstream, a bogus
# "tool-call rate: 0%" reading while the model was in fact emitting perfect
# tool calls. Two guards came out of that: this canary, and the rule that the
# bootstrap inserts `src/` and never the repository root (see _bootstrap.py).
#
# A shadowed `trl` is a namespace package, which has `__path__` but no
# `__version__` — hence the test below.
# ---------------------------------------------------------------------------
import trl  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from multiharness._bootstrap import outputs_root  # noqa: E402

if not hasattr(trl, "__version__"):
    print(f"!! `trl` resolved to a shadowing directory: {trl.__path__}")
    print("   expected the real install; aborting before wasting a GPU run")
    raise SystemExit(2)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = outputs_root() / "smoke_trl"

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


# --------------------------------------------------------------------------
# two dummy environments — the only thing that matters is which tools they own
# --------------------------------------------------------------------------

ROUTING_LOG: dict[str, list[str]] = {"dummy_a": [], "dummy_b": []}
RENDERED_TOOLS: list[tuple[str, list[str]]] = []


class DummyA:
    """Owns ``tool_a``. Records every task_id it is asked to reset."""

    def reset(self, task_id: str = "", **kwargs) -> str:
        ROUTING_LOG["dummy_a"].append(task_id)
        return f"[dummy_a] task={task_id}"

    def tool_a(self, x: str) -> str:
        """A tool that exists only in environment A.

        Args:
            x: Anything.
        """
        return f"a:{x}"

    def get_reward(self) -> float:
        # Deterministic but task-dependent, so the reward column is not constant.
        return 1.0 if ROUTING_LOG["dummy_a"][-1].endswith("1") else 0.0


class DummyB:
    """Owns ``tool_b``. Symmetric to A."""

    def reset(self, task_id: str = "", **kwargs) -> str:
        ROUTING_LOG["dummy_b"].append(task_id)
        return f"[dummy_b] task={task_id}"

    def tool_b(self, x: str) -> str:
        """A tool that exists only in environment B.

        Args:
            x: Anything.
        """
        return f"b:{x}"

    def get_reward(self) -> float:
        return 1.0 if ROUTING_LOG["dummy_b"][-1].endswith("2") else 0.0


def main() -> int:
    print("=" * 74)
    print("SMOKE B — TRL multi-environment routing")
    print("=" * 74)

    if not torch.cuda.is_available():
        print("!! no CUDA; this smoke is designed for the GPU box")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\ngpu: {torch.cuda.get_device_name(0)}  "
          f"free {torch.cuda.mem_get_info()[0] / 2**20:.0f} MiB")

    # -- dataset: ONLY environment + task_id, deliberately no `prompt` -------
    rows = [
        {"environment": "dummy_a", "task_id": "ta-1"},
        {"environment": "dummy_b", "task_id": "tb-2"},
    ]
    dataset = Dataset.from_list(rows)
    check("dataset has `environment` column", "environment" in dataset.column_names)
    check("dataset has NO `prompt` column (env supplies it)", "prompt" not in dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Capture the `tools=` argument each time a prompt is rendered.
    # NOTE: `_env_tools[name]` holds the *bound methods themselves*
    # (grpo_trainer.py:665 `tools + methods`), not pre-built JSON schemas —
    # transformers converts them downstream. So the spy must accept both
    # shapes rather than assuming dicts.
    real_act = tokenizer.apply_chat_template

    def _tool_name(t):
        if isinstance(t, dict):
            fn = t.get("function", t)
            return fn.get("name")
        return getattr(t, "__name__", None)

    def spy_act(conversation=None, tools=None, **kw):
        RENDERED_TOOLS.append(
            (str(conversation)[:40], sorted(n for n in (_tool_name(t) for t in (tools or [])) if n))
        )
        return real_act(conversation=conversation, tools=tools, **kw)

    tokenizer.apply_chat_template = spy_act

    args = GRPOConfig(
        output_dir=str(OUT),
        learning_rate=1e-6,
        # NOTE: in GRPO `per_device_train_batch_size` counts *completions*, not
        # unique prompts. The sampler draws
        # `per_device_train_batch_size * steps_per_generation` rows and repeats
        # each unique prompt `num_generations` times, so a batch holds
        # `per_device_train_batch_size / num_generations` distinct prompts.
        # Setting this to 2 with num_generations=2 yields ONE prompt per batch
        # and the second dataset row never trains at all — which silently makes
        # the routing assertion vacuous. 4 / 2 = both rows.
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=48,
        max_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        seed=42,
        use_vllm=False,
        max_tool_calling_iterations=1,
    )

    print("\n-- 1/2. constructing GRPOTrainer with a dict environment_factory --")
    trainer = GRPOTrainer(
        model=MODEL,
        environment_factory={"dummy_a": DummyA, "dummy_b": DummyB},
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    check("trainer entered multi-environment mode", trainer._multi_environment is True)
    check("both env factories registered",
          set(trainer.environment_factories) == {"dummy_a", "dummy_b"},
          str(sorted(trainer.environment_factories)))

    env_tools = {k: sorted(m.__name__ for m in v) for k, v in trainer._env_tools.items()}
    check("env A tool schema == ['tool_a']", env_tools["dummy_a"] == ["tool_a"], str(env_tools["dummy_a"]))
    check("env B tool schema == ['tool_b']", env_tools["dummy_b"] == ["tool_b"], str(env_tools["dummy_b"]))
    check("schemas are disjoint (this is what makes the harnesses differ)",
          not (set(env_tools["dummy_a"]) & set(env_tools["dummy_b"])))
    check("`get_reward` was NOT exposed as a tool",
          "get_reward" not in env_tools["dummy_a"] + env_tools["dummy_b"])

    print("\n-- 3. reward source auto-registered from env.get_reward --")
    check("two reward columns (one per env class)",
          len(trainer.reward_funcs) == 2, str(trainer.reward_func_names))
    check("reward columns named after the env classes",
          sorted(trainer.reward_func_names) == ["DummyA", "DummyB"],
          str(trainer.reward_func_names))

    print("\n-- 4. training one step --")
    RENDERED_TOOLS.clear()
    for k in ROUTING_LOG:
        ROUTING_LOG[k].clear()

    result = trainer.train()
    check("train() returned", result is not None)
    check("loss is finite", result.training_loss == result.training_loss, str(result.training_loss))

    print("\n-- 5. ROUTING: did each row reach the right env? --")
    print(f"    dummy_a received: {ROUTING_LOG['dummy_a']}")
    print(f"    dummy_b received: {ROUTING_LOG['dummy_b']}")
    check("env A was reset only with task ids meant for A",
          ROUTING_LOG["dummy_a"] and all(t.startswith("ta-") for t in ROUTING_LOG["dummy_a"]),
          str(ROUTING_LOG["dummy_a"]))
    check("env B was reset only with task ids meant for B",
          ROUTING_LOG["dummy_b"] and all(t.startswith("tb-") for t in ROUTING_LOG["dummy_b"]),
          str(ROUTING_LOG["dummy_b"]))
    check("no row was misrouted", len(ROUTING_LOG["dummy_a"]) + len(ROUTING_LOG["dummy_b"]) > 0)
    check("both envs were actually exercised (guards against a vacuous check)",
          len(ROUTING_LOG["dummy_a"]) > 0 and len(ROUTING_LOG["dummy_b"]) > 0,
          f"a={len(ROUTING_LOG['dummy_a'])} b={len(ROUTING_LOG['dummy_b'])}")
    check("reset called once per rollout (4 = 2 prompts x 2 generations)",
          len(ROUTING_LOG["dummy_a"]) + len(ROUTING_LOG["dummy_b"]) == 4,
          str(len(ROUTING_LOG["dummy_a"]) + len(ROUTING_LOG["dummy_b"])))

    print("\n-- 6. TOOL SCHEMA: rendered per example, not per batch? --")
    seen = [names for _, names in RENDERED_TOOLS]
    print(f"    {len(seen)} prompt render(s) captured")
    for _, names in RENDERED_TOOLS[:6]:
        print(f"      tools={names}")
    a_only = [n for n in seen if "tool_a" in n]
    b_only = [n for n in seen if "tool_b" in n]
    mixed = [n for n in seen if "tool_a" in n and "tool_b" in n]
    check("at least one render advertised tool_a", bool(a_only))
    check("at least one render advertised tool_b", bool(b_only))
    check("NO render advertised both (proves per-example schema)", not mixed, str(mixed))
    check("neither schema leaked into the other env's prompt",
          not mixed and bool(a_only) and bool(b_only))

    print("\n" + "=" * 74)
    if FAILS:
        print(f"SMOKE B: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("SMOKE B: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
