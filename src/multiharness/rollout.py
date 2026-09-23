"""Rollout driver — a faithful re-implementation of TRL's tool-calling loop.

Why this exists
---------------
The capability probe and the evaluation sweep both need to run a harness
*without* a trainer. The tempting shortcut is to write a simple
generate-then-check loop, but then the probe would measure a different agent
than the one that actually gets trained, and the Go/No-Go verdict would be
about the wrong system.

So this module mirrors ``GRPOTrainer._tool_call_loop`` (grpo_trainer.py:1958)
in the parts that change behaviour:

  * the prompt is rendered with the environment's *own* tool schema
    (``_tokenize_prompts``, grpo_trainer.py:1751);
  * a turn ends at the first response with no tool calls;
  * tool results are fed back as ``{"role": "tool", "name": ..., "content": ...}``
    (grpo_trainer.py:2032);
  * a tool that raises is a *failed tool call*, not a crash — the error string
    goes back to the model as the tool's content (grpo_trainer.py:2000);
  * the loop stops after ``max_tool_calling_iterations`` (grpo_trainer.py:1970).

One deliberate divergence, documented so it is not mistaken for fidelity:
TRL rebuilds the next prompt by *token concatenation* (prompt + completion +
tool suffix) to keep token-exact TITO for the policy gradient. This driver
re-renders the whole conversation instead. For measuring capability the two
are equivalent — re-rendering is in fact what a real agent harness does at
inference time — but token-level logprobs from this path must never be fed to
a trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import add_response_schema, parse_response

from .harnesses.core import (
    BaseHarnessEnv,
    discover_tools,
    tool_names,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class Rollout:
    """What happened during one agent attempt."""

    __slots__ = (
        "harness",
        "task_id",
        "reward",
        "turns",
        "tool_calls",
        "tool_errors",
        "stopped_reason",
        "transcript",
    )

    def __init__(self, harness: str, task_id: str) -> None:
        self.harness = harness
        self.task_id = task_id
        self.reward = 0.0
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.stopped_reason = "unknown"
        self.transcript: list[str] = []

    def as_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "task_id": self.task_id,
            "reward": self.reward,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "stopped_reason": self.stopped_reason,
        }


class Agent:
    """Loads the model once and runs rollouts against any harness."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 256,
        max_turns: int = 6,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_turns = max_turns
        self.temperature = temperature
        self.top_p = top_p
        self.adapter_dir: str | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Same call TRL makes at grpo_trainer.py:729 so `parse_response` can
        # extract tool calls from generated token ids.
        add_response_schema(self.tokenizer)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        if self.tokenizer.pad_token_id != self.model.config.pad_token_id:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    # -- rendering ---------------------------------------------------------

    def load_adapter(self, adapter_dir: str | Path) -> None:
        """Attach a trained LoRA adapter in place.

        Used by the ablation so the baseline and the trained arm share one
        model instance — reloading the base model between arms would change
        the sampling RNG state and make the comparison less controlled than it
        needs to be.
        """
        from peft import PeftModel

        if getattr(self.model, "peft_config", None) is not None:
            raise RuntimeError("an adapter is already loaded; construct a fresh Agent instead")
        self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        self.model.eval()
        self.adapter_dir = str(adapter_dir)

    def _tool_schemas(self, env: BaseHarnessEnv) -> list | None:
        """Bound methods, exactly as GRPOTrainer passes them (grpo_trainer.py:665)."""
        methods = discover_tools(env)
        return methods or None

    def _render(self, messages: list[dict], env: BaseHarnessEnv) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=messages,
            tools=self._tool_schemas(env),
            add_generation_prompt=True,
            tokenize=False,
        )

    # -- one turn ----------------------------------------------------------

    @torch.no_grad()
    def _generate_batch(self, texts: list[str]) -> list[tuple[list[int], list[int]]]:
        """Generate one turn for a batch of rendered prompts.

        Returns ``(prompt_ids, generated_ids)`` per row. Both are needed by the
        parser: ``prefix=prompt_ids`` is mandatory with a new-style
        ``response_template`` (see ``_parse``), and the response must be parsed
        from its *generated token ids* rather than from a re-encoding of the
        decoded string.
        """
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
            top_p=self.top_p if self.temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1] :]
        attn = enc["attention_mask"]
        # Strip left-padding so `prefix` is the real prompt, not pad + prompt.
        prompt_ids = [
            enc["input_ids"][i][attn[i].bool()].tolist() for i in range(enc["input_ids"].shape[0])
        ]
        return [(prompt_ids[i], gen[i].tolist()) for i in range(len(prompt_ids))]

    def _parse(self, gen_ids: list[int], prefix_ids: list[int]) -> dict:
        """Route a generated response through TRL's own parser.

        Two details are load-bearing and were both learned the hard way:

        1. ``prefix=`` is **mandatory** with a new-style ``response_template``
           (transformers >= 5.13). Omitting it raises ``ValueError`` inside
           ``transformers.tokenization_utils_base.parse_response``, and TRL's
           wrapper catches exactly that class and *silently* returns the raw
           text with no ``tool_calls`` key — which makes every rollout look
           like "the model never called a tool". That is a very convincing
           wrong answer: the model may be emitting perfect tool calls.
        2. Parse the *generated token ids*, not a re-tokenization of the
           decoded string: the parser is a token-level API and re-encoding can
           split or merge tokens at the anchors.
        """
        try:
            return parse_response(self.tokenizer, gen_ids, prefix=prefix_ids)
        except Exception:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            return {"role": "assistant", "content": text}

    # -- the loop ----------------------------------------------------------

    def run(self, env_cls: type[BaseHarnessEnv], task_id: str, n: int = 1) -> list[Rollout]:
        """Run ``n`` independent attempts at ``task_id``.

        Takes a *class*, not an instance: a harness instance owns exactly one
        workdir, so sharing one instance across ``n`` attempts would have every
        reset clobber the previous attempt's workspace and all ``n`` rollouts
        would be graded against whichever workspace was made last. This mirrors
        GRPOTrainer's own per-rollout instance pool (grpo_trainer.py:2313).
        """
        envs = [env_cls() for _ in range(n)]
        results = [Rollout(envs[i].name, task_id) for i in range(n)]

        histories: list[list[dict]] = []
        for i in range(n):
            observation = envs[i].reset(task_id=task_id)
            histories.append([{"role": "user", "content": observation}])

        for turn in range(self.max_turns):
            active = [i for i in range(n) if results[i].stopped_reason == "unknown"]
            if not active:
                break

            texts = [self._render(histories[i], envs[i]) for i in active]
            raws = self._generate_batch(texts)

            for slot, i in enumerate(active):
                results[i].turns = turn + 1
                prompt_ids, gen_ids = raws[slot]
                parsed = self._parse(gen_ids, prompt_ids)
                content = parsed.get("content") or ""
                calls = parsed.get("tool_calls") or []

                if not calls:
                    histories[i].append({"role": "assistant", "content": content})
                    results[i].stopped_reason = "no_tool_call"
                    results[i].transcript.append(f"T{turn + 1} no-tool-call: {content[:200]}")
                    continue

                results[i].tool_calls += len(calls)
                histories[i].append({"role": "assistant", "content": content, "tool_calls": calls})
                for call in calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if not isinstance(args, dict):
                        args = {}
                    ok, payload = self._invoke(envs[i], name, args)
                    if not ok:
                        results[i].tool_errors += 1
                    results[i].transcript.append(
                        f"T{turn + 1} {name}({str(args)[:90]}) -> {str(payload)[:120]}"
                    )
                    histories[i].append(
                        {"role": "tool", "name": name or "unknown", "content": str(payload)}
                    )

                if turn == self.max_turns - 1:
                    results[i].stopped_reason = "max_turns"

        for i, r in enumerate(results):
            r.reward = envs[i].get_reward()
        return results

    def _invoke(self, env: BaseHarnessEnv, name: str | None, args: dict) -> tuple[bool, Any]:
        """Call a tool the way TRL does: unknown name or raised exception both
        become an error *string fed back to the model*, never a crash
        (grpo_trainer.py:1993-2003)."""
        available = {m.__name__: m for m in discover_tools(env)}
        if name not in available:
            return False, {"error": f"Tool {name} not found. Available: {sorted(available)}"}
        try:
            return True, available[name](**args)
        except TypeError as exc:
            return False, {"error": f"bad arguments for {name}: {exc}"}
        except Exception as exc:  # noqa: BLE001 - mirrors TRL's broad catch
            return False, {"error": str(exc)}

    def tool_names(self, env: BaseHarnessEnv) -> list[str]:
        return tool_names(env)
