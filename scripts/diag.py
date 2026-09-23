"""Diagnostic — dump a full, un-truncated transcript for one (harness, task).

Written because the probe's transcript field truncates to 120 chars, which is
enough to see *that* a turn failed but not *why*. The specific question this
answers: after a successful tool call, is the model genuinely emitting EOS
(giving up), or is the driver mis-handling padding and reporting pad tokens as
the model's answer?

Run:
    ./run.sh scripts/diag.py --harness bash_minimal --task t2-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import ALL_HARNESSES
from multiharness.harnesses.core import discover_tools
from multiharness.rollout import Agent
from multiharness.tasks import load as load_tasks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="bash_minimal")
    ap.add_argument("--task", default="t2-01")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    load_tasks()
    cls = ALL_HARNESSES[args.harness]

    agent = Agent(
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    env0 = cls()
    print("=" * 78)
    print(f"DIAG  {args.harness} / {args.task}   n={args.n}")
    print("=" * 78)
    print(f"tools: {[m.__name__ for m in discover_tools(env0)]}")
    print(f"eos={agent.tokenizer.eos_token!r} pad={agent.tokenizer.pad_token!r}")

    obs = env0.reset(task_id=args.task)
    print("\n--- OBSERVATION handed to the model ---")
    print(obs)

    # Re-run the loop but with full-fidelity printing, so nothing is hidden.
    envs = [cls() for _ in range(args.n)]
    histories = [[{"role": "user", "content": envs[i].reset(task_id=args.task)}] for i in range(args.n)]

    for turn in range(args.max_turns):
        print(f"\n{'=' * 78}\nTURN {turn + 1}\n{'=' * 78}")
        texts = [agent._render(histories[i], envs[i]) for i in range(args.n)]
        print("--- rendered prompt (row 0, tail) ---")
        print(texts[0][-700:])

        raws = agent._generate_batch(texts)
        for i in range(args.n):
            prompt_ids, gen_ids = raws[i]
            parsed = agent._parse(gen_ids, prompt_ids)
            calls = parsed.get("tool_calls") or []
            print(f"\n--- row {i}: {len(gen_ids)} tokens generated ---")
            print(f"  first 12 ids : {gen_ids[:12]}")
            print(f"  decoded      : {agent.tokenizer.decode(gen_ids, skip_special_tokens=False)[:400]!r}")
            print(f"  skip_special : {agent.tokenizer.decode(gen_ids, skip_special_tokens=True)[:400]!r}")
            print(f"  parsed content: {parsed.get('content')!r}")
            print(f"  parsed calls  : {calls}")

        # advance every row (this diagnostic does not early-stop)
        for i in range(args.n):
            prompt_ids, gen_ids = raws[i]
            parsed = agent._parse(gen_ids, prompt_ids)
            calls = parsed.get("tool_calls") or []
            histories[i].append(
                {"role": "assistant", "content": parsed.get("content") or "", "tool_calls": calls} if calls
                else {"role": "assistant", "content": parsed.get("content") or ""}
            )
            for call in calls:
                fn = call.get("function", {})
                ok, payload = agent._invoke(envs[i], fn.get("name"), fn.get("arguments") or {})
                print(f"  [row {i}] tool {fn.get('name')} ok={ok} -> {str(payload)[:150]}")
                histories[i].append(
                    {"role": "tool", "name": fn.get("name") or "x", "content": str(payload)}
                )

    print("\n" + "=" * 78)
    for i in range(args.n):
        print(f"row {i}: reward={envs[i].get_reward()}  answer_file="
              f"{(envs[i]._workdir / 'answer.txt').read_text(encoding='utf-8', errors='replace')[:120]!r}"
              if (envs[i]._workdir / "answer.txt").is_file() else f"row {i}: reward={envs[i].get_reward()} no answer.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
