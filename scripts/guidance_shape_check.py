"""Does the guidance actually change the tool shape the model emits?

This is the falsification test for the fix. The environment scan takes tens of
minutes and answers a different question ("can the model score"); this answers
the narrow one that the fix is about: given the *old* guidance and the *new*
guidance, does the model emit a different first tool call?

It exists because the first fix to the guidance was validated by a metric that
went up while nothing improved. The tool-call rate rose 35.4% → 81.2% and the
pass rate stayed at exactly 0.00, because the model was calling
``envtool list_tickets`` *as a tool name* — and a call to a non-existent tool
counts as a tool call. **The check has to be on the shape, not the count.**

So this script renders one prompt per guidance variant, generates a few
completions, and classifies what came back:

  correct      ``bash(command="envtool <tool> ...")`` — the intended shape
  as_tool      ``envtool <tool>(...)`` — the defect: the command used as a name
  other_tool   some other harness tool, e.g. ``shell``
  no_call      prose, or nothing

It prints both arms side by side. The old arm is reconstructed from the text
the old function produced, kept verbatim in ``_OLD_GUIDANCE`` so the comparison
cannot drift from the thing it is comparing against.

Run:
    ./run.sh scripts/guidance_shape_check.py --n 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.rsi import env_adapter, envtask

#: The guidance as it was before the fix, reproduced exactly. Its shape is the
#: point: a bare list of commands, indistinguishable from the harness's own
#: tool-schema block that the chat template renders above it.
_OLD_GUIDANCE = """\
You have a shell, and the environment's tools are installed as the command \
`envtool`. Each tool is one shell command.

The tools, and their arguments:

{listing}

Example: `envtool {first}`, `envtool {mutator} <id> <value>`.

Start by listing the records, then read the one the task describes. The \
instruction identifies it by a property, not by name, so you have to look it up.

**Your work is graded on the state your commands leave behind**, not on the \
commands themselves and not on anything you write to a text file. Do not write \
an answer file. The tools update the state for you, and any sequence of \
commands that produces the right final state scores full marks.
"""


def old_guidance(task: envtask.EnvTask) -> str:
    """Reconstruct the pre-fix text so the two arms are compared fairly."""
    g = task.graph
    names = g.get("nodes") or []
    mutators = [n for n in names if g.get("kinds", {}).get(n) == "mutate"]
    rows = []
    for n in names:
        params = g.get("params", {}).get(n, [])
        args = " ".join(f"<{p}>" for p in params)
        rows.append(f"  envtool {n} {args}".rstrip())
    return _OLD_GUIDANCE.format(
        listing="\n".join(rows),
        first=names[0] if names else "list_items",
        mutator=mutators[0] if mutators else "set_state",
    )


_AS_TOOL = re.compile(r"envtool\s+[a-z_]+")
_AS_CMD = re.compile(r"""command\s*[=:]\s*["']envtool\s+([a-z_]+)""")


def shape(text: str) -> str:
    """Classify one completion by the shape of the call it contains.

    Operates on the raw decoded completion, *including* special tokens, and
    takes a ``<tool_call>`` block as the signal that a call was emitted. This
    mirrors what the real loop's parser keys on: the first version of this
    check decoded with ``skip_special_tokens=True``, which strips the
    ``<tool_call>`` delimiters, and every completion then looked like prose —
    both arms reported 100% ``no_call`` against a scan that showed 100% tool
    calls. A check that disagrees with the thing it checks is measuring itself.
    """
    if "<tool_call>" not in text and "envtool" not in text:
        return "no_call"
    if _AS_CMD.search(text):
        return "correct"
    # `envtool <name>(...)` or `envtool <name>` as the *tool name* of a
    # <tool_call> block: the defect.
    if re.search(r'"name"\s*:\s*"envtool', text):
        return "as_tool"
    if re.search(r"^\s*envtool\s+[a-z_]+\s*\(", text, re.M):
        return "as_tool"
    if '"name"' in text and "envtool" in text and _AS_CMD.search(text) is None:
        return "as_tool"
    if "bash" in text:
        return "bash_other"
    if _AS_TOOL.search(text):
        return "as_tool"
    return "no_call"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=6, help="completions per arm")
    ap.add_argument("--tasks", type=int, default=4, help="distinct tasks to draw from")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()

    tasks = envtask.generate_env_batch(
        args.tasks, seed=11, n_records=4, n_distractors=3, n_steps=4
    )

    from multiharness.harnesses.pool import BashMinimalEnv
    from multiharness.rollout import Agent

    agent = Agent(max_turns=1, max_new_tokens=args.max_new_tokens)
    print("=" * 78)
    print("GUIDANCE SHAPE CHECK — does the prompt change the call the model emits?")
    print("=" * 78)
    print(f"model : {agent.model_id} on {agent.device}")
    print(f"n     : {args.n} completions per arm per task, {len(tasks)} tasks")
    print()

    results: dict[str, dict[str, int]] = {"old": {}, "new": {}}
    for label, render in (("old", old_guidance), ("new", env_adapter.harness_guidance)):
        print(f"--- {label} guidance ---")
        for task in tasks:
            # Built by hand rather than through ``env.reset``: reset resolves the
            # task through the process-global registry, and these tasks are
            # generated in this process and never registered. Registering them
            # would mutate global state that the rest of the run shares, and the
            # thing under test is the *text*, not the reset path.
            body = f"{task.instruction}\n\n{render(task)}".strip()
            msgs = [{"role": "user", "content": body}]
            # A bare harness instance supplies the tool schema the chat template
            # renders above the instruction — which is the second list that the
            # old guidance was being confused with, so it has to be present.
            env = BashMinimalEnv()
            text = agent._render(msgs, env)
            raws = agent._generate_batch([text] * args.n)
            here: dict[str, int] = {}
            for pids, gids in raws:
                # Parsed through the *same* entry point the rollout loop uses,
                # so this check cannot pass while the real path behaves
                # differently. Decoding by hand was the first version's mistake.
                parsed = agent._parse(gids, pids)
                calls = parsed.get("tool_calls") or []
                if calls:
                    args_blob = json.dumps(calls)
                    s = "correct" if '"command"' in args_blob else "as_tool"
                else:
                    s = shape(agent.tokenizer.decode(gids, skip_special_tokens=False))
                here[s] = here.get(s, 0) + 1
                results[label][s] = results[label].get(s, 0) + 1
            print(f"    {task.task_id}  " + "  ".join(
                f"{k}={v}" for k, v in sorted(here.items())))

    print()
    print("=" * 78)
    print("SHAPE OF THE FIRST TOOL CALL")
    print("=" * 78)
    keys = sorted(set(results["old"]) | set(results["new"]))
    total = {"old": sum(results["old"].values()), "new": sum(results["new"].values())}
    print(f"  {'shape':<14}{'old':>12}{'new':>12}")
    for k in keys:
        o = results["old"].get(k, 0)
        n = results["new"].get(k, 0)
        print(f"  {k:<14}{o:>5} ({o / total['old']:>4.0%}){n:>5} ({n / total['new']:>4.0%})")
    print()
    fixed = results["new"].get("correct", 0) / max(total["new"], 1)
    broke = results["new"].get("as_tool", 0) / max(total["new"], 1)
    print(f"correct shape  old {results['old'].get('correct', 0) / max(total['old'], 1):.0%}"
          f"  ->  new {fixed:.0%}")
    print(f"as_tool defect old {results['old'].get('as_tool', 0) / max(total['old'], 1):.0%}"
          f"  ->  new {broke:.0%}")
    if fixed == 0.0:
        print("\n  The fix did not change the emitted shape. Do not report the")
        print("  scan as a capability measurement until this arm moves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
