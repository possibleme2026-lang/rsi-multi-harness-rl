"""Guard: the tools a harness *advertises* must be the tools it *registers*.

Why this exists
---------------
Every harness GUIDANCE string lists its tools in prose ("Available tools:
`bash`, `read_file`, `replace_in_file`."). That prose is the only thing telling
the model what exists, and nothing enforces it against the methods TRL actually
discovers via ``inspect.getmembers(env, ismethod)``.

The failure modes this catches:

  * GUIDANCE promises a tool that was never implemented  -> the model calls it,
    gets "Tool X not found", and the rollout dies at max_turns. This looks
    exactly like "the model is bad at this harness", which is the very signal
    the experiment is trying to measure. A doc bug would be silently scored as
    a capability result.
  * A tool is implemented but absent from GUIDANCE  -> the model never learns it
    exists, so the harness is effectively smaller than intended.

Neither is visible in the pass-rate matrix, which is why it needs its own gate.
Runs in well under a second and needs no model, so it can go first in the
pipeline where it costs nothing.

Run:
    ./run.sh scripts/guard_tool_surface.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness.harnesses import ALL_HARNESSES  # noqa: E402
from multiharness.harnesses.core import discover_tools  # noqa: E402

# Tools that exist to drive the loop or read state rather than to be advertised
# as model-callable surface. `get_reward` is TRL's reward hook (grpo_trainer.py:677)
# and must never be callable by the model.
NON_MODEL_TOOLS = {"get_reward"}

# Matches a backticked identifier: `bash`, `read_file`, ...
BACKTICK = re.compile(r"`([a-z_][a-z0-9_]*)`")

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   ({detail})" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def main() -> int:
    print("=" * 74)
    print("GUARD — advertised tool surface vs registered tool surface")
    print("=" * 74)

    for name, cls in ALL_HARNESSES.items():
        env = cls()
        registered = sorted(t.__name__ for t in discover_tools(env) if t.__name__ not in NON_MODEL_TOOLS)

        # Only identifiers that appear in the "Available tools:" line count as
        # advertised. Scanning the whole GUIDANCE would pick up `answer.txt`
        # and similar path literals that are not tools.
        adv_line = ""
        for line in cls.GUIDANCE.splitlines():
            if "Available tools" in line or "one tool" in line or "exactly one" in line:
                adv_line += line + "\n"
        advertised = sorted(set(BACKTICK.findall(adv_line)))

        print(f"\n-- {name}")
        print(f"   advertised: {advertised}")
        print(f"   registered: {registered}")

        missing = [t for t in advertised if t not in registered]
        extra = [t for t in registered if t not in advertised]

        check(f"{name:<16} every advertised tool is implemented", not missing,
              f"advertised but missing: {missing}")
        check(f"{name:<16} every implemented tool is advertised", not extra,
              f"implemented but unadvertised: {extra}")
        check(f"{name:<16} advertises at least one tool", bool(advertised))
        check(f"{name:<16} get_reward is not model-callable",
              "get_reward" not in [t.__name__ for t in discover_tools(env)])

    # `bash` is the one tool every harness shares. If a harness dropped it the
    # cross-harness comparison would no longer hold the scaffold constant.
    print("\n-- cross-harness invariant")
    for name, cls in ALL_HARNESSES.items():
        reg = {t.__name__ for t in discover_tools(cls())}
        check(f"{name:<16} exposes bash", "bash" in reg)

    print()
    if FAILS:
        print(f"GUARD: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("GUARD: OK — advertised surface matches registered surface on all harnesses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
