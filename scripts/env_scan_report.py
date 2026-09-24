"""Read an environment scan and report *why* each cell scored what it scored.

The scan itself (``probe.py --from-batch outputs/rsi/env_batch.json --out
outputs/rsi/env_scan.json``) reports a pass-rate matrix and four gates. That is
enough to see the floor and not enough to explain it: a matrix of zeros looks
identical whether the model never called a tool, called the wrong tool, called
the right tool and never mutated, or mutated the wrong record.

This script answers the "why" from the artifact alone — no GPU, no re-run — by
classifying every rollout's transcript into a stage on the path from "no tool
call" to "reference behaviour". The stages are ordered, so the report reads as
a funnel and the first big drop is the actual bottleneck:

  no_call        emitted no tool call at all
  bad_args       called a tool, but the arguments were rejected by the harness
  unknown_tool   called a tool name the environment does not define
  query_only     only ever read state; never attempted a mutation
  wrong_target   attempted a mutation and still scored 0
  mutated_ok     attempted a mutation and scored above 0

The last two are read off the *reward*, because the reference trace is not in
the scan artifact. ``wrong_target`` and ``mutated_ok`` therefore mean "a
mutation was attempted and it did / did not pay off", which is not quite the
same as "the right record was / was not hit" — a rollout could hit the right
record with the wrong value and land in ``wrong_target``. The distinction the
buckets *do* make reliably is whether the mutation path pays off at all.

``mutated_ok`` non-empty is the good news and the reason to trust the zeros
elsewhere: it means the whole path works end to end — the agent reached the
environment, mutated the described record, and the checkpoints paid it. A floor
of zeros alongside a non-empty ``mutated_ok`` is a capability floor; a floor of
zeros with ``mutated_ok`` empty is a plumbing question first.

Run:
    ./run.sh scripts/env_scan_report.py
    ./run.sh scripts/env_scan_report.py --scan outputs/rsi/env_scan.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multiharness._bootstrap import outputs_root

#: Read tools as the environment exports them. A call to one of these changes
#: nothing, so a rollout that only ever issued these has not started the task.
QUERY_TOOLS = frozenset({"list_items", "list_tickets", "show", "get", "search", "count_by_sku"})

#: A tool call as it appears in a transcript, in either of the two spellings
#: the harnesses produce: the parsed form ``bash({'command': 'envtool X ...'})``
#: and the bare command text inside a tool-call blob.
_CALL_RE = re.compile(r"envtool\s+([a-z_]+)((?:\s+[^\s'\",}]+)*)")

STAGES = ("no_call", "bad_args", "unknown_tool", "query_only", "wrong_target", "mutated_ok")


def invented_tools(records: list[dict]) -> collections.Counter:
    """Tool names the model emitted that the harness does not have.

    This is the diagnostic that found the real defect. The first environment
    scan scored 0.00 in every cell *and* raised its tool-call rate from 35.4% to
    81.2%, which reads as progress. It was not: the guidance rendered the
    environment's commands as a tool list, the prompt already had one, and the
    model merged them into ``envtool list_tickets`` **as a tool name**. A call to
    a non-existent tool is still a tool call, so the gate passed while nothing
    worked. Counting these names is how that stops being invisible.

    The names are worth reading as a group: ``envtool list_accounts`` and
    ``envtool list_items`` appear on tasks whose environment has no such command
    at all, i.e. the model is not misreading the environment — it is answering
    the *shape* of the prompt rather than its content.
    """
    pat = re.compile(r"Tool (.+?) not found")
    counts: collections.Counter = collections.Counter()
    for r in records:
        for line in r.get("raw", []):
            for m in pat.finditer(str(line)):
                counts[(r["harness"], m.group(1))] += 1
    return counts


def classify(record: dict, query_tools: frozenset[str] = QUERY_TOOLS) -> str:
    """Bucket one rollout. First matching stage wins; the order is the funnel.

    The order matters and got it wrong once. A bogus tool name and a rejected
    argument both produce a ``tool_errors`` count, so a classifier that tests
    ``tool_errors`` before reading the error *text* files every invented tool
    name under ``bad_args`` — which is exactly the mislabelling that made the
    first environment scan look like a model-argument problem rather than the
    prompt-shape problem it was. The error string is therefore read first.
    """
    if record.get("tool_calls", 0) <= 0:
        return "no_call"
    blob = "\n".join(str(t) for t in record.get("raw", []))

    # Read the harness's own error text before inferring anything. "not found"
    # is the signature of a name that is not a tool at all — including
    # `envtool list_tickets`, which the model emits as a *tool name*.
    if "not found" in blob:
        return "unknown_tool"
    if "bad arguments" in blob or "unexpected keyword" in blob:
        return "bad_args"

    calls = _CALL_RE.findall(blob)
    if not calls:
        # A tool call happened and produced no error we recognise: either the
        # arguments never reached dispatch, or the transcript truncated.
        return "unknown_tool"
    if all(c[0] in query_tools for c in calls):
        return "query_only"
    # It attempted a mutation. The reference is not in the scan artifact, so
    # whether it hit the *right* record is read off the reward: a correct
    # mutation would have scored above zero.
    if record.get("reward", 0.0) > 0.0:
        return "mutated_ok"
    return "wrong_target"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default=None,
                    help="scan artifact (default: $MULTIHARNESS_OUT/rsi/env_scan.json)")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many transcripts from the largest non-empty bucket")
    ap.add_argument("--stage", default=None, choices=STAGES,
                    help="print transcripts from this bucket instead of the largest")
    args = ap.parse_args()

    path = Path(args.scan) if args.scan else outputs_root() / "rsi" / "env_scan.json"
    if not path.is_file():
        print(f"no scan at {path}", file=sys.stderr)
        print("produce one with:", file=sys.stderr)
        print("  ./run.sh scripts/probe.py --from-batch outputs/rsi/env_batch.json \\",
              file=sys.stderr)
        print("      --out outputs/rsi/env_scan.json --n 2 --max-turns 6 \\",
              file=sys.stderr)
        print("      --max-new-tokens 256", file=sys.stderr)
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    harnesses = sorted(data["matrix"].keys())

    print("=" * 78)
    print("ENVIRONMENT SCAN — what each cell actually did")
    print("=" * 78)
    print(f"scan      : {path}")
    print(f"model     : {data.get('model')}")
    print(f"rollouts  : {len(records)}  ({data.get('n')} per cell, "
          f"max_turns={data.get('max_turns')}, max_new_tokens={data.get('max_new_tokens')})")
    print()

    by_h = collections.defaultdict(list)
    for r in records:
        by_h[r["harness"]].append(r)

    # ---- the funnel, per harness ----------------------------------------
    print("stage funnel (share of each harness's rollouts)")
    width = max(len(h) for h in harnesses) + 2
    print(f"  {'harness':<{width}}" + "".join(f"{s:>13}" for s in STAGES))
    totals = collections.Counter()
    for h in harnesses:
        stages = [classify(r) for r in by_h[h]]
        counts = collections.Counter(stages)
        totals.update(counts)
        n = len(stages)
        print(f"  {h:<{width}}" + "".join(
            f"{counts.get(s, 0) / n:>12.0%} " for s in STAGES))
    n_all = len(records)
    print(f"  {'ALL':<{width}}" + "".join(f"{totals.get(s, 0) / n_all:>12.0%} " for s in STAGES))
    print()

    # ---- the reward column, so the funnel is read next to the number ------
    print("reward (what the funnel produced)")
    print(f"  {'harness':<{width}}{'pass':>8}{'best':>8}{'calls':>8}{'errs':>8}{'turns':>8}")
    for h in harnesses:
        rs = by_h[h]
        rewards = [r["reward"] for r in rs]
        called = sum(1 for r in rs if r["tool_calls"] > 0)
        print(f"  {h:<{width}}{sum(rewards) / len(rewards):>8.2f}{max(rewards):>8.2f}"
              f"{called / len(rs):>8.0%}{sum(r['tool_errors'] for r in rs):>8}"
              f"{sum(r['turns'] for r in rs) / len(rs):>8.2f}")
    print()

    # ---- the names that were not tools ----------------------------------
    invented = invented_tools(records)
    if invented:
        print("tool names the model emitted that do not exist")
        for (h, name), n in invented.most_common(8):
            print(f"  {n:>3}  {h:<17} {name}")
        print()

    # ---- the finding that matters ---------------------------------------
    print("reading")
    if totals.get("mutated_ok"):
        print(f"  ok  {totals['mutated_ok']} rollout(s) mutated and were paid — the path")
        print("      works end to end, so the zeros beside it are a capability floor")
        print("      and not a plumbing failure.")
    else:
        print("  !! no rollout mutated and scored. The zeros are not yet readable as a")
        print("     capability floor — check the reward path before reporting them.")
    for stage, note in (
        ("no_call", "never engaged; this is a prompting/model limit, not a harness bug"),
        ("bad_args", "the harness rejected the call — look at the argument shape"),
        ("unknown_tool", "called a name outside the tool surface — guidance problem"),
        ("query_only", "read but never mutated; the bottleneck if this dominates"),
        ("wrong_target", "mutated but was not paid — state-reading or value error"),
    ):
        share = totals.get(stage, 0) / n_all
        if share >= 0.15:
            print(f"  {share:>5.0%}  {stage:<13} {note}")
    print()

    # ---- one transcript, on request -------------------------------------
    if args.show:
        biggest = args.stage or max(
            (s for s in STAGES if totals.get(s)),
            key=lambda s: totals[s],
            default=None,
        )
        picked = [r for r in records if classify(r) == biggest][: args.show]
        print(f"--- {len(picked)} transcript(s) from the largest bucket: {biggest} ---")
        for r in picked:
            print(f"\n[{r['harness']} / {r['task_id']}] reward={r['reward']} "
                  f"turns={r['turns']} stopped={r['stopped_reason']}")
            for line in r.get("raw", [])[:6]:
                print(f"  {str(line)[:400]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
