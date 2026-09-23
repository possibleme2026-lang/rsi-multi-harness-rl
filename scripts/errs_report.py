"""Classify tool errors from a scan dump: harness bug vs model misbehaviour.

The scan prints a single `errs=` count per cell, which conflates three very
different things:

  * `Tool X not found`, where X **is** advertised in GUIDANCE
        -> the harness is broken. The model was told a tool exists and it does
           not. Must be fixed before the numbers mean anything.
  * `Tool X not found`, where X is **not** advertised
        -> the model hallucinating a tool (`write_to_file` on a harness that
           only offers `replace_in_file`). That is genuine capability signal
           and must NOT be "fixed" — it is precisely the cross-harness
           adaptation the experiment is trying to measure.
  * `bad arguments for X`
        -> a real tool called with the wrong signature. Also genuine signal.

Only the first class can invalidate the experiment. The classification is done
automatically against each harness's GUIDANCE (and cross-checked against the
registered tool set), because leaving it to the reader means a broken harness
gets mistaken for a model limitation — the exact confusion this experiment is
least able to afford.

Usage:
    ./run.sh scripts/errs_report.py outputs/scan_all.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

NOT_FOUND = re.compile(r"Tool (\S+) not found")
BAD_ARGS = re.compile(r"bad arguments for (\S+)")
BACKTICK = re.compile(r"`([a-z_][a-z0-9_]*)`")

# A tool returning a raw OS error is a harness defect, not model behaviour:
# the message is platform-dependent and gives the model nothing to act on.
# Measured on this host before the fix: 27 of 448 rollouts leaked
# `[Errno 13] Permission denied: ...<workdir>` from a write to an empty path.
OS_LEAK = re.compile(r"\[Errno \d+\]|IsADirectoryError|Permission denied|NotADirectoryError")


def _advertised(cls) -> set[str]:
    """Tool names named in the harness's own GUIDANCE."""
    text = ""
    for line in cls.GUIDANCE.splitlines():
        if "Available tools" in line or "one tool" in line or "exactly one" in line:
            text += line + "\n"
    return set(BACKTICK.findall(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    args = ap.parse_args()

    from multiharness.harnesses import ALL_HARNESSES
    from multiharness.harnesses.core import discover_tools

    data = json.loads(Path(args.scan).read_text(encoding="utf-8"))
    records = data.get("records", [])
    if not records:
        print("no records in scan dump")
        return 1

    advertised = {h: _advertised(cls) for h, cls in ALL_HARNESSES.items()}
    registered = {h: {t.__name__ for t in discover_tools(cls())} for h, cls in ALL_HARNESSES.items()}

    harness_bug: Counter[tuple[str, str]] = Counter()   # advertised but missing
    hallucinated: Counter[tuple[str, str]] = Counter()  # not advertised at all
    badargs: Counter[tuple[str, str]] = Counter()
    os_leak: Counter[tuple[str, str]] = Counter()       # raw OS error surfaced
    other: Counter[tuple[str, str]] = Counter()
    by_harness: Counter[str] = Counter()
    total_err = 0

    for r in records:
        if not r.get("tool_errors"):
            continue
        total_err += r["tool_errors"]
        by_harness[r["harness"]] += r["tool_errors"]
        for line in r.get("raw", []):
            m = NOT_FOUND.search(line)
            if m:
                tool = m.group(1)
                if tool in advertised.get(r["harness"], set()):
                    harness_bug[(r["harness"], tool)] += 1
                else:
                    hallucinated[(r["harness"], tool)] += 1
                continue
            m = BAD_ARGS.search(line)
            if m:
                badargs[(r["harness"], m.group(1))] += 1
                continue
            # Checked before the generic bucket: an OS error is a defect that
            # would otherwise hide in "other" and be read as model behaviour.
            if OS_LEAK.search(line):
                tool = line.split("(")[0].split()[-1] if "(" in line else "?"
                os_leak[(r["harness"], tool)] += 1
                continue
            if "error" in line.lower():
                other[(r["harness"], line[-110:])] += 1

    print("=" * 78)
    print(f"TOOL ERROR REPORT — {args.scan}")
    print("=" * 78)
    print(f"rollouts: {len(records)}   tool_errors: {total_err}")
    print(f"by harness: {dict(by_harness)}")

    print("\n-- class 1: HARNESS BUG (tool advertised but not implemented) --")
    if harness_bug:
        for (h, tool), n in harness_bug.most_common():
            print(f"   {h:<18} {tool:<18} x{n}")
        print("   !! these invalidate the scan — fix the harness and rescan those columns")
    else:
        print("   (none) — no harness advertises a tool it does not implement")

    print("\n-- class 2: hallucinated tool (not advertised) = genuine signal --")
    if hallucinated:
        for (h, tool), n in hallucinated.most_common():
            print(f"   {h:<18} {tool:<18} x{n}"
                  f"   (this harness offers: {sorted(advertised.get(h, []))})")
        print("   do NOT 'fix' these: reaching for a tool the harness does not provide")
        print("   is exactly the cross-harness adaptation under test")
    else:
        print("   (none)")

    print("\n-- class 3: bad arguments = genuine signal --")
    if badargs:
        for (h, tool), n in badargs.most_common():
            print(f"   {h:<18} {tool:<18} x{n}")
    else:
        print("   (none)")

    print("\n-- class 3b: raw OS error leaked to the model = HARNESS DEFECT --")
    if os_leak:
        for (h, tool), n in os_leak.most_common():
            print(f"   {h:<18} {tool:<18} x{n}")
        print("   a platform-dependent OS error (Permission denied / IsADirectoryError)")
        print("   is not a function of the harness, so it cannot be scored as one;")
        print("   the tool must validate its path argument and return a usable message")
    else:
        print("   (none)")

    print("\n-- class 4: other errors --")
    if other:
        for (h, tail), n in other.most_common(12):
            print(f"   {h:<18} x{n}  ...{tail}")
    else:
        print("   (none)")

    # A tool that is registered but never advertised is a third defect: the
    # model cannot use what it was never told about.
    print("\n-- class 5: implemented but unadvertised --")
    unadvertised = []
    for h in advertised:
        extra = registered[h] - advertised[h] - {"bash"}
        if extra:
            unadvertised.append((h, sorted(extra)))
    if unadvertised:
        for h, extra in unadvertised:
            print(f"   {h:<18} {extra}")
        print("   the model will never call these — either advertise or delete them")
    else:
        print("   (none)")

    print()
    defects = bool(harness_bug or os_leak)
    if defects:
        which = []
        if harness_bug:
            which.append("class 1 (advertised-but-missing tools)")
        if os_leak:
            which.append("class 3b (raw OS errors)")
        print(f"VERDICT: scan is INVALID — {' and '.join(which)}")
        print("         fix and rescan the affected harness columns")
        return 2
    print("VERDICT: no harness defects — all recorded errors are model behaviour")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

