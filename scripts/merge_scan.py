"""Merge a partial rescan into an existing scan dump, column by column.

Why this exists
---------------
A scan is ~7 minutes of GPU time across 4 harnesses. When a fix touches only
some harnesses, rescanning all four is pure waste — but naively reusing the old
file is worse, because then the matrix silently mixes results from two
different versions of the code.

This script makes the reuse *explicit and checked*. It takes the columns named
by ``--harnesses`` from the fresh scan and keeps everything else from the old
one, then asserts the two agree on everything that must be held constant:

  * same model, same ``n``, same max_turns / max_new_tokens
  * same task list per harness
  * the replaced harnesses are exactly the requested ones

If any assertion fails the merge is refused, so a stale or mismatched scan
cannot quietly become the training input.

Only harnesses whose *code changed* need rescanning. Verify that claim before
using this: for the workspace-path fix, ``BashMinimalEnv`` (bash only) and
``JsonStrictEnv`` (bash + submit) take no model-supplied path, so their
behaviour is unchanged and their columns are valid to carry over.

Usage:
    # rescan the two harnesses whose path tools changed
    ./run.sh scripts/probe.py \
        --harnesses react_tools,longctx_summary --n 8 --tasks <same> \
        --out outputs/scan_pathfix.json

    ./run.sh scripts/merge_scan.py \
        --base outputs/scan_all.json \
        --fresh outputs/scan_pathfix.json \
        --harnesses react_tools,longctx_summary \
        --out outputs/scan_all.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Keys that describe the measurement setup. They must match exactly between the
# two scans, otherwise the merged matrix would compare incomparable numbers.
INVARIANT_KEYS = ("model", "n", "max_turns", "max_new_tokens")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing scan dump to keep")
    ap.add_argument("--fresh", required=True, help="rescan containing the new columns")
    ap.add_argument("--harnesses", required=True,
                    help="comma-separated harness names to take from --fresh")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    base_path, fresh_path = Path(args.base), Path(args.fresh)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    replace = [h.strip() for h in args.harnesses.split(",") if h.strip()]

    print("=" * 74)
    print("MERGE SCAN — replace selected harness columns")
    print("=" * 74)
    print(f"base  : {base_path}")
    print(f"fresh : {fresh_path}")
    print(f"taking from fresh: {replace}")

    # -- invariants --------------------------------------------------------
    problems: list[str] = []
    for k in INVARIANT_KEYS:
        if base.get(k) != fresh.get(k):
            problems.append(f"{k}: base={base.get(k)!r} fresh={fresh.get(k)!r}")
    if problems:
        print("\n!! refusing to merge — measurement setup differs:")
        for p in problems:
            print(f"   {p}")
        return 1
    print(f"\nsetup matches on {', '.join(INVARIANT_KEYS)}")

    bm, fm = base["matrix"], fresh["matrix"]
    missing = [h for h in replace if h not in fm]
    if missing:
        print(f"\n!! refusing to merge — {missing} not present in the fresh scan")
        return 1
    missing_b = [h for h in replace if h not in bm]
    if missing_b:
        print(f"\n!! refusing to merge — {missing_b} not present in the base scan")
        return 1

    # -- per-harness task lists must line up ------------------------------
    for h in replace:
        if sorted(bm[h]) != sorted(fm[h]):
            print(f"\n!! refusing to merge — task set differs for {h}:")
            print(f"   only in base : {sorted(set(bm[h]) - set(fm[h]))}")
            print(f"   only in fresh: {sorted(set(fm[h]) - set(bm[h]))}")
            return 1
    print("task sets match for every replaced harness")

    # -- the merge itself --------------------------------------------------
    print("\ncolumn deltas (pass rate, base -> fresh):")
    for h in replace:
        deltas = []
        for t in sorted(fm[h]):
            d = fm[h][t] - bm[h][t]
            deltas.append((t, bm[h][t], fm[h][t], d))
        changed = [x for x in deltas if abs(x[3]) > 1e-9]
        mean_b = sum(x[1] for x in deltas) / len(deltas)
        mean_f = sum(x[2] for x in deltas) / len(deltas)
        print(f"  {h:<18} mean {mean_b:.3f} -> {mean_f:.3f}   "
              f"({len(changed)}/{len(deltas)} cells changed)")
        for t, b, f, d in changed:
            print(f"      {t:<7} {b:.2f} -> {f:.2f}  ({d:+.2f})")

    kept = [h for h in bm if h not in replace]
    print(f"\nkeeping from base : {kept}")

    merged = dict(base)
    merged["matrix"] = {h: (fm[h] if h in replace else bm[h]) for h in bm}

    # Records: drop the replaced harnesses' rows and append the fresh ones, so
    # the dump stays internally consistent for errs_report.py.
    base_records = [r for r in base.get("records", []) if r["harness"] not in replace]
    fresh_records = [r for r in fresh.get("records", []) if r["harness"] in replace]
    merged["records"] = base_records + fresh_records
    merged["merged_from"] = {
        "base": str(base_path),
        "fresh": str(fresh_path),
        "replaced_harnesses": replace,
        "kept_harnesses": kept,
    }
    # A merged file is complete by construction.
    merged.pop("partial", None)

    if not args.no_backup and base_path.exists():
        backup = base_path.with_suffix(".premerge.json")
        shutil.copy2(base_path, backup)
        print(f"backup of the previous dump: {backup}")

    Path(args.out).write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"  harnesses : {list(merged['matrix'])}")
    print(f"  records   : {len(merged['records'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
