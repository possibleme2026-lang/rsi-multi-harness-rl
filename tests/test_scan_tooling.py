"""Regression tests for the two analysis scripts that guard the scan.

``errs_report.py`` and ``merge_scan.py`` are both *judgement* tools: one decides
whether a scan dump is valid enough to train on, the other decides whether a
partial rescan may be folded into an old one. Neither decision is visible in a
number, so neither fails loudly on its own — a wrong verdict just quietly
becomes the training input. These tests pin the verdicts.

Both scripts are exercised as subprocesses against synthetic dumps, not by
importing their internals. That is deliberate: the thing under test is the
exit code and the printed verdict, and a test that called a helper directly
would still pass if ``main()`` stopped consulting it.

Zero third-party dependencies, so this runs in the CI core job.

Run:
    ./run.sh tests/test_scan_tooling.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def run(script: str, *args: str) -> tuple[int, str]:
    """Run a repo script in a child process and return (exit code, output)."""
    proc = subprocess.run(
        [sys.executable, "-u", str(_REPO_ROOT / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def record(harness: str, errors: int, raw: list[str]) -> dict:
    """One rollout record in the shape errs_report.py consumes."""
    return {"harness": harness, "tool_errors": errors, "raw": raw}


def scan_dump(records: list[dict], **overrides) -> dict:
    """A minimal scan dump with the invariant keys merge_scan.py compares."""
    dump = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "n": 8,
        "max_turns": 4,
        "max_new_tokens": 192,
        "matrix": {
            "bash_minimal": {"t1-01": 0.5},
            "react_tools": {"t1-01": 0.25},
        },
        "records": records,
    }
    dump.update(overrides)
    return dump


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# errs_report.py — the scan-validity verdict
# ---------------------------------------------------------------------------

def test_errs_report(tmp: Path) -> None:
    print("\n-- errs_report.py: is the scan valid? --")

    # 1. A raw OS error reaching the model is a harness defect. The message is
    #    platform-dependent, so scoring it as harness behaviour would make the
    #    cross-harness comparison depend on the operating system.
    leaky = write(tmp / "leaky.json", scan_dump([
        record("react_tools", 1,
               ["write_file -> [Errno 13] Permission denied: 'C:\\\\ws\\\\out'"]),
    ]))
    rc, out = run("scripts/errs_report.py", str(leaky))
    check("raw OS leak -> exit 2 (scan INVALID)", rc == 2, f"rc={rc}")
    check("raw OS leak -> verdict names the defect", "INVALID" in out and "3b" in out)

    # 2. A tool the harness never advertised is the model reaching for the
    #    wrong scaffold — the adaptation under test. Must NOT be reported as a
    #    defect, or the guard would hide the very signal being measured.
    halluc = write(tmp / "halluc.json", scan_dump([
        record("longctx_summary", 1,
               ["write_to_file -> Tool write_to_file not found. Available: bash, read_file"]),
    ]))
    rc, out = run("scripts/errs_report.py", str(halluc))
    check("hallucinated tool -> exit 0 (scan valid)", rc == 0, f"rc={rc}")
    check("hallucinated tool -> classified as class 2, not a bug",
          "class 2" in out and "genuine signal" in out)
    check("hallucinated tool -> not counted as a harness bug",
          "class 1: HARNESS BUG" in out and "(none) — no harness advertises" in out)

    # 3. An advertised-but-missing tool is a harness bug and must be caught
    #    even when no OS error is present anywhere in the dump.
    advertised_missing = write(tmp / "advertised.json", scan_dump([
        record("react_tools", 1,
               ["finish -> Tool finish not found. Available: bash, read_file"]),
    ]))
    rc, out = run("scripts/errs_report.py", str(advertised_missing))
    check("advertised-but-missing tool -> exit 2", rc == 2, f"rc={rc}")
    check("advertised-but-missing tool -> blamed on the harness",
          "class 1" in out and "react_tools" in out)

    # 4. An empty dump is not a pass. Silence must not read as "no defects".
    empty = write(tmp / "empty.json", scan_dump([]))
    rc, out = run("scripts/errs_report.py", str(empty))
    check("empty dump -> non-zero exit", rc != 0, f"rc={rc}")


# ---------------------------------------------------------------------------
# merge_scan.py — may a partial rescan be folded into an old dump?
# ---------------------------------------------------------------------------

def test_merge_scan(tmp: Path) -> None:
    print("\n-- merge_scan.py: may a partial rescan be merged? --")

    base = write(tmp / "base.json", scan_dump([record("react_tools", 0, [])]))
    fresh = write(tmp / "fresh.json", scan_dump(
        [record("react_tools", 0, [])],
        matrix={"bash_minimal": {"t1-01": 0.5}, "react_tools": {"t1-01": 0.75}},
    ))

    # 1. Same setup -> the merge proceeds, and the requested column is taken
    #    from the fresh scan while the other is kept.
    out_path = tmp / "merged.json"
    rc, out = run("scripts/merge_scan.py",
                  "--base", str(base), "--fresh", str(fresh),
                  "--harnesses", "react_tools", "--out", str(out_path))
    check("matching setup -> merge succeeds", rc == 0, f"rc={rc}")
    merged = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    check("replaced column came from the fresh scan",
          merged.get("matrix", {}).get("react_tools", {}).get("t1-01") == 0.75)
    check("other column was kept from the base scan",
          merged.get("matrix", {}).get("bash_minimal", {}).get("t1-01") == 0.5)
    check("merge provenance is recorded", "merged_from" in merged)
    check("a merged dump is no longer marked partial", "partial" not in merged)

    # 2. A differing measurement setup must be refused. This is the whole
    #    point of the script: two scans taken with different n are not
    #    comparable, and a silent merge would compare them anyway.
    for key, bad in (("n", 4), ("max_turns", 8), ("max_new_tokens", 512), ("model", "other/model")):
        drifted = write(tmp / f"drift-{key}.json",
                        scan_dump([record("react_tools", 0, [])], **{key: bad}))
        rc, out = run("scripts/merge_scan.py",
                      "--base", str(base), "--fresh", str(drifted),
                      "--harnesses", "react_tools", "--out", str(tmp / f"out-{key}.json"))
        check(f"setup drift in {key} -> merge refused", rc != 0, f"rc={rc}")
        check(f"setup drift in {key} -> refusal names the field", key in out)

    # 3. Asking for a harness the fresh scan does not contain must fail, not
    #    silently keep the stale column.
    rc, out = run("scripts/merge_scan.py",
                  "--base", str(base), "--fresh", str(fresh),
                  "--harnesses", "json_strict", "--out", str(tmp / "out-missing.json"))
    check("unknown harness -> merge refused", rc != 0, f"rc={rc}")
    check("unknown harness -> refusal says so", "json_strict" in out)

    # 4. A task-set mismatch means the two scans measured different things.
    narrow = write(tmp / "narrow.json", scan_dump(
        [record("react_tools", 0, [])],
        matrix={"bash_minimal": {"t1-01": 0.5}, "react_tools": {"t1-02": 0.25}},
    ))
    rc, out = run("scripts/merge_scan.py",
                  "--base", str(base), "--fresh", str(narrow),
                  "--harnesses", "react_tools", "--out", str(tmp / "out-narrow.json"))
    check("task-set mismatch -> merge refused", rc != 0, f"rc={rc}")
    check("task-set mismatch -> refusal lists the differing task", "t1-02" in out)

    # 5. A refused merge must not have written its output.
    check("refused merge wrote no output file", not (tmp / "out-narrow.json").exists())


def main() -> int:
    print("=" * 74)
    print("SCAN TOOLING — analysis-script verdicts")
    print("=" * 74)
    with tempfile.TemporaryDirectory(prefix="mh-scan-tooling-") as d:
        tmp = Path(d)
        test_errs_report(tmp)
        test_merge_scan(tmp)

    print("\n" + "=" * 74)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
