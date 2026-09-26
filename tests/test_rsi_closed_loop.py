"""Regression tests for the closed loop: generated batch -> scan -> steer -> train.

Every piece of this loop existed and was tested in isolation, and the loop was
still open, because the failure was in the *ordering* rather than in any one
component:

* `rsi_loop.py` generated a batch and wrote it to ``rsi/batch.json``;
* `pipeline.sh` handed the curriculum ``scan_all.json``, which measures the
  shipped 16 ids;
* the batch shares none of them, so every run printed ``move_count: 0`` with
  the reason "the scan measures 16 task ids, none of which are in the batch";
* `train.py` built its rows from a hardcoded ``TRAIN_TASK_IDS``, so no generated
  task could reach the gradient even if a scan had covered it.

Nothing raised. The curriculum was correct, the scan was correct, the trainer
was correct, and the composition did nothing. These tests pin the three
properties that make the loop closed, so a future edit that reopens it fails
here instead of in a training curve.

They run without a model or a GPU: the batch generator is deterministic in
``--seed`` and registration is pure bookkeeping, so the loop's *wiring* is
fully checkable offline. What cannot be checked here is whether the generated
tasks are learnable — that is a rollout measurement, and it belongs to
``probe.py``.

Zero third-party dependencies, so this runs in the CI core job.

Run:
    ./run.sh tests/test_rsi_closed_loop.py
"""

from __future__ import annotations

import json
import os
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


def run(
    script: str,
    *args: str,
    timeout: int = 180,
    out_root: Path | None = None,
) -> tuple[int, str]:
    """Run a repo script in a child process and return (exit code, output).

    ``out_root`` sets ``MULTIHARNESS_OUT``, which is how the scripts decide where
    artifacts go. Without it a test would write into the repo's real ``outputs/``
    and clobber the artifacts of an actual run.
    """
    env = dict(os.environ)
    if out_root is not None:
        env["MULTIHARNESS_OUT"] = str(out_root)
    proc = subprocess.run(
        [sys.executable, "-u", str(_REPO_ROOT / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO_ROOT),
        timeout=timeout,
        env=env,
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def write_scan(path: Path, task_ids: list[str], harnesses: list[str], passes: int, n: int) -> None:
    """A scan dump with the fields the loop reads: records carry (task_id, reward)."""
    records = []
    for h in harnesses:
        for tid in task_ids:
            for i in range(n):
                records.append(
                    {"harness": h, "task_id": tid, "reward": 1.0 if i < passes else 0.0}
                )
    path.write_text(
        json.dumps({"n": n, "matrix": {}, "records": records, "gates": {}}), encoding="utf-8"
    )


def test_batch_only_is_deterministic(tmp: Path) -> None:
    """`--batch-only` must write exactly the batch the full run would generate.

    The pipeline scans the batch in a *separate process* from the one that later
    steers it. If generation were not reproducible from ``--seed``, the scan
    would measure one batch and the curriculum would steer a different one, and
    the id-overlap check would pass while the loop stayed subtly open.
    """
    print("\n-- --batch-only writes a reproducible batch")

    # `rsi_loop.py` writes to `$MULTIHARNESS_OUT/rsi/`, so the temp root is the
    # env var and the `rsi/` level comes from the script.
    out = tmp
    rsi = out / "rsi"

    code, log = run(
        "scripts/rsi_loop.py",
        "--batch-only",
        "--task-batch", "6",
        "--env-batch", "3",
        "--seed", "11",
        out_root=out,
    )
    check("batch-only exits 0", code == 0, f"exit {code}")
    check("batch-only names the scan command it needs", "probe.py --from-batch" in log)

    batch_a = json.loads((rsi / "batch.json").read_text(encoding="utf-8"))
    env_a = json.loads((rsi / "env_batch.json").read_text(encoding="utf-8"))
    check("string batch written", len(batch_a) == 6, f"{len(batch_a)} tasks")
    check("env batch written", len(env_a) == 3, f"{len(env_a)} tasks")

    # Same seed, fresh process, overwrite the artifacts.
    code, _ = run(
        "scripts/rsi_loop.py",
        "--batch-only",
        "--task-batch", "6",
        "--env-batch", "3",
        "--seed", "11",
        out_root=out,
    )
    batch_b = json.loads((rsi / "batch.json").read_text(encoding="utf-8"))
    check("regeneration is byte-identical", batch_a == batch_b)

    ids = [t["id"] for t in batch_a]
    check("ids are unique", len(set(ids)) == len(ids))
    check(
        "generated ids are disjoint from the shipped suite",
        not (set(ids) & _shipped_ids()),
        "the shipped suite must not leak into a generated batch",
    )


def _shipped_ids() -> set[str]:
    from multiharness.tasks.suite import EVAL_TASK_IDS, TRAIN_TASK_IDS

    return set(TRAIN_TASK_IDS) | set(EVAL_TASK_IDS)


def test_train_refuses_unaligned_scan(tmp: Path) -> None:
    """`--require-signal` must refuse a scan that measures none of the rows.

    This is the exact state the pipeline used to land in by default. Without the
    guard the run *succeeds*: the filter finds no pass rate for any row, keeps
    everything as "unmeasured", and the only symptom is a training curve that
    looks like a capability limit.
    """
    print("\n-- --require-signal refuses a scan of a different task set")

    batch = tmp / "batch.json"
    batch.write_text(
        json.dumps([{"id": "gen-0001", "instruction": "x", "verify": "file_equals",
                     "expected": "x", "reference": ["true"]}]),
        encoding="utf-8",
    )
    # A scan of ids the batch does not contain -- the old default behaviour.
    scan = tmp / "scan_all.json"
    write_scan(scan, ["t1-01", "t1-02"], ["bash_minimal"], passes=1, n=4)

    code, log = run(
        "scripts/train.py",
        "--mode", "multi", "--steps", "1",
        "--batch", str(batch),
        "--scan", str(scan),
        "--require-signal",
    )
    check("refuses and exits non-zero", code != 0, f"exit {code}")
    check("says why", "none of which are among" in log)
    check("names the fix", "--from-batch" in log)

    # And the guard must not fire when the scan does cover the batch.
    aligned = tmp / "scan_batch.json"
    write_scan(aligned, ["gen-0001"], ["bash_minimal"], passes=2, n=4)
    code, log = run(
        "scripts/train.py",
        "--mode", "multi", "--steps", "1",
        "--batch", str(batch),
        "--scan", str(aligned),
        "--require-signal",
        "--dry-run",
    )
    check("aligned scan passes the guard", "scan overlap" in log, "guard did not fire")
    check("dry run exits 0 without a GPU", code == 0, f"exit {code}\n{log[-300:]}")
    check("dry run names the generated task", "gen-0001" in log)
    check(
        "dry run reports the batch as the source, not the shipped suite",
        "source             : " in log and "shipped suite" not in log,
    )


def test_train_registers_batch_tasks(tmp: Path) -> None:
    """Rows naming generated ids must resolve through the task registry.

    `Agent.run` looks task ids up in the process-global registry, so a row whose
    id was never registered raises ``KeyError`` *after* the model has loaded and
    the GPU is busy. Registration is therefore load-bearing, not a formality.
    """
    print("\n-- a generated batch registers and resolves")

    batch = tmp / "batch.json"
    batch.write_text(
        json.dumps(
            [
                {"id": "gen-aaaa", "instruction": "x", "verify": "file_equals",
                 "expected": "x", "reference": ["true"]},
                {"id": "gen-bbbb", "instruction": "y", "verify": "file_equals",
                 "expected": "y", "reference": ["true"]},
            ]
        ),
        encoding="utf-8",
    )

    # Load train.py as a module in-process. It is importable without torch,
    # datasets or trl because the heavy stack is imported inside `main()` past
    # the dry-run exit -- which is the property that lets this test live in the
    # dependency-free CI core job at all.
    import importlib.util

    spec = importlib.util.spec_from_file_location("_train", _REPO_ROOT / "scripts" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    from multiharness.harnesses.core import TASKS, get_task
    from multiharness.tasks import load as load_tasks

    load_tasks()
    before = len(TASKS)
    ids = train.load_generated_batch(batch)

    check("returns every batch id", ids == ["gen-aaaa", "gen-bbbb"], str(ids))
    check("registry grew by the batch size", len(TASKS) - before == 2)
    resolved = all(get_task(i)["id"] == i for i in ids)
    check("every id resolves through get_task", resolved)

    # build_dataset must not need the training stack either.
    rows = train.build_dataset(["bash_minimal", "react_tools"], ids, num_generations=8,
                               per_step_unique=1)
    check("row builder returns plain dicts", isinstance(rows[0], dict) and len(rows) == 4,
          f"{len(rows)} rows")
    check(
        "rows name the generated ids, not the shipped ones",
        {r["task_id"] for r in rows} == set(ids),
    )

    # A malformed batch must fail at load, not at rollout time.
    bad = tmp / "bad.json"
    bad.write_text(json.dumps([{"instruction": "no id here"}]), encoding="utf-8")
    try:
        train.load_generated_batch(bad)
        check("a batch entry without an id is rejected", False, "no exception")
    except ValueError:
        check("a batch entry without an id is rejected", True)

    empty = tmp / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    try:
        train.load_generated_batch(empty)
        check("an empty batch is rejected", False, "no exception")
    except ValueError:
        check("an empty batch is rejected", True)


def test_pipeline_scans_the_batch(tmp: Path) -> None:
    """The pipeline must scan the batch it steers, and fail if the two disagree.

    Asserted against the script text rather than by running it: a full pipeline
    run is tens of minutes of GPU time. The properties pinned are the ones whose
    absence reopened the loop — the batch is generated before it is scanned, the
    scan is of the batch, and the curriculum and the trainer read that same
    scan.

    The second half pins the *last* hop: the steered batch is scanned and becomes
    the training input. Without it the curriculum moved a task, wrote the result
    to a file, and the training arms went on reading the pre-steer batch — the
    loop was wired and never closed.
    """
    print("\n-- pipeline.sh orders generation, scan, steering correctly")

    src = (_REPO_ROOT / "pipeline.sh").read_text(encoding="utf-8")

    i_gen = src.find("--batch-only")
    i_scan = src.find("probe.py --from-batch")
    i_steer = src.find("scripts/rsi_loop.py \"${CURRICULUM_ARGS[@]}\"")
    check("batch is generated", i_gen > 0)
    check("the batch is scanned", i_scan > 0)
    check("generation precedes the scan", 0 < i_gen < i_scan)
    check("the scan precedes steering", 0 < i_scan < i_steer)

    check(
        "curriculum reads the batch scan, not scan_all.json",
        'CURRICULUM_SCAN="$OUT/rsi/scan_batch.json"' in src,
    )
    check(
        "training reads the scan of whatever batch it trains on",
        '"${TRAIN_SCAN_ARGS[@]}"' in src and 'TRAIN_SCAN_ARGS=(--scan "$TRAIN_SCAN_PATH")' in src,
    )
    check(
        "training on the generated batch is the default",
        'TRAIN_ON_BATCH:-1' in src,
        "the shipped suite must be the opt-out, not the default",
    )
    check(
        "an id mismatch is fatal rather than a silent zero-move plan",
        "measures none of the" in src,
    )

    # -- the last hop: the steered batch reaches the trainer ------------------
    i_steered_scan = src.find("--from-batch \"$OUT/rsi/batch_steered.json\"")
    i_train = src.find("scripts/train.py")
    check("the steered batch is scanned", i_steered_scan > 0)
    check("the steered scan precedes training", 0 < i_steered_scan < i_train)
    check(
        "the steered scan is ordered after the steer that produces it",
        0 < i_steer < i_steered_scan,
    )
    check(
        "training can read the steered batch",
        'TRAIN_BATCH_PATH="$OUT/rsi/batch_steered.json"' in src,
        "the curriculum's output has to be reachable by the trainer",
    )
    check(
        "the rescan decision is read from the artifact, not inferred from a log",
        'cur.get("steered", {}).get("moved", 0)' in src,
    )
    check(
        "applying the curriculum is the default",
        'APPLY_CURRICULUM:-1' in src,
        "a curriculum that is planned but never applied cannot move a gradient",
    )
    check(
        "the before/after alignment is reported",
        "scripts/loop_report.py" in src,
    )
    check(
        "a steered-batch scan that misses its ids is fatal",
        "measures none of the" in src and "steered ids" in src,
    )
    check(
        "N_SCAN defaults above the band-resolution floor",
        'N_SCAN="${N_SCAN:-64}"' in src,
        "below 35 a zero-pass cell is unresolved and the curriculum goes inert",
    )
    check(
        "lowering N_SCAN below the floor warns",
        "N_SCAN_MIN_FOR_BANDS" in src and "unresolved" in src,
    )


def main() -> int:
    print("=" * 74)
    print("RSI CLOSED LOOP — generated batch reaches the gradient")
    print("=" * 74)
    with tempfile.TemporaryDirectory(prefix="mh-closed-loop-") as d:
        tmp = Path(d)
        test_batch_only_is_deterministic(tmp)
        test_train_refuses_unaligned_scan(tmp)
        test_train_registers_batch_tasks(tmp)
        test_pipeline_scans_the_batch(tmp)

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
