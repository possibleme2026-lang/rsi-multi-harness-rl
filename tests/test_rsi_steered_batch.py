"""Regression tests for the last hop of the RSI loop: steer -> steered batch -> gradient.

The task axis had four components and every one of them worked:

* ``band.steer`` returns a parameter override for a task that is not teaching;
* ``curriculum.plan_regeneration`` turns a scan into a plan of moves;
* ``task_gen.reparameterise`` turns a move into a new, well-formed task;
* the four gates confirm the replacement is solvable.

And the loop was still open, because ``execute_plan``'s output was written to
``curriculum.json`` and **nothing read it back**. The training arms read
``rsi/batch.json``, the pre-steer batch. A curriculum could move every task in
the batch and not change a single gradient. This was the third instance of the
same defect in this repository — a mechanism that exists, passes its tests, and
is never reached by the thing it was built for — after the missing ``steer`` call
site and the trainer's hardcoded task list.

What is pinned here:

* ``curriculum.steered_batch`` preserves length, order and the kept tasks, and
  replaces exactly the moved ones;
* running the curriculum with ``--apply-curriculum`` writes
  ``rsi/batch_steered.json``, and that file is what the trainer would consume;
* a zero-move plan is diagnosed rather than reported as a bare zero, because
  "nothing needed moving" and "nothing could be measured" are different findings;
* ``band.steer`` moves the verifier in the easier direction, which is the single
  largest difficulty lever and was previously omitted;
* ``loop_report.py`` computes the before/after alignment, and returns non-zero
  only when the curriculum moved the batch the *wrong* way.

Zero third-party dependencies and no GPU: the generator is deterministic in
``--seed`` and the arithmetic is over synthetic scans. What cannot be checked
here is whether the steered tasks are *learnable* — that is a rollout
measurement and it belongs to ``probe.py``.

Run:
    ./run.sh tests/test_rsi_steered_batch.py
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

from multiharness.rsi import band as band_mod
from multiharness.rsi import curriculum as cu
from multiharness.rsi import task_gen

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def run(script: str, *args: str, timeout: int = 240, out_root: Path | None = None) -> tuple[int, str]:
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


def write_scan(path: Path, cells: dict[str, tuple[int, int]], harness: str = "bash_minimal") -> None:
    """A scan dump carrying (passes, n) per task id.

    Records rather than a matrix, because the curriculum aggregates the raw
    records — the matrix stores point estimates and cannot support a verdict.
    """
    records = []
    for tid, (passes, n) in cells.items():
        for i in range(n):
            records.append({"harness": harness, "task_id": tid, "reward": 1.0 if i < passes else 0.0})
    path.write_text(
        json.dumps({"n": max((n for _, n in cells.values()), default=0), "matrix": {}, "records": records}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# the pure function
# --------------------------------------------------------------------------


def test_steered_batch_invariants(tmp: Path) -> None:
    """Length, order and identity: the three ways a batch rewrite goes wrong."""
    print("\n-- steered_batch preserves the batch it is given")

    batch = task_gen.generate_batch(8, seed=11)
    by_id = {t["id"]: t for t in batch}
    # Task 0 out of reach (0/64), task 1 mastered (64/64), the rest unmeasured.
    cells = [
        {"harness": "bash_minimal", "task_id": batch[0]["id"], "passes": 0, "n": 64},
        {"harness": "bash_minimal", "task_id": batch[1]["id"], "passes": 64, "n": 64},
    ]
    plan = cu.plan_regeneration(cells, tasks_by_id=by_id, source="synthetic")
    check("the plan moves both measured tasks", plan.move_count == 2, f"moves={plan.move_count}")

    steered = cu.steered_batch(plan, batch)
    check("length is preserved", len(steered) == len(batch), f"{len(steered)} vs {len(batch)}")
    check("no duplicate ids", len({t["id"] for t in steered}) == len(steered))

    moved = [t for t in steered if "regenerated_from" in t]
    kept = [t for t in steered if "regenerated_from" not in t]
    check("exactly the moved tasks were replaced", len(moved) == 2, f"moved={len(moved)}")
    check("the rest were kept", len(kept) == len(batch) - 2, f"kept={len(kept)}")

    # Kept tasks must be the *same objects*, so a task on the frontier keeps its
    # id and therefore keeps its measurement attached.
    kept_ids_before = [t["id"] for t in batch if t["id"] not in {batch[0]["id"], batch[1]["id"]}]
    check("kept tasks keep their ids in order", [t["id"] for t in kept] == kept_ids_before)

    # Order: the steered batch must be index-aligned with the original, or a
    # downstream consumer that zips the two would pair the wrong rows. Compared
    # on the *kept* positions only — index 0 and 1 were both replaced, so their
    # ids are supposed to differ.
    check(
        "order is preserved (kept tasks stay in place)",
        [t["id"] for t in steered[2:]] == [t["id"] for t in batch[2:]],
    )

    # Provenance, and the new id must actually differ — a "move" that produces
    # the same task is a no-op that would report as a move.
    check(
        "moved tasks record what they came from",
        all(t["regenerated_from"] in {batch[0]["id"], batch[1]["id"]} for t in moved),
    )
    check("a move produces a different id", all(t["id"] != t["regenerated_from"] for t in moved))
    check(
        "moved tasks record the band that moved them",
        {t["regenerated_band"] for t in moved} == {"out_of_reach", "mastered"},
    )
    check(
        "moved tasks record the direction",
        {t["regenerated_direction"] for t in moved} == {"easier", "harder"},
    )

    # The easier direction must actually be easier on the knobs it moves.
    easy = next(t for t in moved if t["regenerated_direction"] == "easier")
    hard = next(t for t in moved if t["regenerated_direction"] == "harder")
    check(
        "the easier move shortens the payload",
        easy["params"]["payload_len"] <= batch[0]["params"]["payload_len"],
    )
    check(
        "the easier move lowers escape density",
        easy["params"]["escape_density"] <= batch[0]["params"]["escape_density"],
    )
    check(
        "the harder move lengthens the payload",
        hard["params"]["payload_len"] >= batch[1]["params"]["payload_len"],
    )

    # A plan with no moves must return the batch unchanged, not an empty list.
    empty_plan = cu.plan_regeneration([], tasks_by_id=by_id, source="synthetic")
    same = cu.steered_batch(empty_plan, batch)
    check("a zero-move plan returns the batch unchanged", [t["id"] for t in same] == [t["id"] for t in batch])


def test_steer_moves_the_verifier(tmp: Path) -> None:
    """`verify_mode` is the largest single difficulty lever and must be steered.

    It was omitted from the override, which meant the "make this easier" rule
    left the biggest knob untouched: `python_exit` asks the agent to write a
    script and carries a full 1.0 of the five parts in ``TaskParams.difficulty``.
    """
    print("\n-- band.steer covers the verifier in the easier direction")

    easy = band_mod.steer(band_mod.Band.OUT_OF_REACH, {"verify_mode": "python_exit"})
    check("out_of_reach sets verify_mode", easy.get("verify_mode") == "file_equals", str(easy))
    check(
        "out_of_reach still moves the original knobs",
        {"payload_len", "escape_density", "steps", "read_source"} <= set(easy),
    )

    hard = band_mod.steer(band_mod.Band.MASTERED, {"verify_mode": "file_equals"})
    check(
        "mastered does not touch verify_mode",
        "verify_mode" not in hard,
        "raising it would change the reward axis, not just difficulty",
    )

    # frontier and unresolved hold: an empty override, so nothing moves.
    check("frontier holds", band_mod.steer(band_mod.Band.FRONTIER, {}) == {})
    check("unresolved holds", band_mod.steer(band_mod.Band.UNRESOLVED, {}) == {})


# --------------------------------------------------------------------------
# the artifact, end to end through the script
# --------------------------------------------------------------------------


def test_apply_writes_the_steered_batch(tmp: Path) -> None:
    """`--apply-curriculum` must leave a batch the trainer can read."""
    print("\n-- --apply-curriculum writes rsi/batch_steered.json")

    out = tmp / "apply"
    out.mkdir()
    code, log = run(
        "scripts/rsi_loop.py", "--batch-only", "--task-batch", "8", "--seed", "11", out_root=out
    )
    check("batch-only exits 0", code == 0, log[-200:])

    batch_path = out / "rsi" / "batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))

    # Two cells that resolve: one hopeless, one mastered. 64 rollouts is what
    # the band definition needs to place a zero-pass cell out of reach.
    scan_path = out / "rsi" / "scan_synthetic.json"
    write_scan(scan_path, {batch[0]["id"]: (0, 64), batch[1]["id"]: (64, 64)})

    code, log = run(
        "scripts/rsi_loop.py",
        "--task-batch", "8", "--seed", "11",
        "--scan", str(scan_path),
        "--apply-curriculum",
        "--rounds", "1",
        out_root=out,
    )
    check("the curriculum run exits 0", code == 0, log[-300:])

    steered_path = out / "rsi" / "batch_steered.json"
    check("batch_steered.json was written", steered_path.is_file())

    if steered_path.is_file():
        steered = json.loads(steered_path.read_text(encoding="utf-8"))
        check("the steered batch has the same size", len(steered) == len(batch), f"{len(steered)} vs {len(batch)}")
        moved = [t for t in steered if "regenerated_from" in t]
        check("something was moved", len(moved) > 0, f"moved={len(moved)}")
        check("something was kept", len(moved) < len(steered), f"kept={len(steered) - len(moved)}")
        check(
            "the trainer could resolve every id",
            all(isinstance(t.get("id"), str) and t["id"] for t in steered),
        )

    cur = json.loads((out / "rsi" / "curriculum.json").read_text(encoding="utf-8"))
    check("the curriculum reports itself applied", cur.get("applied") is True)
    check("the steered batch is recorded in the artifact", "steered" in cur)
    check(
        "the artifact counts match the file",
        cur.get("steered", {}).get("total") == len(batch),
        str(cur.get("steered")),
    )
    check(
        "provenance names every move",
        len(cur.get("provenance", [])) == cur.get("steered", {}).get("moved", -1),
    )
    check(
        "alignment_after is declared unmeasured rather than guessed",
        "unmeasured" in str(cur.get("alignment_after", "")),
    )
    check(
        "alignment_before is a number",
        isinstance(cur.get("alignment_before"), int | float),
        str(cur.get("alignment_before")),
    )

    # The trainer must be able to consume it. --dry-run stops before the model
    # loads, so this is the wiring check without the GPU.
    code, log = run(
        "scripts/train.py", "--mode", "multi", "--steps", "1",
        "--batch", str(steered_path), "--scan", str(scan_path),
        "--require-signal", "--dry-run",
        out_root=out,
    )
    # The steered ids are new, so the pre-steer scan does not cover them: this
    # must be *refused*, which is the guard working.
    check(
        "an unscanned steered batch is refused rather than trained on unmeasured rows",
        code != 0 and "none of which are among" in log,
        f"exit={code}",
    )

    # And with a scan of the steered batch itself, it passes.
    steered_scan = out / "rsi" / "scan_steered_synthetic.json"
    steered_ids = [t["id"] for t in json.loads(steered_path.read_text(encoding="utf-8"))]
    write_scan(steered_scan, {tid: (32, 64) for tid in steered_ids})
    code, log = run(
        "scripts/train.py", "--mode", "multi", "--steps", "1",
        "--batch", str(steered_path), "--scan", str(steered_scan),
        "--require-signal", "--dry-run",
        out_root=out,
    )
    check("a scanned steered batch trains", code == 0, log[-300:])
    check("the trainer names the steered batch as its source", "batch_steered.json" in log)


def test_zero_moves_is_diagnosed(tmp: Path) -> None:
    """Three causes of zero moves, three different responses.

    Only the id-mismatch case used to be visible at the top level. The other two
    surfaced as `move_count: 0`, which reads like "the batch is on target" — the
    one interpretation that requires doing nothing, and the wrong one when the
    real cause is that the scan was too small to place any cell.
    """
    print("\n-- a zero-move plan says which kind of zero it is")

    out = tmp / "diagnosis"
    out.mkdir()
    code, log = run(
        "scripts/rsi_loop.py", "--batch-only", "--task-batch", "6", "--seed", "11", out_root=out
    )
    check("batch-only exits 0", code == 0, log[-200:])
    batch = json.loads((out / "rsi" / "batch.json").read_text(encoding="utf-8"))

    # (a) ids mismatch -- a scan of a different task set.
    mismatch_scan = out / "rsi" / "scan_mismatch.json"
    write_scan(mismatch_scan, {"t1-00000000": (0, 64)})
    run("scripts/rsi_loop.py", "--task-batch", "6", "--seed", "11",
        "--scan", str(mismatch_scan), "--rounds", "1", out_root=out)
    cur = json.loads((out / "rsi" / "curriculum.json").read_text(encoding="utf-8"))
    check(
        "an id mismatch is diagnosed as such",
        cur.get("move_diagnosis", "").startswith("ids_mismatch"),
        str(cur.get("move_diagnosis")),
    )
    check("and it is recorded as skipped", "skipped" in cur)

    # (b) all unresolved -- a scan too small to place any cell.
    tiny_scan = out / "rsi" / "scan_tiny.json"
    write_scan(tiny_scan, {batch[0]["id"]: (0, 8), batch[1]["id"]: (4, 8)})
    run("scripts/rsi_loop.py", "--task-batch", "6", "--seed", "11",
        "--scan", str(tiny_scan), "--rounds", "1", out_root=out)
    cur = json.loads((out / "rsi" / "curriculum.json").read_text(encoding="utf-8"))
    check(
        "an under-powered scan is diagnosed as unresolved, not as 'on target'",
        cur.get("move_diagnosis", "").startswith("all_unresolved"),
        str(cur.get("move_diagnosis")),
    )
    check(
        "the diagnosis names the remedy",
        "scan more" in cur.get("move_diagnosis", ""),
        str(cur.get("move_diagnosis")),
    )
    check(
        "an unresolved zero is NOT recorded as skipped",
        "skipped" not in cur,
        "the scan did measure these tasks; they were just not placed",
    )

    # (c) all frontier -- the batch is where training should happen.
    frontier_scan = out / "rsi" / "scan_frontier.json"
    write_scan(frontier_scan, {t["id"]: (32, 64) for t in batch})
    run("scripts/rsi_loop.py", "--task-batch", "6", "--seed", "11",
        "--scan", str(frontier_scan), "--rounds", "1", out_root=out)
    cur = json.loads((out / "rsi" / "curriculum.json").read_text(encoding="utf-8"))
    check(
        "an on-target batch is diagnosed as frontier",
        cur.get("move_diagnosis", "").startswith("all_frontier"),
        str(cur.get("move_diagnosis")),
    )
    check("a frontier batch reports zero moves", cur["plan"]["move_count"] == 0)

    # A non-zero-move plan must NOT carry a diagnosis: the field means "why was
    # this zero", and a diagnosis on a non-zero plan would be a contradiction.
    real_scan = out / "rsi" / "scan_real.json"
    write_scan(real_scan, {batch[0]["id"]: (0, 64)})
    run("scripts/rsi_loop.py", "--task-batch", "6", "--seed", "11",
        "--scan", str(real_scan), "--rounds", "1", out_root=out)
    cur = json.loads((out / "rsi" / "curriculum.json").read_text(encoding="utf-8"))
    check("a moving plan has no zero-diagnosis", "move_diagnosis" not in cur)
    check("a moving plan reports moves", cur["plan"]["move_count"] > 0)


def test_loop_report(tmp: Path) -> None:
    """The before/after alignment, and the one outcome that exits non-zero."""
    print("\n-- loop_report.py computes the alignment delta")

    out = tmp / "report"
    out.mkdir()
    code, log = run(
        "scripts/rsi_loop.py", "--batch-only", "--task-batch", "8", "--seed", "11", out_root=out
    )
    check("batch-only exits 0", code == 0, log[-200:])
    batch = json.loads((out / "rsi" / "batch.json").read_text(encoding="utf-8"))
    ids = [t["id"] for t in batch]

    before_scan = out / "rsi" / "before.json"
    after_scan = out / "rsi" / "after.json"
    # Before: every task hopeless. After: every task on the frontier. The
    # alignment must rise, and it must rise because of the band change.
    write_scan(before_scan, {tid: (0, 64) for tid in ids})
    write_scan(after_scan, {tid: (32, 64) for tid in ids})

    # An "after" batch that records the moves, so the report can count them.
    steered = [dict(t, regenerated_from=t["id"], regenerated_band="out_of_reach",
                    regenerated_direction="easier") for t in batch]
    steered_path = out / "rsi" / "batch_steered.json"
    steered_path.write_text(json.dumps(steered, indent=2), encoding="utf-8")

    out_json = out / "rsi" / "loop_closed.json"
    code, log = run(
        "scripts/loop_report.py",
        "--before-batch", str(out / "rsi" / "batch.json"),
        "--before-scan", str(before_scan),
        "--after-batch", str(steered_path),
        "--after-scan", str(after_scan),
        "--out", str(out_json),
    )
    check("loop_report exits 0 on a positive delta", code == 0, log[-400:])
    check("it wrote its artifact", out_json.is_file())

    if out_json.is_file():
        rep = json.loads(out_json.read_text(encoding="utf-8"))
        check(
            "the delta is positive",
            rep["alignment_delta"] > 0,
            f"{rep['before']['alignment']} -> {rep['after']['alignment']}",
        )
        check("it counts the moved tasks", rep["moved"] == len(batch), str(rep["moved"]))
        check("it counts the kept tasks", rep["kept"] == 0, str(rep["kept"]))
        check("it records the direction", rep.get("directions") == {"easier": len(batch)})
        check(
            "a real move makes the delta evidence",
            rep["delta_is_evidence"] is True,
            "a delta between two different scans is a measurement",
        )
        check(
            "the interpretation says this is about alignment, not learning",
            "alignment" in rep["interpretation"] and "learning" in rep["interpretation"],
        )

    # The delta must be arithmetic on the two scans, not a decoration: swapping
    # the two scans has to flip its sign. Both batches carry the move markers, so
    # the report sees a real move and reports it as going the wrong way.
    reversed_batch = out / "rsi" / "batch_reversed.json"
    reversed_batch.write_text(
        json.dumps([dict(t, regenerated_from=t["id"], regenerated_band="mastered",
                         regenerated_direction="harder") for t in batch], indent=2),
        encoding="utf-8",
    )
    code, log = run(
        "scripts/loop_report.py",
        "--before-batch", str(reversed_batch),
        "--before-scan", str(after_scan),
        "--after-batch", str(reversed_batch),
        "--after-scan", str(before_scan),
        "--out", str(out / "rsi" / "loop_reversed.json"),
    )
    rep = json.loads((out / "rsi" / "loop_reversed.json").read_text(encoding="utf-8"))
    check("swapping before and after flips the sign", rep["alignment_delta"] < 0)
    check(
        "a negative delta exits non-zero",
        code != 0,
        "moving the batch away from the learnable band is the one outcome worth gating on",
    )
    check(
        "a negative delta is named as a defect",
        "away from the learnable band" in rep["interpretation"],
        rep["interpretation"][:120],
    )

    # The defect the first real run exposed: the pooled delta includes the kept
    # tasks, which are a control group (the same task measured twice). Pooling
    # them lets sampling noise decide the sign of the headline number, and the
    # script then reports noise as a steering defect. Here the moved tasks all
    # improve and the untouched ones all drift down, so the pooled delta is
    # negative while the treatment is positive.
    mixed = out / "rsi"
    # Rebuild with the first half moved and the second kept: the moved tasks
    # rise, the untouched ones drift down, so the pooled delta is negative while
    # the treatment is positive.
    half = len(batch) // 2
    half_batch = [dict(t, regenerated_from=t["id"], regenerated_band="out_of_reach",
                       regenerated_direction="easier") for t in batch[:half]] + \
                 [dict(t) for t in batch[half:]]
    (mixed / "batch_half.json").write_text(json.dumps(half_batch, indent=2), encoding="utf-8")
    write_scan(mixed / "scan_half_before.json",
               {t["id"]: (16, 64) for t in half_batch[:half]}
               | {t["id"]: (32, 64) for t in half_batch[half:]})
    after_half = {t["id"]: (32, 64) for t in half_batch[:half]}
    after_half.update({t["id"]: (0, 64) for t in half_batch[half:]})
    write_scan(mixed / "scan_half_after.json", after_half)
    code, log = run(
        "scripts/loop_report.py",
        "--before-batch", str(mixed / "batch.json"),
        "--before-scan", str(mixed / "scan_half_before.json"),
        "--after-batch", str(mixed / "batch_half.json"),
        "--after-scan", str(mixed / "scan_half_after.json"),
        "--out", str(mixed / "loop_half.json"),
    )
    rep = json.loads((mixed / "loop_half.json").read_text(encoding="utf-8"))
    check(
        "the treatment and the control are reported separately",
        "alignment_delta_moved" in rep and "alignment_delta_kept" in rep,
        "the kept tasks are the noise floor, not a result",
    )
    check(
        "the treatment delta is positive when the moved tasks improve",
        rep["alignment_delta_moved"] > 0,
        str(rep["alignment_delta_moved"]),
    )
    check(
        "the control delta is negative when the untouched tasks drift down",
        rep["alignment_delta_kept"] < 0,
        str(rep["alignment_delta_kept"]),
    )
    check(
        "the pooled delta is not used as the verdict",
        rep["interpretation"].startswith("positive on the treatment")
        or "positive on the treatment" in rep["interpretation"],
        rep["interpretation"][:120],
    )
    check(
        "the sign disagreement is called out rather than hidden",
        "disagrees in sign" in rep["interpretation"],
        rep["interpretation"][-200:],
    )
    check(
        "a non-negative treatment does not fail the run",
        code == 0,
        "exiting on the pooled delta would fail the pipeline over noise",
    )

    # A zero-move report must not claim a curriculum effect.
    no_moves = out / "rsi" / "batch_unchanged.json"
    no_moves.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    code, log = run(
        "scripts/loop_report.py",
        "--before-batch", str(out / "rsi" / "batch.json"),
        "--before-scan", str(before_scan),
        "--after-batch", str(no_moves),
        "--after-scan", str(before_scan),
        "--out", str(out / "rsi" / "loop_nomove.json"),
    )
    rep = json.loads((out / "rsi" / "loop_nomove.json").read_text(encoding="utf-8"))
    check("a zero-move report reports zero moved", rep["moved"] == 0)
    check("a zero-move report exits 0", code == 0)
    check(
        "a zero-move report blames the scan, not the generator",
        "scan" in rep["interpretation"] and "generator" in rep["interpretation"],
        rep["interpretation"][:140],
    )
    # The zero here is arithmetic, not measurement: with no moves the "after"
    # batch is the "before" batch, so the delta is 0 whatever the batch is like.
    # The payload has to say so, or a figure or a README sentence will pick the
    # number up as if the batch had been measured to be well-placed.
    check(
        "a zero-move delta is flagged as not evidence",
        rep["delta_is_evidence"] is False,
        "a delta computed against the same scan twice is a tautology, not a finding",
    )
    check(
        "a zero-move delta is described as zero by construction",
        "by construction" in rep["interpretation"],
        rep["interpretation"][:160],
    )


def main() -> int:
    print("=" * 74)
    print("RSI STEERED BATCH — the curriculum's output reaches the gradient")
    print("=" * 74)
    with tempfile.TemporaryDirectory(prefix="mh-steered-") as d:
        tmp = Path(d)
        test_steered_batch_invariants(tmp)
        test_steer_moves_the_verifier(tmp)
        test_apply_writes_the_steered_batch(tmp)
        test_zero_moves_is_diagnosed(tmp)
        test_loop_report(tmp)

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
