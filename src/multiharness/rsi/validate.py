# Copyright 2026 The rsi-multi-harness-rl Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Four gates a generated task must clear before the policy may train on it.

A generator that is allowed to emit unvalidated tasks makes the whole
experiment unfalsifiable: a task no agent can solve is indistinguishable from
a task the verifier always fails, and a task every agent solves teaches
nothing. SPADE's Environment Designer solves this with two probes — a
reference agent that must score 1, and a no-op agent that must score less —
and those are gates 1 and 2 here. Gates 3 and 4 are additions this repository
needs because it has *five harnesses* where SPADE has one.

    V1  oracle      the generated reference solution scores 1.0
    V2  nop         doing nothing scores < 1.0
    V3  safety      the task declares a known verifier, and every path it
                    mentions resolves inside the work directory
    V4  cross       the task is solvable on *every* trainable harness, and the
                    harnesses do not all score identically

Why V4 is two conditions
------------------------
This is the gate that only exists because the experiment varies the harness as
well as the task, and it is worth being explicit about why it is conjunctive.

*Solvable everywhere* — if a task is solvable on ``bash_minimal`` but not on
``json_strict``, then a low score on that cell measures the harness's
completeness, not the model's skill. The cross-harness gap would then be a
statement about which harnesses are finished, which is a different (and much
less interesting) claim than the one the README makes.

*Not identical everywhere* — if every harness scores exactly the same on a
task, the task carries no information about the harness axis. It is not
*wrong*, but it is not doing the job it was generated for, and a suite full of
such tasks would report a generalization gap of zero for reasons that have
nothing to do with generalization.

Note the asymmetry in how the two are checked. "Solvable everywhere" is a hard
gate — one failing harness rejects the task. "Not identical" is a *variance*
requirement, and variance needs samples, so it can only be checked once the
task has been rolled out; a task that has not been rolled out yet passes V4
provisionally and is flagged ``variance_unmeasured``. Silently treating
"unmeasured" as "identical" would reject most of a fresh batch.

Both gates 1 and 2 need a real shell, so this module runs anywhere with
``bash``; it needs no model and no GPU.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..harnesses.core import ANSWER_NAME, _run_shell_rc, verify

__all__ = [
    "GateResult",
    "TaskVerdict",
    "check_oracle",
    "check_nop",
    "check_safety",
    "check_cross_harness",
    "validate_task",
    "validate_batch",
    "summarise_validation",
    "KNOWN_VERIFY_MODES",
    "TRAINABLE_HARNESSES",
]

#: Every mode ``core.verify`` implements. A task naming anything else would
#: raise at reward time, inside a rollout, which is the worst place to find out.
KNOWN_VERIFY_MODES = frozenset({"file_equals", "file_contains", "python_exit"})

#: Harnesses a generated task must be solvable on. Excludes ``oracle``, which
#: writes the answer itself and so cannot fail, and ``codex_style``, which the
#: harness evolver is forbidden to modify (see ``harness_evolve.py``) and which
#: therefore cannot be held to a moving target.
TRAINABLE_HARNESSES = ("bash_minimal", "react_tools", "json_strict", "longctx_summary")


@dataclass
class GateResult:
    """One gate's outcome on one task."""

    gate: str
    passed: bool
    detail: str = ""
    #: Scores observed, when the gate produced numbers (V1/V2/V4).
    scores: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"gate": self.gate, "passed": self.passed, "detail": self.detail, "scores": self.scores}


@dataclass
class TaskVerdict:
    """All four gates on one task, plus the accept decision."""

    task_id: str
    tier: str
    gates: list[GateResult]

    @property
    def accepted(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def failed_gates(self) -> list[str]:
        return [g.gate for g in self.gates if not g.passed]

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tier": self.tier,
            "accepted": self.accepted,
            "failed_gates": self.failed_gates,
            "gates": [g.as_dict() for g in self.gates],
        }


# --------------------------------------------------------------------------
# a clean work directory with the task's environment materialised
# --------------------------------------------------------------------------


def _materialise(task: dict, root: Path) -> Path:
    """Create ``root`` and lay down the task's ``setup`` files.

    Kept local rather than calling ``BaseHarnessEnv._materialise_task_files``
    because the gates need a *bare* directory: using the harness would also
    write the oracle answer, which is the thing gate 2 is trying to detect.
    """
    root.mkdir(parents=True, exist_ok=True)
    for rel, content in (task.get("setup") or {}).items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(str(content), encoding="utf-8")
    return root


def _run_reference(task: dict, root: Path) -> tuple[bool, str]:
    """Execute the task's reference solution, in order, in ``root``.

    Every command must exit 0. A reference that fails partway would otherwise
    be scored on whatever half-written file it left behind, which is how a
    broken reference can still look like a passing oracle.
    """
    commands = task.get("reference") or []
    if not commands:
        return False, "task carries no reference solution"
    for i, cmd in enumerate(commands):
        out, rc = _run_shell_rc(cmd, cwd=root, timeout=60)
        if rc != 0:
            return False, f"reference step {i + 1}/{len(commands)} exited {rc}: {out.strip()[:160]}"
    return True, f"{len(commands)} reference step(s) exited 0"


# --------------------------------------------------------------------------
# V1 / V2 — the oracle and nop probes
# --------------------------------------------------------------------------


def check_oracle(task: dict, *, root: Path | None = None) -> GateResult:
    """V1: the reference solution must score exactly 1.0.

    The work directory is materialised with the task's ``setup`` files first,
    because the reference of a read-then-write task greps a file the task
    ships. Running it in an empty directory fails for a reason that has nothing
    to do with the task being wrong.

    Two independent derivations have to agree for this to pass. ``expected``
    was computed in Python by the generator; the reference is shell that
    computes the same value a different way. If they disagree the task is
    mis-specified, and the failure is reported as a disagreement rather than
    as "oracle scored 0" so the cause is not buried.
    """
    tmp = root or Path(tempfile.mkdtemp(prefix="mh_v1_"))
    _materialise(task, tmp)
    ok, detail = _run_reference(task, tmp)
    if not ok:
        return GateResult("V1_oracle", False, detail, {"oracle": 0.0})
    score = verify(task, tmp)
    if score != 1.0:
        got = tmp / ANSWER_NAME
        seen = got.read_text(encoding="utf-8", errors="replace")[:120] if got.is_file() else "<no answer.txt>"
        return GateResult(
            "V1_oracle",
            False,
            f"reference ran but verifier scored {score}. "
            f"answer.txt={seen!r} expected={str(task.get('expected'))[:120]!r} "
            f"— the shell and Python derivations disagree, so the task is mis-specified",
            {"oracle": score},
        )
    return GateResult("V1_oracle", True, detail, {"oracle": score})


def check_nop(task: dict, *, root: Path | None = None) -> GateResult:
    """V2: doing nothing must score less than 1.0.

    The failure this catches is a verifier that passes an empty or absent
    answer — ``file_contains`` on an empty needle, or a ``check_script`` that
    forgets to compare. Such a task would report a 100% pass rate for a policy
    that never acted, and it would look like the easiest task in the suite.
    """
    tmp = root or Path(tempfile.mkdtemp(prefix="mh_v2_"))
    _materialise(task, tmp)
    score = verify(task, tmp)
    if score >= 1.0:
        return GateResult(
            "V2_nop",
            False,
            "an empty work directory scored 1.0 — the verifier passes without an answer",
            {"nop": score},
        )
    return GateResult("V2_nop", True, "no-op scores 0.0", {"nop": score})


# --------------------------------------------------------------------------
# V3 — schema and path safety
# --------------------------------------------------------------------------


def check_safety(task: dict) -> GateResult:
    """V3: the task is well-formed and every path it names stays inside the workdir.

    A generated task is data, and data that the harness acts on is a place
    where a generator bug becomes a filesystem write outside the sandbox. The
    checks are the ones that matter for that:

    * the verifier mode is one ``core.verify`` implements, so reward time
      cannot raise;
    * ``id`` is present and non-empty, since the registry keys on it;
    * every ``setup`` key is a relative path with no ``..`` and is not
      absolute, so materialising the task cannot escape the work directory;
    * the reference solution does not contain a path that leaves the work
      directory, for the same reason;
    * an ``expected`` value exists for the modes that need one, and a
      ``check_script`` exists for ``python_exit`` — a mode that silently falls
      back to 0.0 would look like a hard task.
    """
    problems: list[str] = []

    tid = task.get("id")
    if not tid or not isinstance(tid, str):
        problems.append("missing or non-string 'id'")

    mode = task.get("verify", "file_equals")
    if mode not in KNOWN_VERIFY_MODES:
        problems.append(f"unknown verify mode {mode!r}")

    if mode in ("file_equals", "file_contains") and "expected" not in task:
        problems.append(f"verify mode {mode!r} needs an 'expected' value")
    if mode == "python_exit":
        script = task.get("check_script")
        if not script:
            problems.append("verify mode 'python_exit' needs a 'check_script'")
        elif script not in (task.get("setup") or {}):
            # `verify` runs `python <check_script>` in the work directory, and
            # `setup` is the only mechanism that puts a file there. A checker
            # that is not shipped would fail to open, `rc` would be non-zero,
            # and the task would score 0.0 forever while looking merely hard.
            problems.append(
                f"check_script {script!r} is not among the setup files {sorted(task.get('setup') or {})} "
                f"— it would never exist at reward time"
            )

    for rel in (task.get("setup") or {}):
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts:
            problems.append(f"setup path {rel!r} escapes the work directory")

    for i, cmd in enumerate(task.get("reference") or []):
        # Look for an absolute path or a parent traversal in the command. The
        # reference is the only generated string that gets *executed*, so it is
        # the one worth policing.
        if ".." in cmd.replace("...", ""):
            problems.append(f"reference step {i + 1} contains a parent traversal")
        for token in cmd.split():
            if token.startswith("/") and not token.startswith("/tmp/"):
                problems.append(f"reference step {i + 1} writes to absolute path {token!r}")

    if problems:
        return GateResult("V3_safety", False, "; ".join(problems))
    return GateResult("V3_safety", True, f"{len(task.get('setup') or {})} setup file(s), mode {mode}")


# --------------------------------------------------------------------------
# V4 — cross-harness solvability and variance
# --------------------------------------------------------------------------


def check_cross_harness(
    task: dict,
    *,
    scores: dict[str, float] | None = None,
) -> GateResult:
    """V4: solvable on every trainable harness, and not identical across them.

    ``scores`` maps harness name to the pass rate a probe measured. When it is
    omitted the gate can only check the *structural* half — that the task does
    not mention a harness by name in a way that privileges one — and the
    variance half is reported as unmeasured rather than assumed. Callers that
    have rolled the task out should pass their numbers.
    """
    harnesses = sorted((scores or {}).keys())

    if scores is None:
        # Structural half only. A task whose instruction names a tool that only
        # one harness exposes is solvable on that harness alone, and the
        # generator should never produce one.
        text = str(task.get("instruction", ""))
        named = [h for h in TRAINABLE_HARNESSES if h in text]
        if named:
            return GateResult(
                "V4_cross",
                False,
                f"instruction names harness(es) {named} — that privileges them over the others",
            )
        return GateResult(
            "V4_cross",
            True,
            "structurally harness-neutral; variance not yet measured",
            {"variance_unmeasured": 1.0},
        )

    if not harnesses:
        return GateResult("V4_cross", False, "no harness scores supplied")

    unsolvable = [h for h, s in scores.items() if s <= 0.0]
    if unsolvable:
        return GateResult(
            "V4_cross",
            False,
            f"unsolvable on {unsolvable} — a low score there measures harness completeness, "
            f"not model skill",
            dict(scores),
        )

    distinct = len({round(s, 6) for s in scores.values()})
    if distinct == 1:
        return GateResult(
            "V4_cross",
            False,
            f"identical scores on all harnesses ({list(scores.values())[0]:.3f}) — "
            f"the task carries no information about the harness axis",
            dict(scores),
        )

    return GateResult("V4_cross", True, f"{distinct} distinct scores across {len(harnesses)} harnesses", dict(scores))


# --------------------------------------------------------------------------
# the four gates together
# --------------------------------------------------------------------------


def validate_task(task: dict, *, scores: dict[str, float] | None = None, root: Path | None = None) -> TaskVerdict:
    """Run all four gates. A task is accepted only when every one passes.

    Each gate that touches the filesystem gets its **own** directory. Sharing
    one would let gate 1 leave an ``answer.txt`` behind for gate 2 to find, so
    the nop probe would be scoring the oracle's answer — every task would fail
    V2, including the ones whose verifier cannot possibly pass an empty file.
    That failure mode is silent in the sense that matters: the batch simply
    reports a 0% accept rate, and the cause looks like the generator rather
    than like the harness.
    """
    gates = [
        check_oracle(task, root=None if root is None else root / "v1"),
        check_nop(task, root=None if root is None else root / "v2"),
        check_safety(task),
        check_cross_harness(task, scores=scores),
    ]
    return TaskVerdict(task_id=str(task.get("id", "?")), tier=str(task.get("tier", "?")), gates=gates)


def validate_batch(tasks: list[dict], *, roots: Path | None = None) -> list[TaskVerdict]:
    """Validate a batch, reusing one temporary root to avoid thousands of dirs."""
    out: list[TaskVerdict] = []
    for t in tasks:
        base = roots
        if base is not None:
            base = base / str(t["id"])
        out.append(validate_task(t, root=base))
    return out


def summarise_validation(verdicts: list[TaskVerdict]) -> dict:
    """Accept/reject counts and *which gate* did the rejecting.

    Reporting per-gate failure counts rather than a single reject rate is the
    point: "V1 fails on a third of T2 tasks" is a generator bug with a specific
    cause, while "reject rate is 31%" is not actionable.
    """
    per_gate: dict[str, int] = {}
    for v in verdicts:
        for g in v.gates:
            if not g.passed:
                per_gate[g.gate] = per_gate.get(g.gate, 0) + 1
    by_tier: dict[str, dict[str, int]] = {}
    for v in verdicts:
        d = by_tier.setdefault(v.tier, {"total": 0, "accepted": 0})
        d["total"] += 1
        d["accepted"] += int(v.accepted)
    return {
        "total": len(verdicts),
        "accepted": sum(1 for v in verdicts if v.accepted),
        "rejected": sum(1 for v in verdicts if not v.accepted),
        "failures_by_gate": per_gate,
        "by_tier": by_tier,
    }
