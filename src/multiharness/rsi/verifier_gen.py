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

"""Generating the reward, not just the task.

A task without a reward is a prompt. The interesting claim in this repository
is not that tasks can be generated — templated prompts are easy — but that the
*reward function* can be, and that a generated reward can be checked for the
two failure modes that matter:

**It never fires.** A verifier that cannot return 1.0 turns every rollout into
a zero. The task looks hard; the truth is that it is unmeasurable. Gate V1
catches this by running the reference solution.

**It always fires.** A verifier that returns 1.0 for an empty answer makes the
task look solved by a policy that never acted. Gate V2 catches it by scoring an
untouched work directory.

Both are silent in a training log. A constant-zero reward looks like a task
that is too hard and a constant-one reward looks like a task already mastered;
neither announces itself as a bug.

The strictness axis, and where it can live
------------------------------------------
Three reward shapes are worth distinguishing, but they cannot all be built the
same way, and the constraint is not cosmetic:

``exact``
    Equality after stripping surrounding whitespace. Rejects an answer with
    anything appended. Expressible as a digest, so it can ship as a checker.
``normalised``
    Equality after collapsing whitespace runs and case-folding. Accepts
    ``ALPHA  BETA`` for ``alpha beta``. Also a digest, so also shippable.
``substring``
    The expected value appears somewhere in the answer, accepting a preamble.
    **Not expressible as a digest** — testing containment needs the needle
    itself, and the needle is the answer.

That last row is why this module does not implement a substring checker. A
``python_exit`` checker is executed as ``python check.py`` with the work
directory as cwd, and the only way a file reaches that directory is the task's
``setup``; a checker holding the plaintext needle would be readable with
``cat``. The substring reward therefore lives in the ``file_contains`` verify
mode instead, where the comparison happens in the harness process and nothing
is written to disk. :func:`mode_for_strength` is the mapping, and it is total:
every strength has a home.

The same reasoning sizes the digest. For ``exact`` and ``normalised`` the
checker stores a SHA-256 of the normalised answer rather than the answer, so
reading the checker does not yield it. That raises the bar rather than closing
the hole — a 4-character answer has few enough preimages to enumerate — so
:func:`recommend_mode` sends short answers to ``file_equals``, where the
expected value never touches the sandbox at all.
"""

from __future__ import annotations

import hashlib

__all__ = [
    "CHECK_STRENGTHS",
    "CHECKER_NAME",
    "MODE_FOR_STRENGTH",
    "mode_for_strength",
    "recommend_mode",
    "contains_slice",
    "normalise",
    "digest",
    "check_script_source",
    "generate_checker",
    "discrimination_report",
]

#: Filename the generated checker ships under. A fixed name is fine because
#: each rollout gets its own work directory.
CHECKER_NAME = "check.py"

#: How strict a generated reward is. A searchable axis for the same reason the
#: task parameters are: a suite where every reward is exact equality cannot
#: express "any answer that mentions the right value", and one where every
#: reward is a substring match cannot tell a right answer from a right answer
#: with something appended.
CHECK_STRENGTHS = ("exact", "normalised", "substring")

#: The verify mode each strength is implemented in. Total by construction —
#: ``tests/test_rsi_verifier.py`` asserts the two sets match.
MODE_FOR_STRENGTH = {
    "exact": "file_equals",
    "normalised": "python_exit",
    "substring": "file_contains",
}


def mode_for_strength(strength: str) -> str:
    """Which ``core.verify`` mode implements this strength.

    The mapping is not a preference, it is forced by what each mode can
    express without leaking the answer into the agent's work directory. See the
    module docstring for the substring case, which is the non-obvious one.
    """
    if strength not in MODE_FOR_STRENGTH:
        raise ValueError(f"unknown check strength {strength!r}; known: {CHECK_STRENGTHS}")
    return MODE_FOR_STRENGTH[strength]


def normalise(s: str, *, strength: str = "normalised") -> str:
    """The normal form a checker compares in, exposed so callers can predict it.

    ``exact`` only strips the ends; the other two collapse internal whitespace
    and case-fold. Keeping this as the single definition matters because the
    generator hashes one form and the checker compares another — if the two
    ever drifted, every ``normalised`` task would be unpassable while looking
    exactly like a task that is merely hard.
    """
    if strength == "exact":
        return str(s).strip()
    return " ".join(str(s).split()).lower()


def digest(expected: str, strength: str = "normalised") -> str:
    """SHA-256 of the expected answer in the same normal form the checker uses."""
    return hashlib.sha256(normalise(expected, strength=strength).encode("utf-8")).hexdigest()


def recommend_mode(expected: str, *, prefer_strict: bool = False) -> str:
    """Which ``verify`` mode a task with this answer should use.

    The rule is one sentence: **ship a checker only when the answer is long
    enough that its digest is not worth attacking.**

    A short answer graded by ``python_exit`` puts a brute-forceable digest in
    the agent's work directory. The same answer graded by ``file_equals`` puts
    nothing there — the comparison happens in the harness process and the
    expected value never enters the sandbox. So short answers take the
    in-process mode, and long ones may use either.

    ``prefer_strict=True`` forces ``file_equals`` for every answer, which is
    the right setting when a task is graded for a training reward rather than
    for a measurement: a reward that cannot be gamed is worth more than one
    that exercises a code path. The default follows the length rule, so the
    rule is actually reached.
    """
    s = str(expected).strip()
    if prefer_strict or len(s) < 12:
        return "file_equals"
    return "python_exit"


def contains_slice(expected: str) -> str:
    """A contiguous slice of the answer, for the ``file_contains`` mode.

    Takes the middle third rather than the whole string so the mode is
    genuinely more permissive: a contains-check whose needle happened to equal
    the full answer would not be a different reward at all, and the mode would
    be decorative.

    For an answer of three characters or fewer the middle third can be a single
    character, which makes the check nearly free to pass. The whole answer is
    used in that case, and the caller is expected to have chosen the mode
    deliberately — a ``file_contains`` task is a *different question*, not an
    easier one.
    """
    s = str(expected).strip()
    if len(s) <= 3:
        return s
    third = len(s) // 3
    return s[third : 2 * third].strip() or s


def check_script_source(expected: str, strength: str = "exact") -> str:
    """Source of a generated checker, for the strengths a digest can express.

    Raises for ``substring`` rather than producing a checker that cannot work:
    a checker that hashed the answer could only ever test equality, so a
    "substring" checker built this way would silently be an exact checker, and
    every task using it would be graded more strictly than its parameters claim.
    """
    if strength not in CHECK_STRENGTHS:
        raise ValueError(f"unknown check strength {strength!r}; known: {CHECK_STRENGTHS}")
    if strength == "substring":
        raise ValueError(
            "the substring reward cannot be expressed as a digest; use the "
            "'file_contains' verify mode (see verifier_gen.mode_for_strength)"
        )

    norm = "lambda s: s.strip()" if strength == "exact" else "lambda s: ' '.join(s.split()).lower()"
    want = digest(expected, strength)
    return (
        "import hashlib, pathlib, sys\n"
        f"_norm = {norm}\n"
        f"_want = {want!r}\n"
        "p = pathlib.Path('answer.txt')\n"
        "if not p.is_file():\n"
        "    print('FAIL: answer.txt is missing'); sys.exit(2)\n"
        "got = p.read_text(encoding='utf-8', errors='replace')\n"
        "if not got.strip():\n"
        "    print('FAIL: answer.txt is empty'); sys.exit(3)\n"
        "if hashlib.sha256(_norm(got).encode('utf-8')).hexdigest() != _want:\n"
        "    print('FAIL: answer.txt does not match'); sys.exit(1)\n"
        "print('OK')\n"
        "sys.exit(0)\n"
    )


def generate_checker(expected: str, strength: str = "exact") -> str:
    """Alias with a name that reads better at the call site."""
    return check_script_source(expected, strength)


def discrimination_report(task: dict) -> dict:
    """Describe what a task's reward accepts and rejects, without running it.

    The descriptive half of the reward contract; the executable half is gates
    V1 and V2 in ``rsi/validate.py``. They are kept separate so a report can be
    produced with no shell at all, which is what lets the figure tooling run in
    the dependency-free CI job.
    """
    mode = task.get("verify", "file_equals")
    expected = str(task.get("expected", "")).strip()
    return {
        "task_id": task.get("id", "?"),
        "tier": task.get("tier", "?"),
        "mode": mode,
        "expected_len": len(expected),
        "ships_checker": mode == "python_exit",
        "checker_name": task.get("check_script") if mode == "python_exit" else None,
        # A short answer in a mode that ships a checker is the case the module
        # docstring warns about. Surfacing it here means it shows up in a report
        # rather than living only in a comment.
        "digest_brute_forceable": bool(mode == "python_exit" and len(expected) < 12),
    }
