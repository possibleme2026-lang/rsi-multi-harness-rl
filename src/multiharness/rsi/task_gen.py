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

"""Programmatic task generation: the policy writes its own curriculum.

Why this module exists
----------------------
The suite used to be 24 tasks typed out by hand in ``tasks/suite.py``. That
makes the experiment a fixed measurement: it can tell you how four harnesses
differ on *those* 24 tasks and nothing else. It cannot tell you whether the
suite was the right difficulty, and it cannot grow. Worse, a hand-written suite
has hand-written *coverage holes*: every one of the 24 tasks used
``verify: "file_equals"``, so ``file_contains`` and ``python_exit`` were never
exercised at all, and the T2 tier turned out to be unsolvable by the policy
while nobody could tell whether that was the model or the task.

This module replaces the list with a *generator*. A task is now the output of a
parameter vector:

    payload_len      how much content has to survive the round trip
    escape_density   fraction of characters that need quoting or escaping
    steps            how many shell operations the reference solution needs
    read_source      whether the content comes from a shipped file
    verify_mode      how the result is graded

Change a parameter and you get a different task; sweep them and you get a
difficulty curve. That is what makes the suite something the loop can *search*
rather than something a human maintains.

Four artifacts from one parameter vector
----------------------------------------
Following the SPADE formulation, generating a task means generating the whole
trial, not just the question. :func:`generate` returns all four:

1. **The environment** — ``setup``, the files the agent will find on disk.
2. **The task** — ``instruction``, what the agent is asked to do.
3. **The reward** — ``verify`` / ``expected`` / ``check_script``.
4. **The reference solution** — ``reference``, the shell commands that solve it.

The reference is what makes the reward checkable. It is written in *shell*
while ``expected`` is computed in *Python*, so the two are independent
derivations of the same answer: if they disagree, the task is mis-specified and
no agent could have solved it. ``rsi/validate.py`` executes the reference in a
clean work directory and requires the verifier to score it 1.0 — the oracle
gate. A hand-written suite can only assert this by convention; a generated one
can assert it by construction, on every task, every time.

Determinism
-----------
Every function here is a pure function of its parameters and a seed. The same
seed produces byte-identical tasks, which is what lets CI re-derive the suite
and compare it against the artifact that was actually trained on.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace

from . import verifier_gen

__all__ = [
    "ESCAPE_CHARS",
    "PARAM_SPACE",
    "TaskParams",
    "generate",
    "sample_params",
    "generate_batch",
    "summarise_batch",
    "heredoc_delimiter",
]

#: Characters that force a harness to think about quoting. A bash harness and a
#: JSON-tool-call harness need *different* escaping for these, which is exactly
#: why the T3 tier is where the harnesses separate.
ESCAPE_CHARS = ['"', "'", "\\", "$", "`", "%", "!", "*"]

#: A line that is exactly the heredoc delimiter would end the payload early.
#: Payload generation refuses to emit it, so the reference solution below is
#: safe by construction rather than by luck.
_HEREDOC_DELIM = "MHEOF"

_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "harness", "reward", "policy", "rollout", "gradient", "checkpoint",
    "tensor", "kernel", "buffer", "stream", "token", "logits",
]


# --------------------------------------------------------------------------
# the parameter vector
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskParams:
    """One point in task space. Everything about the task derives from this.

    Frozen so a generated task's provenance cannot drift after the fact: the
    parameters recorded in the task dict are the ones that produced it.
    """

    tier: str
    payload_len: int
    escape_density: float
    steps: int
    read_source: bool
    verify_mode: str
    #: Content comes from a shipped file rather than the prompt when True.
    source_name: str = "input.txt"
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "payload_len": self.payload_len,
            "escape_density": round(self.escape_density, 3),
            "steps": self.steps,
            "read_source": self.read_source,
            "verify_mode": self.verify_mode,
            "source_name": self.source_name,
            "seed": self.seed,
        }

    def difficulty(self) -> float:
        """A single scalar for plotting. Deliberately simple and stated.

        Each knob is mapped to ``[0, 1]`` by its own range and averaged with
        equal weight. This is a *display* coordinate, not a claim that the
        knobs are commensurable — the figures plot pass rate against it to
        show a trend, and the trend is what matters, not the units.
        """
        parts = [
            min(1.0, (self.payload_len - 1) / 15.0),
            min(1.0, self.escape_density / 0.6),
            min(1.0, (self.steps - 1) / 2.0),
            1.0 if self.read_source else 0.0,
            0.0 if self.verify_mode == "file_equals" else (0.5 if self.verify_mode == "file_contains" else 1.0),
        ]
        return sum(parts) / len(parts)


#: The searchable space. ``sample_params`` draws from these; the figures sweep
#: them one axis at a time to separate their effects.
PARAM_SPACE: dict[str, object] = {
    "tier": ["T1", "T2", "T3", "T4"],
    "payload_len": [1, 2, 4, 8, 16],
    "escape_density": [0.0, 0.15, 0.35, 0.6],
    "steps": [1, 2, 3],
    "read_source": [False, True],
    "verify_mode": ["file_equals", "file_contains", "python_exit"],
}


# --------------------------------------------------------------------------
# content generation
# --------------------------------------------------------------------------


def _payload(rng: random.Random, length: int, escape_density: float) -> str:
    """Build content with a controlled density of escaping characters.

    Characters are chosen so that the *measured* density matches the requested
    one as closely as possible: with probability ``escape_density`` a position
    is filled from :data:`ESCAPE_CHARS`, otherwise from ordinary word letters.
    The result is assembled into a few space-separated groups rather than one
    long token, because a single 16-character word is a different task (copy
    fidelity) from 16 words (sustained attention).
    """
    chars: list[str] = []
    for _ in range(length):
        if rng.random() < escape_density:
            chars.append(rng.choice(ESCAPE_CHARS))
        else:
            chars.append(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789"))

    text = "".join(chars)
    # Group into 1-4 character chunks so it reads as content, not a hash.
    grouped: list[str] = []
    i = 0
    while i < len(text):
        take = rng.randint(1, 4)
        grouped.append(text[i : i + take])
        i += take
    out = " ".join(grouped)

    # A line equal to the heredoc delimiter would terminate the reference
    # solution's payload early. Refuse by construction.
    if out.strip() == _HEREDOC_DELIM:
        out = out + "x"
    return out


def _source_file(rng: random.Random, name: str, needle: str, value: str, filler: int) -> str:
    """A small key=value file with the needle somewhere in the middle.

    The needle sits neither first nor last, so a ``head -1`` or ``tail -1``
    shortcut fails and the agent has to actually search. That is the property
    that makes this tier different from copy-through, and it is constructed
    rather than hoped for.
    """
    keys = ["mode", "lr", "batch", "retries", "timeout", "workers", "verbose", "shards"]
    rng.shuffle(keys)
    n_rows = max(filler, 3)
    at = max(1, min(n_rows - 2, filler // 2))
    lines: list[str] = []
    for i in range(n_rows):
        if i == at:
            lines.append(f"{needle}{value}")
        else:
            lines.append(f"{keys[i % len(keys)]}={rng.randint(1, 999)}")
    return "\n".join(lines) + "\n"


def heredoc_delimiter() -> str:
    """The delimiter the reference solution uses for a quoted heredoc.

    Exposed so the validator can assert payloads never collide with it, rather
    than the guarantee living only inside :func:`_payload`.
    """
    return _HEREDOC_DELIM


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


def _task_id(p: TaskParams) -> str:
    """Stable id from the parameters, so a re-run reproduces the same task.

    Derived from a hash rather than a counter because the loop generates tasks
    out of order and across processes; a counter would make the id depend on
    evaluation order, and two runs that produced the same task would disagree
    about its name.
    """
    blob = "|".join(f"{k}={v}" for k, v in sorted(p.as_dict().items()))
    return f"{p.tier.lower()}-{hashlib.sha1(blob.encode()).hexdigest()[:8]}"


def generate(params: TaskParams) -> dict:
    """Build one complete trial: environment, task, reward, and reference.

    The returned dict is a superset of what ``core.register_tasks`` expects, so
    it can be registered directly. The extra keys are:

    ``params``
        The parameter vector, for provenance and for plotting.
    ``reference``
        Shell commands that solve the task, run by the oracle gate.
    ``reference_expected``
        What those commands are claimed to produce. Kept separate from
        ``expected`` so the validator can report *which* derivation is wrong
        when the two disagree.
    """
    rng = random.Random(params.seed)
    p = params

    payload = _payload(rng, p.payload_len, p.escape_density)
    needle = "target="
    setup: dict[str, str] = {}
    instruction: str
    expected: str
    reference: list[str]
    verify_mode = p.verify_mode

    # ---------------- environment + task + reference ----------------
    if p.read_source:
        # The content lives on disk; the agent must extract one line.
        raw = _payload(rng, max(2, p.payload_len // 2), p.escape_density)
        setup[p.source_name] = _source_file(rng, p.source_name, needle, raw, filler=6)
        if p.steps >= 2:
            # A second operation: case-fold the extracted value.
            expected = raw.upper()
            transform = " | tr '[:lower:]' '[:upper:]'"
            what = "the part after `target=`, converted to upper case"
        else:
            expected = raw
            transform = ""
            what = "the part after `target=`"
        instruction = (
            f"The file `{p.source_name}` in the current working directory contains several "
            f"lines. Find the line that starts with `{needle}` and write {what} into "
            f"`answer.txt`."
        )
        reference = [
            f"grep '^{needle}' {p.source_name} | cut -d= -f2-{transform} > answer.txt",
        ]
    else:
        # The content is given in the prompt. `steps` controls how much
        # processing is required before it can be written.
        if p.steps == 1:
            expected = payload
            instruction = (
                "Write the following text into a file named `answer.txt` in the current "
                "working directory, exactly as shown:\n\n" + payload
            )
            reference = [f"cat > answer.txt <<'{_HEREDOC_DELIM}'\n{payload}\n{_HEREDOC_DELIM}"]
        elif p.steps == 2:
            expected = payload.replace(" ", "_")
            instruction = (
                "Write the following text into a file named `answer.txt` in the current "
                "working directory, with every space replaced by an underscore:\n\n" + payload
            )
            reference = [
                f"cat <<'{_HEREDOC_DELIM}' | tr ' ' '_' > answer.txt\n{payload}\n{_HEREDOC_DELIM}",
            ]
        else:
            expected = payload.replace(" ", "_").upper()
            instruction = (
                "Write the following text into a file named `answer.txt` in the current "
                "working directory, with every space replaced by an underscore and all "
                "letters converted to upper case:\n\n" + payload
            )
            reference = [
                f"cat <<'{_HEREDOC_DELIM}' | tr ' ' '_' | tr '[:lower:]' '[:upper:]' > answer.txt\n"
                f"{payload}\n{_HEREDOC_DELIM}",
            ]

    # ---------------- reward ----------------
    # `file_contains` and `python_exit` are not variations for their own sake:
    # a contains-check accepts a superset of answers, and a python check can
    # assert a *property* rather than an exact string. Both change what the
    # task is, which is why they are searchable rather than fixed.
    task: dict = {
        "id": _task_id(p),
        "tier": p.tier,
        "flavour": "generated",
        "instruction": instruction,
        "expected": expected,
        "verify": verify_mode,
        "params": p.as_dict(),
        "reference": reference,
    }
    if setup:
        task["setup"] = setup

    # Ask the reward layer which mode this answer may use, rather than trusting
    # the parameter vector. `verify_mode` is drawn independently of the payload,
    # so a third of the batch landed on "python_exit with a 3-character answer"
    # — the one combination `verifier_gen`'s module docstring calls out as
    # wrong, because it writes a brute-forceable digest into the agent's work
    # directory. `recommend_mode` already encodes the rule; the bug was that
    # nothing called it. Measured on 300 generated tasks: 96 (32%) were
    # affected, and 0 are after this.
    #
    # An explicit `python_exit` on a *long* answer is still honoured, so the
    # mode stays reachable when the parameters ask for it deliberately.
    if verify_mode == "python_exit" and verifier_gen.recommend_mode(expected) != "python_exit":
        verify_mode = "file_equals"
        task["verify"] = verify_mode
        task["verify_mode_override"] = "short answer: python_exit would ship a brute-forceable digest"

    if verify_mode == "python_exit":
        # `check_script` is a *filename*, not source: `core.verify` runs
        # `python <check_script>` with the work directory as cwd, and the only
        # way a file gets into that directory is via `setup`. So the generated
        # checker ships as a setup file and the task points at it.
        #
        # The checker stores a SHA-256 rather than the answer, because it lands
        # inside the agent's reach. See `verifier_gen` for why a short answer
        # should not use this mode at all.
        setup.setdefault(verifier_gen.CHECKER_NAME, verifier_gen.generate_checker(expected, "exact"))
        task["setup"] = setup
        task["check_script"] = verifier_gen.CHECKER_NAME
    elif verify_mode == "file_contains":
        # Grade on a distinctive middle slice of the answer, so the check is
        # weaker than equality but still cannot be passed by an empty file.
        task["expected"] = verifier_gen.contains_slice(expected)

    return task


def _contains_slice(expected: str) -> str:
    """Deprecated alias — kept so old call sites keep working.

    The implementation moved to :mod:`verifier_gen`, where it sits next to the
    rest of the reward logic. Two copies of a reward rule is one copy too many:
    the day they disagree, half the suite is graded differently from the other
    half for no visible reason.
    """
    return verifier_gen.contains_slice(expected)


def _check_script(expected: str, p: TaskParams) -> str:
    """Deprecated alias for :func:`verifier_gen.generate_checker`.

    The ``p`` argument is accepted and ignored so existing call sites and the
    tests that exercise them keep working; the checker no longer depends on the
    task's parameters, only on its answer and the chosen strength.
    """
    return verifier_gen.generate_checker(expected, "exact")


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def sample_params(
    rng: random.Random,
    *,
    tier: str | None = None,
    payload_len: int | None = None,
    escape_density: float | None = None,
    steps: int | None = None,
    read_source: bool | None = None,
    verify_mode: str | None = None,
) -> TaskParams:
    """Draw one point from :data:`PARAM_SPACE`, with any axis pinned.

    Pinning an axis is how the sweeps work: hold everything else random, vary
    one knob, and the resulting pass-rate curve is attributable to that knob
    rather than to the corpus.
    """
    space = PARAM_SPACE
    tier_v = tier if tier is not None else rng.choice(space["tier"])  # type: ignore[arg-type]
    len_v = payload_len if payload_len is not None else rng.choice(space["payload_len"])  # type: ignore[arg-type]
    esc_v = escape_density if escape_density is not None else rng.choice(space["escape_density"])  # type: ignore[arg-type]
    steps_v = steps if steps is not None else rng.choice(space["steps"])  # type: ignore[arg-type]
    read_v = read_source if read_source is not None else rng.choice(space["read_source"])  # type: ignore[arg-type]
    mode_v = verify_mode if verify_mode is not None else rng.choice(space["verify_mode"])  # type: ignore[arg-type]

    # Consistency: a T3 task without escaping characters is not a T3 task, and
    # a T4 task that needs one step is not a T4 task. Rather than silently
    # accepting incoherent points, nudge them into agreement — the tier is the
    # label a reader trusts, so it must not lie about the content.
    if tier_v == "T3":
        esc_v = max(esc_v, 0.35)
    if tier_v == "T4":
        steps_v = max(steps_v, 2)
    if tier_v == "T2":
        read_v = True

    return TaskParams(
        tier=tier_v,
        payload_len=len_v,
        escape_density=esc_v,
        steps=steps_v,
        read_source=read_v,
        verify_mode=mode_v,
        seed=rng.randrange(2**31),
    )


def generate_batch(n: int, *, seed: int = 0, **pins) -> list[dict]:
    """``n`` tasks with distinct ids. Ids are deduplicated, not overwritten.

    A collision would silently shrink the batch, and a batch that is smaller
    than requested is a curriculum bug that looks like a scheduling detail, so
    the loop is bounded and the shortfall is visible in the length.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        task = generate(sample_params(rng, **pins))
        if task["id"] in seen:
            continue
        seen.add(task["id"])
        out.append(task)
    return out


def summarise_batch(tasks: list[dict]) -> dict:
    """Coverage report. Used by the validator and by the README's numbers.

    Reports the reward mode the tasks **actually use** (``task["verify"]``), not
    the one their parameter vector asked for. Those two are not the same: the
    generator demotes ``python_exit`` to ``file_equals`` on answers too short to
    resist a brute-forced digest, so a parameter-side count describes a batch
    that was never produced. The parameter-side count is still reported, as
    ``by_verify_mode_requested``, because it is what the search space covered —
    the two together are the only way to see how often the rule fired.
    """
    by_tier: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    by_mode_requested: dict[str, int] = {}
    by_steps: dict[int, int] = {}
    for t in tasks:
        p = t["params"]
        by_tier[p["tier"]] = by_tier.get(p["tier"], 0) + 1
        by_mode[t["verify"]] = by_mode.get(t["verify"], 0) + 1
        by_mode_requested[p["verify_mode"]] = by_mode_requested.get(p["verify_mode"], 0) + 1
        by_steps[p["steps"]] = by_steps.get(p["steps"], 0) + 1
    return {
        "total": len(tasks),
        "by_tier": by_tier,
        "by_verify_mode": by_mode,
        "by_verify_mode_requested": by_mode_requested,
        "mode_overrides": sum(1 for t in tasks if t.get("verify_mode_override")),
        "by_steps": by_steps,
        "with_source_file": sum(1 for t in tasks if "setup" in t),
        "distinct_ids": len({t["id"] for t in tasks}),
    }


def reparameterise(task: dict, **changes) -> dict:
    """Derive a new task by moving one parameter of an existing one.

    This is the harness-evolver's mirror image on the task axis: given a task
    that is currently too easy or too hard, produce a neighbour by changing one
    knob and holding the rest, so the loop can hill-climb difficulty without
    losing track of which knob moved.
    """
    base = TaskParams(**task["params"])
    return generate(replace(base, seed=base.seed + 1, **changes))
