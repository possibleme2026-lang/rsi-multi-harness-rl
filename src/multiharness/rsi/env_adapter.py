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

"""Making a synthesised environment task runnable by the harness pool.

Why this is a separate module
-----------------------------
``harbor_export`` produces a *package* — a directory of files for a container
build. The harness pool runs on the **host**, with a per-rollout temp directory
and no image at all. The two need the same three things (an instruction, a
starting state, executable tools) in different shapes, and this module is the
translation.

The translation is not cosmetic. Three specific mismatches have to be handled,
and each one is a way the environment axis could silently measure nothing:

**The tools must be callable from the agent's shell.** A Harbor package installs
``envtool`` at ``/usr/local/bin``. On the host there is no such directory and no
``python`` on the agent's PATH, so the task ships the same generated ``tools.py``
into the working directory and a ``envtool`` shim beside it. The shim is put on
``PATH`` for the rollout by :func:`environment_path`.

**The state path must be relocatable.** The generated ``tools.py`` defaults to
``/app/state.json``; on the host the state lives in the rollout's own directory.
``tools.py`` reads ``ENVTOOL_STATE``, and this module sets it. That override is
why the *same* generated program can serve both, rather than two implementations
of one tool semantics.

**Grading must read the state, not a string.** Every shipped task is graded by
``core.verify`` on ``answer.txt``. A stateful task is graded by
``core.grade_state`` on ``state.json``. The task dict therefore carries
``verify = "state_checkpoints"`` and the checkpoint list, and the harness layer
gained that mode for it.

What this buys
--------------
The harness pool can attempt environment tasks, which is the precondition for
the measurement this whole axis exists to make: the **cross-harness
generalization gap** on stateful tasks, rather than on string tasks. A gap
measured on "copy this text into a file" is a gap in the submission protocol; a
gap measured on "find the record and change its state" is a gap in the ability
to operate a system, which is the claim the RSI idea rests on.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
from pathlib import Path

from . import envtask, harbor_export

__all__ = [
    "ENVTOOL_SHIM_NAME",
    "SHELL_TOOL",
    "as_harness_task",
    "as_harness_batch",
    "environment_path",
    "install_environment",
    "harness_guidance",
]

#: The command the instruction tells the agent to call. Same name as in the
#: container, so an instruction is true in both places.
ENVTOOL_SHIM_NAME = "envtool"

#: The name of the *harness's* shell tool, which is how the environment's
#: commands are reached. Not a free choice: ``BaseHarnessEnv.bash`` is defined on
#: the base class, so every harness in the pool exposes exactly this name — but
#: that is a property of the pool, not of this module, so it is asserted rather
#: than assumed (``test_rsi_envgen`` checks all of ``TRAIN_HARNESSES``). Measured
#: reason it matters: with the environment's commands presented as if they were
#: harness tools, the model called ``envtool list_tickets(...)`` as a tool name
#: and every harness answered "Tool envtool list_tickets not found".
SHELL_TOOL = "bash"


def harness_guidance(task: envtask.EnvTask) -> str:
    """What the *harness pool* has to tell the agent that Harbor's package does not.

    This exists because the first scan of an environment batch scored **0.00 in
    every cell of every harness** — 96 rollouts, not one point — and the cause
    was not the model. Each harness appends a static ``GUIDANCE`` block written
    for *string* tasks, and for a stateful task every sentence in it is wrong:

    * it says to submit by writing ``answer.txt``, while the verifier reads
      ``state.json`` — so even a perfect solution would score 0.0;
    * it never mentions ``envtool``, so the agent's only route to the environment
      is guessing the tool names. The transcripts show exactly that: ``update_
      inventory_item_by_id: command not found``, ``cat /var/log/syslog``,
      ``echo "set balance to closed"``.

    The Harbor package does not have this problem because ``_instruction_md``
    renders a tool table. That table is generated in ``harbor_export`` and never
    reaches the harness pool — the pool reads a task *dict*, not a package — so
    the two paths diverged and only one of them was ever executed. This function
    is the missing half, and it is attached to the task as ``guidance`` so each
    harness can prefer it over its own static text.

    It is deliberately explicit about the *shape* of a call rather than
    describing the tools in prose: a 0.5B model that is told ``envtool get_item
    <id>`` in a fenced example emits that; one told "there are tools" emits
    invented names.

    **The first version of this function was wrong, and the scan that measured
    it was green on the wrong metric.** It rendered the tools as a list —

    .. code-block:: text

        envtool list_tickets
        envtool set_state <id> <value>

    — which reads as a *tool list*, and the prompt already contains one: the
    tokenizer's chat template renders the harness's own tools (``bash``,
    ``submit``, …) as a schema block. Given two lists of that shape, the model
    merged them and called ``envtool list_tickets`` **as a tool name**:

    .. code-block:: text

        envtool list_tickets({'query': 'state=open owner=east'})
        -> {'error': "Tool envtool list_tickets not found. Available: ['bash']"}

    measured in **every** harness — 9 rollouts in ``json_strict``, 6 in
    ``bash_minimal``, 6 in ``react_tools``, 5 in ``longctx_summary``, and the
    same for ``list_items``/``list_accounts``. The tool-call-rate gate went
    *up* (35.4% → 81.2%) while the pass rate stayed at exactly 0.00, because a
    call to a non-existent tool counts as a tool call. **A gate that a wrong
    call satisfies is not a gate.** The gate is now checked against the stage
    funnel in ``scripts/env_scan_report.py``, which separates "called something"
    from "called a real tool with a real command".

    Two consequences are baked into the text below:

    * the tools are introduced as **shell command lines**, not as tools, and the
      wrong shape is shown explicitly as wrong — a negative example is the only
      thing that reliably stops a small model from repeating a shape it has
      already seen in the prompt;
    * the ``<id>``/``<value>`` placeholders are kept *inside* the command string
      and never presented as named parameters, because ``bash`` takes exactly
      one argument named ``command`` and the model was passing the environment's
      parameters alongside it:

      .. code-block:: text

          bash({'command': 'envtool list_tickets', 'owner': 'east'})
          -> {'error': "bad arguments for bash: ... unexpected keyword argument 'owner'"}

    The name ``bash`` is asserted, not assumed: ``test_rsi_envgen`` checks that
    every harness in ``TRAIN_HARNESSES`` exposes it, so this text cannot quietly
    become false for a harness added later.
    """
    g = task.graph
    names = g.get("nodes") or []
    mutators = [n for n in names if g.get("kinds", {}).get(n) == "mutate"]

    rows = []
    for n in names:
        params = g.get("params", {}).get(n, [])
        args = " ".join(f"<{p}>" for p in params)
        rows.append(f"  {ENVTOOL_SHIM_NAME} {n} {args}".rstrip())
    listing = "\n".join(rows)

    query = names[0] if names else "list_items"
    mutator = mutators[0] if mutators else None
    # A worked example with a *literal* command line, not a tool signature.
    # The id is a placeholder rather than a real record id on purpose: the
    # instruction deliberately does not name the target ("find the ticket whose
    # state is open and owner is east"), so a worked example that named one
    # would either be wrong for the task or would teach the model to skip the
    # lookup the task is testing.
    worked = [f'  {SHELL_TOOL}(command="{ENVTOOL_SHIM_NAME} {query}")']
    if mutator:
        worked.append(
            f'  {SHELL_TOOL}(command="{ENVTOOL_SHIM_NAME} {mutator} '
            '<id> <value>")'
        )

    return (
        f"The environment is a program called `{ENVTOOL_SHIM_NAME}`, run from "
        "the shell. Its commands are **not** tools of this harness — there is no "
        f"tool named `{ENVTOOL_SHIM_NAME}`, and no tool named "
        f"`{ENVTOOL_SHIM_NAME} {query}`.\n"
        "\n"
        f"To use one, call the shell tool `{SHELL_TOOL}` and pass the whole "
        "command as the single string argument `command`:\n"
        "\n"
        + "\n".join(worked) + "\n"
        "\n"
        "That is the right shape. This is the wrong shape, and it is the one to "
        "avoid — it looks like a tool call but there is no such tool:\n"
        "\n"
        f'  {ENVTOOL_SHIM_NAME} {query}(...)   <- wrong, no such tool\n'
        "\n"
        f"`{SHELL_TOOL}` takes exactly one argument, `command`. Do not pass the "
        "environment's parameters to it; they go inside the command string.\n"
        "\n"
        "The commands available, each written as a shell command line:\n"
        "\n"
        f"{listing}\n"
        "\n"
        "Start by listing the records, then read the one the task describes. "
        "The instruction identifies it by a property, not by name, so you have to "
        "look it up.\n"
        "\n"
        "**Your work is graded on the state your commands leave behind**, not on "
        "the commands themselves and not on anything you write to a text file. Do "
        "not write an answer file. The commands update the state for you, and any "
        "sequence of commands that produces the right final state scores full "
        "marks.\n"
    )


def _shim_source() -> str:
    """The host-side ``envtool`` shim.

    Deliberately *not* ``harbor_export._envtool_shim``. That one is a container
    artifact: it hardcodes a fallback of ``/app`` and relies on ``python`` being
    on PATH, both of which are true inside the image and false on the host. This
    one takes its directory from its own location and its interpreter from the
    running one, so it works from any working directory.

    It also does not need to be installed anywhere: :func:`environment_path`
    puts the rollout's own directory on ``PATH``, so the agent types ``envtool``
    and the shell finds it.
    """
    import sys

    py = str(Path(sys.executable)).replace("\\", "/")
    return (
        "#!/bin/sh\n"
        "# Generated by rsi-multi-harness-rl (rsi/env_adapter.py). Do not edit.\n"
        '# Resolve this script\'s own directory, so the shim works wherever the\n'
        "# rollout put it -- there is no /app on the host.\n"
        'HERE=$(cd "$(dirname "$0")" && pwd)\n'
        f'exec "{py}" "$HERE/tools.py" "$@"\n'
    )


def install_environment(task: envtask.EnvTask) -> dict:
    """The environment's files, as the ``setup`` mapping the harness layer writes.

    Returns ``{relpath: contents}`` rather than writing anything, because
    ``BaseHarnessEnv._materialise_task_files`` is the single place that knows how
    a task's files reach a workdir — writing from two places would let the two
    disagree about where a file goes.

    Takes no workdir, deliberately. An earlier version did and ignored it, which
    made ``setup`` *look* rollout-specific while being identical for every
    rollout. That is worse than an unused parameter: it invites a caller to
    believe the files are per-rollout when the only per-rollout thing is where
    the harness chooses to write them, which is handled in ``core``.
    """
    files = harbor_export.package_files(task)
    prefix = f"{task.task_id}/"

    setup: dict[str, str] = {}
    for rel, content in files.items():
        if not rel.startswith(prefix + "environment/assets/"):
            continue
        name = rel[len(prefix + "environment/assets/"):]
        if name == ENVTOOL_SHIM_NAME:
            # Replaced by the host-side shim below; the container one would not
            # run here.
            continue
        setup[name] = content

    # `tools.py` is the one file that must be *adjusted* rather than copied.
    # Its default state path is the container's `/app/state.json`; on the host
    # the rollout's own directory is the right place, and `ENVTOOL_STATE`
    # carries it. Doing this by rewriting the constant would break the
    # container package's hash; the environment variable leaves the file
    # byte-identical to the exported one.
    setup[ENVTOOL_SHIM_NAME] = _shim_source()
    return setup


def environment_path(workdir: Path | None = None) -> dict:
    """The environment variables a rollout needs to use the environment.

    Returns a **serialisable template** when called with no argument, and a
    resolved dict when given a workdir. Both forms exist for a reason:

    * the template is what goes into the task dict, so a batch can be dumped to
      JSON and re-registered in another process — which is how
      ``probe.py --from-batch`` scans an environment batch at all;
    * the resolved dict is what a direct caller (a test, the smoke script) wants
      when it already knows the directory.

    ``core.BaseHarnessEnv._build_env`` expands the template at ``reset``, using
    the *rollout's own* workdir. That is the property that matters: passing a
    precomputed dict would embed whichever directory existed at export time, so
    every rollout would read one shared state file — an environment that looks
    like it works while making all rollouts depend on each other.

    Three variables, and each is load-bearing:

    ``PATH`` gets the workdir prepended. Without it the agent has to type
    ``./envtool``, and the instruction — which says ``envtool``, as it does in
    the container — would be wrong on the host. A task whose instruction does
    not work is a task the model cannot solve for a reason that has nothing to
    do with the model.

    ``ENVTOOL_STATE`` points ``tools.py`` at the rollout's state file. Without
    it the tools read ``/app/state.json``, which does not exist here, so every
    tool call fails on a missing file and the rollout scores 0 for a harness
    reason rather than a policy reason — the exact confound this whole
    experiment is built to avoid.

    ``ENVTOOL_INITIAL`` is the fallback state, used only when no state file
    exists yet. It is what makes an *untouched* rollout score 0.0 by grading the
    initial state rather than by crashing on a missing file.
    """
    if workdir is None:
        # The template form. `_path_prefix` is core's reserved key for entries
        # to prepend to the inherited PATH, and `{workdir}` / `{python_dir}` are
        # substituted there. Kept as literal placeholders so the dict is plain
        # JSON — a callable here would make the batch undumpable.
        return {
            "_path_prefix": ["{workdir}", "{python_dir}"],
            "ENVTOOL_STATE": "{workdir}/state.json",
            "ENVTOOL_INITIAL": "{workdir}/initial_state.json",
        }

    import sys

    parts = [str(workdir), str(Path(sys.executable).parent)]
    if os.environ.get("PATH"):
        parts.append(os.environ["PATH"])
    return {
        "PATH": os.pathsep.join(parts),
        "ENVTOOL_STATE": str(workdir / "state.json"),
        "ENVTOOL_INITIAL": str(workdir / "initial_state.json"),
    }


def as_harness_task(task: envtask.EnvTask) -> dict:
    """One environment task as a task dict the harness pool can run.

    The dict is deliberately the *same shape* as a shipped or generated task:
    ``id``, ``flavour``, ``instruction``, ``verify``, ``setup``. The harness
    layer resolves every task through one registry and one ``reset``, so a new
    kind of task that needed a new code path there would be a harness change
    masquerading as a task — and it would make the cross-harness comparison
    meaningless, since the harnesses would no longer share a substrate.

    ``expected`` is present but not used by the ``state_checkpoints`` mode. It
    is set to the reference's final state as JSON so that a caller inspecting
    the dict (or a future verifier mode) sees what the goal is, and so
    ``task_gen.summarise``-style tooling that expects the key does not crash.
    """
    checkpoints = [c.as_dict() for c in task.checkpoints]
    final = envtask.apply_trace(task.instance, list(task.trace))
    out = {
        "id": task.task_id,
        "tier": "ENV",
        "flavour": "synthesised_env",
        "instruction": task.instruction,
        "verify": "state_checkpoints",
        "checkpoints": checkpoints,
        "expected": json.dumps(final, sort_keys=True),
        "setup": install_environment(task),
        # A serialisable **template**, not a callable and not a resolved dict.
        # `core._build_env` expands it at `reset` against the rollout's own
        # workdir. A callable would make the batch undumpable — measured:
        # `TypeError: Object of type function is not JSON serializable` when
        # writing `env_batch.json`, which is what made the environment batch
        # unscannable. A precomputed dict would be worse: every rollout would
        # share one state file, and the scan would report a plausible number for
        # an experiment that was not run.
        "env": environment_path(),
        # Overrides each harness's static `GUIDANCE`, which is written for string
        # tasks and is actively wrong here (it says to submit `answer.txt` and
        # never mentions the environment's tools). Measured: without this, all 96
        # rollouts across four harnesses scored 0.00 — see `harness_guidance`.
        "guidance": harness_guidance(task),
        "domain": task.instance.spec.domain,
        "difficulty": round(task.instance.spec.difficulty(), 3),
        "checkpoint_count": len(checkpoints),
    }
    return out


def as_harness_batch(tasks: list[envtask.EnvTask]) -> list[dict]:
    """A batch of environment tasks as harness task dicts.

    Raises on a duplicate id rather than letting one overwrite another: the
    registry raises too, but failing here names the batch as the source, and a
    silently shrunk batch is a curriculum bug wearing the costume of a
    scheduling detail.
    """
    out = [as_harness_task(t) for t in tasks]
    ids = [t["id"] for t in out]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate environment task ids in batch: {dupes}")
    return out


def make_executable(path: Path) -> None:
    """Best-effort ``chmod +x``.

    A no-op on Windows, where the filesystem has no POSIX mode bits — and that
    is fine rather than a bug: the shim is invoked through ``sh`` there, and on
    the Linux side the image build sets the bit. Kept as a helper so callers do
    not each have to remember which platforms care.
    """
    with contextlib.suppress(OSError):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
