"""Task suite for the multi-harness experiment.

Every task is graded by ``core.verify`` reading only ``<workdir>/answer.txt``.
The verifier never learns which harness produced the answer — that is the
validity basis for the cross-harness gap.

Three tiers, ordered by how much they stress the *harness* rather than the
model's reasoning:

T1  copy-through      one tool call, content given verbatim in the prompt.
                      The only thing under test is whether the harness's
                      tool-call format is followable.
T2  read-then-write   the content must be pulled out of a file the task
                      ships. Needs a pipeline or two calls.
T3  quoting-stress    the payload contains quotes / backslashes / newlines.
                      This is where harnesses genuinely diverge: a bash-only
                      harness and a JSON-tool-call harness need different
                      escaping, so a model that overfits one format fails
                      the other. This is the load-bearing tier.
"""

from __future__ import annotations

from ..harnesses.core import register_tasks


def _t1(tid: str, payload: str, *, flavour: str = "") -> dict:
    """Content given verbatim; the agent only has to place it in the file."""
    return {
        "id": tid,
        "tier": "T1",
        "flavour": flavour,
        "instruction": (
            f"Write the following text into a file named `answer.txt` in the current "
            f"working directory, exactly as shown:\n\n{payload}"
        ),
        "expected": payload,
        "verify": "file_equals",
    }


def _t2(tid: str, filename: str, content: str, needle: str, *, flavour: str = "") -> dict:
    """A file is shipped; one of its lines must be extracted."""
    return {
        "id": tid,
        "tier": "T2",
        "flavour": flavour,
        "setup": {filename: content},
        "instruction": (
            f"The file `{filename}` in the current working directory contains several "
            f"lines. Find the line that starts with `{needle}` and write **only that "
            f"line's value** (the part after `{needle}`) into `answer.txt`."
        ),
        "expected": _value_after(content, needle),
        "verify": "file_equals",
    }


def _value_after(content: str, needle: str) -> str:
    for line in content.splitlines():
        if line.startswith(needle):
            return line[len(needle) :].strip()
    raise ValueError(f"needle {needle!r} not found in content")


def _t3(tid: str, payload: str, *, flavour: str = "") -> dict:
    """Payload contains characters that need escaping — harness-dependent."""
    return {
        "id": tid,
        "tier": "T3",
        "flavour": flavour,
        "instruction": (
            "Write the following text into a file named `answer.txt` in the current "
            "working directory. Preserve every character exactly, including quotes and "
            "backslashes:\n\n" + payload
        ),
        "expected": payload,
        "verify": "file_equals",
    }


_T1_SPECS = [
    ("t1-01", "hello world"),
    ("t1-02", "harness generalization"),
    ("t1-03", "42"),
    ("t1-04", "multi harness agentic rl"),
    ("t1-05", "the quick brown fox"),
    ("t1-06", "ready"),
    ("t1-07", "alpha beta gamma"),
    ("t1-08", "checkpoint saved"),
    ("t1-09", "reward is one"),
    ("t1-10", "tool call accepted"),
    ("t1-11", "context window"),
    ("t1-12", "baseline pass rate"),
]

_T2_SPECS = [
    (
        "t2-01",
        "config.txt",
        "mode=debug\nseed=1234\nlr=0.0001\nbatch=8\n",
        "seed=",
    ),
    (
        "t2-02",
        "notes.txt",
        "first line\ntarget value here\nlast line\n",
        "target",
    ),
    (
        "t2-03",
        "meta.txt",
        "name=harnessrl\nversion=0.1\nowner=less\n",
        "version=",
    ),
    (
        "t2-04",
        "log.txt",
        "INFO starting\nWARN retrying\nERROR failed hard\nINFO done\n",
        "ERROR",
    ),
    (
        "t2-05",
        "data.txt",
        "red\nblue\ngreen\nyellow\n",
        "green",
    ),
    (
        "t2-06",
        "params.txt",
        "alpha=1\nbeta=2\ngamma=3\n",
        "gamma=",
    ),
]

# Quoting / escaping stress: this is where harnesses actually diverge.
_T3_SPECS = [
    ("t3-01", 'say "hello" now'),
    ("t3-02", "path is C:\\Users\\less"),
    ("t3-03", 'json: {"k": "v"}'),
    ("t3-04", "quote ' and \" mixed"),
    ("t3-05", 'echo "$HOME" literally'),
    ("t3-06", "percent %s and dollar $x"),
]


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    tasks += [_t1(tid, payload, flavour="copy") for tid, payload in _T1_SPECS]
    tasks += [_t2(tid, fn, c, n, flavour="read") for tid, fn, c, n in _T2_SPECS]
    tasks += [_t3(tid, payload, flavour="quote") for tid, payload in _T3_SPECS]
    return tasks


# --------------------------------------------------------------------------
# train / eval split
# --------------------------------------------------------------------------
# The experiment varies two axes independently — the harness and the task — so
# both need a held-out side. Splitting only the harness would let the model
# memorize the tasks and still look like it generalized; splitting only the
# tasks would leave the harness axis untested.
#
# The split is stratified: every tier contributes to both sides, so the eval
# set is not accidentally all one difficulty.

TRAIN_TASK_IDS = [
    # T1 copy-through (8 of 12)
    "t1-01", "t1-02", "t1-03", "t1-04", "t1-05", "t1-06", "t1-07", "t1-08",
    # T2 read-then-write (4 of 6)
    "t2-01", "t2-02", "t2-03", "t2-04",
    # T3 quoting-stress (4 of 6)
    "t3-01", "t3-02", "t3-03", "t3-04",
]

EVAL_TASK_IDS = [
    # T1 (4 of 12)
    "t1-09", "t1-10", "t1-11", "t1-12",
    # T2 (2 of 6)
    "t2-05", "t2-06",
    # T3 (2 of 6)
    "t3-05", "t3-06",
]


def load() -> None:
    """Idempotent registration."""
    from ..harnesses.core import TASKS

    if TASKS:
        return
    register_tasks(build_tasks())


if __name__ == "__main__":
    load()
    from ..harnesses.core import TASKS

    by_tier: dict[str, int] = {}
    for t in TASKS.values():
        by_tier[t["tier"]] = by_tier.get(t["tier"], 0) + 1
    print(f"registered {len(TASKS)} tasks: {by_tier}")
    assert set(TRAIN_TASK_IDS) | set(EVAL_TASK_IDS) == set(TASKS), "split does not cover the suite"
    assert not (set(TRAIN_TASK_IDS) & set(EVAL_TASK_IDS)), "split overlaps"
    print(f"train {len(TRAIN_TASK_IDS)} / eval {len(EVAL_TASK_IDS)}")

