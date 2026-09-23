"""Core primitives for the multi-harness agentic RL experiment.

A *harness* here is a (system-prompt style guidance, tool set, context
policy, submit protocol, error feedback) tuple — the same surface
MiMo-V2.6 varies in its §4.2.5 multi-harness training. Each harness is a
class; TRL's ``GRPOTrainer`` turns its public methods into tools
automatically (``inspect.getmembers(env, ismethod)``, skipping names that
start with ``_`` or are ``reset`` / ``get_reward``), and routes each
dataset row to a harness by the ``environment`` column.

Sandboxing
----------
Execution is a ``subprocess`` with ``cwd=<per-rollout workdir>``. There is
**no container**. We do not claim process-level isolation; see the README
for the honest scope note. What is faithful is the *agent-facing* surface:
tool set, submit protocol, guidance text, and error feedback.

Submit protocol is unified at the file level: every harness ultimately
writes ``<workdir>/answer.txt``. That keeps the verifier completely
harness-agnostic, which is the validity basis for measuring cross-harness
generalization — the harness changes *how* the agent solves, never *how*
it is graded.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_NO_REWARD = object()  # sentinel: reward not computed yet (0.0 is a valid reward)

ANSWER_NAME = "answer.txt"
_SHELL_TIMEOUT_S = 30
_MAX_OUTPUT_CHARS = 8000


# --------------------------------------------------------------------------
# host shell
# --------------------------------------------------------------------------


def _find_bash() -> str | None:
    """Locate a POSIX bash. cmd.exe is unusable: it has no ``echo -n``.

    Resolution order: ``MULTIHARNESS_BASH`` -> ``PATH`` -> the standard
    Git-for-Windows install locations. For a non-standard install, set
    ``MULTIHARNESS_BASH`` to the full path rather than editing this list.
    """
    env = os.environ.get("MULTIHARNESS_BASH")
    if env and Path(env).is_file():
        return env
    found = shutil.which("bash")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


BASH = _find_bash()


def to_bash_path(path: str | Path) -> str:
    """``C:\\Users\\x`` -> ``/c/Users/x`` so bash tools can address it."""
    p = str(path)
    if len(p) > 1 and p[1] == ":":
        return "/" + p[0].lower() + p[2:].replace("\\", "/")
    return p.replace("\\", "/")


def resolve_workspace_path(workdir: Path, path: str) -> tuple[Path | None, str | None]:
    """Resolve a model-supplied *path* to a file inside *workdir*.

    Returns ``(resolved, None)`` on success or ``(None, error_message)`` on a
    rejection the model can act on.

    Why this is not just ``workdir / path``
    ---------------------------------------
    ``Path("/tmp/x") / ""`` is ``Path("/tmp/x")`` — the *directory itself*, not
    an error. So a model that calls ``write_file(path="", content=...)`` or
    ``replace_in_file(path=".")`` produces a write attempt against a directory.
    The raw OS error then leaks through and it is **different on every
    platform**: ``Permission denied`` on Windows, ``IsADirectoryError`` on
    POSIX. Measured on this host: 18 of 208 scan rollouts hit exactly this, all
    of them in ``react_tools``.

    That matters for the experiment, not just for tidiness. The thing being
    measured is the *harness*, so anything a harness returns has to be a
    function of the harness and the model's action — never of the host OS. An
    OS-dependent error string would make the same harness look different on two
    machines, and would be read as cross-harness variance when it is really
    cross-platform variance. It also gives the model nothing to recover from:
    "Permission denied" on a path it never wrote invites a retry loop, whereas
    naming the actual mistake lets it correct in one turn.

    Rejections are returned as an error string rather than raised, matching how
    TRL turns a tool exception into feedback for the model
    (``grpo_trainer.py:1993-2003``).
    """
    raw = "" if path is None else str(path)
    if raw.strip() == "":
        return None, (
            "[error] empty path; pass the file name you want, e.g. "
            f"'{ANSWER_NAME}' — a path is required"
        )
    if raw.strip() in (".", "./", ".\\"):
        return None, (
            "[error] '.' is the working directory, not a file; name the file "
            f"itself, e.g. '{ANSWER_NAME}'"
        )

    candidate = (workdir / raw).resolve()
    # Containment check: `..` must not escape the per-rollout workspace. This
    # is a correctness guard for the measurement, not a security boundary —
    # bash is unjailed by design and can reach anywhere regardless.
    try:
        candidate.relative_to(workdir.resolve())
    except ValueError:
        return None, f"[error] path escapes the working directory: {raw}"

    if candidate.is_dir():
        return None, (
            f"[error] {raw!r} is a directory, not a file; name the file itself, "
            f"e.g. '{ANSWER_NAME}'"
        )
    return candidate, None


def _kill_tree(pid: int) -> None:
    """Kill *pid* and every descendant.

    ``Popen.kill()`` only signals the direct child. A shell command that
    spawned grandchildren (``bash -c 'python x.py'``) leaves them running, and
    on Windows they keep any inherited handles open — which is what turns a
    timeout into a hang (see ``_run_shell``).
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=20,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best-effort cleanup, never fatal
        pass


def _run_shell(command: str, cwd: str | Path, timeout: int = _SHELL_TIMEOUT_S) -> str:
    """Run one shell command in *cwd*. Non-stateful, like mini-swe-agent.

    Output goes to a **temporary file, never a pipe**, and stdin is
    ``DEVNULL``. Both are load-bearing, and both were learned from a live
    hang that cost 84 minutes of idle GPU:

    * Pipes deadlock. ``subprocess.run(capture_output=True, timeout=T)`` kills
      the direct child when the timeout fires and then calls ``communicate()``
      to drain the pipes. If the command spawned a grandchild that inherited
      the write end, the drain never sees EOF and blocks *forever* — the
      timeout fires, and the cleanup itself is what hangs. A file has no EOF
      to wait on, so the read is unconditional and cannot block.
    * A command that reads stdin (bare ``cat``, ``python -c 'input()'``, a
      pager) blocks until stdin closes. The model emits such commands
      occasionally; ``DEVNULL`` turns an infinite hang into an instant EOF.
    """
    if BASH is None:
        return "[error] no POSIX bash found on this host; set MULTIHARNESS_BASH"

    with tempfile.TemporaryFile() as sink:
        try:
            proc = subprocess.Popen(
                [BASH, "-c", command],
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:  # pragma: no cover - host dependent
            return f"[error] could not run command: {exc}"

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc.pid)
            # The tree kill was already issued; if the direct child still has
            # not reaped after 10s, move on rather than blocking here. The
            # original bug was precisely a cleanup path that could block
            # forever, so this second wait must never be unbounded.
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)

        sink.seek(0)
        raw = sink.read()

    out = raw.decode("utf-8", errors="replace")
    if len(out) > _MAX_OUTPUT_CHARS:
        out = out[: _MAX_OUTPUT_CHARS] + "\n... [truncated]"
    if timed_out:
        return f"[error] command timed out after {timeout}s\n{out}".rstrip()
    return out if out.strip() else f"(no output, exit={proc.returncode})"


# --------------------------------------------------------------------------
# verifier — harness-agnostic, reads only <workdir>/answer.txt
# --------------------------------------------------------------------------


def _read_answer(workdir: Path) -> str | None:
    p = workdir / ANSWER_NAME
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _answer_stamp(workdir: Path) -> str | None:
    """Content fingerprint of ``answer.txt``, or ``None`` when absent.

    Used as the reward cache key. Hashing the bytes rather than the mtime
    matters: two consecutive writes inside one filesystem timestamp tick would
    otherwise collide, and the answer file is tiny so the cost is nil.
    """
    p = workdir / ANSWER_NAME
    try:
        data = p.read_bytes()
    except OSError:
        return None
    return hashlib.sha1(data).hexdigest()


def verify(task: dict, workdir: Path) -> float:
    """Return 1.0 / 0.0. Never inspects which harness produced the answer."""
    mode = task.get("verify", "file_equals")

    if mode == "file_equals":
        got = _read_answer(workdir)
        if got is None:
            return 0.0
        expected = str(task["expected"])
        # Compare on stripped content; agents often add a trailing newline.
        return 1.0 if got.strip() == expected.strip() else 0.0

    if mode == "file_contains":
        got = _read_answer(workdir)
        if got is None:
            return 0.0
        return 1.0 if str(task["expected"]).strip() in got else 0.0

    if mode == "python_exit":
        script = task.get("check_script")
        if not script:
            return 0.0
        runner = to_bash_path(sys.executable)
        out = _run_shell(f'"{runner}" {script}', cwd=workdir, timeout=60)
        # _run_shell appends "(no output, exit=N)" when silent; check the file too.
        ok = "[error]" not in out and "Traceback" not in out and "Error" not in out
        return 1.0 if ok else 0.0

    raise ValueError(f"unknown verify mode {mode!r}")


# --------------------------------------------------------------------------
# task registry (populated by tasks/*.py)
# --------------------------------------------------------------------------

TASKS: dict[str, dict] = {}


def register_tasks(tasks: list[dict]) -> None:
    for t in tasks:
        if t["id"] in TASKS:
            raise ValueError(f"duplicate task id {t['id']!r}")
        TASKS[t["id"]] = t


def get_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise KeyError(f"unknown task_id {task_id!r}; known: {sorted(TASKS)[:5]}...")
    return TASKS[task_id]


# --------------------------------------------------------------------------
# tool discovery — mirrors GRPOTrainer exactly
# --------------------------------------------------------------------------


def discover_tools(env: BaseHarnessEnv) -> list:
    """The bound methods TRL will expose as tools, in sorted order.

    This reproduces ``grpo_trainer.py:654`` verbatim (skip ``reset`` /
    ``get_reward`` / ``_``-prefixed). Kept here so the probe and the smoke
    tests agree with the trainer by construction rather than by convention.
    """
    out = []
    for member_name, member in inspect.getmembers(env, predicate=inspect.ismethod):
        if member_name in ("reset", "get_reward") or member_name.startswith("_"):
            continue
        out.append(member)
    return out


def tool_names(env: BaseHarnessEnv) -> list[str]:
    return sorted(m.__name__ for m in discover_tools(env))


# --------------------------------------------------------------------------
# base environment
# --------------------------------------------------------------------------


class BaseHarnessEnv:
    """One harness. Subclasses set ``name`` / ``GUIDANCE`` and the tools.

    Public methods become tools. Keep every internal helper ``_``-prefixed.
    """

    name: str = "base"
    GUIDANCE: str = ""
    SUBMIT_VIA_FILE: bool = True

    def __init__(self) -> None:
        self._task: dict | None = None
        self._workdir: Path | None = None
        self._reward = _NO_REWARD
        self._reward_stamp: float | None = None
        self._turns = 0
        self._submitted: str | None = None

    # -- TRL lifecycle -----------------------------------------------------

    def reset(self, task_id: str | None = None, **kwargs) -> str:
        """Called by GRPOTrainer with every non-``environment`` dataset column."""
        if task_id is None:
            raise ValueError(f"{type(self).__name__}.reset requires task_id")
        self._task = get_task(task_id)
        self._workdir = Path(tempfile.mkdtemp(prefix=f"mh_{self.name}_{task_id}_"))
        self._reward = _NO_REWARD
        self._reward_stamp = None
        self._turns = 0
        self._submitted = None
        self._materialise_task_files()
        return self._instruction()

    @property
    def reward(self) -> float:
        """Cached reward for this rollout. Also available as ``get_reward()``."""
        return self.get_reward()

    def get_reward(self) -> float:
        """TRL reward hook.

        The trainer scans for a *method* named exactly ``get_reward`` and, when
        present, registers it as a reward source for this environment class.
        A ``@property`` would not be picked up (``inspect.ismethod`` misses it),
        so this must stay a plain method.

        The result is cached against the mtime of ``answer.txt`` rather than
        computed once. TRL only asks once per rollout so the distinction is
        invisible there, but evaluation loops re-read the same environment
        after further tool calls — a naive one-shot cache would freeze the
        first (usually failing) verdict and silently report a 0 reward for a
        task the agent subsequently solved.
        """
        if self._workdir is None or self._task is None:
            return 0.0
        stamp = _answer_stamp(self._workdir)
        if self._reward is _NO_REWARD or stamp != self._reward_stamp:
            self._reward = verify(self._task, self._workdir)
            self._reward_stamp = stamp
        return self._reward

    # -- internals ---------------------------------------------------------

    def _instruction(self) -> str:
        task = self._task or {}
        parts = [task.get("instruction", "")]
        if self.GUIDANCE:
            parts.append(self.GUIDANCE.strip())
        return "\n\n".join(p for p in parts if p)

    def _materialise_task_files(self) -> None:
        """Create any files the task ships in its initial workspace."""
        assert self._workdir is not None and self._task is not None
        for rel, content in (self._task.get("setup") or {}).items():
            dest = self._workdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    def _write_answer(self, answer: str) -> None:
        assert self._workdir is not None
        (self._workdir / ANSWER_NAME).write_text(str(answer), encoding="utf-8")
        self._submitted = str(answer)

    def _exec(self, command: str, timeout: int = _SHELL_TIMEOUT_S) -> str:
        assert self._workdir is not None, "reset() must run before tools"
        self._turns += 1
        return _run_shell(command, cwd=self._workdir, timeout=timeout)

    # -- the universal tool ------------------------------------------------

    def bash(self, command: str) -> str:
        """Run a shell command in the task workspace and return stdout+stderr.

        The shell is not stateful between calls. Paths are POSIX style
        (``/c/...`` on Windows). Use it to inspect files, run python, and
        submit by writing the answer file.

        Args:
            command: The shell command to run.

        Returns:
            The command's combined stdout and stderr.
        """
        return self._exec(command)


class OracleHarness(BaseHarnessEnv):
    """Writes the gold answer on reset. Used only to separate "pipeline is
    broken" from "the model cannot do it" — never for training."""

    name = "oracle"
    GUIDANCE = "The answer has already been submitted. Reply with anything."

    def reset(self, task_id: str | None = None, **kwargs) -> str:
        text = super().reset(task_id=task_id, **kwargs)
        assert self._task is not None
        self._write_answer(str(self._task["expected"]))
        return text
