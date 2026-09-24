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
import json
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

    On POSIX this signals the child's **process group**, which is only safe
    because ``_run_shell`` starts the child with ``start_new_session=True`` so
    that group contains nothing but the command's own tree. The guard below
    refuses to signal our own group: without it, a future change that drops
    ``start_new_session`` would make a timeout kill the caller — observed live,
    as a CI job that sat ``in_progress`` for half an hour instead of failing,
    because the step's own shell was killed along with the runaway command.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=20,
            )
        else:
            pgid = os.getpgid(pid)
            if pgid == os.getpgid(0):
                # Same group as us: killpg would take down the caller.
                os.kill(pid, signal.SIGKILL)
            else:
                os.killpg(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best-effort cleanup, never fatal
        pass


def _run_shell(
    command: str,
    cwd: str | Path,
    timeout: int = _SHELL_TIMEOUT_S,
    env: dict[str, str] | None = None,
) -> str:
    """Run one shell command in *cwd* and return its combined output.

    The output-only form. Callers that need the exit status must use
    :func:`_run_shell_rc` — see the note there on why the text is not a
    faithful carrier of it.
    """
    return _run_shell_rc(command, cwd, timeout, env)[0]


def _run_shell_rc(
    command: str,
    cwd: str | Path,
    timeout: int = _SHELL_TIMEOUT_S,
    env: dict[str, str] | None = None,
) -> tuple[str, int | None]:
    """Run one shell command in *cwd*. Non-stateful, like mini-swe-agent.

    Returns ``(output, returncode)``. ``returncode`` is ``None`` when the
    command could not be started at all or was killed by the timeout, and the
    exit status otherwise.

    Why the exit status needs its own channel
    -----------------------------------------
    A *silent* command is rendered as ``"(no output, exit=N)"``, so the same
    string is produced for a success and a failure — ``exit=0`` and ``exit=1``
    differ only in one digit buried in prose. Anything that reads the text to
    decide whether a command succeeded is therefore parsing a human-readable
    message as if it were a status code. ``verify``'s ``python_exit`` mode did
    exactly that, and its check was
    ``"[error]" not in out and "Traceback" not in out and "Error" not in out``
    — all three of which are true for *both* ``exit=0`` and ``exit=1``. Every
    self-generated Python verifier therefore scored 1.0 regardless of what it
    checked, which is a silent false-positive in the one place the experiment
    cannot afford one.

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
        return "[error] no POSIX bash found on this host; set MULTIHARNESS_BASH", None

    # `env=None` inherits the parent's environment, which is what every shipped
    # task wants. A stateful task passes an explicit mapping so `envtool` is on
    # PATH and `tools.py` knows where the state lives. Note this *replaces*
    # rather than merges: the caller is responsible for carrying through PATH,
    # and `env_adapter.environment_path` does.
    run_env = None if env is None else {**env}

    with tempfile.TemporaryFile() as sink:
        try:
            proc = subprocess.Popen(
                [BASH, "-c", command],
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                env=run_env,
                # Its own session, so the child's process group contains only
                # this command's tree. That is what makes `killpg` in
                # `_kill_tree` safe: without it the group is shared with the
                # caller, and a timeout would kill the whole harness process
                # (and, in CI, the runner's own step shell).
                start_new_session=True,
            )
        except OSError as exc:  # pragma: no cover - host dependent
            return f"[error] could not run command: {exc}", None

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
    # A timeout means the process was killed by us, so its status is not a
    # verdict on the command: report None rather than a misleading 0.
    rc = None if timed_out else proc.returncode
    if timed_out:
        return f"[error] command timed out after {timeout}s\n{out}".rstrip(), rc
    if out.strip():
        return out, rc
    # Silent command: surface the status in the text *as well*, because the
    # model reads this string. The second element of the tuple is what code
    # must branch on.
    return f"(no output, exit={rc})", rc


# --------------------------------------------------------------------------
# verifier — harness-agnostic, reads only <workdir>/answer.txt
# --------------------------------------------------------------------------


def _read_answer(workdir: Path) -> str | None:
    p = workdir / ANSWER_NAME
    if not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


#: Where a *stateful* task's final state is read from, and the seed it starts
#: from. Kept here rather than in the rsi layer because the verifier has to
#: reach them without importing a module that imports the verifier — the
#: dependency arrow points ``rsi -> harnesses``, not the other way.
STATE_NAME = "state.json"
INITIAL_STATE_NAME = "initial_state.json"


def _answer_stamp(workdir: Path) -> str | None:
    """Content fingerprint of the graded artifact, or ``None`` when absent.

    Used as the reward cache key. Hashing the bytes rather than the mtime
    matters: two consecutive writes inside one filesystem timestamp tick would
    otherwise collide, and the artifact is tiny so the cost is nil.

    Covers **both** ``answer.txt`` and a stateful task's ``state.json``. Reading
    only the answer file was correct while every task was a question; once a
    task is graded on state, a rollout that never writes ``answer.txt`` would
    have a permanent cache key of ``None``, and ``get_reward`` would freeze its
    first (failing) verdict — a task the agent subsequently solved would keep
    reporting 0.
    """
    parts = []
    for name in (ANSWER_NAME, STATE_NAME):
        p = workdir / name
        try:
            parts.append(hashlib.sha1(p.read_bytes()).hexdigest())
        except OSError:
            parts.append("-")
    return "|".join(parts)


def evaluate_checkpoint(cp: dict, state: dict) -> bool:
    """Evaluate one declarative checkpoint against a final state.

    Total: a missing key, a wrong type, or a malformed state evaluates to
    ``False`` rather than raising. A raise inside ``get_reward`` aborts the
    rollout, so an exception here would make a malformed state look like a
    crashed run instead of a failed attempt — and "the model produced a weird
    state" is a normal, expected outcome that must be *scored*, not thrown.

    This is the **single** implementation. ``rsi/envtask.py`` imports it, and
    ``rsi/harbor_export.py`` inlines its source into the exported verifier; the
    three used to be separate copies, which is a drift risk that fails silently
    (all three keep returning plausible scores). ``tests/test_rsi_envgen.py``
    asserts the inlined copy agrees with this one.
    """
    try:
        a = cp.get("args") or {}
        kind = cp["kind"]

        if kind == "target_field":
            rec = next((r for r in state.get("records", []) if r.get("_id") == a["id"]), None)
            if rec is None:
                return False
            return all(str(rec.get(f)) == str(v) for f, v in a["fields"].items())

        if kind == "target_only":
            return set(state.get("_changed", [])) == {a["id"]}

        if kind == "log_recorded":
            return any(
                e.get("id") == a["id"] and e.get("field") == a["field"]
                for e in state.get("log", [])
            )

        if kind == "meta_consistent":
            entries = state.get("log", [])
            # `>= 1` is load-bearing: "counter == len(log)" is `0 == 0` on an
            # untouched state, i.e. trivially true. That one vacuous predicate
            # was the entire reason the initial state scored 0.2-0.25 instead of
            # 0.0. A state with no change cannot satisfy a condition about a
            # change having been recorded.
            return len(entries) >= 1 and int(state.get("meta", {}).get("changed", -1)) == len(entries)

        if kind == "order_observable":
            got = list(state.get("_order", []))
            # Same guard, same reason: an untouched state has no order, and
            # "no order equals no order" would otherwise be true for free.
            return len(got) > 0 and got == list(a["order"])

        return False
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False


def grade_state(state: dict, checkpoints: list[dict]) -> float:
    """Fraction of checkpoints satisfied. The reward for a stateful task.

    ``0/0`` returns 0.0 rather than 1.0. A task with an empty checkpoint list is
    a generation bug, and scoring it 1.0 would make the bug look like the
    easiest task in the suite — the V2 failure (a reward that always fires)
    arriving through a different door.
    """
    if not checkpoints:
        return 0.0
    passed = sum(1 for cp in checkpoints if evaluate_checkpoint(cp, state))
    return passed / len(checkpoints)


def _read_state(workdir: Path) -> dict | None:
    """The agent's final state, falling back to the seed when it never acted.

    Falling back matters: an agent that does nothing leaves no ``state.json``,
    and returning ``None`` would grade as a crash. Grading the *seed* instead
    produces the honest answer — 0.0, because no checkpoint about a change can
    be satisfied by the state nothing happened to. That also makes the
    no-op case and the "agent never started" case score identically, which they
    should.
    """
    for name in (STATE_NAME, INITIAL_STATE_NAME):
        p = workdir / name
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
    return None


def verify(task: dict, workdir: Path) -> float:
    """Return the reward. Never inspects which harness produced the answer."""
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

    if mode == "state_checkpoints":
        # A *fractional* reward, unlike every other mode. That is deliberate and
        # is the reason the environment axis exists: the GRPO group signal
        # `1 - p^G - (1-p)^G` vanishes as `p` approaches 0, and a 0.5B model on
        # a multi-step stateful task sits near `p = 0`. A boolean reward there
        # makes every group identical and the gradient exactly zero, so the
        # fractional form is a *training* requirement rather than a reporting
        # preference. See `rsi/envtask.py` for the full argument.
        state = _read_state(workdir)
        if state is None:
            return 0.0
        return grade_state(state, task.get("checkpoints") or [])

    if mode == "python_exit":
        script = task.get("check_script")
        if not script:
            return 0.0
        runner = to_bash_path(sys.executable)
        _out, rc = _run_shell_rc(f'"{runner}" {script}', cwd=workdir, timeout=60)
        # The exit status is the verdict. It used to be inferred from the text
        # ("[error]"/"Traceback"/"Error" absent), which passes for a *silent*
        # failure as well as a success: `_run_shell` renders both as
        # "(no output, exit=N)", so `exit=1` carried no traceback and no
        # "[error]" and scored 1.0. Every generated Python verifier was
        # therefore a constant-1.0 oracle. `rc` is the same value the shell
        # itself branched on, so it cannot be fooled by how the output reads.
        #
        # A non-zero exit *with* output is still a failure — the check's own
        # printed reason must not be able to buy a pass.
        return 1.0 if rc == 0 else 0.0

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
        #: Extra environment for tool invocations. Populated by ``reset`` from
        #: the task's ``env`` key, which a stateful task uses to put its own
        #: ``envtool`` on PATH. Empty for every shipped task, so the shipped
        #: behaviour is unchanged.
        self._env: dict[str, str] = {}

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
        # Built after the files exist, because the variables point *at* them.
        self._env = self._build_env()
        return self._instruction()

    def _build_env(self) -> dict[str, str]:
        """The environment for this rollout's tool calls.

        A task may carry an ``env`` key naming the variables it needs; a
        stateful task's adapter supplies ``PATH`` (so ``envtool`` resolves) and
        ``ENVTOOL_STATE`` (so the tools write into *this* rollout's directory
        instead of the container path they default to).

        Three accepted forms, and the third is the one that matters:

        * ``None`` — inherit the process environment.
        * a **callable** — called with this rollout's workdir.
        * a **template dict** — string values may contain ``{workdir}`` and
          ``{python_dir}``, and the reserved key ``_path_prefix`` lists entries
          to prepend to the inherited ``PATH``.

        The template exists because a callable cannot be serialised, and a task
        batch *is* serialised: ``probe.py --from-batch`` reads task dicts from
        JSON, and a callable would either crash the dump or be dropped. Dropping
        it is the dangerous half — without ``ENVTOOL_STATE`` every rollout reads
        the same default path, so all rollouts silently share one state file and
        the scan reports a plausible number for an experiment that was not run.
        Resolution happens here, at ``reset``, so the workdir is the rollout's
        own rather than whichever directory existed at export time.

        Kept as an overridable method rather than inlined into ``reset`` so a
        harness can add its own variables without the base class knowing what
        they mean. Deliberately *not* importing the adapter: the arrow points
        ``rsi -> harnesses``, and the template is the interface that keeps it
        that way.
        """
        assert self._workdir is not None and self._task is not None
        spec = self._task.get("env")
        if spec is None:
            return {}
        if callable(spec):
            spec = spec(self._workdir)

        workdir = self._workdir
        subs = {"workdir": str(workdir), "python_dir": str(Path(sys.executable).parent)}

        def resolve(v: str) -> str:
            try:
                return v.format(**subs)
            except (KeyError, IndexError):
                # A literal brace the task meant literally. Left as-is rather
                # than raising: an env var that fails to expand is not worth
                # aborting a rollout over, and the task is still runnable.
                return v

        out: dict[str, str] = {}
        prefix: list[str] = []
        for k, v in dict(spec).items():
            if k == "_path_prefix":
                prefix = [resolve(str(p)) for p in (v or [])]
                continue
            out[str(k)] = resolve(str(v))

        if prefix:
            inherited = os.environ.get("PATH", "")
            parts = prefix + ([inherited] if inherited else [])
            out["PATH"] = os.pathsep.join(parts)
        return out

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
        """The task text plus the harness's guidance.

        A task may override the guidance with its own ``guidance`` key, and the
        override is not a convenience. Each harness's ``GUIDANCE`` is static text
        written for *string* tasks: it says to submit by writing ``answer.txt``,
        and it lists a fixed tool set. For a stateful environment task both are
        wrong — the verifier reads ``state.json``, and the tools are the
        environment's, not the harness's — so a stateful task that inherited the
        static block was unsolvable by construction while looking merely hard.

        Measured: the first scan of an environment batch scored 0.00 in all 96
        cells across four harnesses. The transcripts show the model inventing
        tool names and echoing a command as if it were a mutation, because the
        only description of the environment it ever received was the instruction
        prose. The Harbor export path renders a tool table and does not have this
        bug; the harness-pool path had no equivalent, which is what this closes.

        The *harness's* own tool description is still true and is not replaced —
        only the submission protocol is, which is why the override lives on the
        task rather than in the harness.
        """
        task = self._task or {}
        parts = [task.get("instruction", "")]
        override = task.get("guidance")
        if override:
            parts.append(str(override).strip())
        elif self.GUIDANCE:
            parts.append(self.GUIDANCE.strip())
        return "\n\n".join(p for p in parts if p)

    def _materialise_task_files(self) -> None:
        """Create any files the task ships in its initial workspace."""
        assert self._workdir is not None and self._task is not None
        for rel, content in (self._task.get("setup") or {}).items():
            dest = self._workdir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            # Best effort: a shipped shell script (a stateful task's `envtool`)
            # needs the bit on POSIX. Windows has no mode bits and this is a
            # no-op there, which is fine — the shim is run through `sh` and the
            # container image sets the bit at build time.
            if rel.endswith(".sh") or content.startswith("#!"):
                with contextlib.suppress(OSError):
                    dest.chmod(dest.stat().st_mode | 0o111)

    def _write_answer(self, answer: str) -> None:
        assert self._workdir is not None
        (self._workdir / ANSWER_NAME).write_text(str(answer), encoding="utf-8")
        self._submitted = str(answer)

    def _exec(self, command: str, timeout: int = _SHELL_TIMEOUT_S) -> str:
        assert self._workdir is not None, "reset() must run before tools"
        self._turns += 1
        return _run_shell(command, cwd=self._workdir, timeout=timeout, env=self._env or None)

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
