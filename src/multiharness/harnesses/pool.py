"""The five harnesses under test.

Each class reproduces the agent-facing surface of a real harness: guidance
text, tool set, submit protocol, error feedback. Four are the training pool;
``codex_style`` is held out.

Design note — the differences are deliberately *structural*, not cosmetic:

  harness            tools                              submit
  -----------------  ---------------------------------  --------------------
  bash_minimal       bash                               write the file
  react_tools        bash, read_file, write_file, finish  finish() tool
  json_strict        bash, submit                       submit() tool
  longctx_summary    bash, read_file, replace_in_file   write the file
  codex_style (ho)   bash, apply_patch                  write the file

The submit channel is what makes a harness a harness: ``react_tools`` and
``json_strict`` never touch ``answer.txt`` themselves, so a model that only
ever learned "write the file with bash" cannot score on them without
adapting. That is the cross-harness signal this experiment measures.
"""

from __future__ import annotations

from .core import ANSWER_NAME, BaseHarnessEnv, _run_shell, resolve_workspace_path

# --------------------------------------------------------------------------
# 1. bash_minimal — mini-swe-agent style: one tool, no ceremony
# --------------------------------------------------------------------------


class BashMinimalEnv(BaseHarnessEnv):
    """Faithful to mini-swe-agent: a single ``bash`` tool and nothing else."""

    name = "bash_minimal"
    GUIDANCE = (
        "You have exactly one tool: `bash`. It runs a shell command and returns "
        "stdout+stderr. There is no other tool.\n"
        f"Submit your final answer by writing it to `{ANSWER_NAME}` in the current "
        f"working directory, for example: echo -n 'VALUE' > {ANSWER_NAME}\n"
        "Stating the answer in prose does not submit it. Only the file counts."
    )


# --------------------------------------------------------------------------
# 2. react_tools — ReAct style: thinking-first guidance, explicit finish tool
# --------------------------------------------------------------------------


class ReactToolsEnv(BaseHarnessEnv):
    """ReAct-flavoured harness: richer toolset, terminates via ``finish``."""

    name = "react_tools"
    GUIDANCE = (
        "Work in explicit steps. Before each tool call, briefly state your reasoning "
        "(Thought), then call exactly one tool (Action).\n"
        "Available tools: `bash`, `read_file`, `write_file`, `finish`.\n"
        "When the answer is ready, call `finish` with the final value. Calling `finish` "
        f"also writes `{ANSWER_NAME}` for you, so you do not need to create it yourself."
    )

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: File path relative to the working directory.

        Returns:
            The file contents, or an error message.
        """
        assert self._workdir is not None
        p, err = resolve_workspace_path(self._workdir, path)
        if err:
            return err
        assert p is not None
        if not p.is_file():
            return f"[error] no such file: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:4000]

    def write_file(self, path: str, content: str) -> str:
        """Write text to a file in the workspace.

        Args:
            path: File path relative to the working directory.
            content: Exact text to write.

        Returns:
            A confirmation message.
        """
        assert self._workdir is not None
        dest, err = resolve_workspace_path(self._workdir, path)
        if err:
            return err
        assert dest is not None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def finish(self, answer: str) -> str:
        """Submit the final answer and end the task.

        Args:
            answer: The final answer value.

        Returns:
            A confirmation message.
        """
        self._write_answer(answer)
        return "finished; answer submitted"


# --------------------------------------------------------------------------
# 3. json_strict — strict JSON tool-call style with a dedicated submit tool
# --------------------------------------------------------------------------


class JsonStrictEnv(BaseHarnessEnv):
    """JSON tool-call harness. Errors are surfaced and the format is enforced."""

    name = "json_strict"
    GUIDANCE = (
        "Every tool call must be a well-formed JSON object with exactly the keys "
        "`name` and `arguments`. Do not emit prose around a tool call.\n"
        "Available tools: `bash`, `submit`.\n"
        "You must finish by calling `submit` with the final answer value. Writing the "
        "file directly is not the submission path in this harness."
    )

    def submit(self, answer: str) -> str:
        """Submit the final answer value. This is the only valid submission path.

        Args:
            answer: The final answer, as an exact string.

        Returns:
            A confirmation message.
        """
        if answer is None or str(answer).strip() == "":
            return "[error] empty answer rejected; call submit again with a value"
        self._write_answer(answer)
        return f"submitted {len(str(answer))} chars"


# --------------------------------------------------------------------------
# 4. longctx_summary — Claude-Code-ish: verbose rules, edit-oriented toolset
# --------------------------------------------------------------------------


class LongCtxEnv(BaseHarnessEnv):
    """Verbose-rules harness with a targeted-edit tool instead of whole-file write."""

    name = "longctx_summary"
    GUIDANCE = (
        "Follow these rules at all times:\n"
        "1. Inspect before you modify. Use `read_file` to see a file's current contents.\n"
        "2. Prefer `replace_in_file` over rewriting a whole file; it is safer and cheaper.\n"
        "3. Keep every intermediate step small and verifiable.\n"
        "4. If a command fails, read the error, adjust, and retry once before moving on.\n"
        "Available tools: `bash`, `read_file`, `replace_in_file`.\n"
        f"When done, ensure `{ANSWER_NAME}` in the working directory holds the final "
        "answer with no extra whitespace."
    )

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from the workspace.

        Args:
            path: File path relative to the working directory.

        Returns:
            The file contents, or an error message.
        """
        assert self._workdir is not None
        p, err = resolve_workspace_path(self._workdir, path)
        if err:
            return err
        assert p is not None
        if not p.is_file():
            return f"[error] no such file: {path}"
        return p.read_text(encoding="utf-8", errors="replace")[:4000]

    def replace_in_file(self, path: str, old: str, new: str) -> str:
        """Replace the first occurrence of ``old`` with ``new`` in a file.

        Creates the file when it does not exist.

        Args:
            path: File path relative to the working directory.
            old: Exact text to find. Must be non-empty.
            new: Replacement text.

        Returns:
            A confirmation or a descriptive error.
        """
        assert self._workdir is not None
        p, err = resolve_workspace_path(self._workdir, path)
        if err:
            return err
        assert p is not None
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(new, encoding="utf-8")
            return "file did not exist; created it with the replacement text"
        text = p.read_text(encoding="utf-8", errors="replace")
        if old == "":
            p.write_text(new, encoding="utf-8")
            return "old was empty; rewrote the file"
        if old not in text:
            return f"[error] the text to replace was not found in {path}"
        p.write_text(text.replace(old, new, 1), encoding="utf-8")
        return f"replaced 1 occurrence in {path}"


# --------------------------------------------------------------------------
# 5. codex_style — HELD OUT: patch-oriented, planning-first
# --------------------------------------------------------------------------


class CodexStyleEnv(BaseHarnessEnv):
    """HELD OUT. Function-call style with a patch tool and multi-step planning."""

    name = "codex_style"
    GUIDANCE = (
        "Plan first, then act. Begin by outlining the steps you will take, then execute "
        "them one at a time.\n"
        "Available tools: `bash`, `apply_patch`.\n"
        "`apply_patch` takes a unified-diff body. Prefer it for creating or changing "
        f"files. The task is complete once `{ANSWER_NAME}` exists in the working "
        "directory with the exact expected content."
    )

    def apply_patch(self, patch: str) -> str:
        """Apply a unified-diff patch inside the workspace.

        A minimal `*** Add File` form is accepted for creating a file:

        ```
        *** Add File: answer.txt
        +the new line
        ```

        Args:
            patch: The patch body.

        Returns:
            A confirmation or a descriptive error.
        """
        assert self._workdir is not None
        if "*** Add File:" in patch:
            lines = patch.splitlines()
            idx = next(i for i, ln in enumerate(lines) if ln.startswith("*** Add File:"))
            rel = lines[idx].split(":", 1)[1].strip()
            body = [ln[1:] for ln in lines[idx + 1 :] if ln.startswith("+")]
            dest, err = resolve_workspace_path(self._workdir, rel)
            if err:
                return err
            assert dest is not None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(body), encoding="utf-8")
            return f"added {rel} ({len(body)} lines)"

        # Fall back to `patch` if the host has it; otherwise report clearly.
        out = _run_shell(
            "command -v patch >/dev/null 2>&1 && patch -p0 -f <<'__PATCH__'\n"
            + patch
            + "\n__PATCH__\n|| echo '[error] patch tool unavailable or patch failed'",
            cwd=self._workdir,
        )
        return out


TRAIN_HARNESSES: dict[str, type[BaseHarnessEnv]] = {
    "bash_minimal": BashMinimalEnv,
    "react_tools": ReactToolsEnv,
    "json_strict": JsonStrictEnv,
    "longctx_summary": LongCtxEnv,
}

HELDOUT_HARNESSES: dict[str, type[BaseHarnessEnv]] = {
    "codex_style": CodexStyleEnv,
}

ALL_HARNESSES = {**TRAIN_HARNESSES, **HELDOUT_HARNESSES}
