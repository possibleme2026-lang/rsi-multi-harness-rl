# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Harness pool** of five agent harnesses that differ in the axes a policy must
  not depend on: system prompt, tool set, and submit protocol. Four are used for
  training (`bash_minimal`, `react_tools`, `json_strict`, `longctx_summary`); one
  (`codex_style`) is held out entirely and only ever appears in evaluation.
- **Task suite** of 24 tasks in three tiers — T1 copy-through, T2
  read-then-write, T3 quoting-stress — split 16/8 into train and eval so that
  both the harness axis *and* the task axis have a held-out side.
- **Harness-agnostic verifier.** Grading reads only `<workdir>/answer.txt`, so
  the grader cannot tell which scaffold produced an answer. This is the validity
  basis for the whole comparison.
- **Capability probe** (`scripts/probe.py`) with four Go/No-Go gates, run before
  any GPU time is spent on training. A failure is a result, not a bug: it says
  the base model cannot drive the harness pool, in which case any gap measured
  afterwards would be noise.
- **Difficulty scan** that measures per-cell pass rates, so rows with provably
  zero reward variance can be dropped before training.
- **Training entry point** (`scripts/train.py`) for both arms of the ablation —
  single-harness and multi-harness — sharing all code except the environment
  list, so the arms differ in exactly one thing.
- **Evaluation sweep** (`scripts/eval.py`) that runs the base model and both
  checkpoints in one process, against held-out tasks *and* the held-out harness,
  and reports the ablation with per-cell live counts.
- **Static guards**, wired into `pipeline.sh` stage 0/1 and into CI:
  - `scripts/guard_tool_surface.py` — every tool a harness advertises in its
    guidance is implemented, and every implemented tool is advertised.
  - `tests/test_path_errors.py` — bad path arguments produce the same
    actionable error on every OS, with no raw OS error and no escape from the
    workspace.
  - `tests/test_shell_timeout.py` — the shell runner cannot deadlock.
  - `tests/test_scan_tooling.py` — the analysis scripts' verdicts, pinned.
- **Two documented correctness fixes** carried in from the development tree,
  each with a regression test: the shell-runner deadlock, and the
  `workdir / ""` path trap.
- **Zero-dependency core.** Harnesses, tasks, the verifier, and all static guards
  import nothing outside the standard library, so CI runs real tests without
  installing torch. Enforced by the `core` job.
- Bilingual README (`README.md` / `README.zh-CN.md`) with a structural-parity
  guard (`tools/check_readme_i18n.py`).

### Fixed

- **Shell-runner deadlock, worth 84 minutes of idle GPU.** The runner used
  `subprocess.run(capture_output=True, timeout=...)`, which kills only the
  direct child when the timeout fires and then drains the pipes. A grandchild
  that inherited the write end keeps them open, so the drain never sees EOF and
  blocks forever — and `TimeoutExpired` is not even raised, because the timeout
  already fired. The symptom was "no output for 84 minutes" with the GPU at 0%.
  The runner now writes to a temporary file, closes stdin, and kills the process
  tree on timeout.

- **`Path(workdir) / ""` is the directory, not an error.** A model calling
  `write_file(path="", content=...)` or `replace_in_file(path=".")` attempted a
  write against the working directory itself, and the raw OS error leaked back
  to the model as the tool's result — `Permission denied` on Windows,
  `IsADirectoryError` on POSIX. Because the message is platform-dependent, it
  polluted the exact axis the experiment measures. All path-taking tools now
  validate through one shared resolver that rejects empty paths, `.`, directory
  targets, and `..` escapes with OS-independent, actionable messages.

- **POSIX absolute paths silently retargeted outside the workspace.** The same
  investigation surfaced a second case in the same class: the model emits paths
  like `/config.txt` or `/Users/qwen/Code/workspace/output.txt`. On Windows
  these resolved to a drive-rooted location outside the task workspace, so the
  write failed with an unrelated access error while the model was told nothing
  useful. The containment check now rejects them.

- **`errs_report.py` returned a false-green verdict.** Its first version
  reported "no harness defects" while 25–27 raw OS errors sat in the catch-all
  bucket. It now has an explicit OS-error class and returns a non-zero exit for
  a scan that is not fit to train on.

### Known limitations

- **The headline ablation is not yet run.** This release ships the
  infrastructure and the pre-registered protocol; the cross-harness gap numbers
  are pending, and the reason is measured rather than hypothetical — see the
  difficulty-gradient finding in the README.
- **52% of scanned training cells are dead.** Of 64 `(harness, task)` cells at
  n=8, 33 have pass rate ≤ 0.05 or ≥ 0.95, so they carry no gradient. The T2 tier
  is dead in all 16 cells and T3 in 13 of 16. The full ablation cannot produce a
  meaningful gap on the current task set until a T2/T3 difficulty gradient
  exists; the signal filter in `train.py` is therefore mandatory, not an
  optimization.
- **Only one base model and one seed.** Qwen2.5-0.5B-Instruct, chosen so the
  whole experiment fits on an 8 GB laptop GPU. Single-seed results are not
  evidence of reproducibility, and no seed sweep has been run.

[Unreleased]: https://github.com/possibleme2026-lang/rsi-multi-harness-rl/commits/main
