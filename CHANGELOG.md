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

### Added — the RSI layer

- **`src/multiharness/rsi/`**, a dependency-free layer that constructs the three
  things this experiment otherwise assumes are given:
  - `task_gen.py` — **axis 1: the task is generated.** Difficulty is a parameter
    vector (`payload_len`, `escape_density`, `steps`, `read_source`), and the
    generator samples it inside the band the steering signal asks for, rather
    than enumerating a fixed list.
  - `verifier_gen.py` — **axis 2: the reward is generated.** Three modes
    (`file_equals`, `file_contains`, `python_exit`), each with a discriminating
    self-proof, and a `recommend_mode` rule that refuses combinations which would
    ship a brute-forceable answer to the agent.
  - `harness_evolve.py` — **axis 3: the harness is evolved**, over a vocabulary
    of `prompt` / `client_tool` / `output_plumbing` / `context_mgmt`, against a
    measured noise floor, with the held-out harness frozen.
  - `validate.py` — gates V1–V4 (oracle, nop, schema/safety, cross-harness).
  - `stats.py` — Wilson intervals, exact rule-of-three, GRPO group-signal
    probability, bootstrap noise floor, and the three-way cell verdict.
  - `band.py` — regret banding (`mastered` / `frontier` / `out_of_reach` /
    `unresolved`) with the steering signal.
  - `ledger.py` — append-only JSONL evidence log, annealed edit budget, prune,
    stall detection.
- **`scripts/rsi_loop.py`** — runs both axes end to end with no model required,
  so the loop is exercisable on CPU.
- **`tools/plot.py`** — every figure regenerated from a recorded artifact, and
  **`tools/check_figures.py`**, which validates the committed PNGs by decoding
  IDAT with nothing but `zlib`. A plotting bug that writes a blank canvas still
  produces a valid PNG of plausible size, so the check is on pixels.
- **`figures` CI job**, separate from `core`, because plotting needs matplotlib
  and the core job's whole value is proving the harness layer needs nothing.

### Fixed — the statistics and generator defects

- **A harness defect that invalidated an entire eval.** `codex_style`'s
  `apply_patch` fallback appended `|| echo '...'` after a heredoc terminator,
  which makes the whole shell command a syntax error — so any unified diff that
  reached the shell was rejected and never applied. The harness's `*** Add File`
  shortcut returns before the shell is reached, so casual testing never hit it:
  the tool was advertised, registered, reachable, and never worked. Fixed by
  grouping the heredoc-fed command in `{ ... }` and reading the exit status from
  `_run_shell_rc` instead of parsing prose (the same mistake `python_exit` made);
  a regression test asserts a valid diff is applied, the file appears, and the
  patched answer scores 1.0.

  **It was not, however, why `codex_style` scored zero, and this entry's first
  draft claimed it was.** Re-running the eval *after* the fix reproduced
  `codex_style` at 0/8 in all three arms including the untrained baseline, which
  rules the syntax error out as the cause. `diag.py` shows the model emits a
  well-formed `apply_patch` call whose `patch` argument is the bare answer string
  `"reward is one"` rather than a unified diff; `patch` correctly answers
  `Only garbage was found in the patch input.` Held fixed, `*** Add File: …` and
  a real unified diff both score 1.0, the bare string scores 0.0. So the held-out
  harness is a **model** limitation, not a plumbing one — and it is still a floor
  effect, because the baseline arm cannot score on it either. That is what makes
  the headline gap metric untestable, and it is a separate finding from the bug.

- **A `DEAD` threshold that was a guess.** `stats.py` said excluding the signal
  band at zero passes takes "roughly `n >= 128`". The true boundary is **73**
  (`wilson_interval(0, 73)[1] = 0.04999`; `n = 72` gives 0.0506). The gap matters:
  73 is a laptop-sized scan, 128 is where you stop and redesign. The docstring
  now states both boundaries and tests pin them (`0/73` DEAD, `0/72` not,
  `1/110` DEAD, `1/109` not).

- **The training filter dropped rows on a point estimate.** `train.py` discarded
  every row with `p <= 0.05 or p >= 0.95` — the retracted rule, still live in the
  entry point after the README had stopped using it. A run was observed printing
  `dropped as dead: 8 rows` before being killed and rewired. It now asks
  `rsi_stats.classify_cell` for a verdict, drops a row only when the interval
  excludes the signal band, and reports under-measured rows as *kept*. Verified:
  at n=8, 28/28 rows kept with 0 dead; at n=32, 0 dead.

- **A generated reward that shipped a brute-forceable digest.** `verify_mode` was
  drawn independently of the payload, so the generator produced `python_exit` on
  three-character answers — the one combination `verifier_gen`'s own docstring
  names as wrong, because `check_script` lands inside the agent's work directory.
  Measured over 300 tasks: **96 (32%) were affected**. Now 0, and an explicit
  long-answer `python_exit` is still honoured.

- **`summarise_batch` reported the wrong mode counts.** Exposed by the fix above:
  it counted `params["verify_mode"]` (requested) rather than what the tasks use,
  so after the fix the coverage report claimed 8 `python_exit` tasks when 1
  existed. It now reports both, with `mode_overrides` explaining the gap.

- **`propose_edits` did not enforce freezing itself.** Only the calling script's
  loop excluded the held-out harnesses, so a caller that forgot would quietly
  turn the held-out result into a training result — and nothing in the output
  would look wrong. Surfaced by an `F841` finding; now enforced in the function.

- **`fig12_training` read a hardcoded, stale artifact path.** It loaded
  `outputs/train-verify/train_summary.json` — an artifact of a one-off
  correctness run — while `pipeline.sh` writes `outputs/train-single-s<N>/` and
  `train-multi-s<N>/`. The figure therefore drew a *different run than the README
  described*, and a training curve is a training curve, so nothing looked wrong.
  It now discovers every `outputs/train-*/train_summary.json`, plots the longest
  as the headline and overlays the arms, so the ablation is visible in the figure
  that is supposed to show it.

- **`rsi_loop.py` ignored `MULTIHARNESS_OUT`.** `probe.py`, `train.py` and
  `eval.py` all resolve their artifact directory through
  `_bootstrap.outputs_root()`, whose docstring calls the override the whole point
  of the helper — but `rsi_loop.py` hardcoded `ROOT / "outputs" / "rsi"`. A
  redirected pipeline therefore wrote its scan, checkpoints and eval to one place
  and the RSI loop's artifacts to another, so the documented override applied to
  only part of a run. Now routed through `outputs_root() / "rsi"`, and the
  "wrote …" lines print the real path instead of a hardcoded one. This matters
  beyond tidiness: it is what lets the `--score rollout` arm write beside the
  replayed one instead of overwriting it.

- **`--score rollout` died on its first rollout.** `Agent.run` resolves a
  `task_id` through `core.get_task`, which reads a process-global registry that
  only `tasks.suite.load()` populates — with the *shipped* 24 tasks. The
  generated batch exists solely in a local list, so the honest scoring mode
  raised `KeyError: unknown task_id 't4-c1ff0a35'` before a single rollout ran.
  Nothing caught it because the path needs a GPU and the CPU test suite only ever
  exercised the replayed mode. Fixed by loading the suite and registering the
  batch; the regression test asserts both, without a model. **This is the failure
  mode the repository keeps finding: the mode that produces the honest number was
  the mode that had never been executed.**

- **A ruff/gitignore coupling.** Ruff skips files listed in `.gitignore`, so
  `outputs/` was lint-exempt by accident — the moment a run was redirected with
  `MULTIHARNESS_OUT`, that directory was no longer gitignored and ruff began
  linting the `check.py` files the *task generator* writes, reporting 12 errors
  in generated code nobody should edit. `exclude = ["outputs", "outputs_*", …]`
  is what was always meant.

- **`rule_of_three_upper` used the wrong base.** It computed
  `1 − confidence^(1/n)`, returning 0.0064 at `n=8` — 49× too small — while its
  own docstring said 0.312. The correct base is `1 − confidence`.

- **`classify_cell` would have called `0/32` live.** A `(hi − lo) <= 0.25`
  shortcut accepted any narrow interval, including `[0, 0.107]` — narrow, and
  entirely outside the signal band. Replaced with containment **and** a width cap
  of half the band.

- **The Wilson shrinkage constant was wrong in three places** (a test comment and
  both READMEs): `0.0545` instead of `0.1622`. Corrected, with a pinning
  assertion so it cannot drift again.

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

- **The headline ablation ran, and it did not settle the question it was built
  for.** Three arms, one seed, 160 rollouts each: baseline `0.1406`, single
  `0.2656`, multi `0.3047` on the train harnesses, with the held-out harness at
  `0.0000` in all three. Training beats the baseline on the harnesses it sees
  (`z = +2.49` and `+3.15` — real), and multi beats single by `d = +0.0391`,
  `z = +0.69` — the predicted direction, indistinguishable from zero. At
  `n = 128` per arm the minimum detectable effect is `0.1572`; 80% power at the
  observed effect needs ~2,096 rollouts per arm, 16× this run. The pre-registered
  hypothesis is **not supported and not refuted**.
- **The held-out axis is a floor effect, so the gap metric could not be tested.**
  `codex_style` scores 0/32 on the *untrained baseline*, so the held-out term is
  pinned at 0 and `d_gap ≡ d_train` exactly — the gap difference carries no
  information the train difference does not. Cause is the model, not the harness:
  it emits a well-formed `apply_patch` call whose `patch` argument is the bare
  answer string rather than a unified diff, which `patch` correctly rejects.
  Holding everything else fixed, `*** Add File: …` and a real diff both score 1.0.
  The fix is a second held-out harness the base model can already drive.
- **No cell can be shown to be dead, which is the opposite of what an earlier
  draft of this file claimed.** That draft reported "52% of scanned training
  cells are dead" from an n=8 scan. Re-running the same 64 cells at n=32 gives
  `live=31, dead=0, under_measured=33`: the earlier figure was 33 cells discarded
  by a point-estimate filter, and a discarded cell is not a dead cell. The
  arithmetic is in the README; the short version is that `p = 0.05` with 8
  generations still carries gradient in 33.7% of groups, and only `p = 0` and
  `p = 1` are provably dead.
- **T2 is a real capability failure and is *not* provably dead.** All four
  harnesses score 0/32 on all four T2 tasks (15 of 16 cells exactly zero, the
  sixteenth at 1/32), so the 0.5B model cannot complete the two-step
  read-then-extract pipeline. But `0/32` has a 95% interval of `[0, 0.107]`, and
  a cell whose true rate is 0.10 still carries signal in 57% of GRPO groups — so
  the honest verdict is "cannot do it reliably", not "carries no gradient".
  Fixing it is the task generator's job, not a hand-written easier tier.
- **Only one base model and one seed.** Qwen2.5-0.5B-Instruct, chosen so the
  whole experiment fits on an 8 GB laptop GPU. Single-seed results are not
  evidence of reproducibility, and no seed sweep has been run.
- **The harness axis has only been scored in `ledger-replay` mode.** The
  artifact records this under `score_mode`, and the README says so beside the
  numbers rather than in a footnote. Replayed scoring exists so the loop's
  budget, guard, noise floor, ledger and prune rule can all be exercised on a
  machine with no GPU — which is the machine CI runs on — but its trajectory is
  synthetic and must not be read as "the harness got better". The `--score
  rollout` arm, which rolls the model out against each candidate, has not been
  run; until it is, the honest statement is that the harness search *works*, not
  that it *helped*.

[Unreleased]: https://github.com/possibleme2026-lang/rsi-multi-harness-rl/commits/main
