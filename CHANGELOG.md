# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Synthesised environment tasks** (`rsi/envgen.py`, `rsi/envtask.py`) — the
  environment axis. A task is a *path through a tool dependency graph*
  (`E = {F_exec, E_doc, Σ_tool}`, EnvScaler arXiv:2601.05808 §2) rather than a
  parameter vector, and the reward is the fraction of K checkpoints satisfied by
  the **final state**, so it is process-agnostic and carries gradient where a
  boolean does not. Four domains (inventory / tickets / accounts / sensors),
  `n_steps` counted in **mutations**, distractor injection so that "read the
  state" is a real requirement.
- **Harbor package export** (`rsi/harbor_export.py`) — a real `task.toml` +
  `instruction.md` + `environment/` + `solution/` + `tests/` package per task,
  with the verifier evaluator extracted from `harnesses.core` via
  `inspect.getsource` so the exported and in-process graders **cannot** drift.
- **Local Harbor runner** (`scripts/harbor_local_run.py`) — executes an exported
  package without Docker by running the oracle, handing `state.json` to a
  separate verifier directory, and grading both that and an untouched state.
  Verified: **30/30 packages OK**, `reward=1.0` on the oracle, `untouched_rc=1`.
  An unexecuted package is a claim; this is what makes it a benchmark.
- **Harness-pool adapter** (`rsi/env_adapter.py`) — makes an environment task
  runnable by the harness pool, with a declarative `env` template resolved at
  `reset` so the batch survives a JSON round trip and each rollout gets its own
  state file.
- **Environment batch dump** (`outputs/rsi/env_batch.json`) so the harness scan
  can run on **stateful** tasks rather than only on string tasks, plus
  `scripts/env_harness_smoke.py` (scripted-policy smoke, 12/12 cells) and
  `scripts/env_reward_ceiling.py` (proves the path reaches 1.0 and that partial
  credit is fractional).
- **Scan-shape analysis** (`scripts/env_scan_report.py`) — reads a scan artifact
  and classifies every rollout by the *shape* of the call it made
  (`no_call` / `bad_args` / `unknown_tool` / `query_only` / `wrong_target` /
  `mutated_ok`), so a pass-rate matrix of zeros can be explained rather than
  merely reported. It is what found the guidance defect above; a matrix of zeros
  looks identical whether the model never called a tool, called a non-existent
  one, or called the right one and stopped.
- **Batch producer** (`scripts/env_batch_make.py`) — regenerates the environment
  batch from a command. The batch that the first scan measured was produced by an
  ad-hoc command that was never written down, so once its producer changed the
  artifact could not be rebuilt; an artifact whose producer is a shell history
  entry is not evidence. Refuses to overwrite a batch whose ids differ.
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

### Measured — environment axis

Numbers, including the ones that are not flattering. Each was produced by a
script in `scripts/`, and each replaced a wrong number that had been reported
before it.

- **Discriminability, 30/30**: reference scores exactly `1.0`, initial state
  exactly `0.0`, distractor trace strictly lower and `<= 0.6`. Measured on the
  batch, not a fixture.
- **Harbor packages, 30/30 execute locally** with `reward=1.0` on the oracle and
  a non-zero exit for an untouched state. Before the tool table was completed,
  **0/30** executed: `unknown tool: restore` on every package.
- **`n_steps` ladder, 60 draws each**: `2 -> (chain 3, ref 2)`, `3 -> (4, 3)`,
  `4 -> (5, 4)`, `5 -> refused on 60/60 seeds`. The knob was wrong three separate
  times before this: chains padded with queries, chains padded with tools the
  reference refuses, and a ceiling that varied with an unrecorded draw.
- **Reward ceiling reached through the harness layer**: `0.00 -> 0.40 -> 1.00`
  as the reference is applied call by call, with a partial attempt scoring
  `0.40`. This is what licenses reading the scan below as a capability result
  rather than a plumbing failure.
- **Environment scan, 96 rollouts** (4 train harnesses x 12 tasks x 2), after the
  guidance fix: **tool-call rate `96.9%`**, multi-turn uptake `100%`, pass rate
  non-zero in two cells (`react_tools` scored `0.20` and `0.10` where the other
  three harnesses scored `0.00`), so `G2` and `G3` now pass and the probe returns
  **GO**. Before the fix the same scan was **`0.00` in every cell**.
- **The guidance fix, measured directly** (`scripts/guidance_shape_check.py`,
  same model, same tasks, one variable): the shape of the emitted call went
  **`as_tool` 83% -> 12%** and **`correct` 17% -> 88%**. This is the measurement
  that justifies the fix; the scan's pass rate alone would not, because a scan
  can move for reasons other than the change under test.
- **The `0.20` spread is NOT a measured cross-harness gap**
  (`scripts/env_spread_significance.py`). It comes from **2 rollouts out of 96**:
  `react_tools` at `1/2` on one task against `0/2` beside it. The Wilson interval
  on that cell is `[0.09, 0.91]`, and the minimum detectable effect at `n=2/arm`
  is `0.40` — larger than the observed `0.20`. So the design **cannot**
  distinguish "no gap" from "a gap this size". It clears the pooled noise floor
  (`0.0129`) and its interval excludes zero, which makes it a *lead*, not a
  finding. Detecting an effect of this size needs **9 rollouts per arm**, 4x the
  current `n`. Reported as a lead with its interval attached.
- **Environment scan, 96 rollouts** (4 train harnesses x 12 tasks x 2), before the
  guidance fix: **pass rate `0.00` in every cell**, tool-call rate `81.2%`,
  multi-turn uptake `100%`. Reported at the time as a **floor**, not a gap:
  `G3 cross-harness spread = 0.00`, so **no cross-harness gap is measurable on
  this batch at this model size** — a floor is uniform across harnesses and a gap
  requires a difference. That reading was correct as far as it went, and the
  floor turned out to be a defect rather than the model: see the guidance entries
  below. The artifact for this run was overwritten by the re-scan before it could
  be archived; the numbers above are the ones extracted from it at the time.

### Fixed — the defects behind those numbers

- **The environment scan's before-picture was overwritten by its own re-scan.**
  Both wrote `outputs/rsi/env_scan.json`, and `probe.py` flushes after every cell,
  so the artifact that was the only evidence of the guidance defect was replaced
  by the run meant to be compared against it. The numbers below survive because
  they were extracted while it existed; the raw records do not. `env_batch_make.py`
  now refuses to overwrite a batch whose ids differ, which is the same hazard one
  step upstream.
- **`env_scan_report`'s `mutated_ok` bucket was documented backwards** — as
  "mutated and still scored 0", when it means "mutated and was paid". The
  mislabelled reading sends you to audit the checkpoints at exactly the moment
  the checkpoints have just been vindicated. It now also states the inference the
  bucket licenses: a non-empty `mutated_ok` means the reward path works end to
  end, so the zeros beside it are a capability floor rather than a plumbing
  failure.
- **`guidance_shape_check`'s first version measured itself.** It decoded the
  completion with `skip_special_tokens=True` — stripping the `<tool_call>`
  delimiters — and matched with its own regex, while the rollout loop parses at
  the token level through `parse_response`. Both arms reported `100% no_call`
  against a scan showing `100%` tool calls. It now routes through `agent._parse`,
  the same entry point the loop uses, so it cannot pass while the real path
  behaves differently.
- **Every harness told the agent to write `answer.txt`.** Each harness appends a
  static `GUIDANCE` block written for string tasks; for a stateful task it names
  the wrong artifact (the verifier reads `state.json`) and never mentions
  `envtool`, so the agent's only route to the environment was guessing tool
  names. Measured: **96/96 rollouts scored 0.00** and tool-call rate was 35.4%.
  Fixed by a per-task `guidance` override carrying the tool table, which
  `core._instruction` prefers over the harness's static text.
- **The first fix to that guidance did not fix it, and the metric it was
  validated on moved the wrong way.** The override rendered the environment's
  commands as a bare list (`envtool list_tickets`, `envtool set_state <id>
  <value>`) — which reads as a *tool list*, and the prompt already contains one,
  because the chat template renders the harness's own tools as a schema block
  above the instruction. Given two lists of that shape the model merged them and
  called `envtool list_tickets` **as a tool name**:

  ```
  envtool list_tickets({'query': 'state=open'})
  -> Tool envtool list_tickets not found. Available: ['bash']
  ```

  Measured across the pool: **70% of all 96 rollouts** were that one shape, and
  the model invented `list_accounts` / `list_items` on environments that have no
  such command — it was answering the shape of the prompt, not its content. The
  tool-call rate rose 35.4% → **81.2%** while the pass rate stayed at exactly
  `0.00`, because **a call to a non-existent tool is still a tool call**. The
  gate that was supposed to detect "the model can engage" was satisfied by the
  failure mode itself. The guidance now introduces the commands as *shell command
  lines*, names the shell tool they are reached through, and shows the wrong
  shape explicitly; `scripts/env_scan_report.py` classifies the shape of every
  call so a rise in the call rate can no longer be read as progress on its own.
- **The exported `tools.py` was missing two tools** (`restore`, `pin`) that the
  in-process grader implements, so the exported environment and the solution
  driving it had drifted — the two-implementations hazard the module docstring
  names, and invisible to any test that compares scores, since both sides keep
  returning plausible numbers. Caught by the local runner, not the suite.
- **A callable in the task dict made the batch undumpable**
  (`TypeError: Object of type function is not JSON serializable`), so the
  environment batch could not be written and the gap could only ever be measured
  on string tasks. Replaced by a declarative `env` template.
- **The tool set was drawn per seed**, so `n_steps`' *legality* depended on an
  unrecorded draw — `n_steps=5` was refused for "3 usable mutators" on 27 of 60
  seeds and "4 usable mutators" on 33. The set is now fixed per domain.
- **`reset_<f2>` could not raise the chain ceiling** because it writes a field
  `set_<f2>` already owns, and two steps writing one field is a redundant step.
  Replaced by `pin`, which writes `_pin`, a field nothing else reaches.
- **`_touched_fields` was referenced before it existed** (module-level
  `NameError`), and its first version's stem split made `count_by_sku` look
  unimplemented on a package that implements it.

### Notes

- `scripts/_probe_steps.py` is a throwaway diagnostic kept in-tree because the
  `n_steps` ladder is the measurement that caught three of the bugs above; it is
  not part of the pipeline.
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
  - `curriculum.py` — **the call site `band.steer` never had.** GenEnv's
    α-Curriculum Reward `exp(−β(p̂−α)²)` with `α` *derived* from this repo's own
    GRPO signal curve rather than copied, a batch-scope difficulty filter, and a
    regeneration plan that is auditable without re-reading the scan.
  - `ledger.py` — append-only JSONL evidence log, annealed edit budget, prune,
    stall detection.
- **`scripts/rsi_loop.py`** — runs both axes end to end with no model required,
  so the loop is exercisable on CPU.
- **A curriculum stage in the loop, and the two bugs it exposed.** `band.steer`
  documented itself as "consumed by the task generator" and the README said it
  "already returns the override", while a grep found only its definition and its
  tests — no pipeline script ever called it. Wiring it up surfaced:
  - **The difficulty filter disabled the steering.** GenEnv's `|p̂ − α| > k_min`
    was applied *per task*; `mastered` is `p > 0.9` and `out_of_reach` is
    `p < 0.1`, so every steerable cell sits ≥ 0.4 from `α = 0.5` — outside a 0.1
    band. The filter rejected exactly the cells the rule existed to move, and the
    first live run reported `moves: 0`. The rule now applies at batch scope, where
    GenEnv wrote it. The arithmetic is pinned in the test.
  - **A mismatched task-id set produced silently wrong overrides.** The shipped
    scan measures the 24-task suite (`t1-01`); a generated batch carries hashed
    ids (`t1-8f87ad9e`); the sets do not intersect. `steer` needs a task's
    parameters, so a missing lookup fell back to defaults and emitted an override
    for a task that does not exist — with nothing in the output looking wrong.
    Now an explicit refusal, with the id overlap reported as a count.
  - `scripts/probe.py --from-batch` and `rsi_loop.py`'s batch dump close the
    loop: a generated batch can be written, registered, and scanned, so the
    curriculum can steer the tasks it actually measured.
  - **Steering needs `n = 64` per cell to become possible at all.** At `n = 32`
    an all-pass cell has a Wilson lower bound of 0.8928, below `MASTERED_ABOVE =
    0.9`, so it is `unresolved` and no move is produced however clear the point
    estimate looks. A fact about the cost of the loop, pinned in the test.
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
  (`wilson_interval(0, 73)[1] = 0.04999`; `n = 72` gives 0.0507). The gap matters:
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
  Measured over 300 tasks at seed 0: **96 (32%) were affected** — 79–99 across
  eight seeds, so the seed is quoted; the `--seed 11` default yields 91. Now 0,
  and an explicit long-answer `python_exit` is still honoured.

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
  `n = 128` per arm the minimum detectable effect is `0.1581`; 80% power at the
  observed effect needs ~2,094 rollouts per arm, 16× this run. The pre-registered
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
- **The harness axis has been scored in both modes, and the synthetic one was
  wrong.** The artifact records the mode under `score_mode`, and the README says
  so beside the numbers rather than in a footnote. Replayed scoring exists so the
  loop's budget, guard, noise floor, ledger and prune rule can all be exercised
  on a machine with no GPU — which is the machine CI runs on — but its trajectory
  is synthetic and must not be read as "the harness got better". Running the
  `rollout` arm showed that the two constants the replayed mode stood in with
  were both unmeasured and both wrong, in opposite directions: a base rate of
  `0.25` for every harness against measured baselines of `0.008 / 0.054 /
  0.029 / 0.029`, and a floor of `0.05` against a bootstrapped `0.0176`. So the
  replayed run started ~8× too high and demanded a ~3× too large margin. The
  measured arm is still running; until it completes, the honest statement is that
  the harness search *works*, not that it *helped* — and the previously reported
  replayed trajectory of `0.471 → 0.528` describes the stand-in function rather
  than a harness.

[Unreleased]: https://github.com/possibleme2026-lang/rsi-multi-harness-rl/commits/main
