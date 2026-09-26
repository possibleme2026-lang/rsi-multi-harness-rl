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

- **The RSI loop was open, and every one of its parts passed its tests.** Three
  defects, all of the same shape — a mechanism that existed, was tested, and was
  never reached:
  1. **The curriculum steered a scan of a different task set.** `pipeline.sh`
     generated a batch and then handed `rsi_loop.py` `scan_all.json`, which
     measures the shipped 16 ids. The batch's ids are generated hashes, so the
     sets are disjoint and the plan was empty on every run — reported correctly
     as `move_count: 0, skipped: "the scan measures 16 task ids, none of which
     are in the batch of 2"`. The correct two-command workaround was written
     into a comment and never executed. Fixed by ordering: `rsi_loop.py
     --batch-only` writes the batch first, `pipeline.sh` scans *that* batch
     (`probe.py --from-batch`), and the curriculum and the trainer both read the
     resulting `rsi/scan_batch.json`. An id mismatch is now fatal rather than a
     silent zero-move plan.
  2. **The trainer could not see a generated task at all.** `train.py` built its
     rows from a hardcoded `TRAIN_TASK_IDS`, so no generated task could reach the
     gradient even once a scan covered it. It now takes `--batch`
     (`rsi/batch.json` and `rsi/env_batch.json` are different task shapes and both
     work) and registers the batch into the process-global registry that
     `Agent.run` resolves ids through — without which a generated id raises
     `KeyError` *after* the model has loaded. `--require-signal` refuses a scan
     that measures none of the rows; `--dry-run` stops before the model loads so
     the wiring is checkable without a GPU.
  3. **The consequence was in the training curve.** All four arms train on the
     same sixteen frozen ids, but `multi` builds one row per (harness, task) pair
     and `single` only one per task, so at a fixed step count the arms see very
     different epoch counts. The logged `epoch` field:

     | arm | rows | steps | epochs | zero-gradient |
     | --- | --- | --- | --- | --- |
     | `multi-s64` | 64 | 64 | 2.00 | 17/64 = 26.6% |
     | `multi-s128` | 64 | 128 | 4.00 | 43/128 = 33.6% |
     | `single-s64` | 16 | 64 | 8.00 | 26/64 = 40.6% |
     | `single-s128` | 16 | 128 | 16.00 | 67/128 = 52.3% |

     Monotone: more epochs over the same sixteen ids means more GRPO groups with
     no within-group variance, hence no gradient. The functional form is *not*
     resolvable from these four points — linear (R²=0.975) and log₂ (R²=0.982)
     both fit, because the epoch values are consecutive powers of two. An earlier
     revision of this entry said "both 128-step arms trained on 64 rows ... 4.0
     epochs", which was wrong for `single` (16 rows, 16.0 epochs).

     What is *not* claimed: 16 epochs of a 0.5B model is not obviously enough to
     have saturated, so this shows the configuration stopped improving, not that
     the task set is its only cause. What the fix does establish is that the
     gradient never saw a generated task — a defect regardless of the plateau.
     `tests/test_rsi_closed_loop.py` pins all three properties: generation before
     the scan, the scan of that batch, and both consumers reading it.
- **The loop was reachable, reached, and still inert — the last hop was missing.**
  Fixing the ordering above made the *scan* measure the batch, so a generated
  task could reach the gradient. A **steered** task still could not, because
  `rsi_loop.py` wrote `execute_plan`'s output into `curriculum.json` and nothing
  read it back:

  ```
  plan_regeneration -> moves -> execute_plan -> fresh tasks -> curriculum.json
                                                                    ^ end of the line
  ```

  `batch.json` was never rewritten, so the training arms read the pre-steer
  batch. A run could move every task in the batch, pass all four gates on every
  replacement, and not change a single gradient. The artifact said so in its own
  words — `measured_effect: "Whether it moved toward alpha needs a rescan of
  these tasks"` — and that rescan was never run, because no script knew the
  steered batch existed. This is the third instance of the same defect (mechanism
  exists, tests pass, never reached) after the missing `steer` call site and the
  trainer's hardcoded task list.

  Fixed in three parts:
  1. `curriculum.steered_batch(plan, batch)` returns the batch with moved tasks
     replaced and the rest kept — length, order and kept ids all preserved, so a
     task on the frontier keeps its id and therefore its measurement.
     `execute_plan` is untouched: one function answers "what did the curriculum
     produce", the other "what does the next round consume".
  2. `rsi_loop.py --apply-curriculum` writes it to `rsi/batch_steered.json`,
     unconditionally, including for a zero-move plan, so downstream has one path
     rather than two.
  3. `pipeline.sh` scans that file and hands it to `train.py`. The rescan is
     skipped when nothing moved, and that decision is read from the artifact
     (`curriculum.json → steered.moved`), not inferred from a log line.

  `scripts/loop_report.py` reports the before/after alignment — the only number
  the task axis produces that **could come out negative**, since every other
  check is a pass/fail on well-formedness and a suite of pass/fail checks cannot
  detect a curriculum that moves tasks the wrong way. It exits non-zero when the
  delta is negative.

  Also fixed alongside it: **a zero-move plan had three causes and only one was
  visible.** `move_count: 0` now carries a top-level `move_diagnosis` —
  `ids_mismatch` (scan the batch you intend to steer), `all_unresolved` (scan
  more; this is about the measurement), or `all_frontier` (nothing to do, the
  batch is on target). The middle case is the one that used to hide, and it is a
  live risk: a band is assigned only when the whole Wilson interval falls inside
  it, and the out-of-reach boundary is `0.1`, so a zero-pass cell needs **n ≥ 35**
  before it can be placed at all (`hi = z²/(n+z²)`; `2/64 → 0.0955` places,
  `3/64 → 0.1182` does not). `N_SCAN` therefore defaults to 64, not 8, and
  `pipeline.sh` warns when it is set below the threshold. `APPLY_CURRICULUM` now
  defaults to 1 — a curriculum that is planned but never applied cannot move a
  gradient.

  `tests/test_rsi_steered_batch.py` pins all of it, including the sign of the
  report's delta under a swapped before/after. `test_rsi_closed_loop.py` gained
  the pipeline-ordering assertions for the new hop.
- **The first measured closed loop found a defect in the report that measured it
  — the treatment and the control were pooled.** The pooled delta came back
  negative (`-0.0031`) and the script's verdict was that the steering rule moved
  tasks the wrong way. Decomposed per task, every one of the 8 replaced tasks had
  a non-negative delta (`+0.0033` on three, exactly `0` on five); the whole
  negative came from three tasks **nobody touched** (`58/256 → 54/256` on
  `t1-9ea9b3bc`, and two smaller drifts). The kept tasks are the *same task
  measured twice*, so they are a free control group and their movement is the
  noise floor at `n=64` — pooling it into the headline let sampling noise decide
  the sign of the one number the script reports as a defect.

  `loop_report.py` now reports three numbers and rests its verdict and its exit
  code on the treatment: `alignment_delta` (pooled, kept for continuity with the
  printed table), `alignment_delta_moved` (**the treatment effect**), and
  `alignment_delta_kept` (the control — the scale the treatment must clear, never
  a result). A sign disagreement between pooled and treatment is now stated
  explicitly rather than silently resolved.

  The honest reading of the first closed loop is therefore **direction right,
  step too small**: all 8 moves went `out_of_reach → easier`, but the batch did
  not move (`out_of_reach 11 → 11`, `frontier 1 → 1`) and the treatment effect
  (`+0.0012`) is an order of magnitude *below* the control noise (`-0.0119`), so
  the rule is not distinguishable from noise at this `n`. `band.steer` halves
  `payload_len`, drops `escape_density` by `0.2` and `steps` by one, which does
  not lift a task off a floor where the 0.5B policy scores 0/256 at the *easiest*
  parameter setting. That is a statement about the step size, not the direction.
- **`band.steer` left the largest difficulty lever untouched.** The
  "make this easier" rule moved `payload_len`, `escape_density`, `steps` and
  `read_source` but not `verify_mode` — and `python_exit` carries a full `1.0` of
  the five equal parts in `TaskParams.difficulty`, more than any other knob. The
  `out_of_reach` override now sets `verify_mode: file_equals`. It is deliberately
  *not* raised in the harder direction: that would change the reward axis rather
  than only difficulty, and the behaviour of that shape has not been measured.
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

- The `n_steps` ladder was the measurement that caught three of the bugs above.
  It lived in `scripts/_probe_steps.py`, a throwaway diagnostic that is no longer
  in-tree — the reference was left behind when the file was removed, so this
  paragraph used to point at a path that does not exist.
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
  measured arm has now completed, and it **inverts** the replayed one: 24 edits
  attempted, **1** accepted (4%), against the replayed 30/10. The single accepted
  edit is `context_mgmt+=keep_last_error` on `react_tools`, scoring
  `0.0542 → 0.0917` (`d=+0.0375`) — the exact component the replayed run pruned
  for zero yield — and its yield per component is `context_mgmt 0.25` with
  everything else `0.0`. The prune set is `prompt` + `output_plumbing`, not
  `context_mgmt`. Of the 23 rejections, 16 were `rejected_worse` and 7
  `rejected_within_noise`, a distinction the replayed mode cannot make because
  its floor was a constant. The honest statement remains that the harness search
  *works* on real scores, not that it *helped*: the trajectory is flat
  (`[0.0917, 0.0917, 0.0917]`), one accepted edit that was never built on, and
  the run was 3 rounds against the replayed run's 6 — so the comparable quantity
  is the accept *rate*, which fell ~8×. The previously reported replayed
  trajectory of `0.471 → 0.528` describes the stand-in function rather than a
  harness.
- **The headline ablation metric is an identity, and the run used to report it
  backwards.** `gap = mean(train) − mean(held-out)`, and the only held-out
  harness scored **0/96** pooled across all three arms (95% CI `[0, 0.0385]`,
  `DEAD` rather than under-measured). With the held-out term at 0, `gap` *is*
  `mean(train)`, so ranking arms by gap ranked them by train mean — and picked
  the arm that improved least, printing "hypothesis NOT supported" for an arm
  the same run called "overfitting the train harnesses". `eval.py` now refuses
  that comparison when the held-out mean is exactly 0.0 and prints the identity
  plus a one-sided bound instead (`gap ≥ 0.2271` single, `≥ 0.2662` multi):
  real and large, but identical to the train term and therefore silent about
  generalization. New `scripts/harness_solvability.py` rules out the harness as
  the cause — the reference solution reaches 1.0 on **120/120** (5 harnesses ×
  24 tasks), `codex_style` included, via its own `apply_patch` — and
  `scripts/diag.py` shows the model emits no tool calls at all. The floor is the
  policy's, not the harness's.
- **`rollouts_for_dead` added to `rsi/stats.py`.** The DEAD boundary (n ≥ 73 at
  zero passes, n ≥ 110 at one) previously existed only as prose in a docstring —
  which is where the incorrect "roughly n ≥ 128" lived. It is now a function,
  with assertions pinning it to the boundary `classify_cell` actually applies.
- **`pool.py` no longer misattributes `codex_style`'s zeros.** It credited the
  `apply_patch` heredoc bug. That bug is real, but it was not the cause:
  `eval_ablation.json` and `eval_ablation_pre_harness_fix.json` are
  byte-identical (`md5 38f7a0dacbeb9eef8e74480608bd7dc1`), so the fix moved
  nothing. A plausible cause near the symptom is not a cause.

[Unreleased]: https://github.com/possibleme2026-lang/rsi-multi-harness-rl/commits/main
