# rsi-multi-harness-rl

**Cross-harness generalization for agentic RL.**

Train one policy across several agent harnesses, then measure how much of it
transfers to a harness it has never seen.

> **Status: infrastructure complete, headline experiment not yet run.**
> This repository ships a verified measurement apparatus and a pre-registered
> protocol. The cross-harness gap numbers are **pending**, and the reason is
> measured rather than hypothetical — 52% of the scanned training cells carry no
> gradient. See [Results](#results).

[中文说明](README.zh-CN.md)

---

## The question

An agentic policy is trained inside a *harness*: a system prompt, a tool set, and
a submit protocol. The harness is normally treated as a product decision made
after training — you train the model, then wrap it in Claude Code, or Codex, or
your own scaffold.

That framing hides a failure mode. If a policy is only ever trained inside one
harness, it can satisfy the reward by memorizing that harness's interface rather
than by understanding the task. The reward goes up; the capability does not
transfer. Then the scaffold changes and the policy has to be retrained.

Xiaomi's MiMo-V2.6 technical report makes this argument directly. §4.2.5 trains
across multiple harnesses, on the reasoning that open-source users build their
own harnesses rather than converging on one, and the release notes state that
multi-harness training improves *"the model's generalization ability across
different frameworks, including unseen ones."* Reported effect: replacing the
harness with ones never seen in training (Codex, Claude Code, mini-swe-agent)
moved average pass rate from roughly 50% to 66%.

This repository is a small, fully reproducible instance of that experiment. One
laptop GPU, a 0.5B model, a task suite small enough to inspect by hand — and the
measurement done properly.

## The measurement

The headline number is a difference of differences:

```
gap = mean(reward | train harnesses) − mean(reward | held-out harness)
```

Read that as an absolute level it is worthless: a base model already has a gap.
So the repository measures three arms with one process, one seed, and shared
harness instances:

| arm | trained on | what it tells you |
| --- | --- | --- |
| baseline | nothing | the gap the base model already has |
| single | one harness | what overfitting one scaffold costs |
| multi | four harnesses | whether mixing them shrinks the held-out gap |

The ablation is `single` vs `multi`, both read against `baseline`. A raw gap
number without the baseline arm is a level, not a result.

Two axes are held out, not one. Splitting only the harness would let the model
memorize the tasks and still look like it generalized; splitting only the tasks
would leave the harness axis untested. So the task suite is split 16/8 and
`codex_style` never appears in training at all.

## Design commitments

Four decisions do the actual work here. Each one exists because the naive version
produced a wrong answer.

**The verifier is harness-agnostic.** Grading reads only `<workdir>/answer.txt`.
The grader cannot tell which scaffold produced an answer, so a difference in score
cannot be an artifact of grading. The harnesses differ in *how* an answer is
submitted — `react_tools` has a `finish` tool, `json_strict` has `submit` — and a
model that only learned "write the file with bash" cannot score on those without
adapting. That is the signal under test.

**Harnesses differ structurally, not cosmetically.** Five harnesses, differing in
tool set and submit protocol:

| harness | tools | submit | role |
| --- | --- | --- | --- |
| `bash_minimal` | `bash` | write the file | train |
| `react_tools` | `bash`, `read_file`, `write_file`, `finish` | `finish()` tool | train |
| `json_strict` | `bash`, `submit` | `submit()` tool | train |
| `longctx_summary` | `bash`, `read_file`, `replace_in_file` | write the file | train |
| `codex_style` | `bash`, `apply_patch` | write the file | **held out** |

**Dead cells are measured, not trained on.** A binary reward has no gradient when
every rollout in a group scores the same. With `G` generations and pass
probability `p`, the probability that a group carries no signal is
`p^G + (1−p)^G` — so cells with `p ≤ 0.05` or `p ≥ 0.95` are provably wasted.
The pipeline measures every cell with a difficulty scan and drops the dead ones
before training. This is a prerequisite, not an optimization.

**A capability probe gates the GPU spend.** Before training, four gates establish
that the base model can drive the harness pool at all. If it cannot, any gap
measured afterwards is noise, and the honest output is "no result" rather than a
number.

| gate | threshold | measured | verdict |
| --- | --- | --- | --- |
| G1 tool-call rate | ≥ 50% | **76.6%** | pass |
| G2 reachable cells (pass@8) | ≥ 1 | **32 of 64** | pass |
| G3 cross-harness spread | > 0 | **0.75** | pass |
| G4 multi-turn uptake | ≥ 50% | **100%** of 392 tool-calling rollouts | pass |

**Go/No-Go: GO.** Qwen2.5-0.5B-Instruct can drive all five harnesses. Mean turns
1.77.

## Results

**The headline ablation is not run yet.** What is measured, at n=8 over 4 train
harnesses × 16 train tasks:

Pass-rate matrix (rows = harness, cols = task):

| task | `bash_minimal` | `react_tools` | `json_strict` | `longctx_summary` |
| --- | --- | --- | --- | --- |
| t1-01 | 0.875 | 0.375 | 0.750 | 0.125 |
| t1-02 | 0.500 | 0.375 | 0.125 | 0.000 |
| t1-03 | 0.375 | 0.625 | 0.250 | 0.250 |
| t1-04 | 0.125 | 0.250 | 0.250 | 0.125 |
| t1-05 | 0.750 | 0.625 | 0.250 | 0.250 |
| t1-06 | 0.500 | 0.375 | 1.000 | 0.250 |
| t1-07 | 0.875 | 0.375 | 0.500 | 0.125 |
| t1-08 | 0.250 | 0.000 | 0.625 | 0.000 |
| t2-01 … t2-04 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-01 | 0.000 | 0.250 | 0.000 | 0.250 |
| t3-02 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-03 | 0.000 | 0.000 | 0.000 | 0.000 |
| t3-04 | 0.000 | 0.125 | 0.000 | 0.000 |

**33 of 64 cells (52%) are dead** — pass rate ≤ 0.05 or ≥ 0.95, therefore no
gradient. The distribution is the finding, and it is worse than a single number
suggests:

| tier | live cells | dead cells |
| --- | --- | --- |
| T1 copy-through | 28 | 4 |
| T2 read-then-write | **0** | **16** |
| T3 quoting-stress | 3 | 13 |

T2 is dead in **all 16 cells**: the 0.5B model cannot complete a two-step
read-then-extract pipeline reliably enough to ever pass, so no T2 cell can
produce a gradient. T3 is dead in 13 of 16 — the tier explicitly designed to be
the load-bearing one, where harnesses genuinely diverge on escaping, is almost
entirely unmeasurable at this model scale.

**Consequence, stated plainly.** Running the full ablation on the current task
set would produce a headline dominated by structural zeros: the eval split's T2
and T3 cells would be all-zero for every arm, and the "gap" would be mostly an
artifact of which tier a task came from. A difficulty gradient for T2/T3 must
exist before the cross-harness gap means anything. That is the next piece of
work, and the reason this release is labelled infrastructure rather than results.

## Reproducing

Requires Python 3.12+ and a Git-Bash shell on Windows. The core — harnesses,
tasks, verifier, all static guards — is **dependency-free**, so the checks below
run without installing torch.

```bash
git clone https://github.com/possibleme2026-lang/rsi-multi-harness-rl.git
cd rsi-multi-harness-rl

# core checks: no torch, no GPU
./run.sh scripts/guard_tool_surface.py
./run.sh tests/test_path_errors.py
./run.sh tests/test_shell_timeout.py
./run.sh tests/test_scan_tooling.py
./run.sh tests/smoke_env.py
```

`run.sh` is the supported entry point, not a convenience. It sets `APPDATA`, the
HuggingFace cache, and clears `PYTHONPATH`, because on Windows each of those
produces a *misleading* error rather than a missing-dependency error when wrong —
`from_pretrained` claiming there is no internet connection, or `import torch`
raising ModuleNotFoundError on a machine where torch is installed.

The full pipeline needs the RL stack (`trl` on the dev line; see
`pyproject.toml`) and a GPU:

```bash
pip install -e ".[rl,dev]"

bash pipeline.sh                    # scan -> train single -> train multi -> eval
STEPS=20 N_EVAL=4 bash pipeline.sh  # a smaller run
SKIP_SCAN=1 bash pipeline.sh        # reuse an existing difficulty scan
```

Everything lands in `outputs/` (override with `MULTIHARNESS_OUT`). The scan takes
about 7 minutes at 0.8 s per cell on an RTX 5060 Laptop; training and evaluation
are longer.

| script | what it does |
| --- | --- |
| `scripts/probe.py` | capability probe + difficulty scan, writes `scan_all.json` |
| `scripts/train.py` | one arm of the ablation (`--mode single` / `--mode multi`) |
| `scripts/eval.py` | baseline + both arms in one process, prints the ablation |
| `scripts/diag.py` | full un-truncated transcript for one (harness, task) |
| `scripts/errs_report.py` | classifies scan tool errors: harness defect vs model behaviour |
| `scripts/guard_tool_surface.py` | advertised tools must equal implemented tools |
| `scripts/merge_scan.py` | fold a partial rescan into an old dump, with invariant checks |

## Layout

```
src/multiharness/
  harnesses/core.py     BaseHarnessEnv, the verifier, the shell runner, path resolver
  harnesses/pool.py     the five harnesses
  tasks/suite.py        the 24 tasks, and the train/eval split
  rollout.py            a standalone re-implementation of TRL's tool-calling loop
  _bootstrap.py         repo root + artifact directory
scripts/                entry points (probe, train, eval, guards)
tests/                  smoke tests and regression tests
tools/                  README i18n parity guard
pipeline.sh             the full run, in dependency order
```

## Two bugs worth documenting

Both were found by measurement disagreeing with expectation, and both have
regression tests. They are in the README because each one silently corrupts a
result rather than crashing.

**A subprocess deadlock worth 84 minutes of idle GPU.** The shell runner used
`subprocess.run(capture_output=True, timeout=...)`. When the timeout fires, that
kills the direct child and then drains the pipes — but a grandchild that inherited
the write end keeps them open, so the drain never sees EOF and blocks forever.
`TimeoutExpired` is not even raised, because the timeout already fired; the
*safety net itself* hangs. The symptom was "no output for 84 minutes" with the GPU
at 0%, and a timeout-only fix would not have helped. The runner now writes to a
temporary file, closes stdin, and kills the process tree.

**`Path(workdir) / ""` is the directory, not an error.** A model calling
`write_file(path="", content=...)` or `replace_in_file(path=".")` attempted a write
against the working directory itself, and the raw OS error leaked back as the
tool's result: `Permission denied` on Windows, `IsADirectoryError` on POSIX. The
message is platform-dependent, which is the real problem — it polluted the exact
axis the experiment measures, so a harness would have scored lower for a reason
having nothing to do with the model. The same investigation surfaced POSIX
absolute paths (`/config.txt`, `/Users/qwen/...`) silently retargeting outside the
workspace on Windows. All path-taking tools now share one resolver that rejects
empty paths, `.`, directory targets, and escapes with OS-independent messages.

A third, related fix: `errs_report.py` originally returned a *false-green*
verdict — "no harness defects" — while 25–27 raw OS errors sat in its catch-all
bucket. It now has an explicit OS-error class and fails the scan. A guard that
cannot fail is worse than no guard.

## Citing

If this repository is useful, cite the work it is built on:

```bibtex
@misc{mimo2026v26pro,
  title={MiMo-V2.6-Pro-RL},
  author={{Xiaomi MiMo Team}},
  year={2026},
  howpublished={\url{https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL}},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
