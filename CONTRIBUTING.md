# Contributing

Thanks for considering a contribution. This document states how to work in this
repository and what a change is expected to carry.

## Project status

The infrastructure is complete and verified; the headline experiment is **not
yet run** (see the README and `CHANGELOG.md`, "Known limitations"). That makes
some contributions more useful than others right now:

- **Most useful:** making the task suite produce a difficulty gradient. 52% of
  scanned cells are dead because the model either always fails or always
  succeeds, and the T2 tier is dead in all 16 cells. Nothing about the
  generalization question can be answered until that changes.
- **Useful:** reproducing the pipeline on other hardware, other base models, or
  additional seeds.
- **Less useful right now:** adding more harnesses. Five already span the axes
  that matter, and a sixth adds a column without answering anything.

## Development environment

Requires Python 3.12 or newer. The core is **dependency-free on purpose** —
harnesses, tasks, the verifier, and every static guard import only the standard
library. Keep it that way: a new third-party import in `src/multiharness/` or in
a static guard breaks the CI core job, which is what makes the harness layer
testable without a GPU or a torch install.

The RL stack is an extra:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # core checks only, no torch
pip install -e ".[rl,dev]"         # to actually train
```

`trl` is pinned to the dev line (`>=1.10.0.dev0`). The multi-environment routing
this experiment depends on — `environment_factory` as a dict, a per-example tool
schema, and an env-owned `get_reward` — is not in the released 0.x series.

> **Windows note.** `run.sh` is a Git-Bash launcher, and it is the supported way
> to run anything here: it sets `APPDATA`, the HuggingFace cache, and
> `PYTHONPATH` correctly. Running the scripts with a bare `python` will fail with
> misleading errors (`import torch` raising ModuleNotFoundError, or
> `from_pretrained` claiming there is no internet connection).

## Running the checks

```bash
./run.sh scripts/guard_tool_surface.py    # tools advertised == tools implemented
./run.sh tests/test_path_errors.py        # no OS-dependent tool errors
./run.sh tests/test_shell_timeout.py      # the runner cannot deadlock
./run.sh tests/test_scan_tooling.py       # analysis verdicts
./run.sh tests/smoke_env.py               # harness layer, no model needed
./run.sh tests/smoke_trl.py               # TRL routing, needs a GPU
python tools/check_readme_i18n.py         # the two READMEs stay in step
```

Everything except `smoke_trl.py` runs without a GPU. All of it must pass before a
change is ready. A change that weakens a check to make it pass is not
acceptable: fix the behaviour, or explain in the pull request why the check's
expectation was wrong.

## What a change should carry

- **A reason.** What question does this answer, or what failure does it prevent?
  "It seemed cleaner" is not enough for a change to the harness pool or the task
  suite, because both are experiment inputs and editing them invalidates every
  number produced before.
- **A test, when the change is a fix.** Every correctness fix in this repository
  has a regression test that fails without it, and the tests are written to fail
  for the right reason — they assert on the *property* (an error is
  OS-independent; a merge with mismatched settings is refused), not on an
  implementation detail.
- **An honest number, when the change affects results.** If you re-run a scan or
  an evaluation, say what changed and by how much. Do not quietly replace a
  result file.

## Things that look like bugs but are not

Two behaviours are intentional. Please do not "fix" them without reading the
reason first:

- **A tool error caused by the model calling a tool the harness never
  advertised.** That is the adaptation under test, not a harness defect.
  `errs_report.py` classifies it as genuine signal on purpose.
- **Low scores on a hard tier.** A task the model always fails is a *measurement*
  of the model, not a broken task. Check the scan dump's live-cell count before
  concluding otherwise.

## Reporting a bug

Include the command you ran, the full output, and the scan or evaluation dump if
one was produced. For a harness-level bug, `scripts/errs_report.py <dump>` output
is the most useful thing you can attach — it separates harness defects from model
behaviour automatically.
