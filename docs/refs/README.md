# Reference implementations: what was borrowed, what was not, and why

Two published systems were read in full before this repository's RSI layer was
designed. Neither is a dependency — nothing here imports them, and the code is
written against this repository's own data shapes. This note exists so that the
relationship is auditable: for each mechanism, whether it was adopted, adapted,
or rejected, and on what grounds.

| | `google-research/rrsi` | `zhengkid/Dream-RSI` |
| --- | --- | --- |
| **What it evolves** | the *harness* (agent scaffold) | the *task* (environment + reward) |
| **Code released** | yes, full | **no** — `Full codebase ⏳ Being prepared` |
| **Read here** | every module | the README only |
| **What this repo took** | noise floor, bootstrap calibration, annealed budget, component novelty, history/prune/stall | the framing that the *task* is the object of search |

The asymmetry matters and is worth stating plainly: **Dream-RSI has no
released code**, so its design could only be read as a description. Where a
claim below is attributed to it, it is a claim about the paper's description,
not about an implementation that was inspected.

---

## Adopted from RRSI

### 1. Noise-adjusted floor — `rrsi/selection.py`

RRSI judges a candidate against `floor = S_star - delta` and rejects anything
below it. The mechanism is the right one: a gain smaller than the measurement
noise is not a gain, and without a floor an evolutionary loop will hill-climb
on sampling error indefinitely.

**Adapted, not copied.** RRSI bands two *harness* scores against each other.
Here the same band is used for two different jobs — classifying a single
*(harness, task)* cell in `rsi/stats.py`, and judging a harness candidate in
`rsi/harness_evolve.py` — and the cell case has no analogue in RRSI at all.
The band arithmetic is the same; the call sites are not.

### 2. Bootstrap calibration — `rrsi/calibrate.py`

`bootstrap_se` resamples trials within each task respecting weights; `calibrate`
takes `delta = z * sd_use`, with `sd_null = stdev(scores) * sqrt(2)` for two or
more base evaluations.

**Adopted in substance.** The `sqrt(2)` for the difference of two independent
estimates and the `z = 2` default are both kept, and both are documented in
`stats.py` at the point of use. The difference is that RRSI's version needs a
set of base evaluations to estimate `sd_null` from; `noise_floor` here pools
every cell's rollouts instead, because the scan produces many cells of unequal
size and pooling gives the larger ones proportionally more weight — the same
convention the evaluation sweep already uses.

### 3. Annealed edit budget — `rrsi/schedule.py`

`b_t = ceil(b_min + (b_max - b_min) * 0.5 * (1 + cos(pi t / T)))`.

**Adopted numerically identically.** `rsi/ledger.py:edit_budget` reproduces the
formula to the digit, including the `round(..., 9)` before the ceiling, which
guards against a floating-point value landing a hair above an integer. Keeping
it identical means a figure of the schedule is comparable across the two
codebases rather than merely similar.

The *interpretation* is stated explicitly here because it is easy to
over-read: the budget bounds `||z||_0`, the number of independent edits in one
proposal, and nothing else. It is not a step size and not a score threshold.

### 4. Component vocabulary and novelty — `rrsi/components.py`

RRSI defines `K = [prompt, control_flow, config, output_plumbing, context_mgmt,
client_tool, skill, memory, subagent]`, a structural subset
`K_STR = [client_tool, skill, memory, subagent]`, and `novelty` as the count of
structural components the incumbent has never had an accepted edit on.

**Adapted to this repository's surface.** A harness here is guidance text, a
tool set, a submission convention, and a context policy, so the vocabulary is
`[prompt, client_tool, output_plumbing, context_mgmt]` — the four that map onto
something an edit can actually change. `control_flow`, `config`, `skill`,
`memory`, and `subagent` have no counterpart in a harness that is a single
`GUIDANCE` string plus a tool list.

The *reason* for restricting novelty to structural components is RRSI's and is
kept: a harness whose guidance was reworded five times has not explored five
regions. Rewarding that as novelty would push the loop to keep rewriting prose
instead of changing the interface.

### 5. History, pruning, and stall detection — `rrsi/history.py`

RRSI keeps a JSONL history with `tried()`, `attempted()`, `accepted_edits()`,
`incumbent_component_counts()`, `yield_g()`, `prune_set()`, `render()`, plus
`stall_flag()` and `exploration()`.

**Adapted, with two deliberate changes.**

*Every attempt is recorded, not only accepted edits.* RRSI records one entry
per accepted edit and reconstructs the rest from the trajectory. The harness
axis here has a much smaller edit space, so the rejections carry proportionally
more information — "this whole neighbourhood is exhausted" is only visible if
the failures are in the ledger.

*Pruning is on yield, not on RRSI's `yield_g`.* The rule here is a component is
pruned only when it has been attempted at least `min_attempts` times **and has
never once been accepted**. A threshold on yield would discard a component that
works occasionally, and discarding on thin evidence is precisely the mistake
the statistics layer exists to correct. The rule is blunter than RRSI's and the
reason is that this project has already been burned by the alternative.

---

## Not adopted, and why

### Dream-RSI's replay simulator

Dream-RSI's central claim is that the task environment itself should be the
object of search, with a history that acts as a replay simulator so the policy
can be re-evaluated on past tasks without regenerating them.

**Rejected on a factual ground, not a judgement.** The mechanism presumes tasks
that have a *history* — a recorded interaction the simulator can replay. The
tasks generated here are single-turn file operations with no interaction to
replay, so there is nothing for a replay simulator to reconstruct. What was
taken from Dream-RSI is the framing: the task is generated, not maintained.

### A hand-written task tier

The first design for this work added a hand-written "T2-lite" tier of easier
tasks to fill the dead cells the scan had found.

**Rejected as the opposite of the requirement.** A hand-written tier is a human
maintaining a curriculum, which is the thing RSI is supposed to replace. The
generator in `rsi/task_gen.py` produces the same coverage by parameter, and can
produce more of it on demand.

### Editing harness source instead of descriptors

RRSI applies edits to a policy's source. Here an edit is a *descriptor* — a
name, a component, and a change — and applying it produces a new guidance
string or tool list.

**Rejected for auditability.** Descriptors can be recorded in the ledger with
their components, replayed to reproduce a result, checked against a guard
before being applied, and plotted, because a descriptor has a name. Editing
source would give a larger search space and an unauditable one. The trade is
stated rather than hidden: this search space is smaller than RRSI's.

---

## What this repository adds

Neither reference has the property that makes this repository's question
answerable, because both fix the axis the other one searches:

* **RRSI** evolves harnesses against a **fixed task set**.
* **SPADE** (the local `reef/recipes/beta/spade/` recipe) generates environments
  against a **fixed agent**, validating them with an oracle probe and a nop
  probe.

This repository varies **both**, which forces two additions:

**Gate V4 — cross-harness solvability and variance.** A generated task must be
solvable on *every* trainable harness, or a low score on one cell measures that
harness's completeness rather than the policy's skill. It must also *not* score
identically on all of them, or it carries no information about the axis the
repository exists to measure. Neither condition has a counterpart in a system
that varies only one axis.

**The two-probe reward check, made executable on every task.** SPADE's oracle
and nop probes are adopted as gates V1 and V2, but applied to *generated* tasks
rather than to hand-written ones. This is what makes the generator's output
trustworthy: a task whose reference solution does not score 1.0 is
mis-specified, and a task whose verifier passes an empty directory is
measuring nothing. Both are checked by construction on every task, every time,
rather than asserted by convention.

---

## Provenance of the borrowed numbers

Every constant that came from a reference is marked at its definition:

| constant | value | source |
| --- | --- | --- |
| `SIGNAL_LO` / `SIGNAL_HI` | 0.05 / 0.95 | this repo's `train.py`, retained as thresholds but no longer used as a filter |
| `MASTERED_ABOVE` | 0.9 | SPADE's regret banding |
| `OUT_OF_REACH_BELOW` | 0.1 | SPADE's regret banding |
| `z` in the noise floor | 2.0 | RRSI `calibrate.py` default |
| `sqrt(2)` in the noise floor | — | RRSI `calibrate.py`, difference of two independent estimates |
| `b_min` / `b_max` | 1 / 3 | RRSI `schedule.py` |
| `DEFAULT_G` | 8 | `train.py --num-generations` |
| rule-of-three confidence | 0.95 | convention; the exact form `1 - (1-c)^(1/n)` is used rather than `3/n` |
