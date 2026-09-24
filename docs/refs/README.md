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

## Environment synthesis: what was read, and what was taken

The loop's second axis is a curriculum, so the environment-synthesis literature
was surveyed before it was written. The survey lives in a separate repository,
[`possibleme2026-lang/awesome-environment`](https://github.com/possibleme2026-lang/awesome-environment)
(107 entries across 9 sections). What follows is the subset that changed a
decision here — for each, whether it was adopted, adapted, or rejected, and on
what grounds.

| work | what it is | disposition here |
| --- | --- | --- |
| **GenEnv** ([2512.19682](https://arxiv.org/abs/2512.19682)) | difficulty-aligned co-evolution between an agent and a generative environment simulator, via an α-Curriculum Reward | **adapted** — the reward and the `k_min` filter are in `rsi/curriculum.py`; α is *derived*, not copied |
| **EnvHarness** ([2608.19880](https://arxiv.org/abs/2608.19880)) | a programmable layer of Stage/Contract/Chain components wrapping a *static* environment, leaving its verifier untouched | **rejected as a mechanism, adopted as a framing** — see below |
| **SPADE** ([2608.19197](https://arxiv.org/abs/2608.19197)) | self-play: an environment designer generates executable environments, a solver agent solves them | **adopted** (pre-existing) — the four gates and the oracle/nop probes are already SPADE's |
| **Efficient Benchmarking of AI Agents** ([2603.23749](https://arxiv.org/abs/2603.23749)) | tasks with intermediate historical pass rates (30–70%) preserve agent *rankings* at 44–70% lower cost | **adopted as a cross-check** — see below |
| **AgentJudgeBench** ([2608.26623](https://arxiv.org/abs/2608.26623)) | LLM-as-judge reliability on dependency-driven tool-calling; hard queries converge to a 77–82% ceiling | **adopted as a constraint on scope** — see below |
| **Agent-World** ([2604.18292](https://arxiv.org/abs/2604.18292)) | 1978 environments / 19,822 tools, tool-dependency graphs, executable Python solutions | **not adopted** — its scale is orthogonal to this repo's question |
| **C-World** ([2601.06328](https://arxiv.org/abs/2601.06328)) | on-demand environment construction, 5,571 tools, transition functions injecting real faults | **not adopted** — tool scale, not harness generalization |
| **TerminalWorld** ([2605.22535](https://arxiv.org/abs/2605.22535)) | reverse-engineers real terminal dependencies into executable sandboxes | **not adopted** — this repo's tasks are single-turn file ops by design |
| **Repo2RLEnv** ([GitHub](https://github.com/huggingface/Repo2RLEnv)) | converts GitHub repos/PRs into Harbor-format RL environments | **not adopted** — see "the Harbor connection" below |

### Adopted from GenEnv — the α-Curriculum Reward

GenEnv rewards an environment policy for emitting tasks the agent solves about
``α`` of the time::

    R_env(p̂) = exp(−β (p̂ − α)²)

with ``α = 0.5``, ``β > 0``, and a difficulty filter that excludes a batch from
the update when ``|p̂ − α| > k_min`` (``k_min = 0.1``).

**Adapted in one substantive way: ``α`` is derived rather than borrowed.**
GenEnv justifies ``α = 0.5`` through ``p(1−p) = 1/4 − (p − 1/2)²``, which is a
statement about Bernoulli *variance*. That is not the quantity this repository
reasons about. What decides whether a cell teaches anything here is the GRPO
group signal probability ``1 − p^G − (1−p)^G``, already implemented and already
measured at ``G = 8`` in ``rsi/stats.py``. So ``curriculum.optimal_alpha``
returns the argmax of *that* curve, and ``alpha_is_derived`` asserts the
relationship. The answer is 0.5 — but it is 0.5 because the curve is symmetric
about 0.5 for every ``G``, which is a fact about this repository's optimizer,
and a test pins it at ``G ∈ {2, 4, 8, 16, 32}``.

The derivation also produced a number worth keeping: at ``G = 8`` the signal
curve is **flat near the top** — ``p = 0.5`` gives 0.9922 and ``p = 0.3``
already gives 0.9423. Over-tuning difficulty toward ``α`` buys almost nothing,
and that is the honest counter-argument to the whole difficulty-alignment idea.
It is recorded in the module docstring rather than left out.

**One design bug, found by running it.** The ``k_min`` filter was first applied
*per task*, which silently disabled the entire curriculum: ``mastered`` is
``p > 0.9`` and ``out_of_reach`` is ``p < 0.1``, so every cell ``steer`` would
act on sits at least 0.4 from ``α = 0.5`` — far outside a 0.1 band. The filter
rejected exactly the cells the steering rule existed to move, and the first live
run reported ``moves: 0``. In GenEnv the reward and the filter both score *one
batch's aggregate success rate*, because what is trained there is an environment
policy that emits batches. There is no such policy here, so the rule was moved
to ``batch_is_misaligned`` at the scope it was written for. Both scopes are
reported on every plan, and the arithmetic of the mistake is pinned in the test.

### Adopted from *Efficient Benchmarking of AI Agents* — as a cross-check, not a rule

That paper's Mid-Range Difficulty Filter keeps tasks with historical pass rates
between 30% and 70%, motivated by Item Response Theory, and reports 44–70%
cheaper evaluation at preserved *ranking* fidelity.

**Not adopted as a threshold, adopted as an independent confirmation.** The
30–70% band and this repository's ``[OUT_OF_REACH_BELOW, MASTERED_ABOVE] =
[0.1, 0.9]`` band are different widths derived from different arguments — one
from IRT, one from GRPO's group-variance collapse. That they both centre on the
middle of the range is weak evidence that the middle is where the signal is,
and it is worth stating because neither band was chosen to agree with the other.
The widths genuinely differ and the disagreement is not resolved here: this
repo's band is wider because the quantity it protects against (a group with
*zero* gradient) is a stricter failure than a task being uninformative.

The paper's other finding is the one that matters more for this repository: it
reports that **absolute score prediction degrades under scaffold-driven
distribution shift while rank-order prediction stays stable.** That is a
directly relevant warning about this repo's headline metric, which is a
difference of absolute rewards across harnesses — the fragile quantity, not the
robust one. It is noted in the README's limitations rather than acted on,
because acting on it would mean changing the metric the ablation was designed
around.

### Adopted from AgentJudgeBench — a constraint on what is claimed

AgentJudgeBench shows that on hard queries without ground truth, six judges
from 20B to frontier scale converge to a **77–82% band** regardless of model
size, and that exposing ground truth *reduces* alignment for some frontier
judges through over-anchoring.

**Adopted as a scope constraint.** This repository uses no LLM judge — grading
is a file comparison or a generated Python checker, which is exactly the
"deterministic programmatic scorer" that paper treats as the reference rather
than as the subject. The finding is recorded here as the reason *not* to
introduce one: the reward is generated, and a generated verifier is audited by
gates V1–V4, which is a stronger guarantee than a judge's alignment rate. The
77–82% ceiling is the argument for keeping the reward programmatic.

### Rejected — EnvHarness's component layer

EnvHarness wraps a frozen environment in Stage/Contract/Chain components that
reshape where an episode starts, what actions are permitted, and what the agent
observes — operating strictly through ``reset()``/``step()`` and leaving the
original verifier untouched. Its stated prerequisite is a **resettable
environment**.

**Rejected on a factual ground.** This repository's "environment" is a
parameter vector that produces a single-turn file operation; there is no
``step()`` loop, no episode state to stage, and no transition function to
contract. EnvHarness's components are transformations of an environment's
state/action/observation/transition terms, and none of those four terms exists
in a task that is "write this string to ``answer.txt``". Its companion EnvRigger
diagnoses a policy's flaws from rollouts and writes components as real Python —
the diagnosis half of that loop *is* implemented here, in the ``out_of_reach``
band plus ``steer``, but the repair half has nothing to attach to.

**What was taken is the framing**, and it is a good one: EnvHarness makes the
symmetry explicit — "Agent = model + Harness", "customised environment = static
environment + EnvHarness". This repository varies the harness *and* the task,
which is the pair EnvHarness's own framing implies but does not itself span. The
rejection is stated as a scope fact, not as a judgement of the work: EnvHarness
is strictly more capable than what is here, on environments that have a
``step()``.

### Not adopted — the environment-generation line as a whole

Agent-World, C-World, AgentMercury, EnvFactory, ScaleEnv, and the rest of
``awesome-environment``'s second section generate *environments at scale*:
thousands of them, with tool-dependency graphs and executable solutions. This
repository generates **12 to 30 tasks** from a five-knob parameter vector and
spends its effort on measuring whether the harness axis generalizes.

The difference is the question, not the ambition. A 1978-environment corpus
would answer "does environment diversity improve an agent"; the ablation here
asks "does training across harnesses transfer to a harness never trained on",
and that needs a *fixed, small, fully-characterised* task set — one where every
cell's pass rate has been measured and every task's reference solution has been
executed. Scale would make the measurement worse, not better: the ablation's
binding constraint is statistical power (the README computes that the observed
effect would need ~2,096 rollouts per arm for 80% power), not task count.

### The Harbor connection, and why it is not taken

``awesome-environment`` lists **Repo2RLEnv**, which converts a GitHub repository
into a Harbor-format task package (instruction + environment + reference
solution + verifier). The format is close to what ``rsi/task_gen.generate``
already produces — environment (``setup``), task (``instruction``), reward
(``verify``/``expected``/``check_script``), and reference (``reference``).

**Not adopted, and the reason is the interesting part.** Emitting Harbor-format
packages would make the generated tasks portable to a container runner, which is
attractive. But the four gates that make a generated task *trustworthy* here —
oracle, nop, safety, cross-harness variance — are run in-process against the
shipped harness pool, and V4 in particular needs all four trainable harnesses
plus the held-out one. A Harbor package carries a verifier but not a harness
pool, so exporting would move the tasks out of the only context in which their
validity has been established. The format is compatible; the *evidence* is not
portable.

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
| `DEFAULT_BETA` | 10.0 | GenEnv `2512.19682`; kept because the reward is only compared against itself |
| `DEFAULT_K_MIN` | 0.1 | GenEnv `2512.19682`, **applied at batch scope** — see the design bug above |
| `DEFAULT_ALPHA` | 0.5 | **derived** — `argmax_p (1 - p^G - (1-p)^G)` at `G = 8`, not copied from GenEnv |
