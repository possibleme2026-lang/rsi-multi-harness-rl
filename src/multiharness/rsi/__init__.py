# Copyright 2026 The rsi-multi-harness-rl Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recursive self-improvement: the loop writes its own tasks, rewards, and harnesses.

The seven modules here are the loop, split by the question each one answers.

``stats``
    *Can this cell teach anything?* Wilson intervals, the GRPO dead-group
    probability, and a three-way verdict that refuses to call a cell dead on
    thin evidence.
``task_gen``
    *What should the policy practise on?* A task is the output of a parameter
    vector, and generating one means generating its environment, its reward,
    and a reference solution that proves it is solvable.
``verifier_gen``
    *How is success measured, and can the measurement be trusted?* The reward
    is generated too, and the two failure modes — never fires, always fires —
    are what gates V1 and V2 probe for.
``validate``
    *Is this generated task usable?* Four gates: oracle, nop, safety, and
    cross-harness solvability plus variance.
``band``
    *Where does it sit on the difficulty curve?* Three bands plus an honest
    ``unresolved``, which is the curriculum's steering signal.
``harness_evolve``
    *Can the interface be improved for this policy?* Descriptor edits over
    guidance, tools, submission, and context, guarded and judged against the
    noise floor.
``ledger``
    *What has been tried, and did it work?* An append-only record of every
    attempt, accepted or not, with the evidence attached.

Two axes, one loop
------------------
The loop moves the task axis and the harness axis independently, which is what
this repository adds to the reference implementations. RRSI evolves harnesses
against a fixed task set; SPADE generates environments against a fixed agent.
Here the policy's curriculum and its interface are both under search, and the
cross-harness variance gate exists because a task that behaves identically on
every harness cannot measure the axis the repository is about.

The import graph is acyclic and the modules are stdlib-only, so the
dependency-free CI job exercises the whole loop except the two gates that need
a shell and the parts that need a model.
"""

from __future__ import annotations

__all__ = [
    "stats",
    "task_gen",
    "verifier_gen",
    "validate",
    "band",
    "harness_evolve",
    "ledger",
]
