# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/possibleme2026-lang/rsi-multi-harness-rl/security/advisories/new)
rather than in a public issue.

Please include what the issue is, how to reproduce it, and what an attacker could
achieve. You will get an acknowledgement, and credit in the advisory unless you
ask otherwise.

## Scope

This project runs model-generated shell commands inside a temporary working
directory. The attack surface is therefore real even though nothing is deployed
as a service. Areas worth reporting:

- **Workspace escape.** Any way to make a harness tool read, write, or execute
  outside its task working directory. This is the highest-value class: the
  resolver in `harnesses/core.py` is the single choke point, and a bypass of it
  is a defect regardless of how the model was prompted into it.
- **Command injection beyond the intended channel.** The `bash` tool is
  *supposed* to run arbitrary commands inside the workspace; that is the point
  of the harness. What is not intended is a way to escape the workspace, persist
  across tasks, or reach the host through a path the task never opened.
- **Cross-task contamination.** Any way for one task's workspace, files, or
  reward state to affect another task's. Each `reset()` is meant to produce a
  fresh directory; leakage would silently invalidate a reward matrix.
- **Verifier bypass.** Any way to score `1.0` without producing the expected
  content in `answer.txt` — including making the verifier raise, cache a stale
  reward, or grade a different path than the one the harness writes.
- **Untrusted input handling.** Scan dumps, task definitions, and patch bodies
  are parsed. A crash is a bug; arbitrary code execution from a crafted dump or
  patch is a vulnerability.

## Design commitments that are not vulnerabilities

Three properties are intentional and should not be reported as bugs:

- **The `bash` tool executes model-generated shell commands.** It runs with the
  task working directory as `cwd`, under the invoking user's privileges, and it
  is not sandboxed beyond that. This is the substrate the experiment measures; a
  harness that cannot run a shell command cannot measure harness adaptation.
  Run this project in a container or a VM if the model is untrusted.
- **A tool error is returned to the model as the tool's result.** A tool that
  raises does not crash the rollout — the error string becomes the tool's
  content, exactly as a real agent harness behaves. That is deliberate.
- **Raw pass rates can be very low, and dead cells are reported, not hidden.**
  A cell where the model always fails carries no gradient; the pipeline reports
  it and the training input drops it. A low number here is a measurement, not a
  denial of service.

## Supported versions

The project is pre-1.0 and has no released versions. Only the `main` branch is
maintained; fixes are made there and no backports are offered.
