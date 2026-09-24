#!/usr/bin/env python3
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

"""Regenerate every figure in ``docs/figures/`` from the recorded artifacts.

Run:
    ./run.sh tools/plot.py

Why this lives in ``tools/``
----------------------------
``.github/workflows/ci.yml`` asserts that everything under ``src/`` and
``scripts/`` imports only the standard library, and that check skips ``tools/``
by design. Plotting needs matplotlib, so the figures cannot be produced from
``scripts/`` without either installing matplotlib in the dependency-free job or
weakening the assertion that makes the core job meaningful. Putting the tool
here keeps both: the core job still proves the harness layer has no third-party
imports, and the figures are still reproducible with one command.

Every figure reads an artifact that a recorded run wrote. Nothing is
synthesised, and a figure whose input is missing is *skipped with a reason*
rather than drawn from invented numbers — a plotted guess is worse than no
plot, because it looks like evidence.

The two figures that read no artifact are the analytic ones (the GRPO signal
curve and the annealed budget), and they are labelled as such: they are
functions, and their inputs are printed on the axes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

# --------------------------------------------------------------------------
# house style — light, because these are read on a white README
# --------------------------------------------------------------------------

INK = "#1a1a1a"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#1f6feb"

#: Categorical palette. Deliberately not the matplotlib default cycle: these
#: figures sit next to each other in one README and the same harness must be
#: the same colour in all of them, or the reader has to re-learn the legend
#: on every scroll.
HARNESS_COLOR = {
    "bash_minimal": "#1f6feb",
    "react_tools": "#d97706",
    "json_strict": "#059669",
    "longctx_summary": "#7c3aed",
    "oracle": "#9ca3af",
    "codex_style": "#be185d",
}

VERDICT_COLOR = {
    "live": "#059669",
    "dead": "#dc2626",
    "under_measured": "#d97706",
}

BAND_COLOR = {
    "mastered": "#1f6feb",
    "frontier": "#059669",
    "out_of_reach": "#dc2626",
    "unresolved": "#d1d5db",
}


def _plt():
    """Import matplotlib with a headless backend and apply the house style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )
    return plt


SKIPPED: list[tuple[str, str]] = []


def _load(rel: str):
    """Read a JSON artifact, or return ``None`` and record why not."""
    p = ROOT / rel
    if not p.is_file():
        SKIPPED.append((rel, "artifact not present"))
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        SKIPPED.append((rel, f"unparseable: {exc}"))
        return None


def _load_jsonl(rel: str) -> list[dict]:
    p = ROOT / rel
    if not p.is_file():
        SKIPPED.append((rel, "artifact not present"))
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _save(plt, fig, name: str, note: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}  ({path.stat().st_size:,} B)  — {note}")


# --------------------------------------------------------------------------
# cell statistics, recomputed from the raw records
# --------------------------------------------------------------------------


def _cells(scan: dict) -> list[dict]:
    """Collapse raw rollout records into per-(harness, task) cells.

    Recomputed here rather than read from the scan's own summary so that the
    figures and ``rsi/stats.py`` cannot disagree: both derive from ``records``,
    which is the only thing the run actually produced.
    """
    agg: dict[tuple[str, str], dict] = {}
    for r in scan.get("records", []):
        key = (str(r.get("harness", "")), str(r.get("task_id", "")))
        c = agg.setdefault(key, {"passes": 0, "n": 0, "turns": 0, "tool_calls": 0, "errors": 0})
        c["n"] += 1
        c["passes"] += int(float(r.get("reward", 0.0)) >= 1.0)
        c["turns"] += int(r.get("turns", 0) or 0)
        c["tool_calls"] += int(r.get("tool_calls", 0) or 0)
        c["errors"] += int(r.get("tool_errors", 0) or 0)
    out = []
    for (h, t), c in agg.items():
        out.append({"harness": h, "task_id": t, **c, "p_hat": c["passes"] / c["n"] if c["n"] else 0.0})
    return sorted(out, key=lambda d: (d["harness"], d["task_id"]))


def _harnesses(cells: list[dict]) -> list[str]:
    return sorted({c["harness"] for c in cells})


def _tasks(cells: list[dict]) -> list[str]:
    return sorted({c["task_id"] for c in cells})


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def fig_scan_matrix(scan: dict) -> None:
    """Pass rate per (harness, task) cell — the experiment's headline result."""
    plt = _plt()
    cells = _cells(scan)
    if not cells:
        SKIPPED.append(("fig01_scan_matrix", "no records"))
        return
    hs, ts = _harnesses(cells), _tasks(cells)
    grid = [[0.0] * len(ts) for _ in hs]
    n_by = {}
    for c in cells:
        i, j = hs.index(c["harness"]), ts.index(c["task_id"])
        grid[i][j] = c["p_hat"]
        n_by[(i, j)] = c["n"]

    n = scan.get("n", "?")
    fig, ax = plt.subplots(figsize=(max(7.0, 0.52 * len(ts) + 2.4), 0.5 * len(hs) + 2.0))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ts)), ts, rotation=90, fontsize=8)
    ax.set_yticks(range(len(hs)), hs, fontsize=9)
    ax.grid(False)
    for i in range(len(hs)):
        for j in range(len(ts)):
            v = grid[i][j]
            ax.text(
                j,
                i,
                f"{v:.2f}" if v > 0 else "0",
                ha="center",
                va="center",
                fontsize=7,
                color="#111111" if 0.25 < v < 0.85 else ("white" if v <= 0.25 else "#111111"),
                fontweight="bold" if v == 0 else "normal",
            )
    ax.set_title(f"Pass rate per (harness, task) cell — n={n} rollouts each", fontsize=11, pad=12)
    ax.set_xlabel("task")
    ax.set_ylabel("harness")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("pass rate", fontsize=9)
    cb.outline.set_edgecolor(GRID)
    _save(plt, fig, "fig01_scan_matrix.png", "pass rate grid; the cross-harness gap is visible column-wise")


def fig_measurement(scan: dict) -> None:
    """Why n=8 was not enough: point estimates with their Wilson intervals."""
    plt = _plt()
    cells = _cells(scan)
    if not cells:
        SKIPPED.append(("fig02_measurement", "no records"))
        return
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.stats import SIGNAL_HI, SIGNAL_LO, wilson_interval

    hs = _harnesses(cells)
    ts = _tasks(cells)
    fig, axes = plt.subplots(1, len(hs), figsize=(3.1 * len(hs), 4.2), sharey=True)
    if len(hs) == 1:
        axes = [axes]
    for ax, h in zip(axes, hs, strict=False):
        sub = {c["task_id"]: c for c in cells if c["harness"] == h}
        xs = [i for i, t in enumerate(ts) if t in sub]
        lo = []
        hi = []
        mid = []
        for i in xs:
            c = sub[ts[i]]
            a, b = wilson_interval(c["passes"], c["n"])
            lo.append(a)
            hi.append(b)
            mid.append(c["p_hat"])
        ax.axhspan(SIGNAL_LO, SIGNAL_HI, color="#ecfdf5", zorder=0)
        ax.axhline(SIGNAL_LO, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.axhline(SIGNAL_HI, color=MUTED, lw=0.9, ls="--", zorder=1)
        ax.vlines(xs, lo, hi, color=HARNESS_COLOR.get(h, ACCENT), lw=2.2, alpha=0.55, zorder=2)
        ax.plot(xs, mid, "o", color=HARNESS_COLOR.get(h, ACCENT), ms=4.5, zorder=3)
        ax.set_title(h, fontsize=10, color=HARNESS_COLOR.get(h, ACCENT))
        ax.set_xticks(range(len(ts)), ts, rotation=90, fontsize=7)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("pass rate  (bar = 95% Wilson interval)")
    fig.suptitle(
        f"Every cell with its confidence interval, n={scan.get('n', '?')} — "
        "a point estimate near a dashed line is not a verdict",
        fontsize=11,
        y=1.0,
    )
    _save(plt, fig, "fig02_measurement.png", "Wilson intervals; the shaded band is where a group can disagree")


def fig_verdicts(scan: dict, prev: dict | None) -> None:
    """How many cells each verdict, at the old sample size and the new one."""
    plt = _plt()
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.stats import CellVerdict, classify_cell

    def counts_of(art: dict) -> dict[str, int]:
        cs = _cells(art) if art else []
        d = {v: 0 for v in CellVerdict}
        for c in cs:
            d[classify_cell(c["passes"], c["n"]).verdict] += 1
        return d

    new = counts_of(scan)
    old = counts_of(prev) if prev else None

    labels = [str(v) for v in CellVerdict]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = range(len(labels))
    w = 0.38
    if old is not None:
        ax.bar([i - w / 2 for i in x], [old[lab] for lab in labels], w, label="n=8", color="#cbd5e1", edgecolor="none")
    ax.bar(
        [i + (w / 2 if old is not None else 0) for i in x],
        [new[lab] for lab in labels],
        w,
        label=f"n={scan.get('n', '?')}",
        color=[VERDICT_COLOR.get(lab, ACCENT) for lab in labels],
        edgecolor="none",
    )
    for i, lab in enumerate(labels):
        off = w / 2 if old is not None else 0
        ax.text(i + off, new[lab] + 0.6, str(new[lab]), ha="center", fontsize=9, color=INK)
        if old is not None:
            ax.text(i - w / 2, old[lab] + 0.6, str(old[lab]), ha="center", fontsize=9, color=MUTED)
    ax.set_xticks(list(x), [lab.replace("_", " ") for lab in labels])
    ax.set_ylabel("cells")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Cell verdicts — under_measured is a verdict, not a rejection", fontsize=11, pad=10)
    _save(plt, fig, "fig03_verdicts.png", "three-way classification, old sample size vs new")


def fig_grpo_signal() -> None:
    """The analytic figure: how often a GRPO group carries any gradient."""
    plt = _plt()
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.stats import SIGNAL_LO, grpo_signal_probability

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ps = [i / 200 for i in range(201)]
    for g, color in ((2, "#d1d5db"), (4, "#93c5fd"), (8, ACCENT), (16, "#1e3a8a")):
        ys = [grpo_signal_probability(p, g) for p in ps]
        ax.plot(ps, ys, color=color, lw=2.0, label=f"G={g}")
        if g == 8:
            y8 = grpo_signal_probability(SIGNAL_LO, 8)
            ax.plot([SIGNAL_LO], [y8], "o", color="#dc2626", ms=7, zorder=5)
            ax.annotate(
                f"p={SIGNAL_LO}: still {y8:.1%} signal\n(the old filter dropped these)",
                xy=(SIGNAL_LO, y8),
                xytext=(0.16, 0.44),
                fontsize=9,
                color="#dc2626",
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2),
            )
    ax.axvspan(0.0, SIGNAL_LO, color="#fef2f2", zorder=0)
    ax.axvspan(1 - SIGNAL_LO, 1.0, color="#fef2f2", zorder=0)
    ax.text(SIGNAL_LO / 2, 0.04, "old\ndead zone", ha="center", fontsize=7.5, color="#dc2626")
    ax.text(1 - SIGNAL_LO / 2, 0.04, "old\ndead zone", ha="center", fontsize=7.5, color="#dc2626")
    ax.set_xlabel("pass rate p")
    ax.set_ylabel("P(group carries a gradient)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(title="group size", fontsize=9, title_fontsize=9)
    ax.set_title("A GRPO group is only dead at p = 0 and p = 1", fontsize=11, pad=10)
    _save(plt, fig, "fig04_grpo_signal.png", "analytic; P(signal) = 1 - (p^G + (1-p)^G)")


def fig_bands(scan: dict) -> None:
    """Where each cell sits on the difficulty curve."""
    plt = _plt()
    cells = _cells(scan)
    if not cells:
        SKIPPED.append(("fig05_bands", "no records"))
        return
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.band import band_of

    hs, ts = _harnesses(cells), _tasks(cells)
    bands = [[None] * len(ts) for _ in hs]
    for c in cells:
        i, j = hs.index(c["harness"]), ts.index(c["task_id"])
        bands[i][j] = band_of(c["passes"], c["n"]).band

    order = ["out_of_reach", "frontier", "mastered", "unresolved"]
    idx = {b: k for k, b in enumerate(order)}
    grid = [[idx.get(bands[i][j], 3) for j in range(len(ts))] for i in range(len(hs))]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.52 * len(ts) + 2.6), 0.5 * len(hs) + 2.4))
    cmap = matplotlib_colors(plt, order)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(order) - 1, aspect="auto")
    ax.set_xticks(range(len(ts)), ts, rotation=90, fontsize=8)
    ax.set_yticks(range(len(hs)), hs, fontsize=9)
    ax.grid(False)
    for i in range(len(hs)):
        for j in range(len(ts)):
            b = bands[i][j] or "?"
            ax.text(j, i, b.replace("_", " ")[:9], ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_title(
        f"Difficulty band per cell, n={scan.get('n', '?')} — "
        "unresolved means the scan cannot place it",
        fontsize=10.5,
        pad=12,
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND_COLOR[b]) for b in order]
    ax.legend(handles, [b.replace("_", " ") for b in order], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    _save(plt, fig, "fig05_bands.png", "regret banding; a band is reported only if the interval is inside it")


def matplotlib_colors(plt, order: list[str]):
    from matplotlib.colors import ListedColormap

    return ListedColormap([BAND_COLOR[b] for b in order])


def fig_stop_reasons(scan: dict) -> None:
    """Why rollouts stopped — the mechanism behind the dead cells."""
    plt = _plt()
    recs = scan.get("records", [])
    if not recs:
        SKIPPED.append(("fig06_stop_reasons", "no records"))
        return
    reasons = sorted({str(r.get("stopped_reason", "?")) for r in recs})
    tiers = sorted({str(r.get("task_id", "?"))[:2] for r in recs})
    counts = {t: {rs: 0 for rs in reasons} for t in tiers}
    for r in recs:
        counts[str(r.get("task_id", "?"))[:2]][str(r.get("stopped_reason", "?"))] += 1

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bottom = [0] * len(tiers)
    palette = ["#1f6feb", "#d97706", "#059669", "#dc2626"]
    totals = {tier: sum(counts[tier].values()) for tier in tiers}
    for k, rs in enumerate(reasons):
        vals = [counts[tier][rs] for tier in tiers]
        ax.bar(tiers, vals, bottom=bottom, label=rs, color=palette[k % len(palette)], edgecolor="none")
        for i, v in enumerate(vals):
            if v > 0.06 * totals[tiers[i]]:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=8, color="white")
        bottom = [b + v for b, v in zip(bottom, vals, strict=False)]
    ax.set_ylabel("rollouts")
    ax.set_xlabel("task tier")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Why rollouts stopped — no_tool_call means the policy answered without acting", fontsize=11, pad=10)
    _save(plt, fig, "fig06_stop_reasons.png", "the failure mechanism, by tier")


def fig_turns(scan: dict) -> None:
    """Turn count and tool-call count, split by outcome."""
    plt = _plt()
    recs = scan.get("records", [])
    if not recs:
        SKIPPED.append(("fig07_turns", "no records"))
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    for ax, key, label in ((axes[0], "turns", "turns taken"), (axes[1], "tool_calls", "tool calls made")):
        vals = sorted({int(r.get(key, 0) or 0) for r in recs})
        win = [sum(1 for r in recs if int(r.get(key, 0) or 0) == v and float(r.get("reward", 0)) >= 1.0) for v in vals]
        lose = [sum(1 for r in recs if int(r.get(key, 0) or 0) == v and float(r.get("reward", 0)) < 1.0) for v in vals]
        ax.bar(vals, win, color="#059669", label="reward 1", edgecolor="none")
        ax.bar(vals, lose, bottom=win, color="#e5e7eb", label="reward 0", edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("rollouts" if ax is axes[0] else "")
        ax.set_xticks(vals)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Effort vs outcome — the zero-reward mass sits at one tool call", fontsize=11, y=1.02)
    _save(plt, fig, "fig07_turns.png", "turn and tool-call distribution split by reward")


def fig_toolcall_reward(scan: dict) -> None:
    """Cross-tab of tool calls against reward, per tier."""
    plt = _plt()
    recs = scan.get("records", [])
    if not recs:
        SKIPPED.append(("fig08_toolcall_reward", "no records"))
        return
    tiers = sorted({str(r.get("task_id", "?"))[:2] for r in recs})
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels, wrong, right = [], [], []
    for t in tiers:
        sub = [r for r in recs if str(r.get("task_id", "?"))[:2] == t]
        for n in sorted({int(r.get("tool_calls", 0) or 0) for r in sub}):
            cell = [r for r in sub if int(r.get("tool_calls", 0) or 0) == n]
            labels.append(f"{t}\n{n} call" + ("s" if n != 1 else ""))
            right.append(sum(1 for r in cell if float(r.get("reward", 0)) >= 1.0))
            wrong.append(sum(1 for r in cell if float(r.get("reward", 0)) < 1.0))
    x = range(len(labels))
    ax.bar(x, right, color="#059669", label="reward 1", edgecolor="none")
    ax.bar(x, wrong, bottom=right, color="#fca5a5", label="reward 0", edgecolor="none")
    for i, (r, w) in enumerate(zip(right, wrong, strict=False)):
        ax.text(i, r + w + max(right + wrong) * 0.015, f"{r}/{r + w}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(list(x), labels, fontsize=7.5)
    ax.set_ylabel("rollouts")
    ax.legend(fontsize=9)
    ax.set_title("Tool calls vs reward — a call is not a solve", fontsize=11, pad=10)
    _save(plt, fig, "fig08_toolcall_reward.png", "per-tier cross-tab of tool calls and reward")


def fig_edit_budget() -> None:
    """The annealed edit budget, from the same function the loop calls."""
    plt = _plt()
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.ledger import edit_budget

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for T, color in ((8, "#d1d5db"), (12, ACCENT), (24, "#1e3a8a")):
        xs = list(range(T + 1))
        ys = [edit_budget(t, T, 1, 3) for t in xs]
        ax.step(xs, ys, where="post", color=color, lw=2.0, label=f"T={T}")
    ax.set_xlabel("round t")
    ax.set_ylabel("edits allowed in one proposal")
    ax.set_yticks([1, 2, 3])
    ax.set_ylim(0.8, 3.3)
    ax.legend(fontsize=9)
    ax.set_title("Annealed edit budget — several edits early, one edit late", fontsize=11, pad=10)
    _save(plt, fig, "fig09_edit_budget.png", "analytic; ceil(b_min + (b_max-b_min)/2 (1+cos(pi t/T)))")


def fig_ledger() -> None:
    """The evolution ledger: what was tried, and whether it cleared the floor.

    Draws the ``rollout`` ledger when one exists, because that is the mode whose
    trajectory is measured rather than synthesised, and falls back to the
    replayed one. Which is which is stated on the figure: the two trajectories
    are both "a rising line", and only one of them means the harness improved.
    """
    plt = _plt()
    replayed = _load_jsonl("outputs/rsi/ledger.jsonl")
    measured = _load_jsonl("outputs_rollout/rsi/ledger.jsonl")
    if measured:
        recs, mode = measured, "rollout (measured)"
    elif replayed:
        recs, mode = replayed, "ledger-replay (synthetic trajectory)"
    else:
        SKIPPED.append(("fig10_ledger", "no ledger recorded"))
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    by_harness: dict[str, list] = {}
    for r in recs:
        by_harness.setdefault(str(r.get("harness", "?")), []).append(r)
    for h, rs in sorted(by_harness.items()):
        rs = sorted(rs, key=lambda d: int(d.get("round", 0)))
        xs = [int(r.get("round", 0)) for r in rs]
        ys = [float(r.get("score_after", 0.0)) for r in rs]
        ax.plot(xs, ys, "-", color=HARNESS_COLOR.get(h, ACCENT), lw=1.4, alpha=0.5)
        for r in rs:
            ax.plot(
                int(r.get("round", 0)),
                float(r.get("score_after", 0.0)),
                "o" if r.get("accepted") else "x",
                color=HARNESS_COLOR.get(h, ACCENT),
                ms=6 if r.get("accepted") else 4.5,
                mew=1.6,
            )
        ax.plot([], [], "-", color=HARNESS_COLOR.get(h, ACCENT), lw=2, label=h)
    ax.plot([], [], "o", color=INK, ms=6, label="accepted")
    ax.plot([], [], "x", color=INK, ms=5, label="rejected")
    ax.set_xlabel("round")
    ax.set_ylabel("candidate score")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Harness evolution trajectory", fontsize=10.5)

    ax2 = axes[1]
    sys.path.insert(0, str(ROOT / "src"))
    from multiharness.rsi.ledger import COMPONENTS

    counts = {c: [0, 0] for c in COMPONENTS}
    for r in recs:
        for c in r.get("components", []):
            counts.setdefault(c, [0, 0])
            counts[c][1] += 1
            if r.get("accepted"):
                counts[c][0] += 1
    comps = [c for c in COMPONENTS if counts.get(c, [0, 0])[1] > 0]
    if comps:
        y = range(len(comps))
        acc = [counts[c][0] for c in comps]
        rej = [counts[c][1] - counts[c][0] for c in comps]
        ax2.barh(list(y), acc, color="#059669", label="accepted", edgecolor="none")
        ax2.barh(list(y), rej, left=acc, color="#e5e7eb", label="rejected", edgecolor="none")
        ax2.set_yticks(list(y), comps, fontsize=8.5)
        ax2.legend(fontsize=8.5)
    ax2.set_xlabel("attempts")
    ax2.set_title("Which component paid off", fontsize=10.5)
    fig.suptitle(
        f"Every attempt is recorded, not only the successes — score mode: {mode}",
        fontsize=10.5,
        y=1.03,
    )
    _save(plt, fig, "fig10_ledger.png", "ledger trajectory and per-component yield")


def fig_validation() -> None:
    """The four gates on a generated batch — real output, not a claim."""
    plt = _plt()
    data = _load("outputs/rsi/validation.json")
    if not data:
        SKIPPED.append(("fig11_validation", "no validation artifact"))
        return
    summary = data.get("summary", {})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    total = summary.get("total", 0)
    accepted = summary.get("accepted", 0)
    ax.bar(["accepted", "rejected"], [accepted, total - accepted], color=["#059669", "#dc2626"], edgecolor="none")
    ax.text(0, accepted + total * 0.02, f"{accepted}/{total}", ha="center", fontsize=10, color=INK)
    ax.set_ylabel("tasks")
    ax.set_title("Generated tasks passing all four gates", fontsize=10.5)

    ax2 = axes[1]
    gates = ["V1_oracle", "V2_nop", "V3_safety", "V4_cross"]
    fails = summary.get("failures_by_gate", {})
    tiers = sorted(summary.get("by_tier", {}).keys())
    if tiers:
        bottom = [0] * len(tiers)
        palette = ["#1f6feb", "#d97706", "#059669", "#dc2626"]
        for k, g in enumerate(gates):
            vals = [fails.get(g, 0) if t == "all" else 0 for t in tiers]
            # Per-tier gate failures are not recorded separately in the summary,
            # so the bars show the totals by gate rather than inventing a split.
            if any(vals):
                ax2.bar(tiers, vals, bottom=bottom, label=g, color=palette[k % 4], edgecolor="none")
        tick_labels = [
            f"{t}\n{summary['by_tier'][t]['accepted']}/{summary['by_tier'][t]['total']}" for t in tiers
        ]
        ax2.set_xticks(range(len(tiers)), tick_labels, fontsize=8)
        ax2.set_ylabel("failures")
    if not any(fails.values()):
        ax2.text(
            0.5,
            0.5,
            "no gate reported a failure",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=10,
            color="#059669",
        )
    ax2.set_title("Failures by gate (a gate that never fires is a gate that works)", fontsize=9.5)
    fig.suptitle("The generator's output is checked before anything trains on it", fontsize=11, y=1.03)
    _save(plt, fig, "fig11_validation.png", "four-gate validation of a generated batch")


def fig_ablation() -> None:
    """The headline result: three arms, and the gap each one leaves behind.

    Nothing consumed ``eval_ablation.json`` before this figure existed, so the
    experiment's actual result was only ever visible in a terminal scrollback.
    The bars are the two numbers that matter — mean reward on the *train*
    harnesses and on the *held-out* harness — and the annotation is their
    difference, because a level without the baseline beside it is not a result.
    """
    plt = _plt()
    data = _load("outputs/eval_ablation.json")
    if not data:
        SKIPPED.append(("fig13_ablation", "no eval artifact — the ablation has not been run"))
        return
    runs = data.get("runs") or {}
    if not runs:
        SKIPPED.append(("fig13_ablation", "eval artifact carries no runs"))
        return

    order = [n for n in ("baseline",) if n in runs] + [n for n in runs if n != "baseline"]
    tr = [runs[n]["mean_train_harness"] for n in order]
    ho = [runs[n]["mean_heldout_harness"] for n in order]
    gaps = [runs[n]["gap"] for n in order]

    # Short tick labels: the arm names are long and run together otherwise.
    def short(name: str) -> str:
        if name == "baseline":
            return "baseline"
        return "multi" if "multi" in name else ("single" if "single" in name else name)

    labels = [short(n) for n in order]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    xs = range(len(order))
    w = 0.36

    ax = axes[0]
    ax.bar([x - w / 2 for x in xs], tr, w, label="train harnesses", color="#1f6feb", edgecolor="none")
    # A zero-height bar is indistinguishable from missing data, so zero is drawn
    # as a hatched outline with its value printed. In this run the held-out mean
    # is exactly 0 in every arm, and that is the single most important fact in
    # the figure — it is why the gap difference equals the train difference.
    ax.bar(
        [x + w / 2 for x in xs], ho, w, label="held-out harness",
        color="#fce7f3", edgecolor="#be185d", hatch="///", linewidth=1.0,
    )
    for x, v in zip(xs, ho, strict=False):
        ax.text(x + w / 2, max(v, 0) + 0.004, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8, color="#be185d")
    for x, v in zip(xs, tr, strict=False):
        ax.text(x - w / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(list(xs), labels, fontsize=9)
    ax.set_ylabel("mean reward")
    ax.set_ylim(0, max(tr + ho + [0.01]) * 1.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Where each arm scores", fontsize=10.5)

    ax2 = axes[1]
    colors = ["#6b7280" if n == "baseline" else ("#059669" if "multi" in n else "#d97706") for n in order]
    ax2.bar(list(xs), gaps, color=colors, edgecolor="none")
    for x, g in zip(xs, gaps, strict=False):
        ax2.text(x, g + (0.004 if g >= 0 else -0.004), f"{g:+.3f}", ha="center",
                 va="bottom" if g >= 0 else "top", fontsize=8.5, color=INK)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xticks(list(xs), labels, fontsize=9)
    ax2.set_ylabel("gap = train - held-out")
    ax2.set_title("The gap, and what training did to it", fontsize=10.5)

    base = runs.get("baseline")
    note = ""
    if base:
        deltas = [f"{short(n)}: {runs[n]['gap'] - base['gap']:+.3f}" for n in order if n != "baseline"]
        if deltas:
            note = "  vs baseline — " + ", ".join(deltas)
    fig.suptitle(
        f"Cross-harness generalization, n={data.get('n', '?')} rollouts/cell, "
        f"seed={data.get('seed', '?')}{note}",
        fontsize=9.5,
        y=1.05,
    )
    _save(plt, fig, "fig13_ablation.png", "baseline vs single vs multi: levels and gaps")


def _find_train_summaries() -> list[Path]:
    """Every pipeline-arm summary under ``outputs/``, longest run first.

    This used to be a single hardcoded path (``outputs/train-verify/``), which
    is an artifact of a one-off correctness run rather than of the pipeline.
    The pipeline writes ``outputs/train-single-s<N>/`` and
    ``outputs/train-multi-s<N>/``, so the hardcoded figure silently drew a
    *different* run than the README described — and nothing about the output
    looked wrong, because a training curve is a training curve.

    Only runs that followed the ablation protocol count, which is two
    conditions: the directory is named ``train-{single,multi}-s<N>``, *and* the
    summary records a ``scan``. The second is not cosmetic — it is what proves
    the run consumed the difficulty scan, and a run with the signal filter off
    is a correctness check rather than an arm. ``train-single-s2`` fails on the
    second condition (it has no ``scan`` key), and ``train-verify`` fails on the
    first. Excluding them matters because overlaying a two-step smoke run on a
    64-step arm would make the figure claim a comparison that was never run.
    """
    found = []
    for p in sorted((ROOT / "outputs").glob("train-*/train_summary.json")):
        if not re.fullmatch(r"train-(single|multi)-s\d+", p.parent.name):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not d.get("scan"):
            continue
        hist = d.get("log_history") or d.get("history") or []
        if hist:
            found.append((len(hist), p, d))
    found.sort(key=lambda t: t[0], reverse=True)
    return found


def fig_training() -> None:
    """The training curve, with the zero-gradient steps marked.

    Draws every arm the pipeline produced, because the ablation is a comparison
    between arms and a single curve cannot show it. The longest run is the
    headline panel; the rest are overlaid so ``single`` and ``multi`` are read
    against each other rather than one at a time.
    """
    plt = _plt()
    runs = _find_train_summaries()
    if not runs:
        SKIPPED.append(("fig12_training", "no training artifact under outputs/train-*/"))
        return

    # Panel 1 shows the longest run in full (with the shaded zero-gradient
    # steps); panels 2-3 overlay all arms so the ablation is visible.
    _, _, headline = runs[0]
    hist = headline.get("log_history") or headline.get("history") or []

    steps = [h.get("step", i + 1) for i, h in enumerate(hist)]
    rew = [h.get("reward") or h.get("rewards") for h in hist]
    frac = [h.get("frac_reward_zero_std") for h in hist]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))

    # -- panel 1: the headline run, with zero-gradient steps shaded ---------
    ax = axes[0]
    pts = [(s, y) for s, y in zip(steps, rew, strict=False) if y is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#059669", lw=2, ms=6)
    # A step where every group had zero reward variance contributes no gradient;
    # marking it is the difference between "loss went to zero" meaning converged
    # and meaning nothing was learned.
    dead_steps = [s for s, f in zip(steps, frac, strict=False) if f is not None and f >= 1.0]
    # Panel 1 is the densest of the three and the shaded spans run its full
    # height, so the explanation goes in a legend below the axes rather than as
    # floating text — text placed inside the axes lands on the curve either way.
    legend_handles = [plt.Line2D([], [], color="#059669", marker="o", lw=2, ms=6, label="mean reward")]
    for s in dead_steps:
        ax.axvspan(s - 0.4, s + 0.4, color="#fee2e2", zorder=0)
    if dead_steps:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor="#fee2e2", edgecolor="none",
                          label=f"frac_reward_zero_std = 1 (no gradient) — "
                                f"{len(dead_steps)} of {len(hist)} steps")
        )
    ax.legend(
        handles=legend_handles,
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
    )
    ax.set_xlabel("step")
    ax.set_ylabel("mean reward")
    ax.set_title(f"{headline.get('mode', '?')} — {len(hist)} steps, {len(dead_steps)} dead", fontsize=10.5)

    # -- panels 2-3: every arm overlaid, so the ablation is legible ---------
    # A single curve cannot show a comparison, and the whole point of running
    # two arms is the comparison. Each arm gets its own colour and a legend.
    for ax, key, label in ((axes[1], "loss", "loss"), (axes[2], "grad_norm", "grad norm")):
        for _, path, d in runs:
            h = d.get("log_history") or d.get("history") or []
            xs = [r.get("step", i + 1) for i, r in enumerate(h)]
            ys = [r.get(key) for r in h]
            p = [(x, y) for x, y in zip(xs, ys, strict=False) if y is not None]
            if not p:
                continue
            tag = path.parent.name
            color = HARNESS_COLOR.get(tag.split("-")[1] if "-" in tag else tag, MUTED)
            ax.plot([q[0] for q in p], [q[1] for q in p], "o-", lw=1.8, ms=4, color=color, label=tag)
        ax.set_xlabel("step")
        ax.set_title(label, fontsize=10.5)
        if len(runs) > 1:
            ax.legend(fontsize=7, frameon=False)

    arms = ", ".join(
        f"{p.parent.name} ({len(d.get('log_history') or d.get('history') or [])} steps)"
        for _, p, d in runs
    )
    fig.suptitle(
        f"Training — {len(runs)} arms: {arms}\n"
        f"headline rows_used={headline.get('rows_used', '?')}, "
        f"dropped_dead={len(headline.get('rows_dropped_dead') or [])}, "
        f"kept_thin={len(headline.get('rows_kept_under_measured') or [])}",
        fontsize=10,
        y=1.06,
    )
    _save(plt, fig, "fig12_training.png", f"{len(runs)} arms, dead steps shaded")


FIGURES = (
    ("fig01_scan_matrix", "pass-rate matrix over harness x task"),
    ("fig02_measurement", "Wilson intervals per cell"),
    ("fig03_verdicts", "live / dead / under_measured counts"),
    ("fig04_grpo_signal", "GRPO group signal probability (analytic)"),
    ("fig05_bands", "difficulty band per cell"),
    ("fig06_stop_reasons", "why rollouts stopped"),
    ("fig07_turns", "turns and tool calls vs reward"),
    ("fig08_toolcall_reward", "tool calls vs reward per tier"),
    ("fig09_edit_budget", "annealed edit budget (analytic)"),
    ("fig10_ledger", "harness evolution ledger"),
    ("fig11_validation", "four-gate validation of generated tasks"),
    ("fig12_training", "training curve"),
    ("fig13_ablation", "baseline vs single vs multi: levels and gaps"),
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate docs/figures/ from the recorded artifacts.")
    ap.add_argument("--scan", default="outputs/scan_all.json", help="the n=32 scan artifact")
    ap.add_argument("--prev", default="outputs/scan_all_n8.json", help="the older, smaller scan, for comparison")
    ap.add_argument("--only", default=None, help="comma-separated figure names to draw")
    args = ap.parse_args()

    scan = _load(args.scan) or {}
    prev = _load(args.prev)

    print("=" * 74)
    print("PLOTTING — every figure is derived from a recorded artifact")
    print("=" * 74)
    if scan:
        print(f"scan      : {args.scan}  n={scan.get('n', '?')}  records={len(scan.get('records', []))}")
    if prev:
        print(f"comparison: {args.prev}  n={prev.get('n', '?')}")
    print()

    wanted = set(args.only.split(",")) if args.only else None

    def run(name: str, fn, *a):
        if wanted is not None and name not in wanted:
            return
        try:
            fn(*a)
        except Exception as exc:  # a broken figure must not hide the others
            SKIPPED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  [SKIP] {name}: {type(exc).__name__}: {exc}")

    run("fig01_scan_matrix", fig_scan_matrix, scan)
    run("fig02_measurement", fig_measurement, scan)
    run("fig03_verdicts", fig_verdicts, scan, prev)
    run("fig04_grpo_signal", fig_grpo_signal)
    run("fig05_bands", fig_bands, scan)
    run("fig06_stop_reasons", fig_stop_reasons, scan)
    run("fig07_turns", fig_turns, scan)
    run("fig08_toolcall_reward", fig_toolcall_reward, scan)
    run("fig09_edit_budget", fig_edit_budget)
    run("fig10_ledger", fig_ledger)
    run("fig11_validation", fig_validation)
    run("fig12_training", fig_training)
    run("fig13_ablation", fig_ablation)

    print()
    if SKIPPED:
        print(f"{len(SKIPPED)} figure(s) or artifact(s) not produced:")
        for name, why in SKIPPED:
            print(f"  - {name}: {why}")
        print()
    made = sorted(p.name for p in OUT.glob("*.png"))
    print(f"docs/figures/ now holds {len(made)} PNG(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
