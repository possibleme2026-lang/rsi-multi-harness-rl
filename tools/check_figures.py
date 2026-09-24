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

"""Assert that every committed figure is a real, non-degenerate image.

Run:
    python tools/check_figures.py

Why this is worth a CI job
--------------------------
A committed PNG is evidence, and evidence that is never checked decays. The
failure mode is specific and quiet: a plotting bug that produces a blank or
near-blank canvas still writes a valid PNG of plausible size, the README still
embeds it, and the figure silently stops supporting the sentence next to it. A
reader cannot tell "this figure shows the result" from "this figure is an empty
white rectangle".

So the check is on the *pixels*, not on the file's existence. Standard library
only — ``zlib`` on the IDAT stream — because this runs in the job that installs
nothing, and a check that needs the plotting library cannot catch a bug in it.

What is asserted, per figure
----------------------------
1. It is a PNG with a plausible header and at least one IDAT chunk.
2. It decodes to a real pixel grid whose dimensions match the header.
3. It is not blank: at least a small fraction of pixels differ from the modal
   colour.
4. It carries enough distinct colours to be a plot rather than a solid fill —
   a figure with two colours is a rectangle with a border.
5. It is not a single row or column, which is what a mis-specified figure size
   produces.

The thresholds are deliberately loose. This is a smoke test for *degenerate*
output, not an aesthetic judgement, and a check that fails on a legitimately
sparse figure would be turned off within a week.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "docs" / "figures"

#: Figures that must exist. A figure removed from this list is a figure the
#: README no longer embeds; the two are checked together by check_readme_i18n.
REQUIRED = (
    "fig01_scan_matrix.png",
    "fig02_measurement.png",
    "fig03_verdicts.png",
    "fig04_grpo_signal.png",
    "fig05_bands.png",
    "fig06_stop_reasons.png",
    "fig07_turns.png",
    "fig08_toolcall_reward.png",
    "fig09_edit_budget.png",
    "fig10_ledger.png",
    "fig11_validation.png",
    "fig12_training.png",
    "fig13_ablation.png",
)

#: A plot of a single series on a white background is still mostly white, so
#: the bar for "not blank" is low. What this catches is a figure that is
#: *entirely* one colour.
MIN_NON_MODAL_FRACTION = 0.005
#: Antialiasing alone produces many shades, but a solid rectangle with a
#: one-pixel border produces about two. Anything under this is not a plot.
MIN_DISTINCT_COLOURS = 24
MIN_SIDE = 120

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    if not ok:
        FAILS.append(label)


def decode_png(path: Path) -> tuple[int, int, list[tuple[int, int, int]]]:
    """Decode a PNG to ``(width, height, pixels)`` using only the stdlib.

    Handles the subset matplotlib writes: 8-bit RGB or RGBA, no interlacing.
    Anything else raises, because a figure this cannot read is a figure whose
    contents cannot be asserted — and silently skipping it would make the whole
    job decorative.
    """
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG (bad signature)")

    pos = 8
    width = height = None
    bit_depth = colour_type = None
    idat = bytearray()
    while pos < len(raw):
        (length,) = struct.unpack(">I", raw[pos : pos + 4])
        ctype = raw[pos + 4 : pos + 8]
        data = raw[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, colour_type, _, _, interlace = struct.unpack(">IIBBBBB", data)
            if interlace != 0:
                raise ValueError("interlaced PNG is not supported")
            if bit_depth != 8:
                raise ValueError(f"unsupported bit depth {bit_depth}")
        elif ctype == b"IDAT":
            idat += data
        elif ctype == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("no IHDR chunk")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(colour_type)
    if channels is None:
        raise ValueError(f"unsupported colour type {colour_type}")

    buf = zlib.decompress(bytes(idat))
    stride = width * channels
    pixels: list[tuple[int, int, int]] = []
    prev = bytearray(stride)
    at = 0
    for _ in range(height):
        ftype = buf[at]
        at += 1
        line = bytearray(buf[at : at + stride])
        at += stride
        # Undo the per-scanline filter. PNG filters are cumulative across rows,
        # so this has to run in order.
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        elif ftype != 0:
            raise ValueError(f"unknown filter type {ftype}")
        prev = line
        if channels >= 3:
            for i in range(0, stride, channels):
                pixels.append((line[i], line[i + 1], line[i + 2]))
        else:
            for i in range(0, stride, channels):
                v = line[i]
                pixels.append((v, v, v))
    return width, height, pixels


def analyse(path: Path) -> dict:
    w, h, px = decode_png(path)
    counts: dict[tuple[int, int, int], int] = {}
    for p in px:
        counts[p] = counts.get(p, 0) + 1
    total = len(px) or 1
    modal, modal_n = max(counts.items(), key=lambda kv: kv[1])
    return {
        "width": w,
        "height": h,
        "pixels": total,
        "distinct": len(counts),
        "modal": modal,
        "non_modal_fraction": 1.0 - (modal_n / total),
    }


def main() -> int:
    print("=" * 74)
    print("CHECK — every committed figure is a real, non-degenerate image")
    print("=" * 74)

    if not FIGDIR.is_dir():
        print(f"  [FAIL] {FIGDIR.relative_to(ROOT)} does not exist")
        return 1

    missing = [n for n in REQUIRED if not (FIGDIR / n).is_file()]
    check(f"all {len(REQUIRED)} required figures are present", not missing, f"missing: {missing}")

    extra = sorted(p.name for p in FIGDIR.glob("*.png") if p.name not in REQUIRED)
    check("no unlisted figures", not extra, f"unlisted: {extra} — add them to REQUIRED or delete them")

    print()
    for name in REQUIRED:
        path = FIGDIR / name
        if not path.is_file():
            continue
        try:
            info = analyse(path)
        except Exception as exc:
            check(f"{name} decodes", False, f"{type(exc).__name__}: {exc}")
            continue
        ok_size = info["width"] >= MIN_SIDE and info["height"] >= MIN_SIDE
        ok_blank = info["non_modal_fraction"] >= MIN_NON_MODAL_FRACTION
        ok_colour = info["distinct"] >= MIN_DISTINCT_COLOURS
        status = "PASS" if (ok_size and ok_blank and ok_colour) else "FAIL"
        print(
            f"  [{status}] {name:<28} {info['width']}x{info['height']}  "
            f"colours={info['distinct']:<6} non-modal={info['non_modal_fraction']:.3f}"
        )
        if not ok_size:
            FAILS.append(f"{name}: {info['width']}x{info['height']} is smaller than {MIN_SIDE}")
        if not ok_blank:
            FAILS.append(f"{name}: {info['non_modal_fraction']:.4f} of pixels differ from the modal colour")
        if not ok_colour:
            FAILS.append(f"{name}: only {info['distinct']} distinct colours")

    print()
    print("=" * 74)
    if FAILS:
        print(f"FAILED — {len(FAILS)} check(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL FIGURES OK — each decodes, and none is blank")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
