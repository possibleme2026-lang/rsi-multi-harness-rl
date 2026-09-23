"""pytest bootstrap for ``tests/``.

The tests in this directory are **standalone scripts**, not pytest test
functions: each has a ``main()``, prints a labelled PASS/FAIL line per
assertion, and returns a process exit code. That shape is deliberate — the
expensive ones (``smoke_trl.py``) need a GPU and a real ``trl`` install, so
they must be runnable from a bare shell without pytest in the picture::

    ./run.sh tests/smoke_env.py

This file exists for the two cases where pytest *is* in the picture:

  * ``import multiharness`` must work during collection, which means ``src/``
    has to be on ``sys.path`` before any test module is imported. Doing it here
    rather than with a ``sys.path`` insert in every file keeps the "insert
    ``src/``, never the repository root" rule (see
    ``multiharness/_bootstrap.py``) in one reviewable place.
  * ``tests`` must be importable as a package for ``-p`` style collection, so
    the directory is added to ``sys.path`` too.

No test functions are collected from here, and none should be added: the
scripts own their own pass/fail reporting, and duplicating that in pytest
assertions would create two definitions of "the test passed".
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_SRC = _REPO_ROOT / "src"

# src/ first — never the repository root. A root on sys.path lets any sibling
# top-level directory shadow an installed package of the same name; that exact
# failure produced a false "tool-call rate: 0%" reading during development.
for _p in (_SRC, _HERE):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
