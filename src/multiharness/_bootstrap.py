"""Shared helpers for the entry points in ``scripts/`` and ``tests/``.

Importing this module requires ``multiharness`` to be importable already, so
each entry point starts with this three-line block *inlined*::

    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

That insert cannot be factored out into this module — you would have to import
this module to get it, which is the problem it solves. It is three lines; the
duplication is cheaper than the indirection.

Why ``src/`` and never the repository root
------------------------------------------
A repository root on ``sys.path`` lets any top-level directory shadow an
installed package of the same name. In the repository this code was extracted
from, a sibling ``trl/`` checkout shadowed the real ``trl`` install exactly
that way, and the resulting ``No module named 'trl'``-shaped confusion
produced a false ``tool-call rate: 0%`` reading — the model was emitting
perfect tool calls the whole time. Inserting ``src/`` keeps the
"run it straight from a clone" convenience without reopening that hole.

What lives here
---------------
:func:`outputs_root` centralises the artifact directory. Each script used to
compute ``Path(__file__).parent / "outputs"`` independently, so moving a script
between directories silently redirected its output. One definition, one
answer, overridable.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ROOT", "outputs_root"]

#: Repository root, from this file's location
#: (``<repo>/src/multiharness/_bootstrap.py`` -> ``<repo>``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def outputs_root() -> Path:
    """Directory for run artifacts: ``<repo>/outputs`` or ``$MULTIHARNESS_OUT``.

    Deliberately does not create the directory — a module-level ``mkdir`` would
    give importing a script the side effect of writing to disk, which breaks
    read-only use and makes ``--help`` create directories. Callers create it
    when they have something to write.
    """
    env = os.environ.get("MULTIHARNESS_OUT")
    return Path(env).expanduser() if env else REPO_ROOT / "outputs"
