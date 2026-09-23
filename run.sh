#!/usr/bin/env bash
# Canonical launcher for every script in this repo.
#
# Usage:
#   ./run.sh scripts/probe.py --n 8
#   ./run.sh tests/smoke_env.py
#   MULTIHARNESS_PYTHON=/usr/bin/python3 ./run.sh tests/test_shell_timeout.py
#
# Why a launcher exists at all
# ----------------------------
# Three environment quirks are silent-failure generators rather than error
# messages, so they are fixed in one place instead of being copy-pasted (and
# eventually forgotten) into every command:
#
#  1. PYTHONPATH is often pre-populated by an outer harness with a shim
#     directory. An *empty* PYTHONPATH is not the same as an unset one — it
#     injects the current directory, which lets a sibling top-level directory
#     shadow an installed package of the same name. Unset outright.
#
#  2. Offline model loading. If HF_HUB_OFFLINE is not set, every
#     ``from_pretrained`` either reaches the network or fails with a
#     misleading "check your internet connection". Point the caches at a real
#     directory and force offline.
#
#  3. APPDATA. When this is launched from a sandbox that clears it, Python
#     computes the wrong user-site directory and ``import torch`` fails with
#     ModuleNotFoundError even though torch is installed. Set it explicitly.
#
# Every path below is overridable; the defaults target the machine this
# experiment was developed on (see README, "Reproducing").
#
#   MULTIHARNESS_PYTHON   python interpreter            (default: /c/Python314/python.exe)
#   MULTIHARNESS_APPDATA  Windows roaming app data      (default: C:\Users\%USERNAME%\AppData\Roaming)
#   MULTIHARNESS_HF_HOME  HuggingFace cache root        (default: $HOME/.cache/huggingface)
#   MULTIHARNESS_BASH     bash used by bash_minimal     (default: auto-detected by core._find_bash)
#   MULTIHARNESS_OUT      artifact directory            (default: <repo>/outputs)

set -euo pipefail

PY="${MULTIHARNESS_PYTHON:-/c/Python314/python.exe}"

unset PYTHONPATH

# Windows-style path for the interpreter; HOME may be a POSIX path in Git Bash.
APPDATA_WIN="${MULTIHARNESS_APPDATA:-}"
if [ -z "$APPDATA_WIN" ] && [ -n "${USERPROFILE:-}" ]; then
  APPDATA_WIN="$(cygpath -w "$USERPROFILE" 2>/dev/null || echo "$USERPROFILE")\\AppData\\Roaming"
fi
[ -n "$APPDATA_WIN" ] && export APPDATA="$APPDATA_WIN"

HF_ROOT="${MULTIHARNESS_HF_HOME:-${HOME:-}/.cache/huggingface}"
if [ -n "$HF_ROOT" ] && [ -d "$HF_ROOT" ]; then
  HF_WIN="$(cygpath -w "$HF_ROOT" 2>/dev/null || echo "$HF_ROOT")"
  export HF_HOME="$HF_WIN"
  export HF_HUB_CACHE="$HF_WIN\\hub"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
fi

# Offline by default: this experiment runs against a pre-populated cache, and
# a silent network fallback changes the artifact without changing the command.
# Set MULTIHARNESS_ONLINE=1 to opt out.
if [ "${MULTIHARNESS_ONLINE:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# Only export MULTIHARNESS_BASH when the caller set it; otherwise let
# core._find_bash() search, so a fresh clone on another machine still works.
if [ -n "${MULTIHARNESS_BASH:-}" ]; then
  export MULTIHARNESS_BASH
fi

export PYTHONUTF8=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec "$PY" -u "$@"
