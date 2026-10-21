#!/usr/bin/env bash
#
# setup_venv.sh - create this project's Python virtual environment (.venv) and install requirements.txt
#
# Usage, from Terminal in the project folder:
#   bash setup_venv.sh            # create .venv (or reuse a healthy one) and install everything
#   source .venv/bin/activate     # activate it in your current shell
#
# Use a specific interpreter:   PYTHON=/opt/homebrew/bin/python3.12 bash setup_venv.sh
# Start over from scratch:      rm -rf .venv && bash setup_venv.sh
#
# Safe to re-run: an existing, working .venv is reused and pip only installs what is missing.
#
# Supported Python: 3.10 - 3.13. A pre-flight resolve of requirements.txt for macOS arm64 (Oct 2026)
# found a ready-made wheel for every dependency on 3.10 - 3.13. On 3.14, zstandard==0.23.0 (required by
# langsmith) has no wheel and must be compiled from C source; 3.9 is end-of-life.

# Refuse to be sourced: 'set -e' and 'exit' would close the Terminal tab that sourced this file.
if [ -n "${ZSH_VERSION:-}" ] || { [ -n "${BASH_VERSION:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" ]; }; then
  echo "Run this with 'bash setup_venv.sh' (not 'source'), then run: source .venv/bin/activate" >&2
  # shellcheck disable=SC2317  # 'return' runs when sourced, 'exit' when executed
  return 1 2>/dev/null || exit 1
fi

set -Eeuo pipefail

readonly VENV_DIR=".venv"
readonly REQ_FILE="requirements.txt"
readonly MIN_MINOR=10   # oldest supported: Python 3.10
readonly MAX_MINOR=13   # newest supported: Python 3.13

info() { printf '\n==> %s\n' "$*"; }
warn() { printf '\nWARNING: %s\n' "$*" >&2; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
on_error() { printf '\nERROR: setup stopped (line %s, exit code %s). The output above shows why.\n' "$1" "$2" >&2; }
trap 'on_error "$LINENO" "$?"' ERR

# Prints "3.12" for the given interpreter, or nothing if it can't run.
py_version() {
  "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true
}

# Succeeds when a "3.X" version string is inside the supported range.
is_supported() {
  local major="${1%%.*}" minor="${1#*.}"
  [ "$major" = "3" ] && [ "$minor" -ge "$MIN_MINOR" ] && [ "$minor" -le "$MAX_MINOR" ]
}

# Prints the interpreter to build the venv with: $PYTHON if set, otherwise the first
# supported one on PATH, preferring 3.12. Prints nothing when there is none.
pick_python() {
  if [ -n "${PYTHON:-}" ]; then
    command -v "$PYTHON" || true
    return
  fi
  local candidate
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && is_supported "$(py_version "$candidate")"; then
      command -v "$candidate"
      return
    fi
  done
}

# Run from the project folder, wherever this script is called from.
cd "$(dirname "$0")"
[ -f "$REQ_FILE" ] || fail "$REQ_FILE not found in $(pwd)."

# 1. Create the virtual environment, or reuse one that already works on this machine.
if [ -e "$VENV_DIR" ]; then
  venv_version="$(py_version "$VENV_DIR/bin/python")"
  [ -n "$venv_version" ] || fail "$VENV_DIR exists, but its Python doesn't run on this machine (broken, or built on another OS). Remove it and re-run: rm -rf $VENV_DIR && bash setup_venv.sh"
  is_supported "$venv_version" || warn "$VENV_DIR uses Python $venv_version, outside the tested range 3.${MIN_MINOR}-3.${MAX_MINOR}. If the install fails, run: rm -rf $VENV_DIR && bash setup_venv.sh"
  info "Reusing existing $VENV_DIR (Python $venv_version)"
else
  python_bin="$(pick_python)"
  if [ -z "$python_bin" ]; then
    if [ -n "${PYTHON:-}" ]; then fail "PYTHON=$PYTHON was not found."; fi
    fail "No Python 3.${MIN_MINOR}-3.${MAX_MINOR} on your PATH. Install one with 'brew install python@3.12', or set PYTHON=/path/to/python3.12."
  fi
  python_version="$(py_version "$python_bin")"
  [ -n "$python_version" ] || fail "$python_bin exists but doesn't run."
  is_supported "$python_version" || warn "Python $python_version is outside the tested range 3.${MIN_MINOR}-3.${MAX_MINOR}; continuing because PYTHON was set."
  info "Creating $VENV_DIR with $python_bin (Python $python_version)"
  "$python_bin" -m venv "$VENV_DIR"
fi

venv_python="$VENV_DIR/bin/python"

# 2. Install the requirements into the venv. Calling the venv's own python means nothing
#    can land in a system, Homebrew or conda Python by mistake, whatever is active right now.
info "Upgrading pip inside $VENV_DIR"
"$venv_python" -m pip install --quiet --disable-pip-version-check --upgrade pip

info "Installing $REQ_FILE (first run on a Mac downloads ~350 MB across ~140 packages; torch is ~130 MB of it)"
"$venv_python" -m pip install --disable-pip-version-check -r "$REQ_FILE"

# 3. Verify: no version conflicts, and every import the project's code uses actually resolves.
info "Checking installed packages for version conflicts"
"$venv_python" -m pip check

info "Importing everything this project's .py files import"
"$venv_python" - <<'PYCODE'
import ast
import importlib
import sys
import warnings
from pathlib import Path

# Needed at runtime but never imported by name in the code (HuggingFaceEmbeddings loads
# sentence_transformers + torch, Chroma loads chromadb); streamlit is listed for the UI.
EXTRA_MODULES = ["sentence_transformers", "torch", "chromadb", "streamlit"]

project_files = sorted(Path(".").glob("*.py"))
local_modules = {path.stem for path in project_files}
stdlib_modules = set(getattr(sys, "stdlib_module_names", ()))

# (module, attribute) pairs; attribute is None for a plain "import x".
targets = {(name, None) for name in EXTRA_MODULES}
for path in project_files:
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            targets.update((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            targets.update((node.module, alias.name) for alias in node.names)

# Skip the project's own modules (importing context.py would download an embedding model).
third_party = sorted(
    ((mod, attr) for mod, attr in targets
     if mod.split(".")[0] not in local_modules | stdlib_modules),
    key=lambda pair: (pair[0], pair[1] or ""),
)

failures = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # LangChain deprecation notices are noise for this check
    for mod, attr in third_party:
        label = f"from {mod} import {attr}" if attr else f"import {mod}"
        try:
            module = importlib.import_module(mod)
            if attr and not hasattr(module, attr):
                importlib.import_module(f"{mod}.{attr}")  # "from package import submodule"
        except Exception as exc:
            failures.append(f"{label}  ->  {type(exc).__name__}: {exc}")

if failures:
    sys.exit("Import check failed:\n  " + "\n  ".join(failures))

import torch

gpu = "yes" if torch.backends.mps.is_available() else "no"
print(f"OK: {len(third_party)} imports | Python {sys.version.split()[0]} | "
      f"torch {torch.__version__} | Apple GPU (MPS) available: {gpu}")
PYCODE

cat <<EOF

Done. Virtual environment ready: $(pwd)/$VENV_DIR

Activate it in your Terminal (a script can't activate your shell for you):
    source "$(pwd)/$VENV_DIR/bin/activate"
Leave it later with:
    deactivate
In Cursor / VS Code: "Python: Select Interpreter" -> $VENV_DIR/bin/python
EOF
