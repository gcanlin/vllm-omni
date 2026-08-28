#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [pytest arguments]

Runs pytest from this checkout's virtual environment. Create it once with:
  uv venv --python 3.12 --seed
  uv pip install -e '.[dev]'

Override the interpreter with PYTHON_BIN=/path/to/python.
Examples:
  $0 tests/entrypoints/test_async_omni_diffusion_config.py::test_stage_override_preserves_model_extras_for_default_diffusion_stage -q
  $0 tests/engine/test_async_omni_engine_input.py -q
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Missing project virtual environment: %s\n' "${PYTHON_BIN}" >&2
  printf "Create it with: cd %s && uv venv --python 3.12 --seed && uv pip install -e '.[dev]'\n" "${REPO_ROOT}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import pytest, vllm, vllm_omni' >/dev/null 2>&1; then
  printf 'The selected interpreter must provide pytest, vllm, and vllm_omni: %s\n' "${PYTHON_BIN}" >&2
  printf "Install project dependencies with: cd %s && uv pip install --python %s -e '.[dev]'\n" "${REPO_ROOT}" "${PYTHON_BIN}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m pytest -o addopts='' -o cache_dir=/tmp/vllm-omni-pytest-cache "$@"