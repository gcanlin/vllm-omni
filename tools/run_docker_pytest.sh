#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
IMAGE="${IMAGE:-vllm-omni-pr5885:f57826a4f-cuda}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [pytest arguments]

Runs a focused pytest target from the current checkout in IMAGE. The runner
installs pytest-mock when needed and explicitly forwards proxy variables.

Required when the image needs Python packages from the network:
  HTTP_PROXY and HTTPS_PROXY

Example:
  HTTP_PROXY=http://proxy.example:8080 HTTPS_PROXY=http://proxy.example:8080 \\
    $0 tests/entrypoints/test_async_omni_diffusion_config.py::test_stage_override_preserves_model_extras_for_default_diffusion_stage -q
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

if [[ -z "${HTTP_PROXY:-}" || -z "${HTTPS_PROXY:-}" ]]; then
  echo 'HTTP_PROXY and HTTPS_PROXY must be set for Docker dependency installation.' >&2
  exit 1
fi

export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

cd "${REPO_ROOT}"
exec docker run --rm \
  -e HTTP_PROXY -e HTTPS_PROXY -e http_proxy -e https_proxy -e NO_PROXY -e no_proxy \
  -v "${REPO_ROOT}:/app/vllm-omni" -w /app/vllm-omni \
  --entrypoint sh "${IMAGE}" \
  -lc 'python -m pip install --quiet pytest-mock && python -m pytest -o addopts="" -o cache_dir=/tmp/vllm-omni-pytest-cache "$@"' sh "$@"