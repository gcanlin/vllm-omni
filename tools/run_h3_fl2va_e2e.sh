#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-vllm-omni-pr5885:f57826a4f-cuda}"
MODEL_ROOT="${MODEL_ROOT:-/mnt/disk1/yuanwu/workspace/vllm-omni/test-report-drafts/pr5885-performance/a0b254ade/disaggregated/model-view}"
HF_CACHE="${HF_CACHE:-/mnt/disk5/HF_CACHE}"
FIRST_FRAME="${FIRST_FRAME:-/mnt/disk1/yuanwu/workspace/vllm-omni/test-report-drafts/pr5885-e2e/fl2va-input.png}"
HOST_GPUS="${HOST_GPUS:-0,2,3,4}"
PORT="${PORT:-18093}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-omni-h3-fl2va-tp4-e2e}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/e2e-artifacts/fl2va-tp4}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-i2va-fl2va-tp4}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--start-only] [--keep-container]

Runs MiniMax H3 FL2VA I2VA on four GPUs with one TP=4 replica and no offload.
Override IMAGE, MODEL_ROOT, HF_CACHE, FIRST_FRAME, HOST_GPUS, PORT, CONTAINER_NAME,
ARTIFACT_DIR, or OUTPUT_BASENAME through environment variables.
EOF
}

START_ONLY=false
KEEP_CONTAINER=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-only) START_ONLY=true ;;
    --keep-container) KEEP_CONTAINER=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

for path in "${MODEL_ROOT}/FL2VA" "${HF_CACHE}" "${FIRST_FRAME}"; do
  [[ -e "${path}" ]] || { printf 'Required path is missing: %s\n' "${path}" >&2; exit 1; }
done
command -v docker >/dev/null || { echo 'docker is required' >&2; exit 1; }

mkdir -p "${ARTIFACT_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d --init \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus "\"device=${HOST_GPUS}\"" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HOME=/mnt/disk5/HF_CACHE \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_MODULES_CACHE=/tmp/hf_modules \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -v "${HF_CACHE}:/mnt/disk5/HF_CACHE:ro" \
  -v "${MODEL_ROOT}:/model:ro" \
  -v "$(dirname "${FIRST_FRAME}"):/inputs:ro" \
  -v "${ARTIFACT_DIR}:/artifacts" \
  "${IMAGE}" \
  vllm-omni serve /model/FL2VA --omni \
    --host 0.0.0.0 --port "${PORT}" --trust-remote-code --task-type fl2va \
    --distributed-executor-backend mp --dtype bfloat16 --tensor-parallel-size 4 \
    --text-encoder-tp-size 4 --vae-patch-parallel-size 4 >/dev/null

printf 'Waiting for %s on port %s...\n' "${CONTAINER_NAME}" "${PORT}"
if ! curl --fail --silent --show-error --retry 150 --retry-connrefused --retry-delay 2 --max-time 600 "http://127.0.0.1:${PORT}/health" >/dev/null; then
  docker logs --tail 500 "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

if [[ "${START_ONLY}" == "true" ]]; then
  echo "Service ready: http://127.0.0.1:${PORT}"
  exit 0
fi

headers="${ARTIFACT_DIR}/${OUTPUT_BASENAME}.headers"
video="${ARTIFACT_DIR}/${OUTPUT_BASENAME}.mp4"
metrics="${ARTIFACT_DIR}/${OUTPUT_BASENAME}.metrics"
probe="${ARTIFACT_DIR}/${OUTPUT_BASENAME}.ffprobe.json"
start="$(date +%s.%N)"
curl --fail --silent --show-error --max-time 1800 -D "${headers}" \
  -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=A fluffy orange tabby cat with bright green eyes walks across a sunlit wooden kitchen floor, pauses beside a blue ceramic bowl, looks directly into the camera, then playfully bats a small red ball. Warm natural daylight, detailed orange fur, realistic cinematic video, smooth camera movement, vivid colors, clear subject.' \
  -F 'fps=24' -F 'num_inference_steps=8' -F 'flow_shift=12.0' -F 'seed=123' \
  -F 'extra_params={"task":"fl2va","duration":4.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" -o "${video}"
end="$(date +%s.%N)"
awk -v start="${start}" -v end="${end}" 'BEGIN { printf "wall_seconds=%.3f\n", end-start }' >"${metrics}"
stat --printf='bytes=%s\n' "${video}" >>"${metrics}"
docker exec "${CONTAINER_NAME}" ffprobe -v error \
  -show_entries format=format_name,duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames \
  -of json "/artifacts/${OUTPUT_BASENAME}.mp4" >"${probe}"
cat "${metrics}"
cat "${probe}"

if [[ "${KEEP_CONTAINER}" != "true" ]]; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi