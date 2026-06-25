#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/mount/lcamai/latest-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mount/lcamai/result-all}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mount/lcamai/river-seg-runtime}"
FRAME_CACHE_ROOT="${FRAME_CACHE_ROOT:-${RUNTIME_ROOT}/frames}"
MANIFEST="${MANIFEST:-${RUNTIME_ROOT}/manifest.json}"
THRESHOLD_JSON="${THRESHOLD_JSON:-${SCRIPT_DIR}/config/thresholds.example.json}"

TEXT_PROMPT="${TEXT_PROMPT:-water}"

GPU_COUNT="${GPU_COUNT:-1}"
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(seq -s ' ' 0 $((GPU_COUNT - 1)))"
fi
read -r -a GPUS <<< "${GPU_IDS}"
NUM_WORKERS="${#GPUS[@]}"

FRAME_COUNT="${FRAME_COUNT:-10}"
LIMIT="${LIMIT:-0}"
SAVE_MASK="${SAVE_MASK:-1}"
SAVE_OVERLAY="${SAVE_OVERLAY:-1}"
LOOP="${LOOP:-0}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_CYCLES="${MAX_CYCLES:-0}"

export PROJECT_ROOT
export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}:${PROJECT_ROOT}/sam3:${PYTHONPATH:-}"

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local worker_root="${RUNTIME_ROOT}/workers/worker_${worker_index}_gpu_${gpu_id}"

  local args=(
    -m river_seg_server.run_segmentation
    --output-root "${OUTPUT_ROOT}"
    --runtime-root "${worker_root}"
    --manifest "${MANIFEST}"
    --threshold-json "${THRESHOLD_JSON}"
    --base-checkpoint "${BASE_CHECKPOINT}"
    --checkpoint "${CHECKPOINT}"
    --text-prompt "${TEXT_PROMPT}"
    --gpus "${gpu_id}"
    --frame-count "${FRAME_COUNT}"
    --limit "${LIMIT}"
    --worker-index "${worker_index}"
    --num-workers "${NUM_WORKERS}"
  )

  if [[ "${SAVE_MASK}" == "1" ]]; then
    args+=(--save-mask)
  fi
  if [[ "${SAVE_OVERLAY}" == "1" ]]; then
    args+=(--save-overlay)
  fi

  "${PYTHON_BIN}" "${args[@]}"
}

merge_outputs() {
  : > "${RUNTIME_ROOT}/results.jsonl"
  : > "${RUNTIME_ROOT}/alarms.jsonl"

  local total_videos=0
  local total_alarms=0
  for i in "${!GPUS[@]}"; do
    local worker_root="${RUNTIME_ROOT}/workers/worker_${i}_gpu_${GPUS[$i]}"
    cat "${worker_root}/results.jsonl" >> "${RUNTIME_ROOT}/results.jsonl"
    if [[ -f "${worker_root}/alarms.jsonl" ]]; then
      cat "${worker_root}/alarms.jsonl" >> "${RUNTIME_ROOT}/alarms.jsonl"
    fi
    local videos
    local alarms
    videos="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${worker_root}/summary.json'))['videos'])")"
    alarms="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${worker_root}/summary.json'))['alarms'])")"
    total_videos=$((total_videos + videos))
    total_alarms=$((total_alarms + alarms))
  done

  "${PYTHON_BIN}" -c "import json, time; json.dump({'archive_root':'${ARCHIVE_ROOT}','frame_cache_root':'${FRAME_CACHE_ROOT}','output_root':'${OUTPUT_ROOT}','gpus':${NUM_WORKERS},'videos':${total_videos},'alarms':${total_alarms},'merged_at':time.strftime('%Y%m%d-%H%M%S')}, open('${RUNTIME_ROOT}/summary.json','w'), ensure_ascii=False, indent=2)"
}

run_once() {
  mkdir -p "${RUNTIME_ROOT}"
  "${PYTHON_BIN}" -m river_seg_server.prepare_lcamai_cycle \
    --archive-root "${ARCHIVE_ROOT}" \
    --frame-cache-root "${FRAME_CACHE_ROOT}" \
    --manifest "${MANIFEST}" \
    --frame-count "${FRAME_COUNT}"

  rm -rf "${RUNTIME_ROOT}/workers"
  mkdir -p "${RUNTIME_ROOT}/workers"

  local pids=()
  for i in "${!GPUS[@]}"; do
    run_worker "${i}" "${GPUS[$i]}" > "${RUNTIME_ROOT}/workers/worker_${i}_gpu_${GPUS[$i]}.log" 2>&1 &
    pids+=("$!")
  done

  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" != "0" ]]; then
    echo "worker failed; see ${RUNTIME_ROOT}/workers/*.log" >&2
    exit 1
  fi

  merge_outputs
}

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo "{\"event\":\"cycle_start\",\"cycle\":${cycle},\"started\":\"$(date '+%Y%m%d-%H%M%S')\",\"workers\":${NUM_WORKERS}}"
  run_once
  if [[ "${LOOP}" != "1" ]]; then
    break
  fi
  if [[ "${MAX_CYCLES}" != "0" && "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
