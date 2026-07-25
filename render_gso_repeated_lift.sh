#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_FILE="${1:-${REPO_ROOT}/gso_objects.md}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(readlink -f "${REPO_ROOT}/output")}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kubruntu-blender-torch:cu118}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"

if [[ ! -f "${ASSET_FILE}" ]]; then
  echo "Asset list not found: ${ASSET_FILE}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "Could not resolve output directory: ${REPO_ROOT}/output" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/spring_mass_repeated_lift"

asset_ids=()
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%$'\r'}"
  line="${line#${line%%[![:space:]]*}}"
  line="${line%${line##*[![:space:]]}}"
  [[ -z "${line}" || "${line}" == \#* ]] && continue

  # Also accept Markdown bullets and inline-code formatting.
  line="${line#- }"
  line="${line#\* }"
  line="${line#\`}"
  line="${line%\`}"
  if [[ ! "${line}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Invalid asset ID in ${ASSET_FILE}: ${line}" >&2
    exit 1
  fi
  asset_ids+=("${line}")
done < "${ASSET_FILE}"

if (( ${#asset_ids[@]} == 0 )); then
  echo "No asset IDs found in ${ASSET_FILE}" >&2
  exit 1
fi

failures=()
for asset_id in "${asset_ids[@]}"; do
  echo "============================================================"
  echo "Rendering ${asset_id}"
  echo "============================================================"

  if docker run --rm \
      --runtime=nvidia \
      --user "$(id -u):$(id -g)" \
      --env "NVIDIA_VISIBLE_DEVICES=${GPU_ID}" \
      --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
      --env KUBRIC_USE_GPU=true \
      --env KUBRIC_CYCLES_BACKEND=CUDA \
      --env PYTHONPATH=/kubric \
      --volume "${REPO_ROOT}:/kubric" \
      --volume "${OUTPUT_ROOT}:/output" \
      --workdir /kubric \
      "${DOCKER_IMAGE}" \
      python3 examples/gso_spring_mass.py \
        --asset_id="${asset_id}" \
        --particle_spacing=0.15 \
        --k_neighbors=64 \
        --spring_stiffness=20 \
        --damping=0.01 \
        --step_rate=2400 \
        --control_vertex_index=0 \
        --control_target_x=0 \
        --control_target_y=0 \
        --camera_count=4 \
        --randomize_cameras \
        --render_depth \
        --render_segmentation \
        --repeat_count=5 \
        --initial_settle_seconds=2.0 \
        --lift_seconds=1.0 \
        --hold_seconds=0.2 \
        --settle_seconds=2.0 \
        --seed="${SEED}" \
        --job-dir=/output/spring_mass_repeated_lift \
        --scratch_dir="/tmp/kubric_spring_mass_repeated_lift_${asset_id}"; then
    echo "Completed ${asset_id}"
  else
    echo "Failed ${asset_id}" >&2
    failures+=("${asset_id}")
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "Failed asset IDs: ${failures[*]}" >&2
  exit 1
fi

echo "Rendered ${#asset_ids[@]} assets successfully."
