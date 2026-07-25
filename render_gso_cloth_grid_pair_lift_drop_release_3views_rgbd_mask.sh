#!/usr/bin/env bash

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-$(readlink -f "${REPO_ROOT}/output")}"
readonly DATASET_NAME="${DATASET_NAME:-cloth_grid_pair_lift_drop_release_rgbd_mask}"
readonly DOCKER_IMAGE="${DOCKER_IMAGE:-kubruntu-blender-torch:cu118}"
readonly GPU_ID="${GPU_ID:-0}"
readonly ASSET_ID="${ASSET_ID:-Cole_Hardware_Dishtowel_Stripe}"
readonly PAIR_LIMIT="${PAIR_LIMIT:-120}"
readonly GRID_SIZE="${GRID_SIZE:-4}"
readonly FRAME_END="${FRAME_END:-168}"
readonly FRAME_RATE="${FRAME_RATE:-24}"
readonly STEP_RATE="${STEP_RATE:-2400}"
readonly RESOLUTION="${RESOLUTION:-256x256}"

readonly OUTPUT_DATASET_DIR="${OUTPUT_ROOT}/${DATASET_NAME}"
mkdir -p "${OUTPUT_DATASET_DIR}"

grid_points=()
for y in $(seq 0 $((GRID_SIZE - 1))); do
  for x in $(seq 0 $((GRID_SIZE - 1))); do
    grid_points+=("x${x}_y${y}")
  done
done

total_unordered_pairs=$(( ${#grid_points[@]} * (${#grid_points[@]} - 1) / 2 ))
if (( PAIR_LIMIT <= 0 || PAIR_LIMIT > total_unordered_pairs )); then
  echo "PAIR_LIMIT must be between 1 and ${total_unordered_pairs}, got ${PAIR_LIMIT}" >&2
  exit 1
fi

has_complete_case() {
  local case_dir="$1"
  local expected_min_count="$((FRAME_END + 1))"
  local view_dir
  local layer
  local pattern
  local count

  for layer in image depth segmentation; do
    case "${layer}" in
      image) pattern="rgba_*.png" ;;
      depth) pattern="depth_*.tiff" ;;
      segmentation) pattern="segmentation_*.png" ;;
    esac
    if [[ ! -d "${case_dir}/${layer}" ]]; then
      return 1
    fi
    count=$(find "${case_dir}/${layer}" \
      -maxdepth 1 \
      -type f \
      -name "${pattern}" \
      | wc -l)
    if [[ "${count}" -lt "${expected_min_count}" ]]; then
      return 1
    fi
  done
}

failures=()
rendered=0
stop=0
for ((i = 0; i < ${#grid_points[@]}; i++)); do
  for ((j = i + 1; j < ${#grid_points[@]}; j++)); do
    point_a="${grid_points[$i]}"
    point_b="${grid_points[$j]}"
    pair_label="p_${point_a}__${point_b}"
    case_dir="${OUTPUT_DATASET_DIR}/${pair_label}"

    if has_complete_case "${case_dir}"; then
      echo "Skip completed case: ${pair_label}"
      rendered=$((rendered + 1))
      if (( rendered >= PAIR_LIMIT )); then
        stop=1
        break
      fi
      continue
    fi

    echo "============================================================"
    echo "Rendering ${ASSET_ID} lift-drop-release pair=${point_a},${point_b} (${rendered}/${PAIR_LIMIT})"
    echo "Output: ${case_dir}"
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
        python3 examples/gso_spring_mass_control.py \
          --asset_id="${ASSET_ID}" \
          --particle_spacing=0.15 \
          --k_neighbors=16 \
          --spring_stiffness=25 \
          --damping=0.01 \
          --step_rate="${STEP_RATE}" \
          --frame_rate="${FRAME_RATE}" \
          --frame_end="${FRAME_END}" \
          --resolution="${RESOLUTION}" \
          --job-dir="/output/${DATASET_NAME}/${pair_label}" \
          --scratch_dir="/tmp/kubric_${DATASET_NAME}_${pair_label}" \
          --device=cuda \
          --control_mode=grid_pair \
          --control_grid_size="${GRID_SIZE}" \
          --control_grid_points="${point_a},${point_b}" \
          --trajectory_mode=lift_drop_release \
          --lift_height=0.8 \
          --lift_seconds=4 \
          --hold_seconds=0.5 \
          --lower_seconds=2.5 \
          --release_height=0.05 \
          --camera_views=front \
          --camera_distance=2.4 \
          --camera_height=1.4 \
          --camera_look_at_z=0.3 \
          --samples_per_pixel=16 \
          --render_depth \
          --render_segmentation \
          --skip_auxiliary_data; then
      echo "Completed lift-drop-release pair=${point_a},${point_b}"
    else
      echo "Failed lift-drop-release pair=${point_a},${point_b}" >&2
      failures+=("${pair_label}")
    fi

    rendered=$((rendered + 1))
    if (( rendered >= PAIR_LIMIT )); then
      stop=1
      break
    fi
  done
  if (( stop )); then
    break
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "Failed pair labels: ${failures[*]}" >&2
  exit 1
fi

echo "Rendered ${rendered} lift-drop-release cloth cases successfully."
