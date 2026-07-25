#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(readlink -f "${REPO_ROOT}/output")}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kubruntu-blender-torch:cu118}"
GPU_ID="${GPU_ID:-0}"
ASSET_ID="${ASSET_ID:-Cole_Hardware_Dishtowel_Stripe}"
SEED="${SEED:-42}"
PAIR_LIMIT="${PAIR_LIMIT:-120}"
GRID_SIZE=4

OUTPUT_DATASET_DIR="${OUTPUT_ROOT}/spring_mass_grid_pair_lift_slow"
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

failures=()
rendered=0
stop=0
for ((i = 0; i < ${#grid_points[@]}; i++)); do
  for ((j = i + 1; j < ${#grid_points[@]}; j++)); do
    point_a="${grid_points[$i]}"
    point_b="${grid_points[$j]}"
    pair_label="p_${point_a}__${point_b}"

    echo "============================================================"
    echo "Rendering ${ASSET_ID} control_pair=${point_a},${point_b} (${rendered}/${PAIR_LIMIT})"
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
          --step_rate=2400 \
          --frame_rate=24 \
          --frame_end=120 \
          --job-dir="/output/spring_mass_grid_pair_lift_slow/${pair_label}" \
          --scratch_dir="/tmp/kubric_spring_mass_grid_pair_lift_slow_${pair_label}" \
          --device=cuda \
          --control_mode=grid_pair \
          --control_grid_size="${GRID_SIZE}" \
          --control_grid_points="${point_a},${point_b}" \
          --lift_height=0.8 \
          --lift_seconds=5 \
          --camera_distance=2.4 \
          --camera_height=1.4 \
          --camera_look_at_z=0.3 \
          --samples_per_pixel=16 \
          --save_simulation_data; then
      echo "Completed control_pair=${point_a},${point_b}"
    else
      echo "Failed control_pair=${point_a},${point_b}" >&2
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

echo "Rendered ${rendered} grid-pair lift videos successfully."
