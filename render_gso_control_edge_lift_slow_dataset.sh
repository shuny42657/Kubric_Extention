#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(readlink -f "${REPO_ROOT}/output")}"
DOCKER_IMAGE="${DOCKER_IMAGE:-kubruntu-blender-torch:cu118}"
GPU_ID="${GPU_ID:-0}"
ASSET_ID="${ASSET_ID:-Cole_Hardware_Dishtowel_Stripe}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_ROOT}/spring_mass_edge_lift"

edge_specs=(
  "x_min 0 min"
  "x_max 0 max"
  "y_min 1 min"
  "y_max 1 max"
)

failures=()
for spec in "${edge_specs[@]}"; do
  read -r edge_label edge_axis edge_side <<<"${spec}"
  echo "============================================================"
  echo "Rendering ${ASSET_ID} edge=${edge_label}"
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
        --frame_end=90 \
        --job-dir="/output/spring_mass_edge_lift_slow/${edge_label}" \
        --scratch_dir="/tmp/kubric_spring_mass_edge_lift_slow${edge_label}" \
        --device=cuda \
        --control_mode=edge \
        --control_edge_axis="${edge_axis}" \
        --control_edge_side="${edge_side}" \
        --control_edge_band_fraction=0.03 \
        --control_edge_max_points=2 \
        --lift_height=0.6 \
        --lift_seconds=4 \
        --camera_distance=2.4 \
        --camera_height=1.4 \
        --camera_look_at_z=0.3 \
        --samples_per_pixel=16 \
        --save_simulation_data; then
    echo "Completed edge=${edge_label}"
  else
    echo "Failed edge=${edge_label}" >&2
    failures+=("${edge_label}")
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "Failed edge labels: ${failures[*]}" >&2
  exit 1
fi

echo "Rendered ${#edge_specs[@]} edge-lift videos successfully."
