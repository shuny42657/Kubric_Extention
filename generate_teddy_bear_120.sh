#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="${IMAGE_NAME:-kubruntu-blender-torch:cu118}"
readonly ASSET_ID="${ASSET_ID:-Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige}"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/output/teddy_bear}"
readonly OUTPUT_ROOT_IN_CONTAINER="${OUTPUT_ROOT_IN_CONTAINER:-/output/teddy_bear}"
readonly TOPDOWN_DIR="${TOPDOWN_DIR:-${SCRIPT_DIR}/output/teddy_bear_topdown_control_points}"
readonly TOPDOWN_DIR_IN_CONTAINER="${TOPDOWN_DIR_IN_CONTAINER:-/output/teddy_bear_topdown_control_points}"
readonly TOPDOWN_CONTROL_POINTS="${TOPDOWN_CONTROL_POINTS:-${TOPDOWN_DIR}/topdown_control_points.npz}"
readonly TOPDOWN_CONTROL_POINTS_IN_CONTAINER="${TOPDOWN_CONTROL_POINTS_IN_CONTAINER:-${TOPDOWN_DIR_IN_CONTAINER}/topdown_control_points.npz}"
readonly PAIR_LIMIT="${PAIR_LIMIT:-120}"
readonly NUM_CONTROL_POINTS="${NUM_CONTROL_POINTS:-16}"
readonly FPS="${FPS:-24}"
readonly CRF="${CRF:-18}"
readonly PRESET="${PRESET:-medium}"
readonly OVERWRITE_MP4="${OVERWRITE_MP4:-1}"
readonly FRAME_END="${FRAME_END:-60}"
readonly EXPECTED_FRAME_COUNT="${EXPECTED_FRAME_COUNT:-$((FRAME_END + 1))}"

mkdir -p "${OUTPUT_ROOT}" "${TOPDOWN_DIR}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found in PATH" >&2
  exit 1
fi

docker_kubric() {
  docker run --rm \
    --runtime=nvidia \
    --user "$(id -u):$(id -g)" \
    --env NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    --env KUBRIC_USE_GPU=true \
    --env KUBRIC_CYCLES_BACKEND=CUDA \
    --env PYTHONPATH=/kubric \
    --volume "${SCRIPT_DIR}:/kubric" \
    --volume "${SCRIPT_DIR}/output:/output" \
    --workdir /kubric \
    "${IMAGE_NAME}" \
    "$@"
}

create_mp4() {
  local run_output_dir="$1"
  local pair_label="$2"
  local input_pattern="${run_output_dir}/image/rgba_%05d.png"
  local first_frame="${run_output_dir}/image/rgba_00000.png"
  local output_video="${run_output_dir}/render.mp4"
  local overwrite_flag="-y"

  if [[ "${OVERWRITE_MP4}" == "0" ]]; then
    overwrite_flag="-n"
  fi

  if [[ ! -f "${first_frame}" ]]; then
    echo "Missing rendered frames for ${pair_label}: ${first_frame}" >&2
    return 1
  fi

  if [[ -f "${output_video}" && "${OVERWRITE_MP4}" == "0" ]]; then
    echo "Skip existing MP4: ${output_video}"
    return 0
  fi

  echo "Create MP4: ${pair_label}"
  ffmpeg ${overwrite_flag} \
    -framerate "${FPS}" \
    -i "${input_pattern}" \
    -c:v libx264 \
    -preset "${PRESET}" \
    -crf "${CRF}" \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "${output_video}"
}

has_complete_frames() {
  local run_output_dir="$1"
  local frame_count

  if [[ ! -f "${run_output_dir}/image/rgba_00000.png" ]]; then
    return 1
  fi

  frame_count=$(find "${run_output_dir}/image" \
    -maxdepth 1 \
    -type f \
    -name 'rgba_*.png' \
    | wc -l)

  [[ "${frame_count}" -ge "${EXPECTED_FRAME_COUNT}" ]]
}

is_completed_case() {
  local run_output_dir="$1"

  [[ -s "${run_output_dir}/render.mp4" ]] && has_complete_frames "${run_output_dir}"
}

if [[ ! -f "${TOPDOWN_CONTROL_POINTS}" ]]; then
  echo "Create top-down control point candidates: ${TOPDOWN_CONTROL_POINTS}"
  docker_kubric \
    python3 examples/gso_topdown_control_points.py \
      --asset_id="${ASSET_ID}" \
      --num_control_points="${NUM_CONTROL_POINTS}" \
      --grid_resolution=160 \
      --job-dir="${TOPDOWN_DIR_IN_CONTAINER}" \
      --scratch_dir=/tmp/kubric_teddy_bear_topdown_control_points \
      --seed=42
else
  echo "Use existing top-down control point candidates: ${TOPDOWN_CONTROL_POINTS}"
fi

pair_count=0
for point_a in $(seq 0 "$((NUM_CONTROL_POINTS - 2))"); do
  for point_b in $(seq "$((point_a + 1))" "$((NUM_CONTROL_POINTS - 1))"); do
    if (( pair_count >= PAIR_LIMIT )); then
      echo "Reached PAIR_LIMIT=${PAIR_LIMIT}."
      echo "Completed ${pair_count} pair runs."
      exit 0
    fi

    pair_label=$(printf "p_%02d__p_%02d" "${point_a}" "${point_b}")
    run_output_dir="${OUTPUT_ROOT}/${pair_label}"
    run_output_dir_in_container="${OUTPUT_ROOT_IN_CONTAINER}/${pair_label}"
    control_trajectory="${run_output_dir}/control_trajectory_${pair_label}.npz"
    control_trajectory_in_container="${run_output_dir_in_container}/control_trajectory_${pair_label}.npz"

    if is_completed_case "${run_output_dir}"; then
      echo "Skip completed case: ${pair_label}"
      pair_count=$((pair_count + 1))
      continue
    fi

    if has_complete_frames "${run_output_dir}"; then
      echo "Frames already complete; create MP4 if needed: ${pair_label}"
      create_mp4 "${run_output_dir}" "${pair_label}"
      pair_count=$((pair_count + 1))
      continue
    fi

    mkdir -p "${run_output_dir}"

    echo "Create trajectory: ${pair_label} (point_a=${point_a}, point_b=${point_b})"
    docker_kubric \
      python3 examples/create_topdown_pair_control_trajectory.py \
        --topdown_control_points="${TOPDOWN_CONTROL_POINTS_IN_CONTAINER}" \
        --output="${control_trajectory_in_container}" \
        --point_a="${point_a}" \
        --point_b="${point_b}" \
        --frame_start=0 \
        --frame_end="${FRAME_END}" \
        --frame_rate=24 \
        --lift_height=0.6 \
        --lift_seconds=4.0

    echo "Render: ${pair_label}"
    docker_kubric \
      python3 examples/gso_spring_mass_control.py \
        --seed="${pair_count}" \
        --asset_id="${ASSET_ID}" \
        --particle_spacing=0.25 \
        --k_neighbors=16 \
        --sampled_surface_particle_count=3000 \
        --max_particles=5000 \
        --spring_stiffness=25 \
        --damping=0.001 \
        --step_rate=4800 \
        --frame_rate=24 \
        --frame_end="${FRAME_END}" \
        --job-dir="${run_output_dir_in_container}" \
        --scratch_dir="/tmp/kubric_teddy_bear_${pair_label}" \
        --device=cuda \
        --control_trajectory="${control_trajectory_in_container}" \
        --control_physics_mode=soft \
        --control_attachment_k=40 \
        --control_attachment_radius=0.25 \
        --control_stiffness=20 \
        --control_damping=0.05 \
        --camera_distance=2.4 \
        --camera_height=1.4 \
        --camera_look_at_z=0.3 \
        --samples_per_pixel=16 \
        --save_simulation_data

    create_mp4 "${run_output_dir}" "${pair_label}"

    echo "Completed: ${pair_label}"
    pair_count=$((pair_count + 1))
  done
done

echo "Completed ${pair_count} pair runs."
