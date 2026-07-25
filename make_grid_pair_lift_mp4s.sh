#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(readlink -f "${REPO_ROOT}/output")}"
DATASET_DIR="${DATASET_DIR:-${OUTPUT_ROOT}/spring_mass_grid_pair_lift_slow}"
FPS="${FPS:-24}"
CRF="${CRF:-18}"
PRESET="${PRESET:-medium}"
OVERWRITE="${OVERWRITE:-1}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found in PATH" >&2
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory not found: ${DATASET_DIR}" >&2
  exit 1
fi

overwrite_flag="-y"
if [[ "${OVERWRITE}" == "0" ]]; then
  overwrite_flag="-n"
fi

case_dirs=()
while IFS= read -r -d '' case_dir; do
  case_dirs+=("${case_dir}")
done < <(find "${DATASET_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'p_*' -print0 | sort -z)

if (( ${#case_dirs[@]} == 0 )); then
  echo "No case directories found under ${DATASET_DIR}" >&2
  exit 1
fi

failures=()
completed=0
for case_dir in "${case_dirs[@]}"; do
  case_name="$(basename "${case_dir}")"
  input_pattern="${case_dir}/image/rgba_%05d.png"
  first_frame="${case_dir}/image/rgba_00000.png"
  output_video="${case_dir}/render.mp4"

  if [[ ! -f "${first_frame}" ]]; then
    echo "Skipping ${case_name}: missing ${first_frame}" >&2
    failures+=("${case_name}:missing_frames")
    continue
  fi

  echo "============================================================"
  echo "Creating MP4 for ${case_name}"
  echo "============================================================"

  if ffmpeg ${overwrite_flag} \
      -framerate "${FPS}" \
      -i "${input_pattern}" \
      -c:v libx264 \
      -preset "${PRESET}" \
      -crf "${CRF}" \
      -pix_fmt yuv420p \
      -movflags +faststart \
      "${output_video}"; then
    completed=$((completed + 1))
  else
    echo "Failed ${case_name}" >&2
    failures+=("${case_name}:ffmpeg")
  fi
done

if (( ${#failures[@]} > 0 )); then
  echo "Failures: ${failures[*]}" >&2
  echo "Created ${completed}/${#case_dirs[@]} videos." >&2
  exit 1
fi

echo "Created ${completed} MP4 videos under ${DATASET_DIR}."
