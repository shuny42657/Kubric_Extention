#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/render_gso_cloth_grid_pair_lift_drop_release_3views_rgbd_mask.sh" "$@"
