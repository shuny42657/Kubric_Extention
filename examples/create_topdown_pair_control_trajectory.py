# Copyright 2026 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates a two-point lift trajectory from top-down control point candidates."""

import argparse
import pathlib

import numpy as np


def _smoothstep(x):
  x = np.clip(x, 0., 1.)
  return x * x * (3. - 2. * x)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--topdown_control_points", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--point_a", type=int, default=0)
  parser.add_argument("--point_b", type=int, default=1)
  parser.add_argument("--frame_start", type=int, default=0)
  parser.add_argument("--frame_end", type=int, default=90)
  parser.add_argument("--frame_rate", type=int, default=24)
  parser.add_argument("--lift_height", type=float, default=0.6)
  parser.add_argument("--lift_seconds", type=float, default=4.0)
  args = parser.parse_args()

  if args.point_a == args.point_b:
    raise ValueError("point_a and point_b must be different")
  if args.lift_seconds <= 0:
    raise ValueError("lift_seconds must be positive")
  if args.frame_end < args.frame_start:
    raise ValueError("frame_end must be >= frame_start")

  with np.load(args.topdown_control_points, allow_pickle=False) as data:
    nearest_vertex_indices = data["nearest_vertex_indices"].astype(np.int32)
    nearest_vertex_positions = data[
        "nearest_vertex_positions_world"].astype(np.float32)
    selected_positions = data["selected_positions_world"].astype(np.float32)

  point_indices = np.asarray([args.point_a, args.point_b], dtype=np.int32)
  if point_indices.min() < 0 or point_indices.max() >= len(nearest_vertex_indices):
    raise ValueError(
        f"point indices must be in [0, {len(nearest_vertex_indices) - 1}]")

  control_vertex_indices = nearest_vertex_indices[point_indices]
  initial_positions = nearest_vertex_positions[point_indices]
  selected_surface_positions = selected_positions[point_indices]

  frame_indices = np.arange(args.frame_start, args.frame_end + 2, dtype=np.int32)
  elapsed = (frame_indices - args.frame_start).astype(np.float64) / args.frame_rate
  progress = _smoothstep(elapsed / args.lift_seconds).astype(np.float32)
  positions = np.repeat(initial_positions[None, :, :], len(frame_indices), axis=0)
  positions[:, :, 2] += args.lift_height * progress[:, None]

  output = pathlib.Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      str(output),
      control_vertex_indices=control_vertex_indices.astype(np.int32),
      positions_world=positions.astype(np.float32),
      frame_indices=frame_indices,
      frame_rate=np.asarray(args.frame_rate, dtype=np.float32),
      topdown_point_indices=point_indices,
      topdown_selected_positions_world=selected_surface_positions,
      description=np.asarray(
          "two top-down control point nearest-vertex smooth lift"),
  )


if __name__ == "__main__":
  main()
