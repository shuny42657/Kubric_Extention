# Copyright 2026 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Creates a stationary trajectory containing all top-down control candidates."""

import argparse
import pathlib

import numpy as np


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--topdown_control_points", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--frame_start", type=int, default=0)
  parser.add_argument("--frame_end", type=int, default=1)
  parser.add_argument("--frame_rate", type=int, default=24)
  args = parser.parse_args()

  if args.frame_end < args.frame_start:
    raise ValueError("frame_end must be >= frame_start")

  with np.load(args.topdown_control_points, allow_pickle=False) as data:
    nearest_vertex_indices = data["nearest_vertex_indices"].astype(np.int32)
    nearest_vertex_positions = data[
        "nearest_vertex_positions_world"].astype(np.float32)

  frame_indices = np.arange(args.frame_start, args.frame_end + 1, dtype=np.int32)
  positions = np.repeat(
      nearest_vertex_positions[None, :, :], len(frame_indices), axis=0)

  output = pathlib.Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      str(output),
      control_vertex_indices=nearest_vertex_indices,
      positions_world=positions.astype(np.float32),
      frame_indices=frame_indices,
      frame_rate=np.asarray(args.frame_rate, dtype=np.float32),
      topdown_point_indices=np.arange(
          len(nearest_vertex_indices), dtype=np.int32),
      description=np.asarray(
          "stationary trajectory containing all top-down control candidates"),
  )


if __name__ == "__main__":
  main()
