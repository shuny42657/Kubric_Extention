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

"""Adapters from rigid-body simulation output to vertex animation."""

from typing import Mapping, Sequence

import numpy as np
import pyquaternion as pyquat

from kubric.core.animation import VertexAnimation
from kubric.core.objects import PhysicalObject


def rigid_body_to_vertex_animation(
    asset: PhysicalObject,
    rest_vertices: np.ndarray,
    pose_animation: Mapping[str, Sequence],
    frame_start: int,
) -> VertexAnimation:
  """Converts Kubric WXYZ rigid poses into world-space mesh vertices.

  Args:
    asset: Object whose scale is applied to the rest vertices.
    rest_vertices: Render-mesh vertices in the asset's local coordinates.
    pose_animation: Animation returned for `asset` by `PyBullet.run()`.
    frame_start: Frame corresponding to the first pose.

  Returns:
    A renderer-independent world-space vertex animation.
  """
  if not isinstance(asset, PhysicalObject):
    raise TypeError("asset must be a PhysicalObject")

  rest_vertices = np.asarray(rest_vertices, dtype=np.float64)
  if rest_vertices.ndim != 2 or rest_vertices.shape[1] != 3:
    raise ValueError(
        f"rest_vertices must have shape (num_vertices, 3), got {rest_vertices.shape}")
  if rest_vertices.shape[0] == 0:
    raise ValueError("rest_vertices must contain at least one vertex")
  if not np.all(np.isfinite(rest_vertices)):
    raise ValueError("rest_vertices must contain only finite values")

  missing_keys = {"position", "quaternion"} - set(pose_animation)
  if missing_keys:
    raise KeyError(f"pose_animation is missing keys: {sorted(missing_keys)}")

  positions = np.asarray(pose_animation["position"], dtype=np.float64)
  quaternions = np.asarray(pose_animation["quaternion"], dtype=np.float64)
  if positions.ndim != 2 or positions.shape[1] != 3:
    raise ValueError(f"position must have shape (num_frames, 3), got {positions.shape}")
  if quaternions.ndim != 2 or quaternions.shape[1] != 4:
    raise ValueError(
        f"quaternion must have shape (num_frames, 4), got {quaternions.shape}")
  if len(positions) == 0:
    raise ValueError("pose_animation must contain at least one frame")
  if len(positions) != len(quaternions):
    raise ValueError(
        "position and quaternion must have the same number of frames, "
        f"got {len(positions)} and {len(quaternions)}")
  if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
    raise ValueError("position and quaternion must contain only finite values")

  scaled_vertices = rest_vertices * np.asarray(asset.scale, dtype=np.float64)
  vertices = np.empty((len(positions), len(rest_vertices), 3), dtype=np.float32)
  for frame_idx, (position, quaternion) in enumerate(zip(positions, quaternions)):
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
      raise ValueError(f"quaternion at frame index {frame_idx} has zero length")
    rotation = pyquat.Quaternion(*(quaternion / norm)).rotation_matrix
    vertices[frame_idx] = scaled_vertices @ rotation.T + position

  return VertexAnimation(
      asset=asset,
      frame_start=frame_start,
      vertices=vertices,
  )
