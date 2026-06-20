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

"""Renderer-independent vertex animation data."""

from dataclasses import dataclass

import numpy as np

from kubric.core.objects import PhysicalObject


@dataclass(frozen=True, eq=False)
class VertexAnimation:
  """A fixed-topology mesh animation expressed in world coordinates.

  `vertices` has shape `(num_frames, num_vertices, 3)`. Vertex indices must
  correspond to the target asset's render mesh and remain stable over time.
  """

  asset: PhysicalObject
  frame_start: int
  vertices: np.ndarray
  coordinate_space: str = "world"

  def __post_init__(self):
    if not isinstance(self.asset, PhysicalObject):
      raise TypeError("asset must be a PhysicalObject")
    if not isinstance(self.frame_start, (int, np.integer)):
      raise TypeError("frame_start must be an integer")
    if self.coordinate_space != "world":
      raise ValueError("Only world-space vertex animations are supported")

    vertices = np.asarray(self.vertices, dtype=np.float32)
    if vertices.ndim != 3 or vertices.shape[2] != 3:
      raise ValueError(
          "vertices must have shape (num_frames, num_vertices, 3), "
          f"got {vertices.shape}")
    if vertices.shape[0] == 0:
      raise ValueError("vertices must contain at least one frame")
    if vertices.shape[1] == 0:
      raise ValueError("vertices must contain at least one vertex")
    if not np.all(np.isfinite(vertices)):
      raise ValueError("vertices must contain only finite values")

    vertices = np.array(vertices, dtype=np.float32, copy=True)
    vertices.setflags(write=False)
    object.__setattr__(self, "frame_start", int(self.frame_start))
    object.__setattr__(self, "vertices", vertices)

  @property
  def num_frames(self) -> int:
    return self.vertices.shape[0]

  @property
  def num_vertices(self) -> int:
    return self.vertices.shape[1]

  @property
  def frame_end(self) -> int:
    return self.frame_start + self.num_frames - 1

  def vertices_at(self, frame: int) -> np.ndarray:
    """Returns vertices at an exact animation frame."""
    if frame < self.frame_start or frame > self.frame_end:
      raise IndexError(
          f"frame {frame} is outside [{self.frame_start}, {self.frame_end}]")
    return self.vertices[frame - self.frame_start]
