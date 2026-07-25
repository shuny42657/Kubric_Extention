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

"""Tests for renderer-independent vertex animation data and adapters."""

import numpy as np
import pytest

import kubric as kb
from kubric.simulator.rigid_body import rigid_body_to_vertex_animation


def test_vertex_animation_properties_and_frame_access():
  asset = kb.Cube()
  vertices = np.arange(18, dtype=np.float32).reshape(2, 3, 3)

  animation = kb.VertexAnimation(asset, frame_start=4, vertices=vertices)

  assert animation.num_frames == 2
  assert animation.num_vertices == 3
  assert animation.frame_end == 5
  np.testing.assert_array_equal(animation.vertices_at(5), vertices[1])
  assert animation.vertices.flags.writeable is False


@pytest.mark.parametrize("vertices", [
    np.zeros((2, 3)),
    np.zeros((2, 3, 4)),
    np.zeros((0, 3, 3)),
    np.zeros((2, 0, 3)),
    np.full((2, 3, 3), np.nan),
])
def test_vertex_animation_rejects_invalid_vertices(vertices):
  with pytest.raises(ValueError):
    kb.VertexAnimation(kb.Cube(), frame_start=0, vertices=vertices)


def test_rigid_body_to_vertex_animation_applies_scale_rotation_translation():
  asset = kb.Cube(scale=(2., 3., 4.))
  rest_vertices = np.array([
      [1., 0., 0.],
      [0., 1., 0.],
  ])
  # Identity followed by a 90 degree rotation around Z, in WXYZ order.
  pose_animation = {
      "position": [[1., 2., 3.], [-1., 1., 0.]],
      "quaternion": [[1., 0., 0., 0.], [np.sqrt(.5), 0., 0., np.sqrt(.5)]],
  }

  animation = rigid_body_to_vertex_animation(
      asset, rest_vertices, pose_animation, frame_start=7)

  np.testing.assert_allclose(animation.vertices[0], [
      [3., 2., 3.],
      [1., 5., 3.],
  ], atol=1e-6)
  np.testing.assert_allclose(animation.vertices[1], [
      [-1., 3., 0.],
      [-4., 1., 0.],
  ], atol=1e-6)
  assert animation.frame_start == 7


def test_rigid_body_to_vertex_animation_validates_pose_lengths():
  with pytest.raises(ValueError, match="same number of frames"):
    rigid_body_to_vertex_animation(
        kb.Cube(),
        np.zeros((8, 3)),
        {
            "position": [[0., 0., 0.]],
            "quaternion": [[1., 0., 0., 0.], [1., 0., 0., 0.]],
        },
        frame_start=0,
    )
