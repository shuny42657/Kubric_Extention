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

"""Tests for the non-learnable spring-mass simulator."""

import numpy as np
import pytest

import kubric as kb
from kubric.simulator.spring_mass import fill_mesh_with_particles
from kubric.simulator.spring_mass import SpringMassConfig
from kubric.simulator.spring_mass import SpringMassSimulator


TETRAHEDRON_VERTICES = np.asarray([
    [0., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
])
TETRAHEDRON_FACES = np.asarray([
    [0, 2, 1],
    [0, 1, 3],
    [0, 3, 2],
    [1, 2, 3],
])


def test_fill_mesh_keeps_unique_surface_vertices_first_and_points_inside():
  config = SpringMassConfig(
      particle_spacing=0.2,
      surface_sample_spacing=0.3,
      k_neighbors=3,
      max_particles=1000,
      ground_axis=None,
  )

  particles = fill_mesh_with_particles(
      TETRAHEDRON_VERTICES, TETRAHEDRON_FACES, config)

  np.testing.assert_array_equal(particles[:4], TETRAHEDRON_VERTICES)
  assert len(particles) > len(TETRAHEDRON_VERTICES)
  # The tetrahedron is x >= 0, y >= 0, z >= 0, x + y + z <= 1.
  assert np.all(particles >= -1e-6)
  assert np.all(particles.sum(axis=1) <= 1. + 1e-6)


def test_fill_mesh_rejects_open_surface():
  config = SpringMassConfig(particle_spacing=0.2, k_neighbors=3)
  with pytest.raises(ValueError, match="non-watertight"):
    fill_mesh_with_particles(
        TETRAHEDRON_VERTICES, TETRAHEDRON_FACES[:-1], config)


def test_fill_mesh_welds_duplicate_surface_vertices():
  duplicated_vertices = TETRAHEDRON_VERTICES[TETRAHEDRON_FACES].reshape((-1, 3))
  duplicated_faces = np.arange(len(duplicated_vertices)).reshape((-1, 3))
  config = SpringMassConfig(
      particle_spacing=0.25,
      surface_sample_spacing=0.4,
      k_neighbors=3,
      max_particles=1000,
  )

  particles = fill_mesh_with_particles(
      duplicated_vertices, duplicated_faces, config)

  surface_particles = {tuple(vertex) for vertex in particles[:4]}
  assert surface_particles == {tuple(vertex) for vertex in TETRAHEDRON_VERTICES}
  for vertex in TETRAHEDRON_VERTICES:
    assert np.any(np.all(np.isclose(particles, vertex), axis=1))


def test_spring_mass_returns_vertex_animation_on_cpu():
  pytest.importorskip("torch")
  scene = kb.Scene(
      frame_start=0,
      frame_end=2,
      frame_rate=10,
      step_rate=100,
      gravity=(0., 0., -1.),
  )
  asset = kb.FileBasedObject(
      asset_id="tetrahedron",
      position=(0., 0., 2.),
      mass=1.,
      friction=0.,
      restitution=0.,
  )
  config = SpringMassConfig(
      particle_spacing=0.3,
      surface_sample_spacing=0.4,
      k_neighbors=3,
      spring_stiffness=20.,
      damping=0.1,
      ground_axis=None,
      max_particles=1000,
      seed=3,
  )
  simulator = SpringMassSimulator(scene, config=config, device="cpu")

  animation = simulator.run(
      asset, TETRAHEDRON_VERTICES, TETRAHEDRON_FACES)

  assert animation.vertices.shape == (3, 4, 3)
  np.testing.assert_allclose(
      animation.vertices[0], TETRAHEDRON_VERTICES + (0., 0., 2.), atol=1e-6)
  assert np.all(animation.vertices[1, :, 2] < animation.vertices[0, :, 2])
  assert simulator.last_edges.shape[1] == 2
