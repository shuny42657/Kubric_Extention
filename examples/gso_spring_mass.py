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

"""Fills a GSO mesh, simulates it with GPU springs, and renders deformation."""

import logging

import numpy as np

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import SpringMassConfig
from kubric.simulator import SpringMassSimulator


parser = kb.ArgumentParser()
parser.add_argument(
    "--gso_assets",
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json",
)
parser.add_argument("--asset_id", type=str, default=None)
parser.add_argument("--particle_spacing", type=float, default=0.1)
parser.add_argument("--k_neighbors", type=int, default=16)
parser.add_argument("--spring_stiffness", type=float, default=200.0)
parser.add_argument("--damping", type=float, default=0.5)
parser.set_defaults(
    frame_end=24,
    frame_rate=24,
    step_rate=240,
    resolution="256x256",
    seed=42,
)
FLAGS = parser.parse_args()


scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64, use_denoising=True)
simulator = SpringMassSimulator(
    scene,
    config=SpringMassConfig(
        particle_spacing=FLAGS.particle_spacing,
        k_neighbors=FLAGS.k_neighbors,
        spring_stiffness=FLAGS.spring_stiffness,
        damping=FLAGS.damping,
        total_mass=1.0,
        initial_velocity=(0.5, 0., 0.),
        ground_axis=2,
        ground_height=0.,
        restitution=0.2,
        friction=0.3,
        seed=FLAGS.seed,
    ),
    device="cuda",
)

floor = kb.Cube(
    name="floor",
    scale=(3., 3., 0.1),
    position=(0., 0., -0.1),
    static=True,
    material=kb.PrincipledBSDFMaterial(color=kb.Color(0.3, 0.3, 0.3)),
)
scene.add(floor)
scene.camera = kb.PerspectiveCamera(position=(3.2, -4.8, 3.0))
scene.camera.look_at((0., 0., 1.))
scene.add(kb.DirectionalLight(
    name="sun", position=(-3., -4., 6.), look_at=(0., 0., 0.), intensity=2.0))
scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)

with kb.AssetSource.from_manifest(FLAGS.gso_assets, scratch_dir) as gso:
  asset_ids = sorted(gso._assets)  # pylint: disable=protected-access
  asset_id = FLAGS.asset_id or rng.choice(asset_ids)
  if asset_id not in gso._assets:  # pylint: disable=protected-access
    raise ValueError(f"Unknown GSO asset ID: {asset_id!r}")

  logging.info("Using GSO asset %r", asset_id)
  obj = gso.create(asset_id=asset_id)
  bounds = np.asarray(obj.bounds)
  scale = 1.0 / np.max(bounds[1] - bounds[0])
  obj.scale = (scale, scale, scale)
  obj.position = (0., 0., 1.5)
  obj.quaternion = kb.random_rotation(rng=rng)
  scene.add(obj)

  vertices, faces = renderer.get_mesh_geometry(obj)
  vertex_animation = simulator.run(
      asset=obj,
      vertices=vertices,
      faces=faces,
      frame_start=0,
      frame_end=scene.frame_end + 1,
  )
  renderer.add_vertex_animation(obj, vertex_animation)

  logging.info(
      "Spring graph contains %d particles and %d edges",
      len(simulator.last_initial_particles), len(simulator.last_edges))
  renderer.save_state(output_dir / "gso_spring_mass.blend")
  frames = renderer.render(return_layers=("rgba",))
  kb.write_image_dict({"rgba": frames["rgba"]}, output_dir)
  kb.write_json({
      "asset_id": asset_id,
      "num_particles": len(simulator.last_initial_particles),
      "num_springs": len(simulator.last_edges),
  }, output_dir / "metadata.json")
