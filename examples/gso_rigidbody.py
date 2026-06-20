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

"""Drops one Google Scanned Object onto a floor and renders the simulation."""

import logging

import numpy as np

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import PyBullet


parser = kb.ArgumentParser()
parser.add_argument(
    "--gso_assets",
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json",
    help="Path to the GSO asset manifest.",
)
parser.add_argument(
    "--asset_id",
    type=str,
    default=None,
    help="GSO asset ID. A random asset is selected when omitted.",
)
parser.set_defaults(frame_end=24, frame_rate=24, resolution="256x256", seed=42)
FLAGS = parser.parse_args()


scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64, use_denoising=True)

# The cube's top surface is at z=0.
floor = kb.Cube(
    name="floor",
    scale=(3, 3, 0.1),
    position=(0, 0, -0.1),
    static=True,
    friction=0.8,
    restitution=0.1,
    material=kb.PrincipledBSDFMaterial(color=kb.Color(0.3, 0.3, 0.3)),
)
scene.add(floor)

scene.camera = kb.PerspectiveCamera(position=(4, -6, 3.5))
scene.camera.look_at((0, 0, 1))
scene.add(kb.DirectionalLight(
    name="sun",
    position=(-3, -4, 6),
    look_at=(0, 0, 0),
    intensity=2.0,
))
scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)

with kb.AssetSource.from_manifest(FLAGS.gso_assets, scratch_dir) as gso:
  asset_ids = sorted(gso._assets)  # pylint: disable=protected-access
  asset_id = FLAGS.asset_id or rng.choice(asset_ids)
  if asset_id not in gso._assets:  # pylint: disable=protected-access
    raise ValueError(f"Unknown GSO asset ID: {asset_id!r}")
  logging.info("Using GSO asset '%s'", asset_id)
  obj = gso.create(asset_id=asset_id)

  # Normalize the largest object dimension to one scene unit while preserving
  # the same uniform scale in Blender and PyBullet.
  bounds = np.asarray(obj.bounds)
  scale = 1.0 / np.max(bounds[1] - bounds[0])
  obj.scale = (scale, scale, scale)
  obj.position = (0, 0, 2.5)
  obj.quaternion = kb.random_rotation(rng=rng)
  obj.velocity = (0.5, 0, 0)
  obj.angular_velocity = (1, 2, 1)
  obj.friction = 0.5
  obj.restitution = 0.3
  scene.add(obj)

  logging.info("Running rigid-body simulation ...")
  _, collisions = simulator.run(frame_start=0, frame_end=scene.frame_end + 1)

  renderer.save_state(output_dir / "gso_rigidbody.blend")
  logging.info("Rendering frames to '%s' ...", output_dir)
  frames = renderer.render(return_layers=("rgba",))
  kb.write_image_dict({"rgba": frames["rgba"]}, output_dir)
  kb.write_json({
      "asset_id": asset_id,
      "collisions": kb.process_collisions(collisions, scene),
  }, output_dir / "metadata.json")
