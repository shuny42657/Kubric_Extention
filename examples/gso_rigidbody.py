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
import shlex
import sys

import numpy as np
import pyquaternion as pyquat

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import PyBullet
from kubric.simulator import rigid_body_to_vertex_animation


def _get_run_metadata(flags):
  argv = list(sys.argv)
  return {
      "argv": argv,
      "command": " ".join(shlex.quote(arg) for arg in argv),
      "flags": dict(vars(flags)),
  }


def _rigid_body_vertex_velocities(asset, rest_vertices, pose_animation):
  """Computes instantaneous world-space vertex velocities for a rigid body."""
  scaled_vertices = rest_vertices * np.asarray(asset.scale, dtype=np.float64)
  linear_velocities = np.asarray(pose_animation["velocity"], dtype=np.float64)
  angular_velocities = np.asarray(
      pose_animation["angular_velocity"], dtype=np.float64)
  quaternions = np.asarray(pose_animation["quaternion"], dtype=np.float64)
  velocities = np.empty(
      (len(linear_velocities), len(rest_vertices), 3), dtype=np.float32)

  for frame_idx, (linear_velocity, angular_velocity, quaternion) in enumerate(
      zip(linear_velocities, angular_velocities, quaternions)):
    norm = np.linalg.norm(quaternion)
    if norm <= 1e-12:
      raise ValueError(f"quaternion at frame index {frame_idx} has zero length")
    rotation = pyquat.Quaternion(*(quaternion / norm)).rotation_matrix
    vertex_offsets_world = scaled_vertices @ rotation.T
    velocities[frame_idx] = (
        linear_velocity + np.cross(angular_velocity, vertex_offsets_world))

  return velocities


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
parser.add_argument(
    "--render_depth",
    action="store_true",
    help="Write depth TIFF frames in addition to RGBA frames.",
)
parser.add_argument(
    "--render_segmentation",
    action="store_true",
    help="Write segmentation PNG frames in addition to RGBA frames.",
)
parser.add_argument(
    "--motion_mode",
    choices=["rigid_body", "free_fall"],
    default="rigid_body",
    help="Motion preset: rigid_body uses initial linear and angular velocity; "
         "free_fall starts from rest and only falls under gravity.",
)
parser.set_defaults(frame_end=24, frame_rate=24, resolution="256x256", seed=42)
FLAGS = parser.parse_args()
render_layers = ["rgba"]
if FLAGS.render_depth:
  render_layers.append("depth")
if FLAGS.render_segmentation:
  render_layers.append("segmentation")


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
  if FLAGS.motion_mode == "free_fall":
    obj.velocity = (0, 0, 0)
    obj.angular_velocity = (0, 0, 0)
  else:
    obj.velocity = (0.5, 0, 0)
    obj.angular_velocity = (1, 2, 1)
  obj.friction = 0.5
  obj.restitution = 0.3
  scene.add(obj)

  logging.info("Running rigid-body simulation ...")
  pose_animations, collisions = simulator.run(
      frame_start=0, frame_end=scene.frame_end + 1)
  rest_vertices, faces = renderer.get_mesh_geometry(obj)
  vertex_animation = rigid_body_to_vertex_animation(
      asset=obj,
      rest_vertices=rest_vertices,
      pose_animation=pose_animations[obj],
      frame_start=0,
  )
  renderer.add_vertex_animation(obj, vertex_animation)
  mesh_vertices_path = output_dir / "mesh_vertices.npz"
  np.savez_compressed(
      str(mesh_vertices_path),
      vertices_world=vertex_animation.vertices,
      velocities_world=_rigid_body_vertex_velocities(
          obj, rest_vertices, pose_animations[obj]),
      frame_indices=np.arange(
          vertex_animation.frame_start, vertex_animation.frame_end + 1,
          dtype=np.int32),
      rest_vertices_local=rest_vertices.astype(np.float32),
      faces=faces.astype(np.int32),
      object_positions=np.asarray(
          pose_animations[obj]["position"], dtype=np.float32),
      object_quaternions_wxyz=np.asarray(
          pose_animations[obj]["quaternion"], dtype=np.float32),
      object_velocities=np.asarray(
          pose_animations[obj]["velocity"], dtype=np.float32),
      object_angular_velocities=np.asarray(
          pose_animations[obj]["angular_velocity"], dtype=np.float32),
  )

  renderer.save_state(output_dir / "gso_rigidbody.blend")
  logging.info("Rendering frames to '%s' ...", output_dir)
  frames = renderer.render(return_layers=render_layers)
  kb.write_image_dict(frames, output_dir)
  kb.write_json({
      "asset_id": asset_id,
      "run": _get_run_metadata(FLAGS),
      "mesh_vertices_file": mesh_vertices_path.name,
      "num_mesh_vertices": vertex_animation.num_vertices,
      "num_mesh_faces": len(faces),
      "collisions": kb.process_collisions(collisions, scene),
  }, output_dir / "metadata.json")
