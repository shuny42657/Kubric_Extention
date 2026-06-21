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

import hashlib
import logging
import math

import numpy as np

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import RepeatedLiftConfig
from kubric.simulator import SpringMassConfig
from kubric.simulator import SpringMassSimulator


CAMERA_FOLLOW_EMA_ALPHA = 0.1


def _add_camera_follow_animation(camera, initial_look_at, vertex_animation):
  """Animates camera orientation to follow the mesh bbox center with EMA."""
  smoothed_look_at = np.asarray(initial_look_at, dtype=np.float64).copy()
  look_at_animation = []
  previous_quaternion = None

  for frame_offset, frame_vertices in enumerate(vertex_animation.vertices):
    frame = vertex_animation.frame_start + frame_offset
    object_center = 0.5 * (
        np.min(frame_vertices, axis=0) + np.max(frame_vertices, axis=0))
    smoothed_look_at += CAMERA_FOLLOW_EMA_ALPHA * (
        object_center - smoothed_look_at)

    camera.look_at(smoothed_look_at)
    quaternion = np.asarray(camera.quaternion, dtype=np.float64)
    if previous_quaternion is not None and np.dot(
        quaternion, previous_quaternion) < 0.:
      quaternion = -quaternion
      camera.quaternion = quaternion
    camera.keyframe_insert("quaternion", frame)

    previous_quaternion = quaternion
    look_at_animation.append(smoothed_look_at.copy())

  return np.asarray(look_at_animation)


def _get_camera_metadata(camera, look_at, resolution):
  """Returns explicit intrinsics and extrinsics for the current camera pose."""
  width, height = resolution
  camera_to_world = np.asarray(camera.matrix_world, dtype=np.float64)
  world_to_camera_blender = np.linalg.inv(camera_to_world)
  blender_to_opencv = np.diag([1., -1., -1., 1.])
  world_to_camera_opencv = blender_to_opencv @ world_to_camera_blender

  sensor_height = camera.sensor_width * height / width
  focal_x_pixels = camera.focal_length * width / camera.sensor_width
  focal_y_pixels = camera.focal_length * height / sensor_height
  principal_x = width / 2.
  principal_y = height / 2.
  intrinsic_pixels_opencv = np.asarray([
      [focal_x_pixels, 0., principal_x],
      [0., focal_y_pixels, principal_y],
      [0., 0., 1.],
  ])

  return {
      "index": None,
      "output_directory": None,
      "intrinsics": {
          "resolution_pixels": [int(width), int(height)],
          "focal_length_mm": float(camera.focal_length),
          "sensor_width_mm": float(camera.sensor_width),
          "sensor_height_mm": float(sensor_height),
          "field_of_view_x_radians": float(camera.field_of_view),
          "field_of_view_y_radians": float(
              2. * np.arctan2(sensor_height / 2., camera.focal_length)),
          "principal_point_pixels": [float(principal_x), float(principal_y)],
          "matrix_normalized_kubric": np.asarray(
              camera.intrinsics, dtype=np.float64).tolist(),
          "matrix_pixels_opencv": intrinsic_pixels_opencv.tolist(),
      },
      "extrinsics": {
          "position_world": np.asarray(
              camera.position, dtype=np.float64).tolist(),
          "look_at_world": np.asarray(look_at, dtype=np.float64).tolist(),
          "quaternion_wxyz": np.asarray(
              camera.quaternion, dtype=np.float64).tolist(),
          "camera_to_world_blender": camera_to_world.tolist(),
          "world_to_camera_blender": world_to_camera_blender.tolist(),
          "world_to_camera_opencv": world_to_camera_opencv.tolist(),
          "rotation_world_to_camera_opencv": (
              world_to_camera_opencv[:3, :3].tolist()),
          "translation_world_to_camera_opencv": (
              world_to_camera_opencv[:3, 3].tolist()),
      },
      "coordinate_conventions": {
          "blender_camera": "+X right, +Y up, -Z forward",
          "opencv_camera": "+X right, +Y down, +Z forward",
          "quaternion_order": "wxyz",
          "matrix_vectors": "column vectors",
      },
  }


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
parser.add_argument("--control_vertex_index", type=int, default=0)
parser.add_argument("--control_target_x", type=float, default=0.0)
parser.add_argument("--control_target_y", type=float, default=0.0)
parser.add_argument("--camera_count", type=int, default=4)
parser.add_argument("--randomize_cameras", action="store_true")
parser.add_argument("--camera_follow_obj", action="store_true")
parser.add_argument("--repeat_count", type=int, default=5)
parser.add_argument("--initial_settle_seconds", type=float, default=2.0)
parser.add_argument("--lift_seconds", type=float, default=1.0)
parser.add_argument("--hold_seconds", type=float, default=0.2)
parser.add_argument("--settle_seconds", type=float, default=2.0)
parser.set_defaults(
    frame_end=24,
    frame_rate=24,
    step_rate=240,
    resolution="256x256",
    seed=42,
)
FLAGS = parser.parse_args()
if FLAGS.repeat_count < 0:
  raise ValueError("repeat_count cannot be negative")
fixed_camera_positions = (
    (0.0, -4.0, 2.6),
    (4.0, 0.0, 2.6),
    (0.0, 4.0, 2.6),
    (-4.0, 0.0, 2.6),
)
fixed_camera_look_at = (0., 0., 1.)
if not 1 <= FLAGS.camera_count <= len(fixed_camera_positions):
  raise ValueError(
      f"camera_count must be between 1 and {len(fixed_camera_positions)}")
simulation_seconds = (
    FLAGS.initial_settle_seconds + FLAGS.repeat_count * (
        FLAGS.lift_seconds + FLAGS.hold_seconds + FLAGS.settle_seconds))
FLAGS.frame_end = math.ceil(simulation_seconds * FLAGS.frame_rate)
repeated_lift = None
if FLAGS.repeat_count > 0:
  repeated_lift = RepeatedLiftConfig(
      control_vertex_index=FLAGS.control_vertex_index,
      repeat_count=FLAGS.repeat_count,
      initial_settle_seconds=FLAGS.initial_settle_seconds,
      lift_seconds=FLAGS.lift_seconds,
      hold_seconds=FLAGS.hold_seconds,
      settle_seconds=FLAGS.settle_seconds,
      target_horizontal=(FLAGS.control_target_x, FLAGS.control_target_y),
  )


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
scene.add(kb.DirectionalLight(
    name="sun", position=(-3., -4., 6.), look_at=(0., 0., 0.), intensity=2.0))
scene.ambient_illumination = kb.Color(0.1, 0.1, 0.1)

with kb.AssetSource.from_manifest(FLAGS.gso_assets, scratch_dir) as gso:
  asset_ids = sorted(gso._assets)  # pylint: disable=protected-access
  asset_id = str(FLAGS.asset_id or rng.choice(asset_ids))
  if asset_id not in gso._assets:  # pylint: disable=protected-access
    raise ValueError(f"Unknown GSO asset ID: {asset_id!r}")
  asset_output_dir = output_dir / asset_id
  asset_output_dir.mkdir(parents=True, exist_ok=True)

  if FLAGS.randomize_cameras:
    seed_material = f"{scene.metadata['seed']}:{asset_id}".encode("utf-8")
    camera_seed = int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8], "little")
    camera_rng = np.random.default_rng(camera_seed)
    base_azimuth = camera_rng.uniform(0., 2. * np.pi)
    nominal_azimuths = (
        base_azimuth + np.arange(FLAGS.camera_count) *
        (2. * np.pi / FLAGS.camera_count))
    azimuth_jitter = np.deg2rad(camera_rng.uniform(
        -8., 8., size=FLAGS.camera_count))
    azimuths = nominal_azimuths + azimuth_jitter
    radii = camera_rng.uniform(3.8, 4.2, size=FLAGS.camera_count)
    heights = camera_rng.uniform(2.45, 2.75, size=FLAGS.camera_count)
    shared_look_at = np.asarray([
        camera_rng.uniform(-0.1, 0.1),
        camera_rng.uniform(-0.1, 0.1),
        camera_rng.uniform(0.9, 1.1),
    ])
    camera_positions = [
        (radius * np.cos(azimuth), radius * np.sin(azimuth), height)
        for radius, azimuth, height in zip(radii, azimuths, heights)]
    camera_look_ats = [
        shared_look_at + camera_rng.uniform(-0.04, 0.04, size=3)
        for _ in range(FLAGS.camera_count)]
  else:
    camera_seed = None
    camera_positions = fixed_camera_positions[:FLAGS.camera_count]
    camera_look_ats = [fixed_camera_look_at] * FLAGS.camera_count

  cameras = []
  for camera_index, (position, look_at) in enumerate(
      zip(camera_positions, camera_look_ats)):
    camera = kb.PerspectiveCamera(
        name=f"camera_{camera_index:02d}", position=position)
    camera.look_at(look_at)
    scene.add(camera)
    cameras.append(camera)
  scene.camera = cameras[0]

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
      repeated_lift=repeated_lift,
  )
  renderer.add_vertex_animation(obj, vertex_animation)

  camera_look_at_animations = [None] * len(cameras)
  if FLAGS.camera_follow_obj:
    for camera_index, camera in enumerate(cameras):
      camera_look_at_animations[camera_index] = _add_camera_follow_animation(
          camera, camera_look_ats[camera_index], vertex_animation)

  logging.info(
      "Spring graph contains %d particles and %d edges",
      len(simulator.last_initial_particles), len(simulator.last_edges))
  renderer.save_state(asset_output_dir / "gso_spring_mass.blend")
  for camera_index, camera in enumerate(cameras):
    scene.camera = camera
    camera_output_dir = (
        asset_output_dir if len(cameras) == 1
        else asset_output_dir / f"camera_{camera_index:02d}")
    renderer.scratch_dir = scratch_dir / f"camera_{camera_index:02d}"
    logging.info(
        "Rendering camera %d from %s to %s",
        camera_index, camera.position, camera_output_dir)
    frames = renderer.render(return_layers=("rgba",))
    kb.write_image_dict({"rgba": frames["rgba"]}, camera_output_dir)
  cameras_metadata = []
  for camera_index, camera in enumerate(cameras):
    look_at_animation = camera_look_at_animations[camera_index]
    metadata_look_at = (
        camera_look_ats[camera_index]
        if look_at_animation is None else look_at_animation[-1])
    camera_metadata = _get_camera_metadata(
        camera, metadata_look_at, scene.resolution)
    camera_metadata["index"] = camera_index
    camera_metadata["output_directory"] = (
        "." if len(cameras) == 1 else f"camera_{camera_index:02d}")
    if look_at_animation is not None:
      camera_metadata["extrinsics"]["frame"] = vertex_animation.frame_end
      camera_metadata["look_at_animation"] = {
          "frame_start": vertex_animation.frame_start,
          "ema_alpha": CAMERA_FOLLOW_EMA_ALPHA,
          "look_at_world": look_at_animation.tolist(),
      }
    cameras_metadata.append(camera_metadata)
  kb.write_json({
      "asset_id": asset_id,
      "num_particles": len(simulator.last_initial_particles),
      "num_springs": len(simulator.last_edges),
      "control_vertex_index": (
          None if repeated_lift is None else FLAGS.control_vertex_index),
      "control_particle_index": simulator.last_control_particle_index,
      "control_target_xy": (
          None if repeated_lift is None
          else [FLAGS.control_target_x, FLAGS.control_target_y]),
      "randomize_cameras": FLAGS.randomize_cameras,
      "camera_follow_obj": FLAGS.camera_follow_obj,
      "camera_seed": camera_seed,
      "cameras": cameras_metadata,
  }, asset_output_dir / "metadata.json")
