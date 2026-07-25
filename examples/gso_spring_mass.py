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
import pathlib
import shlex
import sys

import numpy as np

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import RepeatedLiftConfig
from kubric.simulator import SpringMassConfig
from kubric.simulator import SpringMassSimulator


CAMERA_FOLLOW_EMA_ALPHA = 0.1
IMAGE_FILE_TEMPLATES = {
    "rgba": "image/rgba_{:05d}.png",
    "depth": "depth/depth_{:05d}.tiff",
    "segmentation": "segmentation/segmentation_{:05d}.png",
}


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


def _as_absolute_string(path):
  path = str(path)
  if path.startswith("gs://"):
    return path
  return str(pathlib.Path(path).resolve())


def _get_run_metadata(flags):
  argv = list(sys.argv)
  return {
      "argv": argv,
      "command": " ".join(shlex.quote(arg) for arg in argv),
      "flags": dict(vars(flags)),
  }


def _lift_phase_at_time(repeated_lift, time_seconds):
  if repeated_lift is None:
    return "none"
  if time_seconds < repeated_lift.initial_settle_seconds:
    return "initial_settle"

  cycle_time = time_seconds - repeated_lift.initial_settle_seconds
  cycle_duration = (
      repeated_lift.lift_seconds
      + repeated_lift.hold_seconds
      + repeated_lift.settle_seconds)
  if repeated_lift.repeat_count <= 0 or (
      cycle_time >= repeated_lift.repeat_count * cycle_duration):
    return "done"

  phase_time = cycle_time % cycle_duration
  if phase_time < repeated_lift.lift_seconds:
    return "lift"
  if phase_time < repeated_lift.lift_seconds + repeated_lift.hold_seconds:
    return "hold"
  return "settle"


def _get_control_state_by_frame(repeated_lift, frame_indices, frame_rate):
  frame_indices = np.asarray(frame_indices, dtype=np.int32)
  phases = np.asarray([
      _lift_phase_at_time(repeated_lift, frame / frame_rate)
      for frame in frame_indices
  ])
  is_grasped = np.isin(phases, ["lift", "hold"])
  return {
      "frame_indices": frame_indices,
      "phase": phases,
      "is_grasped": is_grasped,
      "grasped_frame_ranges": _get_true_ranges(frame_indices, is_grasped),
  }


def _get_true_ranges(frame_indices, mask):
  ranges = []
  start = None
  previous = None
  for frame, is_true in zip(frame_indices.tolist(), mask.tolist()):
    if is_true and start is None:
      start = frame
    if not is_true and start is not None:
      ranges.append([start, previous])
      start = None
    previous = frame
  if start is not None:
    ranges.append([start, previous])
  return ranges


def _get_processed_camera_view(camera, resolution, image_path):
  """Returns the compact per-frame camera view metadata."""
  width, height = resolution
  camera_to_world = np.asarray(camera.matrix_world, dtype=np.float64)
  world_to_camera_blender = np.linalg.inv(camera_to_world)
  blender_to_opencv = np.diag([1., -1., -1., 1.])
  world_to_camera_opencv = blender_to_opencv @ world_to_camera_blender
  camera_to_world_opencv = np.linalg.inv(world_to_camera_opencv)

  sensor_height = camera.sensor_width * height / width
  focal_x_pixels = camera.focal_length * width / camera.sensor_width
  focal_y_pixels = camera.focal_length * height / sensor_height
  principal_x = width / 2.
  principal_y = height / 2.

  return {
      "camera_index": None,
      "image_path": image_path,
      "fxfycxcy": [
          float(focal_x_pixels),
          float(focal_y_pixels),
          float(principal_x),
          float(principal_y),
      ],
      "w2c": world_to_camera_opencv.tolist(),
      "c2w": camera_to_world_opencv.tolist(),
  }


def _write_processed_camera_metadata(
    output_dir,
    asset_id,
    cameras,
    resolution,
    frame_start,
    frame_end,
    multi_camera,
):
  """Writes frame-major camera metadata in the metadata_processed.json format."""
  frame_numbers = list(range(frame_start, frame_end + 1))
  denom = max(1, len(frame_numbers) - 1)
  frames = []
  for frame_index, frame_number in enumerate(frame_numbers):
    views = []
    for camera_index, camera in enumerate(cameras):
      camera_dir = (
          output_dir if not multi_camera
          else output_dir / f"camera_{camera_index:02d}")
      image_path = _as_absolute_string(
          camera_dir / IMAGE_FILE_TEMPLATES["rgba"].format(frame_index))
      with camera.at_frame(frame_number):
        view = _get_processed_camera_view(camera, resolution, image_path)
      view["camera_index"] = camera_index
      views.append(view)
    frames.append({
        "frame_index": frame_index,
        "time_normalized": -1.0 + 2.0 * frame_index / denom,
        "look_at_animation_index": frame_number,
        "views": views,
    })

  kb.write_json({
      "format": "svsm_dynamic_multiview_v1",
      "scene_name": asset_id,
      "num_frames": len(frame_numbers),
      "num_cameras": len(cameras),
      "camera_convention": "opencv: +X right, +Y down, +Z forward",
      "frames": frames,
  }, output_dir / "metadata_processed.json")


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
parser.add_argument("--render_depth", action="store_true")
parser.add_argument("--render_segmentation", action="store_true")
parser.add_argument(
    "--motion_mode",
    choices=["spring_mass", "free_fall"],
    default="spring_mass",
    help=(
        "Motion preset: spring_mass enables repeated lift; free_fall "
        "disables it and starts from rest."),
)
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
render_layers = ["rgba"]
if FLAGS.render_depth:
  render_layers.append("depth")
if FLAGS.render_segmentation:
  render_layers.append("segmentation")
if FLAGS.motion_mode == "free_fall":
  FLAGS.repeat_count = 0
else:
  simulation_seconds = (
      FLAGS.initial_settle_seconds + FLAGS.repeat_count * (
          FLAGS.lift_seconds + FLAGS.hold_seconds + FLAGS.settle_seconds))
  FLAGS.frame_end = math.ceil(simulation_seconds * FLAGS.frame_rate)
repeated_lift = None
if FLAGS.motion_mode == "spring_mass" and FLAGS.repeat_count > 0:
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
        initial_velocity=(
            (0., 0., 0.)
            if FLAGS.motion_mode == "free_fall" else (0.5, 0., 0.)),
        ground_axis=2,
        ground_height=0.,
        restitution=0.2,
        friction=0.3,
        record_all_particles=True,
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
  asset_output_dir = output_dir
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
  mesh_vertices_path = asset_output_dir / "mesh_vertices.npz"
  mesh_frame_indices = np.arange(
      vertex_animation.frame_start, vertex_animation.frame_end + 1,
      dtype=np.int32)
  mesh_control_state = _get_control_state_by_frame(
      repeated_lift, mesh_frame_indices, scene.frame_rate)
  np.savez_compressed(
      str(mesh_vertices_path),
      vertices_world=vertex_animation.vertices,
      velocities_world=simulator.last_render_velocity_trajectory,
      frame_indices=mesh_frame_indices,
      rest_vertices_local=vertices.astype(np.float32),
      faces=faces.astype(np.int32),
      control_phase=mesh_control_state["phase"],
      control_is_grasped=mesh_control_state["is_grasped"],
  )
  spring_mass_particles_path = asset_output_dir / "spring_mass_particles.npz"
  np.savez_compressed(
      str(spring_mass_particles_path),
      particle_positions_world=simulator.last_particle_trajectory,
      frame_indices=mesh_frame_indices,
      initial_particle_positions_world=(
          simulator.last_initial_particles.astype(np.float32)),
      edges=simulator.last_edges.astype(np.int32),
      edge_columns=np.asarray([
          "source_particle_index", "target_particle_index"]),
  )
  render_frame_indices = np.arange(
      scene.frame_start, scene.frame_end + 1, dtype=np.int32)
  render_control_state = _get_control_state_by_frame(
      repeated_lift, render_frame_indices, scene.frame_rate)
  control_point_metadata = None
  if repeated_lift is not None:
    control_vertex_trajectory = (
        vertex_animation.vertices[:, FLAGS.control_vertex_index, :])
    control_particle_index = simulator.last_control_particle_index
    control_point_metadata = {
        "vertex_index": FLAGS.control_vertex_index,
        "particle_index": control_particle_index,
        "initial_vertex_position_world": (
            control_vertex_trajectory[0].astype(np.float64).tolist()),
        "initial_particle_position_world": (
            simulator.last_initial_particles[control_particle_index]
            .astype(np.float64).tolist()),
        "target_xy": [FLAGS.control_target_x, FLAGS.control_target_y],
        "frame_start": vertex_animation.frame_start,
        "frame_end": vertex_animation.frame_end,
        "mesh_frames": {
            "frame_indices": mesh_control_state["frame_indices"].tolist(),
            "phase": mesh_control_state["phase"].tolist(),
            "is_grasped": mesh_control_state["is_grasped"].tolist(),
            "grasped_frame_ranges": (
                mesh_control_state["grasped_frame_ranges"]),
        },
        "rendered_frames": {
            "frame_indices": render_control_state["frame_indices"].tolist(),
            "phase": render_control_state["phase"].tolist(),
            "is_grasped": render_control_state["is_grasped"].tolist(),
            "grasped_frame_ranges": (
                render_control_state["grasped_frame_ranges"]),
        },
        "trajectory_world": (
            control_vertex_trajectory.astype(np.float64).tolist()),
    }
  renderer.add_vertex_animation(obj, vertex_animation)

  camera_look_at_animations = [None] * len(cameras)
  if FLAGS.camera_follow_obj:
    for camera_index, camera in enumerate(cameras):
      camera_look_at_animations[camera_index] = _add_camera_follow_animation(
          camera, camera_look_ats[camera_index], vertex_animation)

  logging.info(
      "Spring graph contains %d particles and %d edges",
      len(simulator.last_initial_particles), len(simulator.last_edges))
  spring_edges = simulator.last_edges.astype(np.int32)
  spring_rest_vectors = (
      simulator.last_initial_particles[spring_edges[:, 1]] -
      simulator.last_initial_particles[spring_edges[:, 0]])
  spring_rest_lengths = np.linalg.norm(spring_rest_vectors, axis=1)
  kb.write_json({
      "format": "kubric_spring_mass_model_v1",
      "asset_id": asset_id,
      "coordinate_frame": "world",
      "topology": "undirected_knn",
      "k_neighbors": simulator.config.k_neighbors,
      "num_particles": len(simulator.last_initial_particles),
      "num_springs": len(spring_edges),
      "particle_positions_world": (
          simulator.last_initial_particles.astype(np.float64)),
      "edges": spring_edges,
      "edge_columns": ["source_particle_index", "target_particle_index"],
      "rest_lengths": spring_rest_lengths.astype(np.float64),
      "config": {
          "particle_spacing": simulator.config.particle_spacing,
          "surface_sample_spacing": simulator.config.surface_sample_spacing,
          "weld_tolerance": simulator.config.weld_tolerance,
          "spring_stiffness": simulator.config.spring_stiffness,
          "damping": simulator.config.damping,
          "total_mass": simulator.config.total_mass,
          "initial_velocity": simulator.config.initial_velocity,
          "ground_axis": simulator.config.ground_axis,
          "ground_height": simulator.config.ground_height,
          "restitution": simulator.config.restitution,
          "friction": simulator.config.friction,
          "require_watertight": simulator.config.require_watertight,
          "seed": simulator.config.seed,
      },
  }, asset_output_dir / "spring_mass_metatada.json")
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
    frames = renderer.render(return_layers=render_layers)
    kb.write_image_dict(
        frames, camera_output_dir, file_templates=IMAGE_FILE_TEMPLATES)
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
  _write_processed_camera_metadata(
      output_dir=asset_output_dir,
      asset_id=asset_id,
      cameras=cameras,
      resolution=scene.resolution,
      frame_start=scene.frame_start,
      frame_end=scene.frame_end,
      multi_camera=len(cameras) > 1,
  )
  kb.write_json({
      "asset_id": asset_id,
      "run": _get_run_metadata(FLAGS),
      "mesh_vertices_file": mesh_vertices_path.name,
      "spring_mass_particles_file": spring_mass_particles_path.name,
      "num_mesh_vertices": vertex_animation.num_vertices,
      "num_mesh_faces": len(faces),
      "num_particles": len(simulator.last_initial_particles),
      "num_springs": len(simulator.last_edges),
      "control_vertex_index": (
          None if repeated_lift is None else FLAGS.control_vertex_index),
      "control_particle_index": simulator.last_control_particle_index,
      "control_target_xy": (
          None if repeated_lift is None
          else [FLAGS.control_target_x, FLAGS.control_target_y]),
      "control_point": control_point_metadata,
      "randomize_cameras": FLAGS.randomize_cameras,
      "camera_follow_obj": FLAGS.camera_follow_obj,
      "camera_seed": camera_seed,
      "cameras": cameras_metadata,
  }, asset_output_dir / "metadata.json")
