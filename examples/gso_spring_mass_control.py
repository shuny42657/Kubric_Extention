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

"""Renders a GSO spring-mass object driven by external control-point motion.

The control trajectory is loaded from an .npz file with:
  control_vertex_indices: int array with shape (num_points,)
  positions_world: float array with shape (num_frames, num_points, 3)
  frame_indices: optional int array
  frame_rate: optional scalar

If --control_trajectory is omitted, this script creates a simple trajectory
that smoothly lifts either one vertex or a row of vertices along one side of
the object upward, then keeps them held at the final position for the rest of
the render.
"""

import logging
import pathlib
import shlex
import sys

import numpy as np

import kubric as kb
from kubric.renderer import Blender
from kubric.simulator import ControlTrajectoryConfig
from kubric.simulator import SpringMassConfig
from kubric.simulator import SpringMassSimulator

try:
  import torch
except ImportError:
  torch = None


IMAGE_FILE_TEMPLATES = {
    "rgba": "image/rgba_{:05d}.png",
    "depth": "depth/depth_{:05d}.tiff",
    "segmentation": "segmentation/segmentation_{:05d}.png",
}


def _smoothstep(x):
  x = np.clip(x, 0., 1.)
  return x * x * (3. - 2. * x)


def _get_run_metadata(flags):
  argv = list(sys.argv)
  return {
      "argv": argv,
      "command": " ".join(shlex.quote(arg) for arg in argv),
      "flags": dict(vars(flags)),
  }


def _release_torch_cuda_cache():
  if torch is not None and torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _place_object_on_ground(obj, local_bounds, scale, ground_height=0.0):
  scaled_bounds = np.asarray(local_bounds, dtype=np.float64) * scale
  obj.position = (
      0.,
      0.,
      ground_height - float(scaled_bounds[0, 2]) + 1e-3,
  )


def _ensure_mesh_above_ground(obj, vertices, ground_height=0.0, clearance=1e-3):
  world_vertices = _to_world_vertices(obj, vertices)
  min_z = float(np.min(world_vertices[:, 2]))
  target_min_z = ground_height + clearance
  if min_z < target_min_z:
    obj.position = (
        float(obj.position[0]),
        float(obj.position[1]),
        float(obj.position[2]) + target_min_z - min_z,
    )
    world_vertices = _to_world_vertices(obj, vertices)
    min_z = float(np.min(world_vertices[:, 2]))
  if min_z < ground_height - 1e-6:
    raise RuntimeError(
        f"Object mesh penetrates the ground plane: min_z={min_z}, "
        f"ground_height={ground_height}")
  return min_z


def _select_edge_control_vertices(
    world_vertices,
    side_axis,
    side,
    band_fraction,
    max_points,
):
  if side_axis not in (0, 1):
    raise ValueError("control_edge_axis must be 0 or 1")
  if side not in ("min", "max"):
    raise ValueError("control_edge_side must be 'min' or 'max'")
  if not 0. < band_fraction <= 0.5:
    raise ValueError("control_edge_band_fraction must be in (0, 0.5]")
  if max_points <= 0:
    raise ValueError("control_edge_max_points must be positive")

  coordinates = world_vertices[:, side_axis]
  lower = float(np.min(coordinates))
  upper = float(np.max(coordinates))
  extent = upper - lower
  if extent <= 0:
    raise ValueError("Cannot select an edge from a degenerate bbox")
  band_width = extent * band_fraction
  if side == "min":
    candidate_indices = np.flatnonzero(coordinates <= lower + band_width)
  else:
    candidate_indices = np.flatnonzero(coordinates >= upper - band_width)
  if len(candidate_indices) == 0:
    raise ValueError("No vertices found in the requested control edge band")

  tangent_axis = 1 - side_axis
  tangent_coordinates = world_vertices[candidate_indices, tangent_axis]
  sorted_candidates = candidate_indices[np.argsort(tangent_coordinates)]
  if len(sorted_candidates) <= max_points:
    return sorted_candidates.astype(np.int32)

  sample_positions = np.linspace(
      0, len(sorted_candidates) - 1, max_points).round().astype(np.int32)
  return sorted_candidates[sample_positions].astype(np.int32)


def _parse_grid_point_spec(point_spec):
  try:
    x_part, y_part = point_spec.split("_")
    if not x_part.startswith("x") or not y_part.startswith("y"):
      raise ValueError
    return int(x_part[1:]), int(y_part[1:])
  except ValueError as exc:
    raise ValueError(
        "Grid point spec must look like 'x0_y1', got "
        f"{point_spec!r}") from exc


def _select_grid_pair_control_vertices(world_vertices, grid_size, grid_points):
  if grid_size < 2:
    raise ValueError("control_grid_size must be at least 2")
  point_specs = [spec.strip() for spec in grid_points.split(",") if spec.strip()]
  if len(point_specs) != 2:
    raise ValueError("control_grid_points must contain exactly two points")

  lower = np.min(world_vertices[:, :2], axis=0)
  upper = np.max(world_vertices[:, :2], axis=0)
  selected_indices = []
  for point_spec in point_specs:
    grid_x, grid_y = _parse_grid_point_spec(point_spec)
    if not 0 <= grid_x < grid_size or not 0 <= grid_y < grid_size:
      raise ValueError(
          f"Grid point {point_spec!r} is outside a {grid_size}x{grid_size} grid")
    fraction = np.asarray([grid_x, grid_y], dtype=np.float64) / (grid_size - 1)
    target_xy = lower + fraction * (upper - lower)
    distances = np.sum((world_vertices[:, :2] - target_xy[None, :]) ** 2, axis=1)
    nearest_order = np.argsort(distances)
    for vertex_index in nearest_order:
      vertex_index = int(vertex_index)
      if vertex_index not in selected_indices:
        selected_indices.append(vertex_index)
        break
  if len(selected_indices) != 2:
    raise ValueError("Could not select two distinct grid control vertices")
  return np.asarray(selected_indices, dtype=np.int32)


def _create_default_lift_npz(
    path,
    initial_control_positions,
    control_vertex_indices,
    frame_start,
    frame_end,
    frame_rate,
    lift_height,
    lift_seconds,
    flip_after_lift=False,
    flip_center_xy=(0., 0.),
    flip_rotation_degrees=180.,
    flip_seconds=2.0,
    flip_extra_height=0.,
    trajectory_mode="lift",
    hold_seconds=0.0,
    lower_seconds=2.0,
    release_height=0.05,
):
  frame_indices = np.arange(frame_start, frame_end + 1, dtype=np.int32)
  elapsed = (frame_indices - frame_start).astype(np.float64) / frame_rate
  initial_control_positions = np.asarray(
      initial_control_positions, dtype=np.float32)
  control_vertex_indices = np.asarray(control_vertex_indices, dtype=np.int32)
  positions = np.repeat(
      initial_control_positions[None, :, :],
      len(frame_indices),
      axis=0,
  )
  control_active = np.ones(
      (len(frame_indices), len(control_vertex_indices)), dtype=np.bool_)
  if trajectory_mode == "lift":
    lift_progress = _smoothstep(elapsed / lift_seconds)
    positions[:, :, 2] += (
        lift_height * lift_progress).astype(np.float32)[:, None]
    description = "default smooth lift; final control points remain held"
  elif trajectory_mode == "lift_drop_release":
    if lower_seconds <= 0:
      raise ValueError("lower_seconds must be positive")
    if hold_seconds < 0:
      raise ValueError("hold_seconds cannot be negative")
    if release_height < 0:
      raise ValueError("release_height cannot be negative")
    lift_progress = _smoothstep(elapsed / lift_seconds)
    lower_start_seconds = lift_seconds + hold_seconds
    lower_progress = _smoothstep(
        (elapsed - lower_start_seconds) / lower_seconds)
    height_offset = (
        lift_height * (1.0 - lower_progress) +
        release_height * lower_progress)
    before_lower = elapsed < lower_start_seconds
    height_offset[before_lower] = lift_height * lift_progress[before_lower]
    release_seconds = lower_start_seconds + lower_seconds
    released = elapsed >= release_seconds
    height_offset[released] = release_height
    positions[:, :, 2] += height_offset.astype(np.float32)[:, None]
    control_active[released, :] = False
    description = (
        "smooth lift, lower near ground, then release control constraints")
  else:
    raise ValueError(f"Unknown trajectory_mode: {trajectory_mode!r}")

  if flip_after_lift and trajectory_mode == "lift":
    if flip_seconds <= 0:
      raise ValueError("flip_seconds must be positive")
    flip_progress = _smoothstep((elapsed - lift_seconds) / flip_seconds)
    angles = np.deg2rad(flip_rotation_degrees * flip_progress)
    cos_angles = np.cos(angles).astype(np.float32)
    sin_angles = np.sin(angles).astype(np.float32)
    center = np.asarray(flip_center_xy, dtype=np.float32)
    relative_xy = initial_control_positions[:, :2] - center[None, :]
    rotated_x = (
        relative_xy[None, :, 0] * cos_angles[:, None] -
        relative_xy[None, :, 1] * sin_angles[:, None])
    rotated_y = (
        relative_xy[None, :, 0] * sin_angles[:, None] +
        relative_xy[None, :, 1] * cos_angles[:, None])
    positions[:, :, 0] = center[0] + rotated_x
    positions[:, :, 1] = center[1] + rotated_y
    positions[:, :, 2] += (
        flip_extra_height * np.sin(np.pi * flip_progress)
    ).astype(np.float32)[:, None]
    description = (
        "default smooth lift followed by z-axis rotation; final control "
        "points remain held")
  elif flip_after_lift:
    raise ValueError("flip_after_lift is only supported with trajectory_mode=lift")
  np.savez_compressed(
      str(path),
      control_vertex_indices=control_vertex_indices,
      positions_world=positions.astype(np.float32),
      control_active=control_active,
      frame_indices=frame_indices,
      frame_rate=np.asarray(frame_rate, dtype=np.float32),
      description=np.asarray(description),
  )


def _load_control_trajectory(path, default_frame_rate):
  with np.load(str(path), allow_pickle=False) as data:
    if "control_vertex_indices" not in data:
      raise ValueError("control trajectory npz must contain control_vertex_indices")
    if "positions_world" in data:
      positions = data["positions_world"]
    elif "positions" in data:
      positions = data["positions"]
    else:
      raise ValueError(
          "control trajectory npz must contain positions_world or positions")
    indices = tuple(int(index) for index in data["control_vertex_indices"])
    frame_rate = (
        float(data["frame_rate"]) if "frame_rate" in data else default_frame_rate)
    if "frame_indices" in data:
      frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
      frame_start = int(frame_indices[0])
    else:
      frame_start = 0
      frame_indices = np.arange(
          frame_start, frame_start + len(positions), dtype=np.int32)
    active = data["control_active"] if "control_active" in data else None
  config = ControlTrajectoryConfig(
      control_vertex_indices=indices,
      positions=positions,
      active=active,
      frame_start=frame_start,
      frame_rate=frame_rate,
  )
  return config, frame_indices


def _load_simulation_data(path):
  with np.load(str(path), allow_pickle=False) as data:
    return {key: np.array(data[key]) for key in data.files}


def _scalar(data, key):
  return data[key].item()


def _string(data, key):
  return str(_scalar(data, key))


def _save_simulation_data(
    path,
    flags,
    scene,
    asset,
    vertices,
    faces,
    simulator,
    control_trajectory,
    control_frame_indices,
):
  """Stores enough simulation input state to replay the spring-mass run."""
  np.savez_compressed(
      str(path),
      format=np.asarray("kubric_spring_mass_control_replay_v1"),
      asset_id=np.asarray(flags.asset_id),
      gso_assets=np.asarray(flags.gso_assets),
      object_position=np.asarray(asset.position, dtype=np.float32),
      object_quaternion=np.asarray(asset.quaternion, dtype=np.float32),
      object_scale=np.asarray(asset.scale, dtype=np.float32),
      object_mass=np.asarray(asset.mass, dtype=np.float32),
      object_friction=np.asarray(asset.friction, dtype=np.float32),
      object_restitution=np.asarray(asset.restitution, dtype=np.float32),
      rest_vertices_local=vertices.astype(np.float32),
      faces=faces.astype(np.int32),
      initial_particle_positions_world=(
          simulator.last_initial_particles.astype(np.float32)),
      surface_mapping=simulator.last_surface_mapping.astype(np.int32),
      spring_edges=simulator.last_edges.astype(np.int32),
      control_vertex_indices=np.asarray(
          control_trajectory.control_vertex_indices, dtype=np.int32),
      control_particle_indices=simulator.last_control_particle_indices.astype(
          np.int32),
      control_attachment_indices=(
          np.empty((0, 0), dtype=np.int32)
          if simulator.last_control_attachment_indices is None
          else simulator.last_control_attachment_indices),
      control_positions_world=control_trajectory.positions.astype(np.float32),
      control_active=control_trajectory.active.astype(np.bool_),
      control_frame_indices=control_frame_indices.astype(np.int32),
      control_frame_start=np.asarray(control_trajectory.frame_start,
                                     dtype=np.int32),
      control_frame_rate=np.asarray(control_trajectory.frame_rate,
                                    dtype=np.float32),
      scene_frame_start=np.asarray(scene.frame_start, dtype=np.int32),
      scene_frame_end=np.asarray(scene.frame_end, dtype=np.int32),
      scene_frame_rate=np.asarray(scene.frame_rate, dtype=np.int32),
      scene_step_rate=np.asarray(scene.step_rate, dtype=np.int32),
      scene_gravity=np.asarray(scene.gravity, dtype=np.float32),
      particle_spacing=np.asarray(simulator.config.particle_spacing,
                                  dtype=np.float32),
      k_neighbors=np.asarray(simulator.config.k_neighbors, dtype=np.int32),
      sampled_surface_particle_count=np.asarray(
          -1 if simulator.config.sampled_surface_particle_count is None
          else simulator.config.sampled_surface_particle_count,
          dtype=np.int32),
      max_particles=np.asarray(simulator.config.max_particles, dtype=np.int32),
      spring_stiffness=np.asarray(simulator.config.spring_stiffness,
                                  dtype=np.float32),
      damping=np.asarray(simulator.config.damping, dtype=np.float32),
      control_mode=np.asarray(simulator.config.control_mode),
      control_attachment_k=np.asarray(
          simulator.config.control_attachment_k, dtype=np.int32),
      control_attachment_radius=np.asarray(
          -1 if simulator.config.control_attachment_radius is None
          else simulator.config.control_attachment_radius,
          dtype=np.float32),
      control_stiffness=np.asarray(
          simulator.config.control_stiffness, dtype=np.float32),
      control_damping=np.asarray(
          simulator.config.control_damping, dtype=np.float32),
      total_mass=np.asarray(simulator.config.total_mass, dtype=np.float32),
      initial_velocity=np.asarray(simulator.config.initial_velocity,
                                  dtype=np.float32),
      ground_axis=np.asarray(simulator.config.ground_axis, dtype=np.int32),
      ground_height=np.asarray(simulator.config.ground_height,
                               dtype=np.float32),
      restitution=np.asarray(simulator.config.restitution, dtype=np.float32),
      friction=np.asarray(simulator.config.friction, dtype=np.float32),
      require_watertight=np.asarray(simulator.config.require_watertight),
      seed=np.asarray(simulator.config.seed, dtype=np.int32),
  )


def _to_world_vertices(asset, vertices):
  scaled = vertices * np.asarray(asset.scale, dtype=np.float64)
  rotation = np.asarray(asset.matrix_world, dtype=np.float64)[:3, :3]
  return scaled @ rotation.T + np.asarray(asset.position, dtype=np.float64)


parser = kb.ArgumentParser()
parser.add_argument(
    "--gso_assets",
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json",
)
parser.add_argument(
    "--asset_id",
    type=str,
    default="Cole_Hardware_Dishtowel_Stripe",
    help=(
        "GSO asset ID. This corresponds to "
        "https://app.gazebosim.org/GoogleResearch/fuel/models/"
        "Cole_Hardware_Dishtowel_Stripe in the public GSO manifest."),
)
parser.add_argument("--control_trajectory", type=str, default=None)
parser.add_argument(
    "--control_mode",
    choices=["edge", "single", "grid_pair"],
    default="edge",
    help="Default trajectory type used when --control_trajectory is omitted.")
parser.add_argument("--control_vertex_index", type=int, default=0)
parser.add_argument("--control_grid_size", type=int, default=4)
parser.add_argument("--control_grid_points", type=str, default="x0_y0,x1_y0")
parser.add_argument(
    "--control_edge_axis",
    type=int,
    default=1,
    help="World horizontal axis used to choose the lifted side: 0=x, 1=y.")
parser.add_argument(
    "--control_edge_side",
    choices=["min", "max"],
    default="max",
    help="Which side of the selected axis is lifted.")
parser.add_argument("--control_edge_band_fraction", type=float, default=0.03)
parser.add_argument("--control_edge_max_points", type=int, default=12)
parser.add_argument("--lift_height", type=float, default=0.8)
parser.add_argument("--lift_seconds", type=float, default=3.0)
parser.add_argument(
    "--trajectory_mode",
    choices=["lift", "lift_drop_release"],
    default="lift")
parser.add_argument(
    "--hold_seconds",
    type=float,
    default=0.0,
    help="Hold duration after lifting before lowering in lift_drop_release mode.")
parser.add_argument(
    "--lower_seconds",
    type=float,
    default=2.0,
    help="Duration for lowering control points before release.")
parser.add_argument(
    "--release_height",
    type=float,
    default=0.05,
    help="Height above each initial control position at which control releases.")
parser.add_argument(
    "--flip_after_lift",
    action="store_true",
    help="After lifting, rotate the controlled edge around the world z axis.")
parser.add_argument("--flip_seconds", type=float, default=2.0)
parser.add_argument("--flip_rotation_degrees", type=float, default=180.0)
parser.add_argument(
    "--flip_distance_fraction",
    type=float,
    default=1.1,
    help="Deprecated; ignored. Flip now uses z-axis rotation.")
parser.add_argument(
    "--flip_extra_height",
    type=float,
    default=0.2,
    help="Extra arc height during the flip phase.")
parser.add_argument("--particle_spacing", type=float, default=0.08)
parser.add_argument("--k_neighbors", type=int, default=16)
parser.add_argument("--sampled_surface_particle_count", type=int, default=None)
parser.add_argument("--max_particles", type=int, default=20000)
parser.add_argument("--spring_stiffness", type=float, default=200.0)
parser.add_argument("--damping", type=float, default=0.5)
parser.add_argument("--control_physics_mode", choices=["hard", "soft"], default="hard")
parser.add_argument("--control_attachment_k", type=int, default=12)
parser.add_argument("--control_attachment_radius", type=float, default=None)
parser.add_argument("--control_stiffness", type=float, default=50.0)
parser.add_argument("--control_damping", type=float, default=0.1)
parser.add_argument("--require_watertight", action="store_true")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--camera_distance", type=float, default=2.8)
parser.add_argument("--camera_height", type=float, default=1.6)
parser.add_argument("--camera_look_at_z", type=float, default=0.35)
parser.add_argument(
    "--camera_views",
    type=str,
    default="front",
    help=(
        "Comma-separated camera views to render. Supported values are "
        "front,left,right. Multiple views are written under camera_XX folders."))
parser.add_argument("--samples_per_pixel", type=int, default=64)
parser.add_argument(
    "--save_simulation_data",
    action="store_true",
    help="Save a replayable spring-mass simulation input package.")
parser.add_argument(
    "--skip_auxiliary_data",
    action="store_true",
    help=(
        "Skip mesh_vertices.npz, spring_mass_particles.npz, and .blend output. "
        "Image, depth, segmentation, and metadata outputs are still written."))
parser.add_argument(
    "--load_simulation_data",
    type=str,
    default=None,
    help="Load a replayable spring-mass simulation input package.")
parser.add_argument(
    "--load_spring_mass_model",
    type=str,
    default=None,
    help=(
        "Load only the spring-mass model state from a simulation_data.npz "
        "package. The control trajectory is still taken from "
        "--control_trajectory or generated from the current flags."))
parser.add_argument("--render_depth", action="store_true")
parser.add_argument("--render_segmentation", action="store_true")
parser.set_defaults(
    frame_start=0,
    frame_end=96,
    frame_rate=24,
    step_rate=240,
    resolution="256x256",
    seed=42,
)
FLAGS = parser.parse_args()

REPLAY_DATA = None
SPRING_MASS_MODEL_DATA = None
if (FLAGS.load_simulation_data is not None and
    FLAGS.load_spring_mass_model is not None):
  raise ValueError(
      "Specify only one of --load_simulation_data and --load_spring_mass_model")
if FLAGS.load_simulation_data is not None:
  REPLAY_DATA = _load_simulation_data(FLAGS.load_simulation_data)
  FLAGS.asset_id = _string(REPLAY_DATA, "asset_id")
  FLAGS.gso_assets = _string(REPLAY_DATA, "gso_assets")
  FLAGS.frame_start = int(_scalar(REPLAY_DATA, "scene_frame_start"))
  FLAGS.frame_end = int(_scalar(REPLAY_DATA, "scene_frame_end"))
  FLAGS.frame_rate = int(_scalar(REPLAY_DATA, "scene_frame_rate"))
  FLAGS.step_rate = int(_scalar(REPLAY_DATA, "scene_step_rate"))
  FLAGS.particle_spacing = float(_scalar(REPLAY_DATA, "particle_spacing"))
  FLAGS.k_neighbors = int(_scalar(REPLAY_DATA, "k_neighbors"))
  if "sampled_surface_particle_count" in REPLAY_DATA:
    value = int(_scalar(REPLAY_DATA, "sampled_surface_particle_count"))
    FLAGS.sampled_surface_particle_count = None if value < 0 else value
  if "max_particles" in REPLAY_DATA:
    FLAGS.max_particles = int(_scalar(REPLAY_DATA, "max_particles"))
  FLAGS.spring_stiffness = float(_scalar(REPLAY_DATA, "spring_stiffness"))
  FLAGS.damping = float(_scalar(REPLAY_DATA, "damping"))
  if "control_mode" in REPLAY_DATA:
    FLAGS.control_physics_mode = _string(REPLAY_DATA, "control_mode")
  if "control_attachment_k" in REPLAY_DATA:
    FLAGS.control_attachment_k = int(_scalar(REPLAY_DATA, "control_attachment_k"))
  if "control_attachment_radius" in REPLAY_DATA:
    value = float(_scalar(REPLAY_DATA, "control_attachment_radius"))
    FLAGS.control_attachment_radius = None if value < 0 else value
  if "control_stiffness" in REPLAY_DATA:
    FLAGS.control_stiffness = float(_scalar(REPLAY_DATA, "control_stiffness"))
  if "control_damping" in REPLAY_DATA:
    FLAGS.control_damping = float(_scalar(REPLAY_DATA, "control_damping"))
  FLAGS.require_watertight = bool(_scalar(REPLAY_DATA, "require_watertight"))
  FLAGS.seed = int(_scalar(REPLAY_DATA, "seed"))
  FLAGS.gravity = tuple(REPLAY_DATA["scene_gravity"].astype(float).tolist())
elif FLAGS.load_spring_mass_model is not None:
  SPRING_MASS_MODEL_DATA = _load_simulation_data(FLAGS.load_spring_mass_model)
  FLAGS.asset_id = _string(SPRING_MASS_MODEL_DATA, "asset_id")
  FLAGS.gso_assets = _string(SPRING_MASS_MODEL_DATA, "gso_assets")
  FLAGS.particle_spacing = float(
      _scalar(SPRING_MASS_MODEL_DATA, "particle_spacing"))
  FLAGS.k_neighbors = int(_scalar(SPRING_MASS_MODEL_DATA, "k_neighbors"))
  if "sampled_surface_particle_count" in SPRING_MASS_MODEL_DATA:
    value = int(_scalar(
        SPRING_MASS_MODEL_DATA, "sampled_surface_particle_count"))
    FLAGS.sampled_surface_particle_count = None if value < 0 else value
  if "max_particles" in SPRING_MASS_MODEL_DATA:
    FLAGS.max_particles = int(_scalar(SPRING_MASS_MODEL_DATA, "max_particles"))
  FLAGS.spring_stiffness = float(
      _scalar(SPRING_MASS_MODEL_DATA, "spring_stiffness"))
  FLAGS.damping = float(_scalar(SPRING_MASS_MODEL_DATA, "damping"))
  FLAGS.require_watertight = bool(
      _scalar(SPRING_MASS_MODEL_DATA, "require_watertight"))
  FLAGS.seed = int(_scalar(SPRING_MASS_MODEL_DATA, "seed"))
  FLAGS.gravity = tuple(
      SPRING_MASS_MODEL_DATA["scene_gravity"].astype(float).tolist())

if FLAGS.lift_seconds <= 0:
  raise ValueError("lift_seconds must be positive")
if FLAGS.hold_seconds < 0:
  raise ValueError("hold_seconds cannot be negative")
if FLAGS.lower_seconds <= 0:
  raise ValueError("lower_seconds must be positive")
if FLAGS.release_height < 0:
  raise ValueError("release_height cannot be negative")
if FLAGS.flip_seconds <= 0:
  raise ValueError("flip_seconds must be positive")
if FLAGS.control_edge_axis not in (0, 1):
  raise ValueError("control_edge_axis must be 0 or 1")

CAMERA_VIEWS = [view.strip() for view in FLAGS.camera_views.split(",")
                if view.strip()]
if not CAMERA_VIEWS:
  raise ValueError("--camera_views must contain at least one view")
for view in CAMERA_VIEWS:
  if view not in ("front", "left", "right"):
    raise ValueError(
        "--camera_views only supports front,left,right; got "
        f"{view!r}")

render_layers = ["rgba"]
if FLAGS.render_depth:
  render_layers.append("depth")
if FLAGS.render_segmentation:
  render_layers.append("segmentation")

scene, _, output_dir, scratch_dir = kb.setup(FLAGS)
MODEL_DATA = REPLAY_DATA if REPLAY_DATA is not None else SPRING_MASS_MODEL_DATA
renderer = Blender(
    scene,
    scratch_dir,
    samples_per_pixel=FLAGS.samples_per_pixel,
    use_denoising=True)
simulator = SpringMassSimulator(
    scene,
    config=SpringMassConfig(
        particle_spacing=FLAGS.particle_spacing,
        k_neighbors=FLAGS.k_neighbors,
        sampled_surface_particle_count=FLAGS.sampled_surface_particle_count,
        max_particles=FLAGS.max_particles,
        spring_stiffness=FLAGS.spring_stiffness,
        damping=FLAGS.damping,
        control_mode=FLAGS.control_physics_mode,
        control_attachment_k=FLAGS.control_attachment_k,
        control_attachment_radius=FLAGS.control_attachment_radius,
        control_stiffness=FLAGS.control_stiffness,
        control_damping=FLAGS.control_damping,
        total_mass=(
            float(_scalar(MODEL_DATA, "total_mass"))
            if MODEL_DATA is not None else 1.0),
        initial_velocity=(
            tuple(MODEL_DATA["initial_velocity"].astype(float).tolist())
            if MODEL_DATA is not None else (0., 0., 0.)),
        ground_axis=(
            int(_scalar(MODEL_DATA, "ground_axis"))
            if MODEL_DATA is not None else 2),
        ground_height=(
            float(_scalar(MODEL_DATA, "ground_height"))
            if MODEL_DATA is not None else 0.),
        restitution=(
            float(_scalar(MODEL_DATA, "restitution"))
            if MODEL_DATA is not None else 0.0),
        friction=(
            float(_scalar(MODEL_DATA, "friction"))
            if MODEL_DATA is not None else 0.5),
        require_watertight=FLAGS.require_watertight,
        record_all_particles=True,
        seed=FLAGS.seed,
    ),
    device=FLAGS.device,
)

floor = kb.Cube(
    name="floor",
    scale=(3., 3., 0.1),
    position=(0., 0., -0.1),
    static=True,
    material=kb.PrincipledBSDFMaterial(color=kb.Color(0.35, 0.35, 0.35)),
)
scene.add(floor)
scene.add(kb.DirectionalLight(
    name="sun", position=(-3., -4., 6.), look_at=(0., 0., 0.), intensity=2.0))
scene.ambient_illumination = kb.Color(0.15, 0.15, 0.15)

def _camera_position_for_view(view):
  if view == "front":
    return (0., -FLAGS.camera_distance, FLAGS.camera_height)
  if view == "left":
    return (-FLAGS.camera_distance, 0., FLAGS.camera_height)
  if view == "right":
    return (FLAGS.camera_distance, 0., FLAGS.camera_height)
  raise ValueError(f"Unsupported camera view: {view}")


def _set_camera_view(camera, view):
  camera.position = _camera_position_for_view(view)
  camera.look_at((0., 0., FLAGS.camera_look_at_z))


camera = kb.PerspectiveCamera(name="camera")
_set_camera_view(camera, CAMERA_VIEWS[0])
scene.camera = camera
scene.add(camera)

with kb.AssetSource.from_manifest(FLAGS.gso_assets, scratch_dir) as gso:
  if FLAGS.asset_id not in gso._assets:  # pylint: disable=protected-access
    raise ValueError(f"Unknown GSO asset ID: {FLAGS.asset_id!r}")

  obj = gso.create(asset_id=FLAGS.asset_id)
  if MODEL_DATA is None:
    bounds = np.asarray(obj.bounds, dtype=np.float64)
    scale = 1.0 / np.max(bounds[1] - bounds[0])
    obj.scale = (scale, scale, scale)
    _place_object_on_ground(obj, bounds, scale)
  else:
    obj.position = tuple(MODEL_DATA["object_position"].astype(float).tolist())
    obj.quaternion = tuple(
        MODEL_DATA["object_quaternion"].astype(float).tolist())
    obj.scale = tuple(MODEL_DATA["object_scale"].astype(float).tolist())
    obj.mass = float(_scalar(MODEL_DATA, "object_mass"))
    obj.friction = float(_scalar(MODEL_DATA, "object_friction"))
    obj.restitution = float(_scalar(MODEL_DATA, "object_restitution"))
  scene.add(obj)

  if MODEL_DATA is None:
    vertices, faces = renderer.get_mesh_geometry(obj)
    initial_min_z = _ensure_mesh_above_ground(obj, vertices, ground_height=0.0)
    logging.info("Placed object with render-mesh min z = %.6f", initial_min_z)
  else:
    vertices = MODEL_DATA["rest_vertices_local"].astype(np.float32)
    faces = MODEL_DATA["faces"].astype(np.int64)
  if FLAGS.control_vertex_index < 0 or FLAGS.control_vertex_index >= len(vertices):
    raise ValueError("control_vertex_index is outside the render mesh range")

  control_trajectory_path = (
      pathlib.Path(FLAGS.control_trajectory)
      if FLAGS.control_trajectory else output_dir / "control_trajectory.npz")
  if REPLAY_DATA is not None:
    control_trajectory = ControlTrajectoryConfig(
        control_vertex_indices=tuple(
            REPLAY_DATA["control_vertex_indices"].astype(int).tolist()),
        positions=REPLAY_DATA["control_positions_world"].astype(np.float32),
        active=(
            REPLAY_DATA["control_active"].astype(np.bool_)
            if "control_active" in REPLAY_DATA else None),
        frame_start=int(_scalar(REPLAY_DATA, "control_frame_start")),
        frame_rate=float(_scalar(REPLAY_DATA, "control_frame_rate")),
    )
    control_frame_indices = REPLAY_DATA["control_frame_indices"].astype(np.int32)
    control_trajectory_path = pathlib.Path(FLAGS.load_simulation_data)
  elif FLAGS.control_trajectory is None:
    world_vertices = _to_world_vertices(obj, vertices)
    bbox_lower = np.min(world_vertices, axis=0)
    bbox_upper = np.max(world_vertices, axis=0)
    flip_center_xy = 0.5 * (bbox_lower[:2] + bbox_upper[:2])
    if FLAGS.control_mode == "single":
      control_vertex_indices = np.asarray(
          [FLAGS.control_vertex_index], dtype=np.int32)
    elif FLAGS.control_mode == "grid_pair":
      control_vertex_indices = _select_grid_pair_control_vertices(
          world_vertices=world_vertices,
          grid_size=FLAGS.control_grid_size,
          grid_points=FLAGS.control_grid_points,
      )
    else:
      control_vertex_indices = _select_edge_control_vertices(
          world_vertices=world_vertices,
          side_axis=FLAGS.control_edge_axis,
          side=FLAGS.control_edge_side,
          band_fraction=FLAGS.control_edge_band_fraction,
          max_points=FLAGS.control_edge_max_points,
      )
    logging.info(
        "Using %d control vertices: %s",
        len(control_vertex_indices), control_vertex_indices.tolist())
    _create_default_lift_npz(
        path=control_trajectory_path,
        initial_control_positions=world_vertices[control_vertex_indices],
        control_vertex_indices=control_vertex_indices,
        frame_start=scene.frame_start,
        frame_end=scene.frame_end + 1,
        frame_rate=scene.frame_rate,
        lift_height=FLAGS.lift_height,
        lift_seconds=FLAGS.lift_seconds,
        flip_after_lift=FLAGS.flip_after_lift,
        flip_center_xy=flip_center_xy,
        flip_rotation_degrees=FLAGS.flip_rotation_degrees,
        flip_seconds=FLAGS.flip_seconds,
        flip_extra_height=FLAGS.flip_extra_height,
        trajectory_mode=FLAGS.trajectory_mode,
        hold_seconds=FLAGS.hold_seconds,
        lower_seconds=FLAGS.lower_seconds,
        release_height=FLAGS.release_height,
    )
    logging.info("Wrote default control trajectory to %s", control_trajectory_path)

  if REPLAY_DATA is None:
    control_trajectory, control_frame_indices = _load_control_trajectory(
        control_trajectory_path, default_frame_rate=scene.frame_rate)
  vertex_animation = simulator.run(
      asset=obj,
      vertices=vertices,
      faces=faces,
      frame_start=scene.frame_start,
      frame_end=scene.frame_end + 1,
      control_trajectory=control_trajectory,
      initial_particles_world=(
          None if MODEL_DATA is None
          else MODEL_DATA["initial_particle_positions_world"]),
      surface_mapping=(
          None if MODEL_DATA is None else MODEL_DATA["surface_mapping"]),
      edges=None if MODEL_DATA is None else MODEL_DATA["spring_edges"],
  )
  renderer.add_vertex_animation(obj, vertex_animation)

  simulation_data_path = None
  if FLAGS.save_simulation_data and not FLAGS.skip_auxiliary_data:
    simulation_data_path = output_dir / "simulation_data.npz"
    _save_simulation_data(
        path=simulation_data_path,
        flags=FLAGS,
        scene=scene,
        asset=obj,
        vertices=vertices,
        faces=faces,
        simulator=simulator,
        control_trajectory=control_trajectory,
        control_frame_indices=control_frame_indices,
    )
    logging.info("Wrote replayable simulation data to %s", simulation_data_path)

  mesh_vertices_path = None
  particles_path = None
  if not FLAGS.skip_auxiliary_data:
    mesh_frame_indices = np.arange(
        vertex_animation.frame_start, vertex_animation.frame_end + 1,
        dtype=np.int32)
    mesh_vertices_path = output_dir / "mesh_vertices.npz"
    np.savez_compressed(
        str(mesh_vertices_path),
        vertices_world=vertex_animation.vertices,
        velocities_world=simulator.last_render_velocity_trajectory,
        frame_indices=mesh_frame_indices,
        rest_vertices_local=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        control_vertex_indices=np.asarray(
            control_trajectory.control_vertex_indices, dtype=np.int32),
        control_active=control_trajectory.active.astype(np.bool_),
    )

    particles_path = output_dir / "spring_mass_particles.npz"
    np.savez_compressed(
        str(particles_path),
        particle_positions_world=simulator.last_particle_trajectory,
        frame_indices=mesh_frame_indices,
        initial_particle_positions_world=(
            simulator.last_initial_particles.astype(np.float32)),
        surface_mapping=simulator.last_surface_mapping.astype(np.int32),
        edges=simulator.last_edges.astype(np.int32),
        control_particle_indices=simulator.last_control_particle_indices,
        control_active=control_trajectory.active.astype(np.bool_),
        control_attachment_indices=(
            np.empty((0, 0), dtype=np.int32)
            if simulator.last_control_attachment_indices is None
            else simulator.last_control_attachment_indices),
    )

    renderer.save_state(output_dir / "gso_spring_mass_control.blend")
  camera_metadata = []
  for camera_index, camera_view in enumerate(CAMERA_VIEWS):
    _set_camera_view(camera, camera_view)
    _release_torch_cuda_cache()
    frames = renderer.render(return_layers=render_layers)
    camera_output_dir = (
        output_dir if len(CAMERA_VIEWS) == 1
        else output_dir / f"camera_{camera_index:02d}")
    kb.write_image_dict(
        frames, camera_output_dir, file_templates=IMAGE_FILE_TEMPLATES)
    camera_metadata.append({
        "index": camera_index,
        "name": f"camera_{camera_index:02d}",
        "view": camera_view,
        "position": list(_camera_position_for_view(camera_view)),
        "look_at": [0., 0., FLAGS.camera_look_at_z],
    })

  kb.write_json({
      "format": "kubric_spring_mass_control_v1",
      "asset_id": FLAGS.asset_id,
      "run": _get_run_metadata(FLAGS),
      "control_trajectory_file": str(control_trajectory_path),
      "mesh_vertices_file": (
          None if mesh_vertices_path is None else mesh_vertices_path.name),
      "spring_mass_particles_file": (
          None if particles_path is None else particles_path.name),
      "simulation_data_file": (
          None if simulation_data_path is None else simulation_data_path.name),
      "control_vertex_indices": list(control_trajectory.control_vertex_indices),
      "control_particle_indices": (
          simulator.last_control_particle_indices.astype(int).tolist()),
      "num_mesh_vertices": vertex_animation.num_vertices,
      "num_mesh_faces": len(faces),
      "num_particles": len(simulator.last_initial_particles),
      "num_springs": len(simulator.last_edges),
      "cameras": camera_metadata,
      "ground": {"axis": 2, "height": 0.0},
      "config": {
          "particle_spacing": simulator.config.particle_spacing,
          "k_neighbors": simulator.config.k_neighbors,
          "sampled_surface_particle_count": (
              simulator.config.sampled_surface_particle_count),
          "spring_stiffness": simulator.config.spring_stiffness,
          "damping": simulator.config.damping,
          "control_mode": simulator.config.control_mode,
          "control_attachment_k": simulator.config.control_attachment_k,
          "control_attachment_radius": simulator.config.control_attachment_radius,
          "control_stiffness": simulator.config.control_stiffness,
          "control_damping": simulator.config.control_damping,
          "require_watertight": simulator.config.require_watertight,
      },
  }, output_dir / "metadata.json")
