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

"""GPU-capable fixed-topology spring-mass simulation."""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import numpy as np
import pyquaternion as pyquat

from kubric import core

try:
  import torch
except ImportError:  # Keep the rest of Kubric importable without PyTorch.
  torch = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpringMassConfig:
  """Configuration for a non-learnable spring-mass simulation.

  Distances are expressed in Kubric world units and time in seconds.
  `particle_spacing` controls both volumetric filling and surface sampling.
  """

  particle_spacing: float = 0.08
  k_neighbors: int = 16
  spring_stiffness: float = 200.0
  damping: float = 0.5
  total_mass: Optional[float] = None
  initial_velocity: Tuple[float, float, float] = (0., 0., 0.)
  surface_sample_spacing: Optional[float] = None
  sampled_surface_particle_count: Optional[int] = None
  max_particles: int = 20000
  max_fill_candidates: int = 2000000
  weld_tolerance: Optional[float] = None
  point_in_mesh_chunk_size: int = 128
  knn_chunk_size: int = 2048
  ground_axis: Optional[int] = 2
  ground_height: float = 0.0
  restitution: Optional[float] = None
  friction: Optional[float] = None
  control_mode: str = "hard"
  control_attachment_k: int = 12
  control_attachment_radius: Optional[float] = None
  control_stiffness: float = 50.0
  control_damping: float = 0.1
  require_watertight: bool = True
  record_all_particles: bool = False
  seed: int = 0

  def __post_init__(self):
    if self.particle_spacing <= 0:
      raise ValueError("particle_spacing must be positive")
    if self.surface_sample_spacing is not None and self.surface_sample_spacing <= 0:
      raise ValueError("surface_sample_spacing must be positive")
    if (self.sampled_surface_particle_count is not None and
        self.sampled_surface_particle_count <= 1):
      raise ValueError("sampled_surface_particle_count must be greater than one")
    if self.k_neighbors <= 0:
      raise ValueError("k_neighbors must be positive")
    if self.spring_stiffness < 0 or self.damping < 0:
      raise ValueError("spring_stiffness and damping cannot be negative")
    if self.total_mass is not None and self.total_mass <= 0:
      raise ValueError("total_mass must be positive")
    if self.max_particles <= 1:
      raise ValueError("max_particles must be greater than one")
    if (self.sampled_surface_particle_count is not None and
        self.sampled_surface_particle_count > self.max_particles):
      raise ValueError(
          "sampled_surface_particle_count cannot exceed max_particles")
    if self.max_fill_candidates <= 0:
      raise ValueError("max_fill_candidates must be positive")
    if self.weld_tolerance is not None and self.weld_tolerance <= 0:
      raise ValueError("weld_tolerance must be positive")
    if self.point_in_mesh_chunk_size <= 0 or self.knn_chunk_size <= 0:
      raise ValueError("chunk sizes must be positive")
    if self.ground_axis not in (None, 0, 1, 2):
      raise ValueError("ground_axis must be None, 0, 1, or 2")
    if len(self.initial_velocity) != 3:
      raise ValueError("initial_velocity must contain three values")
    for name, value in (("restitution", self.restitution),
                        ("friction", self.friction)):
      if value is not None and not 0. <= value <= 1.:
        raise ValueError(f"{name} must be between zero and one")
    if self.control_mode not in ("hard", "soft"):
      raise ValueError("control_mode must be 'hard' or 'soft'")
    if self.control_attachment_k <= 0:
      raise ValueError("control_attachment_k must be positive")
    if (self.control_attachment_radius is not None and
        self.control_attachment_radius <= 0):
      raise ValueError("control_attachment_radius must be positive")
    if self.control_stiffness < 0 or self.control_damping < 0:
      raise ValueError("control stiffness and damping cannot be negative")


@dataclass(frozen=True)
class RepeatedLiftConfig:
  """Schedule for repeatedly lifting and releasing one surface vertex.

  The control point is the welded simulation particle corresponding to
  `control_vertex_index` in the input render mesh. If `target_height` is not
  specified, it is lifted back to its height at the start of the simulation.
  `target_horizontal` gives the target coordinates on the other two axes.
  """

  control_vertex_index: int
  repeat_count: int = 5
  initial_settle_seconds: float = 2.0
  lift_seconds: float = 1.0
  hold_seconds: float = 0.2
  settle_seconds: float = 2.0
  target_height: Optional[float] = None
  target_horizontal: Tuple[float, float] = (0., 0.)
  vertical_axis: int = 2

  def __post_init__(self):
    if self.control_vertex_index < 0:
      raise ValueError("control_vertex_index cannot be negative")
    if self.repeat_count <= 0:
      raise ValueError("repeat_count must be positive")
    durations = (
        self.initial_settle_seconds, self.lift_seconds, self.hold_seconds,
        self.settle_seconds)
    if not all(np.isfinite(duration) for duration in durations):
      raise ValueError("lift schedule durations must be finite")
    if self.initial_settle_seconds < 0 or self.hold_seconds < 0:
      raise ValueError("initial settle and hold durations cannot be negative")
    if self.lift_seconds <= 0:
      raise ValueError("lift_seconds must be positive")
    if self.settle_seconds < 0:
      raise ValueError("settle_seconds cannot be negative")
    if self.vertical_axis not in (0, 1, 2):
      raise ValueError("vertical_axis must be 0, 1, or 2")
    if self.target_height is not None and not np.isfinite(self.target_height):
      raise ValueError("target_height must be finite")
    if (len(self.target_horizontal) != 2 or
        not all(np.isfinite(value) for value in self.target_horizontal)):
      raise ValueError("target_horizontal must contain two finite values")


@dataclass(frozen=True)
class ControlTrajectoryConfig:
  """World-space kinematic constraints for one or more render vertices.

  `positions` is sampled per frame and has shape `(num_frames, num_points, 3)`.
  A shape of `(num_frames, 3)` is accepted for a single control point. Values
  are linearly interpolated during substeps and clamped to the first/last sample
  outside the provided frame range.
  `active` optionally gates the constraint per frame and control point. Missing
  values keep the legacy behavior where all controls remain active forever.
  """

  control_vertex_indices: Tuple[int, ...]
  positions: np.ndarray
  active: Optional[np.ndarray] = None
  frame_start: int = 0
  frame_rate: Optional[float] = None

  def __post_init__(self):
    indices = tuple(int(index) for index in self.control_vertex_indices)
    if not indices:
      raise ValueError("control_vertex_indices must not be empty")
    if any(index < 0 for index in indices):
      raise ValueError("control_vertex_indices cannot contain negatives")
    if len(set(indices)) != len(indices):
      raise ValueError("control_vertex_indices must be unique")

    positions = np.asarray(self.positions, dtype=np.float32)
    if positions.ndim == 2 and len(indices) == 1 and positions.shape[1] == 3:
      positions = positions[:, None, :]
    if positions.ndim != 3 or positions.shape[1:] != (len(indices), 3):
      raise ValueError(
          "positions must have shape (num_frames, num_points, 3), "
          f"got {positions.shape}")
    if positions.shape[0] == 0:
      raise ValueError("positions must contain at least one frame")
    if not np.all(np.isfinite(positions)):
      raise ValueError("positions must be finite")
    if self.frame_rate is not None and self.frame_rate <= 0:
      raise ValueError("frame_rate must be positive")

    if self.active is None:
      active = np.ones(positions.shape[:2], dtype=np.bool_)
    else:
      active = np.asarray(self.active, dtype=np.bool_)
      if active.ndim == 1:
        active = np.repeat(active[:, None], len(indices), axis=1)
      if active.shape != positions.shape[:2]:
        raise ValueError(
            "active must have shape (num_frames, num_points), "
            f"got {active.shape}")

    positions = np.array(positions, dtype=np.float32, copy=True)
    positions.setflags(write=False)
    active = np.array(active, dtype=np.bool_, copy=True)
    active.setflags(write=False)
    object.__setattr__(self, "control_vertex_indices", indices)
    object.__setattr__(self, "positions", positions)
    object.__setattr__(self, "active", active)
    object.__setattr__(self, "frame_start", int(self.frame_start))


class SpringMassSimulator:
  """Simulates a filled triangle mesh as particles connected by springs.

  Coincident render vertices are welded to shared surface particles. Additional
  particles are sampled on the surface and strictly inside the closed mesh.
  Surface-particle displacement is mapped back to every original render vertex,
  preserving renderer vertex ordering, topology, and UV seams.
  """

  def __init__(
      self,
      scene: core.Scene,
      config: Optional[SpringMassConfig] = None,
      device: str = "cuda",
  ):
    if torch is None:
      raise ImportError(
          "SpringMassSimulator requires PyTorch. Use the CUDA-enabled "
          "Kubric Docker image or install torch separately.")
    self.scene = scene
    self.config = config or SpringMassConfig()
    self.device = torch.device(device)
    if self.device.type == "cuda" and not torch.cuda.is_available():
      raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")

    self.last_initial_particles = None
    self.last_edges = None
    self.last_particle_trajectory = None
    self.last_render_velocity_trajectory = None
    self.last_surface_mapping = None
    self.last_control_particle_index = None
    self.last_control_particle_indices = None
    self.last_control_attachment_indices = None

  def run(
      self,
      asset: core.PhysicalObject,
      vertices: np.ndarray,
      faces: np.ndarray,
      frame_start: Optional[int] = None,
      frame_end: Optional[int] = None,
      repeated_lift: Optional[RepeatedLiftConfig] = None,
      control_trajectory: Optional[ControlTrajectoryConfig] = None,
      initial_particles_world: Optional[np.ndarray] = None,
      surface_mapping: Optional[np.ndarray] = None,
      edges: Optional[np.ndarray] = None,
  ) -> core.VertexAnimation:
    """Runs the simulation and returns world-space render-mesh vertices.

    Args:
      asset: Asset represented by `vertices` and `faces`.
      vertices: Render mesh vertices in the asset's local coordinates.
      faces: Triangular render mesh faces indexing `vertices`.
      frame_start: First recorded frame, inclusive.
      frame_end: Last recorded frame, inclusive.
      repeated_lift: Optional repeated lift-and-release schedule.
      control_trajectory: Optional always-on world-space control trajectory.
      initial_particles_world: Optional replay particle cloud in world space.
      surface_mapping: Optional render-vertex to particle mapping for replay.
      edges: Optional spring edges for replay.
    """
    if not isinstance(asset, core.PhysicalObject):
      raise TypeError("asset must be a PhysicalObject")
    if asset.static:
      raise ValueError("Cannot run spring-mass simulation on a static asset")

    vertices, faces = _validate_mesh(vertices, faces)
    frame_start = self.scene.frame_start if frame_start is None else frame_start
    frame_end = self.scene.frame_end if frame_end is None else frame_end
    if frame_end < frame_start:
      raise ValueError("frame_end must be greater than or equal to frame_start")
    if repeated_lift is not None and control_trajectory is not None:
      raise ValueError("Specify only one of repeated_lift or control_trajectory")

    world_vertices = _to_world_vertices(asset, vertices)
    replay_inputs = (initial_particles_world, surface_mapping, edges)
    preserve_render_vertex_indices = None
    if repeated_lift is not None:
      preserve_render_vertex_indices = [repeated_lift.control_vertex_index]
    elif control_trajectory is not None:
      preserve_render_vertex_indices = list(
          control_trajectory.control_vertex_indices)

    if any(value is not None for value in replay_inputs):
      if not all(value is not None for value in replay_inputs):
        raise ValueError(
            "initial_particles_world, surface_mapping, and edges must be "
            "provided together")
      particles = np.asarray(initial_particles_world, dtype=np.float32)
      surface_mapping = np.asarray(surface_mapping, dtype=np.int64)
      edges = np.asarray(edges, dtype=np.int64)
      if particles.ndim != 2 or particles.shape[1] != 3:
        raise ValueError("initial_particles_world must have shape (N, 3)")
      if surface_mapping.shape != (len(vertices),):
        raise ValueError("surface_mapping must have shape (num_vertices,)")
      if (surface_mapping.min() < 0 or
          surface_mapping.max() >= len(particles)):
        raise ValueError("surface_mapping contains out-of-range particle ids")
      if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
      if edges.min() < 0 or edges.max() >= len(particles):
        raise ValueError("edges contain out-of-range particle ids")
    else:
      particles, surface_mapping = _prepare_particle_cloud(
          world_vertices, faces, self.config, preserve_render_vertex_indices)
    if len(particles) <= self.config.k_neighbors:
      raise ValueError(
          f"Need more than {self.config.k_neighbors} particles, got {len(particles)}")

    positions = torch.as_tensor(
        particles, dtype=torch.float32, device=self.device).clone()
    velocities = torch.zeros_like(positions)
    velocities += torch.as_tensor(
        self.config.initial_velocity, dtype=torch.float32, device=self.device)
    if edges is None:
      edges = _build_knn_edges(
          positions, self.config.k_neighbors, self.config.knn_chunk_size)
    else:
      edges = torch.as_tensor(edges, dtype=torch.long, device=self.device)
    rest_vectors = positions[edges[:, 1]] - positions[edges[:, 0]]
    rest_lengths = torch.linalg.vector_norm(rest_vectors, dim=1)
    valid_edges = rest_lengths > 1e-8
    edges = edges[valid_edges]
    rest_lengths = rest_lengths[valid_edges]
    if len(edges) == 0:
      raise ValueError("KNN graph contains no non-zero-length springs")

    total_mass = asset.mass if self.config.total_mass is None else self.config.total_mass
    if total_mass <= 0:
      raise ValueError("The simulated object must have positive mass")
    particle_mass = float(total_mass) / len(particles)
    gravity = torch.as_tensor(
        self.scene.gravity, dtype=torch.float32, device=self.device)
    steps_per_frame = self.scene.step_rate // self.scene.frame_rate
    dt = 1.0 / self.scene.step_rate
    restitution = (asset.restitution if self.config.restitution is None
                   else self.config.restitution)
    friction = asset.friction if self.config.friction is None else self.config.friction

    lift_controller = None
    soft_controller = None
    self.last_control_particle_index = None
    self.last_control_particle_indices = None
    self.last_control_attachment_indices = None
    if repeated_lift is not None:
      if repeated_lift.control_vertex_index >= len(vertices):
        raise ValueError(
            "control_vertex_index is outside the render mesh vertex range")
      control_particle_index = int(
          surface_mapping[repeated_lift.control_vertex_index])
      lift_controller = _RepeatedLiftController(
          config=repeated_lift,
          particle_index=control_particle_index,
          initial_position=positions[control_particle_index].clone(),
      )
      self.last_control_particle_index = control_particle_index
      self.last_control_particle_indices = np.asarray(
          [control_particle_index], dtype=np.int32)
    elif control_trajectory is not None:
      if max(control_trajectory.control_vertex_indices) >= len(vertices):
        raise ValueError(
            "control_vertex_indices contains an index outside the render mesh "
            "vertex range")
      control_particle_indices = np.asarray([
          int(surface_mapping[vertex_index])
          for vertex_index in control_trajectory.control_vertex_indices
      ], dtype=np.int64)
      if self.config.control_mode == "soft":
        soft_controller = _SoftControlTrajectoryController(
            config=control_trajectory,
            center_particle_indices=control_particle_indices,
            initial_positions=positions,
            device=self.device,
            scene_frame_rate=self.scene.frame_rate,
            simulation_frame_start=frame_start,
            attachment_k=self.config.control_attachment_k,
            attachment_radius=self.config.control_attachment_radius,
            stiffness=self.config.control_stiffness,
            damping=self.config.control_damping,
        )
        self.last_control_attachment_indices = (
            soft_controller.attachment_indices.detach().cpu().numpy().astype(
                np.int32))
      else:
        lift_controller = _ControlTrajectoryController(
            config=control_trajectory,
            particle_indices=control_particle_indices,
            device=self.device,
            scene_frame_rate=self.scene.frame_rate,
            simulation_frame_start=frame_start,
        )
      self.last_control_particle_index = int(control_particle_indices[0])
      self.last_control_particle_indices = control_particle_indices.astype(
          np.int32)

    render_trajectory = []
    render_velocity_trajectory = []
    particle_trajectory = [] if self.config.record_all_particles else None
    surface_mapping_tensor = torch.as_tensor(
        surface_mapping, dtype=torch.long, device=self.device)
    initial_surface_particles = positions[surface_mapping_tensor].clone()
    original_surface_vertices = torch.as_tensor(
        world_vertices, dtype=torch.float32, device=self.device)
    if (lift_controller is not None and
        isinstance(lift_controller, _ControlTrajectoryController)):
      lift_controller.apply_initial(positions, velocities)
    with torch.no_grad():
      for frame in range(frame_start, frame_end + 1):
        surface_displacement = (
            positions[surface_mapping_tensor] - initial_surface_particles)
        render_positions = original_surface_vertices + surface_displacement
        render_velocities = velocities[surface_mapping_tensor]
        render_trajectory.append(render_positions.detach().cpu().numpy().copy())
        render_velocity_trajectory.append(
            render_velocities.detach().cpu().numpy().copy())
        if particle_trajectory is not None:
          particle_trajectory.append(positions.detach().cpu().numpy().copy())
        if frame == frame_end:
          break

        for _ in range(steps_per_frame):
          extra_forces = None
          if soft_controller is not None:
            extra_forces = soft_controller.compute_forces(
                positions, velocities, dt)
          positions, velocities = _integrate_step(
              positions=positions,
              velocities=velocities,
              edges=edges,
              rest_lengths=rest_lengths,
              particle_mass=particle_mass,
              spring_stiffness=self.config.spring_stiffness,
              damping=self.config.damping,
              gravity=gravity,
              dt=dt,
              extra_forces=extra_forces,
          )
          _apply_ground_collision(
              positions, velocities, self.config.ground_axis,
              self.config.ground_height, restitution, friction)
          if lift_controller is not None:
            lift_controller.advance(positions, velocities, dt)
          if soft_controller is not None:
            soft_controller.advance(dt)

        if not torch.isfinite(positions).all():
          raise RuntimeError(
              f"Spring-mass simulation became non-finite at frame {frame + 1}; "
              "reduce stiffness or increase scene.step_rate")

    self.last_initial_particles = particles.copy()
    self.last_edges = edges.detach().cpu().numpy()
    self.last_surface_mapping = surface_mapping.copy()
    self.last_particle_trajectory = (
        None if particle_trajectory is None
        else np.stack(particle_trajectory).astype(np.float32))
    self.last_render_velocity_trajectory = (
        np.stack(render_velocity_trajectory).astype(np.float32))

    logger.info(
        "Simulated %s with %d surface vertices, %d particles, and %d springs on %s",
        asset.uid, len(world_vertices), len(particles), len(edges), self.device)
    return core.VertexAnimation(
        asset=asset,
        frame_start=frame_start,
        vertices=np.stack(render_trajectory).astype(np.float32),
    )


class _RepeatedLiftController:
  """Applies a kinematic constraint to one particle on a timed schedule."""

  def __init__(self, config, particle_index, initial_position):
    self.config = config
    self.particle_index = particle_index
    self.target_height = (
        float(initial_position[config.vertical_axis])
        if config.target_height is None else float(config.target_height))
    self.phase = "initial_settle"
    self.phase_elapsed = 0.0
    self.completed_lifts = 0
    self.lift_start = None
    self.lift_target = None
    self.previous_target = None
    if config.initial_settle_seconds == 0:
      self.phase = "lift"
      self._set_lift_start(initial_position)

  def advance(self, positions, velocities, dt):
    if self.phase == "done":
      return

    if self.phase == "initial_settle":
      self.phase_elapsed += dt
      if self._duration_reached(self.config.initial_settle_seconds):
        self._begin_lift(positions)
      return

    if self.phase == "lift":
      if self.lift_start is None:
        self._capture_lift_start(positions)
      self.phase_elapsed = min(
          self.phase_elapsed + dt, self.config.lift_seconds)
      progress = self.phase_elapsed / self.config.lift_seconds
      smooth_progress = progress * progress * (3.0 - 2.0 * progress)
      target = self.lift_start + (
          self.lift_target - self.lift_start) * smooth_progress
      self._constrain(positions, velocities, target, dt)
      if self._duration_reached(self.config.lift_seconds):
        self.phase = "hold"
        self.phase_elapsed = 0.0
      return

    if self.phase == "hold":
      positions[self.particle_index] = self.previous_target
      velocities[self.particle_index].zero_()
      self.phase_elapsed += dt
      if self._duration_reached(self.config.hold_seconds):
        self.completed_lifts += 1
        self.phase = "settle"
        self.phase_elapsed = 0.0
      return

    if self.phase == "settle":
      self.phase_elapsed += dt
      if self._duration_reached(self.config.settle_seconds):
        if self.completed_lifts >= self.config.repeat_count:
          self.phase = "done"
        else:
          self._begin_lift(positions)

  def _begin_lift(self, positions):
    self.phase = "lift"
    self.phase_elapsed = 0.0
    self._capture_lift_start(positions)

  def _capture_lift_start(self, positions):
    self._set_lift_start(positions[self.particle_index])

  def _set_lift_start(self, position):
    self.lift_start = position.clone()
    self.lift_target = self.lift_start.clone()
    vertical_axis = self.config.vertical_axis
    self.lift_target[vertical_axis] = self.target_height
    horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
    for axis, value in zip(horizontal_axes, self.config.target_horizontal):
      self.lift_target[axis] = value
    self.previous_target = self.lift_start.clone()

  def _constrain(self, positions, velocities, target, dt):
    velocities[self.particle_index] = (target - self.previous_target) / dt
    positions[self.particle_index] = target
    self.previous_target = target.clone()

  def _duration_reached(self, duration):
    return self.phase_elapsed >= duration - 1e-12


class _ControlTrajectoryController:
  """Applies always-on kinematic constraints from sampled world positions."""

  def __init__(
      self,
      config,
      particle_indices,
      device,
      scene_frame_rate,
      simulation_frame_start,
  ):
    self.config = config
    self.particle_indices = torch.as_tensor(
        particle_indices, dtype=torch.long, device=device)
    self.positions = torch.as_tensor(
        config.positions, dtype=torch.float32, device=device)
    self.active = torch.as_tensor(
        config.active, dtype=torch.bool, device=device)
    self.frame_rate = (
        float(scene_frame_rate) if config.frame_rate is None
        else float(config.frame_rate))
    self.time_seconds = float(simulation_frame_start) / float(scene_frame_rate)
    self.previous_target = None

  def apply_initial(self, positions, velocities):
    active = self._active_at_current_time()
    if not torch.any(active):
      self.previous_target = None
      return
    target = self._target_at_current_time()
    positions[self.particle_indices[active]] = target[active]
    velocities[self.particle_indices[active]].zero_()
    self.previous_target = target.clone()

  def advance(self, positions, velocities, dt):
    self.time_seconds += dt
    active = self._active_at_current_time()
    if not torch.any(active):
      self.previous_target = None
      return
    target = self._target_at_current_time()
    if self.previous_target is None:
      velocities[self.particle_indices[active]].zero_()
    else:
      velocities[self.particle_indices[active]] = (
          target[active] - self.previous_target[active]) / dt
    positions[self.particle_indices[active]] = target[active]
    self.previous_target = target.clone()

  def _sample_indices_and_weight(self):
    sample_position = (
        self.time_seconds * self.frame_rate - float(self.config.frame_start))
    max_index = self.positions.shape[0] - 1
    clamped = min(max(sample_position, 0.0), float(max_index))
    lower_index = int(np.floor(clamped))
    upper_index = min(lower_index + 1, max_index)
    return lower_index, upper_index, clamped - lower_index

  def _active_at_current_time(self):
    lower_index, _, _ = self._sample_indices_and_weight()
    return self.active[lower_index]

  def _target_at_current_time(self):
    lower_index, upper_index, weight = self._sample_indices_and_weight()
    if upper_index == lower_index:
      return self.positions[lower_index]
    return (
        self.positions[lower_index] * (1.0 - weight) +
        self.positions[upper_index] * weight)


class _SoftControlTrajectoryController:
  """Pulls local particle patches toward moving virtual control anchors."""

  def __init__(
      self,
      config,
      center_particle_indices,
      initial_positions,
      device,
      scene_frame_rate,
      simulation_frame_start,
      attachment_k,
      attachment_radius,
      stiffness,
      damping,
  ):
    self.config = config
    self.positions = torch.as_tensor(
        config.positions, dtype=torch.float32, device=device)
    self.active = torch.as_tensor(
        config.active, dtype=torch.bool, device=device)
    self.frame_rate = (
        float(scene_frame_rate) if config.frame_rate is None
        else float(config.frame_rate))
    self.time_seconds = float(simulation_frame_start) / float(scene_frame_rate)
    self.stiffness = float(stiffness)
    self.damping = float(damping)
    self.previous_targets = self._target_at_current_time().clone()

    center_particle_indices = torch.as_tensor(
        center_particle_indices, dtype=torch.long, device=device)
    attachment_rows = []
    offset_rows = []
    initial_anchor_positions = initial_positions[center_particle_indices]
    for center_particle_index, anchor_position in zip(
        center_particle_indices.tolist(), initial_anchor_positions):
      distances = torch.linalg.vector_norm(
          initial_positions - initial_positions[center_particle_index], dim=1)
      order = torch.argsort(distances)
      if attachment_radius is not None:
        within_radius = order[distances[order] <= float(attachment_radius)]
        if len(within_radius) >= attachment_k:
          order = within_radius
      selected = order[:attachment_k]
      if not torch.any(selected == center_particle_index):
        selected = torch.cat([
            torch.as_tensor([center_particle_index], dtype=torch.long, device=device),
            selected[:-1],
        ])
      attachment_rows.append(selected)
      offset_rows.append(initial_positions[selected] - anchor_position)

    self.attachment_indices = torch.stack(attachment_rows, dim=0)
    self.initial_offsets = torch.stack(offset_rows, dim=0)

  def compute_forces(self, positions, velocities, dt):
    active = self._active_at_current_time()
    if not torch.any(active):
      self.previous_targets = self._target_at_current_time().clone()
      return torch.zeros_like(positions)
    target = self._target_at_current_time()
    target_velocity = (target - self.previous_targets) / dt
    particle_targets = target[:, None, :] + self.initial_offsets
    attached_positions = positions[self.attachment_indices]
    attached_velocities = velocities[self.attachment_indices]
    attached_target_velocities = target_velocity[:, None, :].expand_as(
        attached_velocities)
    attachment_forces = (
        self.stiffness * (particle_targets - attached_positions) +
        self.damping * (attached_target_velocities - attached_velocities))
    attachment_forces = attachment_forces * active[:, None, None]
    forces = torch.zeros_like(positions)
    forces.index_add_(
        0,
        self.attachment_indices.reshape(-1),
        attachment_forces.reshape(-1, 3))
    self.previous_targets = target.clone()
    return forces

  def advance(self, dt):
    self.time_seconds += dt

  def _target_at_current_time(self):
    lower_index, upper_index, weight = self._sample_indices_and_weight()
    if upper_index == lower_index:
      return self.positions[lower_index]
    return (
        self.positions[lower_index] * (1.0 - weight) +
        self.positions[upper_index] * weight)

  def _active_at_current_time(self):
    lower_index, _, _ = self._sample_indices_and_weight()
    return self.active[lower_index]

  def _sample_indices_and_weight(self):
    sample_position = (
        self.time_seconds * self.frame_rate - float(self.config.frame_start))
    max_index = self.positions.shape[0] - 1
    clamped = min(max(sample_position, 0.0), float(max_index))
    lower_index = int(np.floor(clamped))
    upper_index = min(lower_index + 1, max_index)
    return lower_index, upper_index, clamped - lower_index


def fill_mesh_with_particles(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: SpringMassConfig,
) -> np.ndarray:
  """Returns welded surface vertices followed by sampled surface/interior points."""
  particles, _ = _prepare_particle_cloud(vertices, faces, config)
  return particles


def _prepare_particle_cloud(vertices, faces, config, preserve_render_vertex_indices=None):
  vertices, faces = _validate_mesh(vertices, faces)
  weld_tolerance = config.weld_tolerance or config.particle_spacing * 1e-4
  volume_vertices, volume_faces, surface_mapping = _weld_mesh(
      vertices, faces, weld_tolerance)
  if config.require_watertight and not _is_watertight(volume_faces):
    raise ValueError("Cannot fill a non-watertight mesh")

  rng = np.random.default_rng(config.seed)
  if config.sampled_surface_particle_count is not None:
    preserve_welded_indices = None
    if preserve_render_vertex_indices is not None:
      preserve_render_vertex_indices = np.asarray(
          preserve_render_vertex_indices, dtype=np.int64)
      if (preserve_render_vertex_indices.size and
          (preserve_render_vertex_indices.min() < 0 or
           preserve_render_vertex_indices.max() >= len(vertices))):
        raise ValueError("preserve_render_vertex_indices contains out-of-range ids")
      preserve_welded_indices = surface_mapping[preserve_render_vertex_indices]
    selected_indices = _sample_surface_vertices_farthest(
        volume_vertices,
        config.sampled_surface_particle_count,
        rng,
        preserve_welded_indices)
    particles = volume_vertices[selected_indices].astype(np.float32)
    sampled_surface_mapping = _nearest_point_indices_chunked(
        volume_vertices[surface_mapping],
        particles,
        config.knn_chunk_size).astype(np.int64)
    if len(particles) > config.max_particles:
      raise ValueError(
          f"Sampled surface generated {len(particles)} particles, exceeding "
          f"max_particles={config.max_particles}")
    return particles, sampled_surface_mapping

  surface_spacing = config.surface_sample_spacing or config.particle_spacing
  available_samples = config.max_particles - len(volume_vertices)
  if available_samples <= 0:
    raise ValueError(
        f"Welded mesh already has {len(volume_vertices)} vertices, exceeding "
        f"max_particles={config.max_particles}")
  surface_samples = _sample_mesh_surface(
      volume_vertices, volume_faces, surface_spacing, rng, available_samples)

  lower = volume_vertices.min(axis=0)
  upper = volume_vertices.max(axis=0)
  axes = [np.arange(lo + config.particle_spacing * 0.5, hi,
                    config.particle_spacing, dtype=np.float64)
          for lo, hi in zip(lower, upper)]
  if any(len(axis) == 0 for axis in axes):
    interior_samples = np.empty((0, 3), dtype=np.float64)
  else:
    candidate_count = int(np.prod([len(axis) for axis in axes]))
    if candidate_count > config.max_fill_candidates:
      raise ValueError(
          f"Volume grid requires {candidate_count} candidates, exceeding "
          f"max_fill_candidates={config.max_fill_candidates}; "
          "increase particle_spacing")
    candidates = np.stack(
        np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    inside = _points_inside_mesh(
        candidates, volume_vertices, volume_faces,
        config.point_in_mesh_chunk_size)
    interior_samples = candidates[inside]

  particles = _append_unique_points(
      volume_vertices,
      np.concatenate([surface_samples, interior_samples], axis=0),
      tolerance=config.particle_spacing * 0.1)
  if len(particles) > config.max_particles:
    raise ValueError(
        f"Mesh filling generated {len(particles)} particles, exceeding "
        f"max_particles={config.max_particles}; increase particle_spacing")
  return particles.astype(np.float32), surface_mapping


def _sample_surface_vertices_farthest(points, sample_count, rng, preserve_indices=None):
  points = np.asarray(points, dtype=np.float64)
  sample_count = min(int(sample_count), len(points))
  if preserve_indices is None:
    selected = []
  else:
    selected = []
    for index in np.asarray(preserve_indices, dtype=np.int64).tolist():
      if index not in selected:
        selected.append(index)
  if len(selected) > sample_count:
    raise ValueError(
        "sampled_surface_particle_count is smaller than the number of "
        "preserved control particles")

  if not selected:
    center = np.mean(points, axis=0)
    selected.append(int(np.argmin(np.sum((points - center[None, :]) ** 2, axis=1))))

  min_distances = np.full(len(points), np.inf, dtype=np.float64)
  for index in selected:
    distances = np.sum((points - points[index][None, :]) ** 2, axis=1)
    min_distances = np.minimum(min_distances, distances)
  min_distances[selected] = -np.inf

  while len(selected) < sample_count:
    next_index = int(np.argmax(min_distances))
    selected.append(next_index)
    distances = np.sum((points - points[next_index][None, :]) ** 2, axis=1)
    min_distances = np.minimum(min_distances, distances)
    min_distances[selected] = -np.inf
  return np.asarray(selected, dtype=np.int64)


def _nearest_point_indices_chunked(query_points, reference_points, chunk_size):
  query_points = np.asarray(query_points, dtype=np.float32)
  reference_points = np.asarray(reference_points, dtype=np.float32)
  nearest = np.empty(len(query_points), dtype=np.int64)
  for start in range(0, len(query_points), chunk_size):
    end = min(start + chunk_size, len(query_points))
    delta = query_points[start:end, None, :] - reference_points[None, :, :]
    distances = np.sum(delta * delta, axis=2)
    nearest[start:end] = np.argmin(distances, axis=1)
  return nearest


def _validate_mesh(vertices, faces):
  vertices = np.asarray(vertices, dtype=np.float64)
  faces = np.asarray(faces, dtype=np.int64)
  if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
    raise ValueError("vertices must have shape (N, 3) with N >= 4")
  if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
    raise ValueError("faces must have non-empty shape (F, 3)")
  if not np.all(np.isfinite(vertices)):
    raise ValueError("vertices must be finite")
  if faces.min() < 0 or faces.max() >= len(vertices):
    raise ValueError("faces contain out-of-range vertex indices")
  return vertices, faces


def _is_watertight(faces):
  edges = np.concatenate([
      faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
  edges.sort(axis=1)
  _, counts = np.unique(edges, axis=0, return_counts=True)
  return bool(len(counts) and np.all(counts == 2))


def _weld_mesh(vertices, faces, tolerance):
  """Creates a topology-only mesh with coincident vertices merged."""
  keys = np.round(vertices / tolerance).astype(np.int64)
  _, representative_indices, sorted_inverse = np.unique(
      keys, axis=0, return_index=True, return_inverse=True)
  preserve_order = np.argsort(representative_indices)
  sorted_to_preserved = np.empty(len(preserve_order), dtype=np.int64)
  sorted_to_preserved[preserve_order] = np.arange(len(preserve_order))
  representative_indices = representative_indices[preserve_order]
  inverse = sorted_to_preserved[sorted_inverse]
  welded_vertices = vertices[representative_indices]
  welded_faces = inverse[faces]

  nondegenerate = ((welded_faces[:, 0] != welded_faces[:, 1]) &
                   (welded_faces[:, 1] != welded_faces[:, 2]) &
                   (welded_faces[:, 2] != welded_faces[:, 0]))
  welded_faces = welded_faces[nondegenerate]
  canonical_faces = np.sort(welded_faces, axis=1)
  _, unique_indices = np.unique(canonical_faces, axis=0, return_index=True)
  welded_faces = welded_faces[np.sort(unique_indices)]
  if len(welded_faces) == 0:
    raise ValueError("mesh contains no faces after welding")
  return welded_vertices, welded_faces, inverse


def _sample_mesh_surface(vertices, faces, spacing, rng, max_samples):
  triangles = vertices[faces]
  cross = np.cross(triangles[:, 1] - triangles[:, 0],
                   triangles[:, 2] - triangles[:, 0])
  areas = np.linalg.norm(cross, axis=1) * 0.5
  total_area = areas.sum()
  if total_area <= 0:
    raise ValueError("mesh surface area must be positive")
  sample_count = max(1, int(np.ceil(total_area / (spacing * spacing))))
  if sample_count > max_samples:
    raise ValueError(
        f"Surface sampling requires {sample_count} particles but only "
        f"{max_samples} fit within max_particles; increase surface_sample_spacing")
  face_ids = rng.choice(len(faces), size=sample_count, p=areas / total_area)
  selected = triangles[face_ids]
  uv = rng.random((sample_count, 2))
  reflect = uv.sum(axis=1) > 1.
  uv[reflect] = 1. - uv[reflect]
  return (selected[:, 0] +
          uv[:, :1] * (selected[:, 1] - selected[:, 0]) +
          uv[:, 1:] * (selected[:, 2] - selected[:, 0]))


def _points_inside_mesh(points, vertices, faces, chunk_size):
  """Classifies points with odd-even ray casting using a non-axis ray."""
  if len(points) == 0:
    return np.zeros(0, dtype=bool)
  triangles = vertices[faces]
  v0 = triangles[:, 0]
  edge1 = triangles[:, 1] - v0
  edge2 = triangles[:, 2] - v0
  direction = np.array([1., 0.37139068, 0.52912831], dtype=np.float64)
  direction /= np.linalg.norm(direction)
  h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
  determinant = np.einsum("fi,fi->f", edge1, h)
  valid_faces = np.abs(determinant) > 1e-12
  inverse_determinant = np.zeros_like(determinant)
  inverse_determinant[valid_faces] = 1. / determinant[valid_faces]

  inside = np.empty(len(points), dtype=bool)
  for start in range(0, len(points), chunk_size):
    chunk = points[start:start + chunk_size]
    s = chunk[:, None, :] - v0[None, :, :]
    u = np.einsum("pfi,fi->pf", s, h) * inverse_determinant[None, :]
    q = np.cross(s, edge1[None, :, :])
    v = np.einsum("i,pfi->pf", direction, q) * inverse_determinant[None, :]
    distance = np.einsum("fi,pfi->pf", edge2, q) * inverse_determinant[None, :]
    intersections = (valid_faces[None, :] &
                     (u >= -1e-10) & (v >= -1e-10) &
                     (u + v <= 1. + 1e-10) & (distance > 1e-10))
    inside[start:start + len(chunk)] = intersections.sum(axis=1) % 2 == 1
  return inside


def _append_unique_points(base, additions, tolerance):
  result = [point.copy() for point in base]
  occupied = {tuple(np.round(point / tolerance).astype(np.int64))
              for point in base}
  for point in additions:
    key = tuple(np.round(point / tolerance).astype(np.int64))
    if key not in occupied:
      occupied.add(key)
      result.append(point.copy())
  return np.asarray(result, dtype=np.float64)


def _to_world_vertices(asset, vertices):
  scaled = vertices * np.asarray(asset.scale, dtype=np.float64)
  rotation = pyquat.Quaternion(*asset.quaternion).rotation_matrix
  return scaled @ rotation.T + np.asarray(asset.position, dtype=np.float64)


def _build_knn_edges(points, k_neighbors, chunk_size):
  neighbor_chunks = []
  for start in range(0, len(points), chunk_size):
    end = min(start + chunk_size, len(points))
    distances = torch.cdist(points[start:end], points)
    local_rows = torch.arange(end - start, device=points.device)
    distances[local_rows, torch.arange(start, end, device=points.device)] = torch.inf
    neighbor_chunks.append(torch.topk(
        distances, k=k_neighbors, dim=1, largest=False, sorted=True).indices)
  neighbors = torch.cat(neighbor_chunks, dim=0)
  source = torch.arange(len(points), device=points.device).unsqueeze(1)
  source = source.expand_as(neighbors).reshape(-1)
  target = neighbors.reshape(-1)
  edges = torch.stack([torch.minimum(source, target),
                       torch.maximum(source, target)], dim=1)
  return torch.unique(edges, dim=0)


def _integrate_step(
    positions,
    velocities,
    edges,
    rest_lengths,
    particle_mass,
    spring_stiffness,
    damping,
    gravity,
    dt,
    extra_forces=None,
):
  source, target = edges[:, 0], edges[:, 1]
  delta = positions[target] - positions[source]
  lengths = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-8)
  directions = delta / lengths.unsqueeze(1)
  relative_velocity = velocities[target] - velocities[source]
  axial_velocity = (relative_velocity * directions).sum(dim=1)
  magnitudes = (spring_stiffness * (lengths - rest_lengths) +
                damping * axial_velocity)
  edge_forces = magnitudes.unsqueeze(1) * directions

  forces = gravity.unsqueeze(0).expand_as(positions) * particle_mass
  forces = forces.clone()
  if extra_forces is not None:
    forces = forces + extra_forces
  forces.index_add_(0, source, edge_forces)
  forces.index_add_(0, target, -edge_forces)
  velocities = velocities + forces * (dt / particle_mass)
  positions = positions + velocities * dt
  return positions, velocities


def _apply_ground_collision(
    positions, velocities, axis, height, restitution, friction):
  if axis is None:
    return
  collided = positions[:, axis] < height
  if not torch.any(collided):
    return
  positions[collided, axis] = height
  normal_velocity = velocities[collided, axis]
  velocities[collided, axis] = torch.where(
      normal_velocity < 0., -restitution * normal_velocity, normal_velocity)
  tangent_axes = [candidate for candidate in range(3) if candidate != axis]
  for tangent_axis in tangent_axes:
    velocities[collided, tangent_axis] *= max(0., 1. - friction)
