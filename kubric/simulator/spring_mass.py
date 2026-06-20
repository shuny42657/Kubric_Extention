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
  max_particles: int = 20000
  max_fill_candidates: int = 2000000
  weld_tolerance: Optional[float] = None
  point_in_mesh_chunk_size: int = 128
  knn_chunk_size: int = 2048
  ground_axis: Optional[int] = 2
  ground_height: float = 0.0
  restitution: Optional[float] = None
  friction: Optional[float] = None
  require_watertight: bool = True
  record_all_particles: bool = False
  seed: int = 0

  def __post_init__(self):
    if self.particle_spacing <= 0:
      raise ValueError("particle_spacing must be positive")
    if self.surface_sample_spacing is not None and self.surface_sample_spacing <= 0:
      raise ValueError("surface_sample_spacing must be positive")
    if self.k_neighbors <= 0:
      raise ValueError("k_neighbors must be positive")
    if self.spring_stiffness < 0 or self.damping < 0:
      raise ValueError("spring_stiffness and damping cannot be negative")
    if self.total_mass is not None and self.total_mass <= 0:
      raise ValueError("total_mass must be positive")
    if self.max_particles <= 1:
      raise ValueError("max_particles must be greater than one")
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

  def run(
      self,
      asset: core.PhysicalObject,
      vertices: np.ndarray,
      faces: np.ndarray,
      frame_start: Optional[int] = None,
      frame_end: Optional[int] = None,
  ) -> core.VertexAnimation:
    """Runs the simulation and returns world-space render-mesh vertices.

    Args:
      asset: Asset represented by `vertices` and `faces`.
      vertices: Render mesh vertices in the asset's local coordinates.
      faces: Triangular render mesh faces indexing `vertices`.
      frame_start: First recorded frame, inclusive.
      frame_end: Last recorded frame, inclusive.
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

    world_vertices = _to_world_vertices(asset, vertices)
    particles, surface_mapping = _prepare_particle_cloud(
        world_vertices, faces, self.config)
    if len(particles) <= self.config.k_neighbors:
      raise ValueError(
          f"Need more than {self.config.k_neighbors} particles, got {len(particles)}")

    positions = torch.as_tensor(
        particles, dtype=torch.float32, device=self.device).clone()
    velocities = torch.zeros_like(positions)
    velocities += torch.as_tensor(
        self.config.initial_velocity, dtype=torch.float32, device=self.device)
    edges = _build_knn_edges(
        positions, self.config.k_neighbors, self.config.knn_chunk_size)
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

    render_trajectory = []
    particle_trajectory = [] if self.config.record_all_particles else None
    surface_mapping_tensor = torch.as_tensor(
        surface_mapping, dtype=torch.long, device=self.device)
    initial_surface_particles = positions[surface_mapping_tensor].clone()
    original_surface_vertices = torch.as_tensor(
        world_vertices, dtype=torch.float32, device=self.device)
    with torch.no_grad():
      for frame in range(frame_start, frame_end + 1):
        surface_displacement = (
            positions[surface_mapping_tensor] - initial_surface_particles)
        render_positions = original_surface_vertices + surface_displacement
        render_trajectory.append(render_positions.detach().cpu().numpy().copy())
        if particle_trajectory is not None:
          particle_trajectory.append(positions.detach().cpu().numpy().copy())
        if frame == frame_end:
          break

        for _ in range(steps_per_frame):
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
          )
          _apply_ground_collision(
              positions, velocities, self.config.ground_axis,
              self.config.ground_height, restitution, friction)

        if not torch.isfinite(positions).all():
          raise RuntimeError(
              f"Spring-mass simulation became non-finite at frame {frame + 1}; "
              "reduce stiffness or increase scene.step_rate")

    self.last_initial_particles = particles.copy()
    self.last_edges = edges.detach().cpu().numpy()
    self.last_particle_trajectory = (
        None if particle_trajectory is None
        else np.stack(particle_trajectory).astype(np.float32))

    logger.info(
        "Simulated %s with %d surface vertices, %d particles, and %d springs on %s",
        asset.uid, len(world_vertices), len(particles), len(edges), self.device)
    return core.VertexAnimation(
        asset=asset,
        frame_start=frame_start,
        vertices=np.stack(render_trajectory).astype(np.float32),
    )


def fill_mesh_with_particles(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: SpringMassConfig,
) -> np.ndarray:
  """Returns welded surface vertices followed by sampled surface/interior points."""
  particles, _ = _prepare_particle_cloud(vertices, faces, config)
  return particles


def _prepare_particle_cloud(vertices, faces, config):
  vertices, faces = _validate_mesh(vertices, faces)
  weld_tolerance = config.weld_tolerance or config.particle_spacing * 1e-4
  volume_vertices, volume_faces, surface_mapping = _weld_mesh(
      vertices, faces, weld_tolerance)
  if config.require_watertight and not _is_watertight(volume_faces):
    raise ValueError("Cannot fill a non-watertight mesh")

  rng = np.random.default_rng(config.seed)
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
