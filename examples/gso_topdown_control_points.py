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

"""Computes top-down control point candidates for a non-planar GSO object."""

import sys
import shlex

import numpy as np

import imageio
import kubric as kb
from kubric.renderer import Blender


def _get_run_metadata(flags):
  argv = list(sys.argv)
  return {
      "argv": argv,
      "command": " ".join(shlex.quote(arg) for arg in argv),
      "flags": dict(vars(flags)),
  }


def _to_world_vertices(asset, vertices):
  scaled = vertices * np.asarray(asset.scale, dtype=np.float64)
  rotation = np.asarray(asset.matrix_world, dtype=np.float64)[:3, :3]
  return scaled @ rotation.T + np.asarray(asset.position, dtype=np.float64)


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


def _top_surface_intersections(world_vertices, faces, grid_resolution, chunk_size):
  lower = np.min(world_vertices[:, :2], axis=0)
  upper = np.max(world_vertices[:, :2], axis=0)
  xs = np.linspace(lower[0], upper[0], grid_resolution)
  ys = np.linspace(lower[1], upper[1], grid_resolution)
  query_xy = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape((-1, 2))

  triangles = world_vertices[faces]
  x0, y0, z0 = triangles[:, 0, 0], triangles[:, 0, 1], triangles[:, 0, 2]
  x1, y1, z1 = triangles[:, 1, 0], triangles[:, 1, 1], triangles[:, 1, 2]
  x2, y2, z2 = triangles[:, 2, 0], triangles[:, 2, 1], triangles[:, 2, 2]
  denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
  valid_triangles = np.abs(denominator) > 1e-12

  hit_positions = []
  hit_face_indices = []
  hit_barycentric = []
  hit_query_xy = []
  face_ids = np.arange(len(faces))
  for start in range(0, len(query_xy), chunk_size):
    chunk = query_xy[start:start + chunk_size]
    qx = chunk[:, 0:1]
    qy = chunk[:, 1:2]
    inv_den = np.zeros_like(denominator)
    inv_den[valid_triangles] = 1.0 / denominator[valid_triangles]

    bary0 = ((y1 - y2)[None, :] * (qx - x2[None, :]) +
             (x2 - x1)[None, :] * (qy - y2[None, :])) * inv_den[None, :]
    bary1 = ((y2 - y0)[None, :] * (qx - x2[None, :]) +
             (x0 - x2)[None, :] * (qy - y2[None, :])) * inv_den[None, :]
    bary2 = 1.0 - bary0 - bary1
    inside = (
        valid_triangles[None, :] &
        (bary0 >= -1e-10) &
        (bary1 >= -1e-10) &
        (bary2 >= -1e-10))
    z = (
        bary0 * z0[None, :] +
        bary1 * z1[None, :] +
        bary2 * z2[None, :])
    z = np.where(inside, z, -np.inf)
    best_local_faces = np.argmax(z, axis=1)
    best_z = z[np.arange(len(chunk)), best_local_faces]
    hits = np.isfinite(best_z)
    if not np.any(hits):
      continue
    selected_faces = best_local_faces[hits]
    selected_bary = np.stack([
        bary0[hits, selected_faces],
        bary1[hits, selected_faces],
        bary2[hits, selected_faces],
    ], axis=1)
    selected_xy = chunk[hits]
    hit_query_xy.append(selected_xy)
    hit_face_indices.append(face_ids[selected_faces])
    hit_barycentric.append(selected_bary)
    hit_positions.append(np.concatenate([
        selected_xy,
        best_z[hits, None],
    ], axis=1))

  if not hit_positions:
    raise RuntimeError("No top-down ray intersections found")
  return (
      np.concatenate(hit_positions, axis=0),
      np.concatenate(hit_query_xy, axis=0),
      np.concatenate(hit_face_indices, axis=0),
      np.concatenate(hit_barycentric, axis=0),
  )


def _farthest_point_sample_xy(points_xy, sample_count):
  if sample_count <= 0:
    raise ValueError("num_control_points must be positive")
  if len(points_xy) < sample_count:
    raise ValueError(
        f"Need at least {sample_count} top-down candidates, got {len(points_xy)}")
  center = np.mean(points_xy, axis=0)
  first_index = int(np.argmin(np.sum((points_xy - center[None, :]) ** 2, axis=1)))
  selected = [first_index]
  min_distances = np.sum((points_xy - points_xy[first_index][None, :]) ** 2, axis=1)
  for _ in range(1, sample_count):
    next_index = int(np.argmax(min_distances))
    selected.append(next_index)
    distances = np.sum((points_xy - points_xy[next_index][None, :]) ** 2, axis=1)
    min_distances = np.minimum(min_distances, distances)
  return np.asarray(selected, dtype=np.int32)


def _nearest_vertices(world_vertices, selected_positions):
  nearest = []
  for position in selected_positions:
    distances = np.sum((world_vertices - position[None, :]) ** 2, axis=1)
    nearest.append(int(np.argmin(distances)))
  return np.asarray(nearest, dtype=np.int32)


_DIGITS_3X5 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}


def _draw_disk(image, center, radius, color):
  height, width = image.shape[:2]
  cx, cy = center
  x_min = max(0, int(np.floor(cx - radius)))
  x_max = min(width - 1, int(np.ceil(cx + radius)))
  y_min = max(0, int(np.floor(cy - radius)))
  y_max = min(height - 1, int(np.ceil(cy + radius)))
  for y in range(y_min, y_max + 1):
    for x in range(x_min, x_max + 1):
      if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
        image[y, x] = color


def _draw_digit_text(image, text, origin, color, scale=2):
  x0, y0 = origin
  cursor_x = int(x0)
  for char in str(text):
    pattern = _DIGITS_3X5.get(char)
    if pattern is None:
      cursor_x += 4 * scale
      continue
    for row, line in enumerate(pattern):
      for col, value in enumerate(line):
        if value == "1":
          y_start = int(y0) + row * scale
          x_start = cursor_x + col * scale
          image[y_start:y_start + scale, x_start:x_start + scale] = color
    cursor_x += 4 * scale


def _xy_to_pixels(xy, lower, upper, image_size, margin):
  usable = image_size - 2 * margin
  span = np.maximum(upper - lower, 1e-12)
  normalized = (xy - lower[None, :]) / span[None, :]
  pixels = np.empty_like(normalized)
  pixels[:, 0] = margin + normalized[:, 0] * usable
  pixels[:, 1] = image_size - margin - normalized[:, 1] * usable
  return pixels


def _write_topdown_visualization(
    path,
    candidate_xy,
    selected_xy,
    image_size,
    margin,
):
  lower = np.min(candidate_xy, axis=0)
  upper = np.max(candidate_xy, axis=0)
  image = np.full((image_size, image_size, 3), 255, dtype=np.uint8)

  candidate_pixels = _xy_to_pixels(candidate_xy, lower, upper, image_size, margin)
  for center in candidate_pixels:
    _draw_disk(image, center, radius=1.0, color=np.asarray([210, 210, 210]))

  selected_pixels = _xy_to_pixels(selected_xy, lower, upper, image_size, margin)
  for point_index, center in enumerate(selected_pixels):
    _draw_disk(image, center, radius=5.0, color=np.asarray([230, 40, 40]))
    _draw_disk(image, center, radius=2.0, color=np.asarray([255, 255, 255]))
    label_origin = (
        int(np.clip(center[0] + 7, 0, image_size - 16)),
        int(np.clip(center[1] - 7, 0, image_size - 12)),
    )
    _draw_digit_text(
        image,
        str(point_index),
        label_origin,
        color=np.asarray([20, 20, 20]),
        scale=2,
    )
  imageio.imwrite(str(path), image)


parser = kb.ArgumentParser()
parser.add_argument(
    "--gso_assets",
    type=str,
    default="gs://kubric-public/assets/GSO/GSO.json",
)
parser.add_argument(
    "--asset_id",
    type=str,
    default="Lovable_Huggable_Cuddly_Boutique_Teddy_Bear_Beige",
)
parser.add_argument("--num_control_points", type=int, default=16)
parser.add_argument("--grid_resolution", type=int, default=160)
parser.add_argument("--ray_chunk_size", type=int, default=512)
parser.add_argument("--output_name", type=str, default="topdown_control_points")
parser.add_argument("--visualization_size", type=int, default=1024)
parser.add_argument("--visualization_margin", type=int, default=32)
parser.set_defaults(
    frame_start=0,
    frame_end=1,
    frame_rate=24,
    step_rate=240,
    resolution="256x256",
    seed=42,
)
FLAGS = parser.parse_args()

if FLAGS.grid_resolution < 2:
  raise ValueError("grid_resolution must be at least 2")

scene, _, output_dir, scratch_dir = kb.setup(FLAGS)
renderer = Blender(scene, scratch_dir, samples_per_pixel=1, use_denoising=False)

with kb.AssetSource.from_manifest(FLAGS.gso_assets, scratch_dir) as gso:
  if FLAGS.asset_id not in gso._assets:  # pylint: disable=protected-access
    raise ValueError(f"Unknown GSO asset ID: {FLAGS.asset_id!r}")

  obj = gso.create(asset_id=FLAGS.asset_id)
  bounds = np.asarray(obj.bounds, dtype=np.float64)
  scale = 1.0 / np.max(bounds[1] - bounds[0])
  obj.scale = (scale, scale, scale)
  _place_object_on_ground(obj, bounds, scale)
  scene.add(obj)

  vertices, faces = renderer.get_mesh_geometry(obj)
  initial_min_z = _ensure_mesh_above_ground(obj, vertices, ground_height=0.0)
  world_vertices = _to_world_vertices(obj, vertices)

  candidate_positions, candidate_xy, candidate_faces, candidate_barycentric = (
      _top_surface_intersections(
          world_vertices=world_vertices,
          faces=faces,
          grid_resolution=FLAGS.grid_resolution,
          chunk_size=FLAGS.ray_chunk_size))
  selected_candidate_indices = _farthest_point_sample_xy(
      candidate_xy, FLAGS.num_control_points)
  selected_positions = candidate_positions[selected_candidate_indices]
  selected_xy = candidate_xy[selected_candidate_indices]
  selected_faces = candidate_faces[selected_candidate_indices]
  selected_barycentric = candidate_barycentric[selected_candidate_indices]
  nearest_vertex_indices = _nearest_vertices(world_vertices, selected_positions)
  nearest_vertex_positions = world_vertices[nearest_vertex_indices]

  npz_path = output_dir / f"{FLAGS.output_name}.npz"
  visualization_path = output_dir / f"{FLAGS.output_name}.png"
  _write_topdown_visualization(
      path=visualization_path,
      candidate_xy=candidate_xy,
      selected_xy=selected_xy,
      image_size=FLAGS.visualization_size,
      margin=FLAGS.visualization_margin,
  )
  np.savez_compressed(
      str(npz_path),
      asset_id=np.asarray(FLAGS.asset_id),
      world_vertices=world_vertices.astype(np.float32),
      faces=faces.astype(np.int32),
      candidate_positions_world=candidate_positions.astype(np.float32),
      candidate_xy=candidate_xy.astype(np.float32),
      candidate_face_indices=candidate_faces.astype(np.int32),
      candidate_barycentric=candidate_barycentric.astype(np.float32),
      selected_candidate_indices=selected_candidate_indices.astype(np.int32),
      selected_positions_world=selected_positions.astype(np.float32),
      selected_xy=selected_xy.astype(np.float32),
      selected_face_indices=selected_faces.astype(np.int32),
      selected_barycentric=selected_barycentric.astype(np.float32),
      nearest_vertex_indices=nearest_vertex_indices.astype(np.int32),
      nearest_vertex_positions_world=nearest_vertex_positions.astype(np.float32),
  )

  points_metadata = []
  for point_index in range(FLAGS.num_control_points):
    points_metadata.append({
        "point_index": point_index,
        "selected_position_world": selected_positions[point_index].tolist(),
        "selected_xy": selected_xy[point_index].tolist(),
        "face_index": int(selected_faces[point_index]),
        "barycentric": selected_barycentric[point_index].tolist(),
        "nearest_vertex_index": int(nearest_vertex_indices[point_index]),
        "nearest_vertex_position_world": (
            nearest_vertex_positions[point_index].tolist()),
    })

  kb.write_json({
      "format": "kubric_topdown_control_points_v1",
      "asset_id": FLAGS.asset_id,
      "run": _get_run_metadata(FLAGS),
      "num_control_points": FLAGS.num_control_points,
      "grid_resolution": FLAGS.grid_resolution,
      "num_topdown_candidates": int(len(candidate_positions)),
      "object_position": np.asarray(obj.position, dtype=np.float64).tolist(),
      "object_quaternion": np.asarray(obj.quaternion, dtype=np.float64).tolist(),
      "object_scale": np.asarray(obj.scale, dtype=np.float64).tolist(),
      "render_mesh_min_z": float(initial_min_z),
      "npz_file": npz_path.name,
      "visualization_file": visualization_path.name,
      "points": points_metadata,
  }, output_dir / f"{FLAGS.output_name}.json")
