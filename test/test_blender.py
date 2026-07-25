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

import numpy as np
import pytest

from kubric import core
from kubric.renderer import blender
from kubric.renderer import blender_utils
from kubric.safeimport.bpy import bpy


def test_prepare_blender_object():
  @blender_utils.prepare_blender_object
  def add_asset(self, asset):
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.active_object
    return cube

  cube_asset = core.Cube()
  cube_obj = add_asset(None, cube_asset)

  assert cube_obj.name.split('.')[0] == cube_asset.uid
  assert cube_obj.rotation_mode == "QUATERNION"
  assert cube_obj in bpy.context.scene.collection.objects.values()


def test_blender_scene_properties(tmp_path):
  scene = core.Scene(
      frame_start=2,
      frame_end=3,
      frame_rate=5,
      resolution=(7, 11),
  )
  renderer = blender.Blender(scene, tmp_path)
  assert renderer in scene.views
  assert renderer.scene == scene

  assert renderer.blender_scene.frame_start == 2
  assert renderer.blender_scene.frame_end == 3
  assert renderer.blender_scene.render.fps == 5
  assert renderer.blender_scene.render.resolution_x == 7
  assert renderer.blender_scene.render.resolution_y == 11


def test_blender_camera_on_init(tmp_path):
  cam = core.PerspectiveCamera(position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3,
                               sensor_width=4)
  renderer = blender.Blender(core.Scene(camera=cam), tmp_path)

  assert renderer in cam.linked_objects
  blender_cam = cam.linked_objects[renderer]
  assert renderer.blender_scene.camera == blender_cam
  assert blender_cam in renderer.blender_scene.collection.objects.values()
  assert tuple(blender_cam.location) == (1, 2, 3)
  assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
  assert blender_cam.data.lens == 3
  assert blender_cam.data.sensor_width == 4


def test_blender_camera_assign_after_init(tmp_path):
  scene = core.Scene()
  renderer = blender.Blender(scene, tmp_path)

  cam = core.PerspectiveCamera(position=(1, 2, 3), quaternion=(0, 1, 0, 0), focal_length=3,
                               sensor_width=4)

  scene.camera = cam

  assert renderer in cam.linked_objects
  blender_cam = cam.linked_objects[renderer]
  assert renderer.blender_scene.camera == blender_cam
  assert blender_cam in renderer.blender_scene.collection.objects.values()
  assert tuple(blender_cam.location) == (1, 2, 3)
  assert tuple(blender_cam.rotation_quaternion) == (0, 1, 0, 0)
  assert blender_cam.data.lens == 3
  assert blender_cam.data.sensor_width == 4


def test_blender_adaptive_sampling_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_set_adaptive_sampling(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.adaptive_sampling = False
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_init_adaptive_sampling(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, adaptive_sampling=False)
  assert renderer.adaptive_sampling is False
  assert renderer.blender_scene.cycles.use_adaptive_sampling is False


def test_blender_vertex_animation(tmp_path):
  scene = core.Scene(frame_start=1, frame_end=2)
  cube = core.Cube(scale=2., position=(5., 0., 0.))
  scene.add(cube)
  renderer = blender.Blender(scene, tmp_path)
  rest_vertices = renderer.get_mesh_vertices(cube)
  assert rest_vertices.shape == (8, 3)

  frame_vertices = np.stack([
      rest_vertices + (1., 2., 3.),
      rest_vertices + (4., 5., 6.),
  ])
  animation = core.VertexAnimation(cube, frame_start=1, vertices=frame_vertices)
  renderer.add_vertex_animation(cube, animation)

  blender_obj = cube.linked_objects[renderer]
  assert tuple(blender_obj.location) == (0., 0., 0.)
  assert tuple(blender_obj.rotation_quaternion) == (1., 0., 0., 0.)
  assert tuple(blender_obj.scale) == (1., 1., 1.)
  assert len(blender_obj.data.shape_keys.key_blocks) == 3

  for frame, expected_vertices in enumerate(frame_vertices, start=1):
    renderer.blender_scene.frame_set(frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = blender_obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    actual_vertices = np.empty(len(evaluated_mesh.vertices) * 3)
    evaluated_mesh.vertices.foreach_get("co", actual_vertices)
    evaluated_obj.to_mesh_clear()
    np.testing.assert_allclose(
        actual_vertices.reshape((-1, 3)), expected_vertices, atol=1e-6)


def test_blender_get_mesh_geometry_triangulates_faces(tmp_path):
  scene = core.Scene()
  cube = core.Cube()
  scene.add(cube)
  renderer = blender.Blender(scene, tmp_path)

  vertices, faces = renderer.get_mesh_geometry(cube)

  assert vertices.shape == (8, 3)
  assert faces.shape == (12, 3)


def test_blender_vertex_animation_rejects_vertex_count_mismatch(tmp_path):
  scene = core.Scene()
  cube = core.Cube()
  scene.add(cube)
  renderer = blender.Blender(scene, tmp_path)
  animation = core.VertexAnimation(
      cube, frame_start=1, vertices=np.zeros((1, 7, 3)))

  with pytest.raises(ValueError, match="Vertex count mismatch"):
    renderer.add_vertex_animation(cube, animation)


def test_blender_principled_material_socket_compatibility(tmp_path):
  scene = core.Scene()
  renderer = blender.Blender(scene, tmp_path)
  material = core.PrincipledBSDFMaterial(
      specular=0.2,
      specular_tint=core.Color(0.2, 0.4, 0.6),
      transmission=0.1,
      transmission_roughness=0.3,
      emission=core.Color(0.1, 0.2, 0.3),
  )

  scene.add(core.Cube(material=material))

  assert renderer in material.linked_objects


def test_blender_use_denoising_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.use_denoising is True
  assert renderer.blender_scene.cycles.use_denoising is True


def test_blender_set_use_denoising(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.use_denoising = False
  assert renderer.use_denoising is False
  assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_use_denoising_init(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, use_denoising=False)
  assert renderer.use_denoising is False
  assert renderer.blender_scene.cycles.use_denoising is False


def test_blender_samples_per_pixel_default(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  assert renderer.samples_per_pixel == 128
  assert renderer.blender_scene.cycles.samples == 128


def test_blender_set_samples_per_pixel(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path)
  renderer.samples_per_pixel = 64
  assert renderer.samples_per_pixel == 64
  assert renderer.blender_scene.cycles.samples == 64


def test_blender_samples_per_pixel_init(tmp_path):
  renderer = blender.Blender(core.Scene(), tmp_path, samples_per_pixel=256)
  assert renderer.samples_per_pixel == 256
  assert renderer.blender_scene.cycles.samples == 256
