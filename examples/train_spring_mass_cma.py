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

"""Fits spring-mass parameters with CMA-ES from saved simulation data."""

import argparse
import json
import logging
import pathlib

import numpy as np

import kubric as kb
from kubric.simulator import ControlTrajectoryConfig
from kubric.simulator import SpringMassConfig
from kubric.simulator import SpringMassSimulator

try:
  import torch
except ImportError:
  torch = None


def _scalar(data, key):
  return data[key].item()


def _load_npz(path):
  with np.load(str(path), allow_pickle=False) as data:
    return {key: np.array(data[key]) for key in data.files}


def _write_json(data, path):
  def convert(value):
    if isinstance(value, np.ndarray):
      return value.tolist()
    if isinstance(value, np.generic):
      return value.item()
    return value
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, default=convert, indent=2)
    f.write("\n")


class CMAES:
  """Small full-covariance CMA-ES implementation for low-dimensional search."""

  def __init__(self, mean, sigma, population_size=None, seed=0):
    self.rng = np.random.default_rng(seed)
    self.mean = np.asarray(mean, dtype=np.float64)
    self.dimension = len(self.mean)
    self.sigma = float(sigma)
    self.population_size = population_size or (4 + int(3 * np.log(self.dimension)))
    self.mu = self.population_size // 2
    weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
    self.weights = weights / np.sum(weights)
    self.mu_eff = 1.0 / np.sum(self.weights ** 2)

    n = self.dimension
    self.cc = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
    self.cs = (self.mu_eff + 2) / (n + self.mu_eff + 5)
    self.c1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
    self.cmu = min(
        1 - self.c1,
        2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
        ((n + 2) ** 2 + self.mu_eff),
    )
    self.damps = (
        1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.cs)

    self.pc = np.zeros(n)
    self.ps = np.zeros(n)
    self.covariance = np.eye(n)
    self.generation = 0
    self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

  def ask(self):
    eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-20)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    noise = self.rng.standard_normal((self.population_size, self.dimension))
    samples = self.mean[None, :] + self.sigma * (noise @ transform.T)
    return samples

  def tell(self, samples, losses):
    order = np.argsort(losses)
    selected = samples[order[:self.mu]]
    old_mean = self.mean.copy()
    self.mean = np.sum(selected * self.weights[:, None], axis=0)

    y = (self.mean - old_mean) / self.sigma
    inv_sqrt_cov = self._inv_sqrt_covariance()
    self.ps = (
        (1 - self.cs) * self.ps +
        np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (inv_sqrt_cov @ y))
    ps_norm = np.linalg.norm(self.ps)
    hsig = float(
        ps_norm /
        np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) /
        self.chi_n <
        (1.4 + 2 / (self.dimension + 1)))
    self.pc = (
        (1 - self.cc) * self.pc +
        hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y)

    artmp = (selected - old_mean[None, :]) / self.sigma
    rank_mu = sum(
        weight * np.outer(step, step)
        for weight, step in zip(self.weights, artmp))
    self.covariance = (
        (1 - self.c1 - self.cmu) * self.covariance +
        self.c1 * (
            np.outer(self.pc, self.pc) +
            (1 - hsig) * self.cc * (2 - self.cc) * self.covariance) +
        self.cmu * rank_mu)
    self.covariance = 0.5 * (self.covariance + self.covariance.T)
    self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chi_n - 1))
    self.generation += 1

  def _inv_sqrt_covariance(self):
    eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-20)
    return eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _make_scene(data):
  return kb.Scene(
      frame_start=int(_scalar(data, "scene_frame_start")),
      frame_end=int(_scalar(data, "scene_frame_end")),
      frame_rate=int(_scalar(data, "scene_frame_rate")),
      step_rate=int(_scalar(data, "scene_step_rate")),
      gravity=tuple(data["scene_gravity"].astype(float).tolist()),
  )


def _make_asset(data):
  return kb.FileBasedObject(
      asset_id=str(_scalar(data, "asset_id")),
      position=tuple(data["object_position"].astype(float).tolist()),
      quaternion=tuple(data["object_quaternion"].astype(float).tolist()),
      scale=tuple(data["object_scale"].astype(float).tolist()),
      mass=float(_scalar(data, "object_mass")),
      friction=float(_scalar(data, "object_friction")),
      restitution=float(_scalar(data, "object_restitution")),
  )


def _simulate(data, spring_stiffness, damping, device):
  scene = _make_scene(data)
  asset = _make_asset(data)
  simulator = SpringMassSimulator(
      scene,
      config=SpringMassConfig(
          particle_spacing=float(_scalar(data, "particle_spacing")),
          k_neighbors=int(_scalar(data, "k_neighbors")),
          spring_stiffness=float(spring_stiffness),
          damping=float(damping),
          total_mass=float(_scalar(data, "total_mass")),
          initial_velocity=tuple(data["initial_velocity"].astype(float).tolist()),
          ground_axis=int(_scalar(data, "ground_axis")),
          ground_height=float(_scalar(data, "ground_height")),
          restitution=float(_scalar(data, "restitution")),
          friction=float(_scalar(data, "friction")),
          require_watertight=bool(_scalar(data, "require_watertight")),
          record_all_particles=False,
          seed=int(_scalar(data, "seed")),
      ),
      device=device,
  )
  control_trajectory = ControlTrajectoryConfig(
      control_vertex_indices=tuple(
          data["control_vertex_indices"].astype(int).tolist()),
      positions=data["control_positions_world"].astype(np.float32),
      frame_start=int(_scalar(data, "control_frame_start")),
      frame_rate=float(_scalar(data, "control_frame_rate")),
  )
  return simulator.run(
      asset=asset,
      vertices=data["rest_vertices_local"].astype(np.float32),
      faces=data["faces"].astype(np.int64),
      frame_start=scene.frame_start,
      frame_end=scene.frame_end + 1,
      control_trajectory=control_trajectory,
      initial_particles_world=data["initial_particle_positions_world"],
      surface_mapping=data["surface_mapping"],
      edges=data["spring_edges"],
  )


def _trajectory_loss(predicted, target, frame_stride):
  pred = predicted.vertices[::frame_stride]
  tgt = target[::frame_stride]
  frames = min(len(pred), len(tgt))
  pred = pred[:frames]
  tgt = tgt[:frames]
  if pred.shape != tgt.shape:
    raise ValueError(f"trajectory shape mismatch: {pred.shape} vs {tgt.shape}")
  error = pred - tgt
  return float(np.mean(np.sum(error * error, axis=-1)))


def _release_torch_cuda_cache():
  if torch is not None and torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--simulation_data", required=True)
  parser.add_argument("--target_mesh_vertices", default=None)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--generations", type=int, default=20)
  parser.add_argument("--population_size", type=int, default=8)
  parser.add_argument("--sigma", type=float, default=0.5)
  parser.add_argument("--frame_stride", type=int, default=2)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--initial_spring_stiffness", type=float, default=None)
  parser.add_argument("--initial_damping", type=float, default=None)
  parser.add_argument("--min_spring_stiffness", type=float, default=0.1)
  parser.add_argument("--max_spring_stiffness", type=float, default=200.0)
  parser.add_argument("--min_damping", type=float, default=1e-4)
  parser.add_argument("--max_damping", type=float, default=5.0)
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)
  simulation_data_path = pathlib.Path(args.simulation_data)
  data = _load_npz(simulation_data_path)
  if args.target_mesh_vertices is None:
    target_path = simulation_data_path.parent / "mesh_vertices.npz"
  else:
    target_path = pathlib.Path(args.target_mesh_vertices)
  target_data = _load_npz(target_path)
  target_vertices = target_data["vertices_world"].astype(np.float32)

  initial_stiffness = (
      float(_scalar(data, "spring_stiffness"))
      if args.initial_spring_stiffness is None
      else args.initial_spring_stiffness)
  initial_damping = (
      float(_scalar(data, "damping"))
      if args.initial_damping is None
      else args.initial_damping)
  mean = np.log10([initial_stiffness, initial_damping])
  lower = np.log10([args.min_spring_stiffness, args.min_damping])
  upper = np.log10([args.max_spring_stiffness, args.max_damping])
  optimizer = CMAES(
      mean=mean,
      sigma=args.sigma,
      population_size=args.population_size,
      seed=args.seed,
  )

  output_dir = pathlib.Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  history = []
  best = {"loss": float("inf")}
  for generation in range(args.generations):
    samples = optimizer.ask()
    losses = []
    for candidate_index, sample in enumerate(samples):
      clipped = np.clip(sample, lower, upper)
      stiffness, damping = np.power(10.0, clipped)
      try:
        animation = _simulate(data, stiffness, damping, args.device)
        loss = _trajectory_loss(animation, target_vertices, args.frame_stride)
        _release_torch_cuda_cache()
      except Exception as exc:  # pylint: disable=broad-except
        _release_torch_cuda_cache()
        logging.warning(
            "candidate failed gen=%d idx=%d stiffness=%g damping=%g: %s",
            generation, candidate_index, stiffness, damping, exc)
        loss = 1e30
      losses.append(loss)
      record = {
          "generation": generation,
          "candidate": candidate_index,
          "spring_stiffness": stiffness,
          "damping": damping,
          "loss": loss,
      }
      history.append(record)
      if loss < best["loss"]:
        best = record.copy()
        _write_json(best, output_dir / "best.json")
    optimizer.tell(samples, np.asarray(losses, dtype=np.float64))
    logging.info(
        "generation=%d best_loss=%g best_stiffness=%g best_damping=%g",
        generation, best["loss"], best["spring_stiffness"], best["damping"])
    _write_json({
        "best": best,
        "history": history,
        "optimizer": {
            "mean_log10": optimizer.mean,
            "sigma": optimizer.sigma,
            "covariance": optimizer.covariance,
        },
    }, output_dir / "training_state.json")

  _write_json({
      "best": best,
      "history": history,
      "simulation_data": str(simulation_data_path),
      "target_mesh_vertices": str(target_path),
      "frame_stride": args.frame_stride,
  }, output_dir / "result.json")


if __name__ == "__main__":
  main()
