"""Checks CUDA visibility for PyTorch operations and Blender Cycles."""

import bpy
import torch


def main():
  if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access CUDA")

  points = torch.rand((1, 32, 3), device="cuda")
  distances = torch.cdist(points, points)
  neighbors = torch.topk(distances, k=4, largest=False).indices
  if not neighbors.is_cuda:
    raise RuntimeError("PyTorch KNN operations did not run on CUDA")

  preferences = bpy.context.preferences.addons["cycles"].preferences
  preferences.compute_device_type = "CUDA"
  preferences.get_devices()
  cuda_devices = [device for device in preferences.devices
                  if device.type == "CUDA"]
  if not cuda_devices:
    available = [(device.name, device.type) for device in preferences.devices]
    raise RuntimeError(f"Cycles cannot access CUDA. Available devices: {available}")

  for device in preferences.devices:
    device.use = device in cuda_devices
  bpy.context.scene.cycles.device = "GPU"

  print("PyTorch CUDA:", torch.version.cuda)
  print("PyTorch device:", torch.cuda.get_device_name(0))
  print("PyTorch capability:", torch.cuda.get_device_capability(0))
  print("Cycles CUDA devices:", [device.name for device in cuda_devices])


if __name__ == "__main__":
  main()
