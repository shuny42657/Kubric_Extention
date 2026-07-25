# CUDA-enabled Kubric image for:
#   1. Blender 2.93 Cycles rendering on NVIDIA GPUs.
#   2. Kubric SpringMassSimulator with CUDA-enabled PyTorch.
#
# The image is intentionally built for NVIDIA Volta (Tesla V100, sm_70).
# Build from the Kubric repository root:
#
#   docker build \
#     -f docker/BlenderCuda118Torch.Dockerfile \
#     -t kubruntu-blender-torch:cu118 .
#
# Run with:
#
#   docker run --rm --runtime=nvidia \
#     --env NVIDIA_VISIBLE_DEVICES=0 \
#     --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
#     --env KUBRIC_USE_GPU=true \
#     --env KUBRIC_CYCLES_BACKEND=CUDA \
#     --env PYTHONPATH=/kubric \
#     --volume "$PWD:/kubric" \
#     kubruntu-blender-torch:cu118 \
#     python3 docker/verify_cuda118.py

ARG CUDA_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu20.04
ARG KUBRIC_BLENDER_IMAGE=kubricdockerhub/blender:latest
ARG KUBRIC_IMAGE=kubricdockerhub/kubruntu:latest


# Keep the complete CUDA toolkit in a standalone stage so that it can be
# copied into the existing Kubric images, which already provide Python 3.9
# and Blender's non-CUDA build/runtime dependencies.
FROM ${CUDA_IMAGE} AS cuda-toolkit


FROM ${KUBRIC_BLENDER_IMAGE} AS cycles-build

USER root

ARG BLENDER_VERSION=blender-v2.93-release
ARG BUILD_JOBS=4
ARG CUDA_ARCH=sm_70

COPY --from=cuda-toolkit /usr/local/cuda-11.8 /usr/local/cuda-11.8

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64

RUN ln -sfn ${CUDA_HOME} /usr/local/cuda && \
    apt-get update && \
    apt-get install --yes --no-install-recommends cmake subversion && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /blenderpy

# This follows docker/Blender.Dockerfile, but explicitly compiles Cycles CUDA
# kernels for V100 instead of relying on runtime kernel compilation.
RUN git clone --branch ${BLENDER_VERSION} --depth 1 \
      https://github.com/blender/blender.git && \
    mkdir -p lib && \
    svn checkout \
      https://svn.blender.org/svnroot/bf-blender/tags/blender-2.93-release/lib/linux_centos7_x86_64 \
      lib/linux_centos7_x86_64 && \
    cd blender && \
    make bpy NPROCS=${BUILD_JOBS} BUILD_CMAKE_ARGS="\
      -DWITH_CYCLES=ON \
      -DWITH_CYCLES_DEVICE_CUDA=ON \
      -DWITH_CYCLES_CUDA_BINARIES=ON \
      -DWITH_CYCLES_DEVICE_OPTIX=OFF \
      -DCYCLES_CUDA_BINARIES_ARCH=${CUDA_ARCH} \
      -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}" && \
    test -f \
      /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages/2.93/scripts/addons/cycles/lib/kernel_${CUDA_ARCH}.cubin && \
    test -f \
      /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages/2.93/scripts/addons/cycles/lib/filter_${CUDA_ARCH}.cubin


FROM ${KUBRIC_IMAGE} AS runtime

USER root

ARG PYTORCH_VERSION=2.1.2
ARG TORCH_CUDA_ARCH_LIST=7.0

# Retaining nvcc in the final image makes future CUDA simulation extensions
# reproducible without another base-image build.
COPY --from=cuda-toolkit /usr/local/cuda-11.8 /usr/local/cuda-11.8
COPY --from=cycles-build /blenderpy/build_linux_bpy/bin/bpy.so \
  /usr/local/lib/python3.9/dist-packages/bpy.so
COPY --from=cycles-build \
  /blenderpy/lib/linux_centos7_x86_64/python/lib/python3.9/site-packages/2.93 \
  /usr/local/lib/python3.9/dist-packages/2.93

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV KUBRIC_CYCLES_BACKEND=CUDA
ENV PYTHONPATH=/kubric

RUN ln -sfn ${CUDA_HOME} /usr/local/cuda && \
    echo "${CUDA_HOME}/lib64" > /etc/ld.so.conf.d/cuda-11-8.conf && \
    ldconfig

# PyTorch wheels contain their CUDA runtime. Kubric's spring-mass simulator
# implements chunked KNN with torch.cdist/topk, so no external CUDA extension
# is needed for simulation.
RUN python3 -m pip install --no-cache-dir \
      "torch==${PYTORCH_VERSION}+cu118" \
      --index-url https://download.pytorch.org/whl/cu118

RUN python3 -c "import bpy, torch; \
assert torch.version.cuda == '11.8', torch.version.cuda; \
print('Blender:', bpy.app.version_string); \
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)"

WORKDIR /kubric
