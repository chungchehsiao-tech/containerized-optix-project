# ==============================================================================
# BASE IMAGE LICENSING NOTICE:
# This Dockerfile uses the NVIDIA CUDA base images (11.8.0).
# The use of these base images is governed by the NVIDIA Deep Learning 
# Container License. By building or running this container, you agree to 
# NVIDIA's terms: https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
# ==============================================================================

# ==============================================================================
# STAGE 1: The Builder
# ==============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the source code (ignores /build and /deps via .dockerignore)
COPY . .

# SECURITY & COMPLIANCE CHECK:
# We require the user to have their legally obtained OptiX headers in the local folder
# If the directory is empty, the build fails with a helpful error message.
RUN if [ ! -f "/app/deps/OptiX_SDK_7.2/include/optix.h" ]; then \
    echo "ERROR: OptiX SDK not found. Due to NVIDIA EULA, you must legally download the SDK and place it in this folder."; \
    echo "Please see README.md for instructions."; \
    exit 1; \
    fi

# Configure and Build
RUN cmake -B build -DOPTIX_INCLUDE_DIR=/app/deps/Optix_SDK_7.2/include -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release

# ==============================================================================
# STAGE 2: Runtime (Slim, ~300MB version)
# ==============================================================================
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

WORKDIR /app

# SECURITY: Create a non-root user and group
RUN groupadd -r optixgroup && useradd -r -g optixgroup optixuser

# Teleport the compiled executable from the builder
COPY --from=builder /app/build/OptixApp /app/OptixApp

# SECURITY: Change ownership and switch to non-root user
RUN chown -R optixuser:optixgroup /app
USER optixuser

CMD ["./OptixApp"]