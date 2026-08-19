#!/bin/bash

set -e

# Anchor to engine/. Without this, running the script from the repo root makes
# the wipe below delete the WRONG build/ and points `cmake ..` at the wrong tree.
cd "$(dirname "$0")"

# The wipe is intentional: every backend script configures into the shared
# engine/build/, and a stale CMakeCache from another backend poisons the
# configure. This is always a full rebuild.
rm -rf build
mkdir build
cd build

cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --config Release -j "$(nproc)"
