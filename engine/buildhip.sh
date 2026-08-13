#!/bin/bash

rm -rf build
mkdir build
cd build

if command -v rocminfo; then export GFX_NAME=$(rocminfo | awk '/ *Name: +gfx[1-9]/ {print $2; exit}'); else echo "rocminfo missing!"; fi
if [ -z "${GFX_NAME}" ]; then 
   echo "Warn: Couldn't detect AMD GPU for HIP! Using fallback value (gfx1030)."; 
   GFX_NAME="gfx1030";
else 
   echo "Building for GPU arch: ${GFX_NAME}"; 
fi

cmake .. -DGGML_HIP=ON -DGPU_TARGETS=$GFX_NAME -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j `nproc --ignore=1`
