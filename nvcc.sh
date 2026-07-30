#!/usr/bin/env bash
set -e

# Install location
CUDA_HOME="$HOME/cuda-13.0"

mkdir -p "$HOME/downloads"
cd "$HOME/downloads"

# Download CUDA 13.0 toolkit runfile
wget -O cuda_13.0.0_linux.run \
https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_580.65.06_linux.run

chmod +x cuda_13.0.0_linux.run

# Install ONLY the toolkit (no driver)
./cuda_13.0.0_linux.run \
    --silent \
    --toolkit \
    --toolkitpath="$CUDA_HOME"

# Add to PATH if not already present
grep -qxF "export CUDA_HOME=$CUDA_HOME" ~/.bashrc || echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
grep -qxF 'export PATH=$CUDA_HOME/bin:$PATH' ~/.bashrc || echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
grep -qxF 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' ~/.bashrc || echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Load for current shell
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo
echo "======================================="
echo "CUDA installed to: $CUDA_HOME"
echo "======================================="
echo

nvcc --version