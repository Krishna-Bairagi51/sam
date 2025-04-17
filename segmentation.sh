#!/bin/bash
#!/usr/bin/env bash
# Auto‑detect CUDA_HOME if not already set
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  echo "🚀 CUDA_HOME auto‑set to $CUDA_HOME"
fi
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting segmentation setup script..."

# Clone GroundingDINO repository
echo "Cloning GroundingDINO..."
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
echo "Checking out specific GroundingDINO commit..."
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251

# Install GroundingDINO with verbose output for build process
echo "Installing GroundingDINO (editable mode, verbose)..."
pip install -e .
python setup.py build install

# Verify that the C++/CUDA extensions were compiled successfully
echo "Verifying GroundingDINO custom ops compilation..."
if python -c "import sys; sys.path.append('.'); from groundingdino.models import _C; print('Successfully imported GroundingDINO _C module.')" ; then
    echo "GroundingDINO custom ops check PASSED."
else
    echo "ERROR: Failed to build or import GroundingDINO C++/CUDA extensions!"
    echo "Check the build log above for errors (nvcc, gcc, g++, CUDA)."
    exit 1 # Fail the Docker build if extensions are missing
fi

# Navigate back to the WORKDIR (/app)
cd ..
echo "Current directory: $(pwd)"

# Install segment-anything
echo "Installing segment-anything..."
pip install --no-cache-dir 'git+https://github.com/facebookresearch/segment-anything.git'

# Create weights directory
mkdir -p weights # Use -p to avoid error if directory exists
cd weights
echo "Current directory: $(pwd)"

# Download model weights
echo "Downloading GroundingDINO weights..."
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
echo "Downloading SAM weights..."
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Navigate back to the WORKDIR (/app)
cd ..
echo "Current directory: $(pwd)"

echo "Setup script completed successfully."
