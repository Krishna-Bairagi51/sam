#!/usr/bin/env bash
set -e

# Auto-detect CUDA_HOME if not provided
if [ -z "$CUDA_HOME" ] && command -v nvcc &>/dev/null; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  echo "🚀 CUDA_HOME set to $CUDA_HOME"
fi

# Update paths for CUDA binaries and libraries
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Debug info (optional)
echo "Running nvcc version:"
nvcc --version

# Execute the Python handler
exec python3 -u rp_handler.py
