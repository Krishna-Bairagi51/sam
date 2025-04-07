#!/bin/bash
cd GroundingDINO
git checkout -q 57535c5a79791cb76e36fdb64975271354f10251
pip install -q -e .

# Navigate back to root directory
cd ..

# Install segment-anything
pip install 'git+https://github.com/facebookresearch/segment-anything.git'


# Create weights directory and download GroundingDINO weights
mkdir weights
cd weights

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    echo "ERROR: Failed to download groundingdino_swint_ogc.pth"
    exit 1
fi

# Download Segment Anything model weights
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
if [ ! -f "sam_vit_h_4b8939.pth" ]; then
    echo "ERROR: Failed to download sam_vit_h_4b8939.pth"
    exit 1
fi

# Navigate back to root directory
echo "Setup completed successfully."
