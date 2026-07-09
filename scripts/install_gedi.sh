#!/usr/bin/env bash
# =============================================================================
# install_gedi.sh — Install GeDi geometric descriptors inside the OSCAR container
# =============================================================================
#
# Run this script INSIDE the OSCAR Docker container:
#   docker compose run --rm oscar bash scripts/install_gedi.sh
#
# Or manually step-by-step if you hit issues (see comments below).
#
# Prerequisites:
#   - CUDA 12.2 dev headers (nvcc) — requires cuda:12.2.0-devel base image
#   - PyTorch 2.x with CUDA support
#   - Open3D >= 0.17
#
# What this script does:
#   1. Clones the GeDi repo to /gedi
#   2. Installs torchgeometry (or kornia as fallback)
#   3. Compiles pointnet2_ops_lib (CUDA C++ extension)
#   4. Downloads the pretrained GeDi checkpoint
#   5. Verifies the installation
# =============================================================================

set -e

GEDI_DIR="/gedi"
CHECKPOINT_DIR="${GEDI_DIR}/data/chkpts/3dmatch"

echo "============================================="
echo "  GeDi Installation for OSCAR+"
echo "============================================="

# -----------------------------------------------------------------------
# Step 1: Clone GeDi repo
# -----------------------------------------------------------------------
if [ -d "${GEDI_DIR}" ] && [ -f "${GEDI_DIR}/gedi.py" ]; then
    echo "[1/5] GeDi repo already exists at ${GEDI_DIR}, skipping clone."
else
    echo "[1/5] Cloning GeDi repo..."
    git clone https://github.com/fabiopoiesi/gedi.git "${GEDI_DIR}"
fi

cd "${GEDI_DIR}"

# -----------------------------------------------------------------------
# Step 2: Install torchgeometry
# -----------------------------------------------------------------------
echo "[2/5] Installing torchgeometry..."
# torchgeometry 0.1.2 is the version GeDi was tested with.
# It's an older package (predecessor to kornia). If it fails to install,
# we patch GeDi to use kornia instead.
if pip install torchgeometry==0.1.2 2>/dev/null; then
    echo "  torchgeometry 0.1.2 installed."
else
    echo "  torchgeometry failed. Installing kornia and patching GeDi..."
    pip install kornia
    # Patch gedi.py to use kornia instead of torchgeometry
    if grep -q "import torchgeometry as tgm" "${GEDI_DIR}/gedi.py"; then
        sed -i 's/import torchgeometry as tgm/import kornia.geometry as tgm/' \
            "${GEDI_DIR}/gedi.py"
        # kornia renamed quaternion_to_angle_axis → quaternion_to_axis_angle
        # and angle_axis_to_rotation_matrix → axis_angle_to_rotation_matrix
        sed -i 's/tgm\.quaternion_to_angle_axis/tgm.conversions.quaternion_to_axis_angle/' \
            "${GEDI_DIR}/gedi.py"
        sed -i 's/tgm\.angle_axis_to_rotation_matrix/tgm.conversions.axis_angle_to_rotation_matrix/' \
            "${GEDI_DIR}/gedi.py"
        echo "  Patched gedi.py to use kornia."
    fi
fi

# -----------------------------------------------------------------------
# Step 3: Compile pointnet2_ops_lib
# -----------------------------------------------------------------------
echo "[3/5] Compiling pointnet2_ops_lib..."
# This requires nvcc (CUDA dev headers). If using cuda:12.2.0-runtime,
# this step WILL FAIL. You need cuda:12.2.0-devel-ubuntu22.04.
if ! command -v nvcc &>/dev/null; then
    echo "  ERROR: nvcc not found. Cannot compile pointnet2_ops_lib."
    echo "  The Dockerfile must use nvidia/cuda:12.2.0-devel-ubuntu22.04"
    echo "  instead of nvidia/cuda:12.2.0-runtime-ubuntu22.04."
    echo ""
    echo "  Alternatively, you can try installing CUDA toolkit:"
    echo "    apt-get install -y nvidia-cuda-toolkit"
    echo ""
    echo "  Skipping compilation. GeDi will not work (FPFH fallback active)."
else
    cd "${GEDI_DIR}/backbones/pointnet2_ops_lib"
    # Clean any previous builds
    rm -rf build dist *.egg-info
    pip install .
    echo "  pointnet2_ops_lib compiled and installed."
fi

# -----------------------------------------------------------------------
# Step 4: Download pretrained checkpoint
# -----------------------------------------------------------------------
echo "[4/5] Downloading GeDi checkpoint..."
if [ -f "${CHECKPOINT_DIR}/chkpt.tar" ]; then
    echo "  Checkpoint already exists, skipping download."
else
    cd "${GEDI_DIR}"
    # GeDi provides a download script
    if [ -f "download_data.py" ]; then
        pip install gdown 2>/dev/null || true
        python3 download_data.py
    else
        echo "  WARNING: download_data.py not found. Manual download needed."
        echo "  Expected location: ${CHECKPOINT_DIR}/chkpt.tar"
    fi
fi

# -----------------------------------------------------------------------
# Step 5: Verify installation
# -----------------------------------------------------------------------
echo "[5/5] Verifying GeDi installation..."
cd "${GEDI_DIR}"
python3 -c "
import sys
sys.path.insert(0, '${GEDI_DIR}')
try:
    from gedi import GeDi
    print('  ✓ GeDi class imported successfully.')
except Exception as e:
    print(f'  ✗ GeDi import failed: {e}')
    sys.exit(1)

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')

try:
    from backbones.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule
    print('  ✓ pointnet2_ops imported successfully.')
except Exception as e:
    print(f'  ✗ pointnet2_ops import failed: {e}')
    print('    This means GeDi descriptors won'\''t work.')
    print('    FPFH fallback will be used instead.')

import os
chkpt = '${CHECKPOINT_DIR}/chkpt.tar'
if os.path.isfile(chkpt):
    print(f'  ✓ Checkpoint found: {chkpt}')
else:
    print(f'  ✗ Checkpoint not found: {chkpt}')
"

echo ""
echo "============================================="
echo "  GeDi installation complete."
echo ""
echo "  To use GeDi in the pipeline, run with:"
echo "    --geometry-reranking \\"
echo "    --gedi-repo /gedi \\"
echo "    --gedi-checkpoint /gedi/data/chkpts/3dmatch/chkpt.tar"
echo "============================================="
