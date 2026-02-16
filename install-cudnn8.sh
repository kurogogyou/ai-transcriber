#!/bin/bash
# Install cuDNN 8 libraries alongside cuDNN 9 in the venv.
#
# PyTorch 2.4 requires cuDNN 9 (nvidia-cudnn-cu11==9.x) but CTranslate2 4.x
# requires cuDNN 8. Since the .so files use different version suffixes (.so.8
# vs .so.9), they can coexist in the same directory.
#
# This script downloads the cuDNN 8 wheel and copies just the .so.8 files into
# the existing nvidia/cudnn/lib/ directory.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDNN_LIB="$SCRIPT_DIR/venv/lib/python3.12/site-packages/nvidia/cudnn/lib"

if [ -f "$CUDNN_LIB/libcudnn_ops_infer.so.8" ]; then
    echo "cuDNN 8 libraries already installed, skipping."
    exit 0
fi

if [ ! -d "$CUDNN_LIB" ]; then
    echo "ERROR: $CUDNN_LIB not found. Run 'pip install -r requirements.txt' first."
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading nvidia-cudnn-cu11==8.9.6.50 wheel..."
pip download nvidia-cudnn-cu11==8.9.6.50 -d "$TMPDIR/wheel" --no-deps -q

echo "Extracting cuDNN 8 libraries..."
unzip -o "$TMPDIR/wheel/"*.whl "nvidia/cudnn/lib/libcudnn*.so.8" -d "$TMPDIR/extracted" -q

cp "$TMPDIR/extracted/nvidia/cudnn/lib/"libcudnn*.so.8 "$CUDNN_LIB/"

echo "Done. cuDNN 8 libraries installed to $CUDNN_LIB"
ls "$CUDNN_LIB"/libcudnn*.so.8
