#!/bin/bash
# Run COLMAP for each N3DV scene using the first frame from all cameras.
# Usage: bash run_colmap_n3dv.sh [data/n3dv] [--no_gpu] [--skip_matching]

set -e

DATA_DIR="${1:-data/n3dv}"
EXTRA_ARGS="${@:2}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

for SCENE_PATH in "$DATA_DIR"/*/; do
    SCENE_PATH="${SCENE_PATH%/}"
    SCENE_NAME="$(basename "$SCENE_PATH")"

    # Skip non-directories (e.g. zip files)
    [ -d "$SCENE_PATH" ] || continue

    # Check that at least cam00 exists
    if [ ! -d "$SCENE_PATH/cam00" ]; then
        echo "[SKIP] $SCENE_NAME — no cam folders found"
        continue
    fi

    echo "========================================"
    echo "[SCENE] $SCENE_NAME"
    echo "========================================"

    INPUT_DIR="$SCENE_PATH/input"
    mkdir -p "$INPUT_DIR"

    # Copy first frame from each cam as camXX.png into input/
    for CAM_DIR in "$SCENE_PATH"/cam*/; do
        CAM_DIR="${CAM_DIR%/}"
        CAM_NAME="$(basename "$CAM_DIR")"
        SRC="$CAM_DIR/images/0000.png"
        DST="$INPUT_DIR/${CAM_NAME}.png"
        if [ -f "$SRC" ]; then
            cp "$SRC" "$DST"
        else
            echo "  [WARN] $SRC not found, skipping"
        fi
    done

    echo "  Copied $(ls "$INPUT_DIR" | wc -l) frames into $INPUT_DIR"

    # Run convert.py (gaussian-splatting colmap converter)
    python "$SCRIPT_DIR/convert.py" -s "$SCENE_PATH" \
        --colmap_executable "mamba run -n 3dgs colmap" \
        $EXTRA_ARGS

    # Move sparse/0/ → sparse_/ as expected by QUEEN's dataset reader
    SPARSE_SRC="$SCENE_PATH/sparse/0"
    SPARSE_DST="$SCENE_PATH/sparse_"
    if [ -d "$SPARSE_SRC" ]; then
        rm -rf "$SPARSE_DST"
        mv "$SPARSE_SRC" "$SPARSE_DST"
        echo "  Moved sparse/0 → sparse_"
    else
        echo "  [WARN] $SPARSE_SRC not found after convert.py"
    fi

    echo "  [DONE] $SCENE_NAME"
done

echo ""
echo "All scenes processed."
