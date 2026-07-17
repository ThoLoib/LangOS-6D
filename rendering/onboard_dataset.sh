#!/usr/bin/env bash
# =============================================================================
# rendering/onboard_dataset.sh — One-command CAD gallery onboarding
# =============================================================================
#
# Orchestrates the full onboarding pipeline for a new CAD gallery:
#   1. (Optional) Prepare BOP PLY models → object_database/ layout
#   2. Render reference images (42 icosphere views via Blender)
#   3. Generate partial point clouds (per-view, for ULIP-2 partial mode)
#   4. Generate LLaVA text descriptions
#
# Each step is idempotent: existing outputs are skipped unless --overwrite
# is set.  Steps can be run individually via --step.
#
# Usage:
#   # Full onboarding (all steps):
#   bash rendering/onboard_dataset.sh --dataset tless
#
#   # Single step:
#   bash rendering/onboard_dataset.sh --dataset tless --step render
#
#   # Re-render existing dataset with 42 views:
#   bash rendering/onboard_dataset.sh --dataset ycbv_gso --step render --overwrite
#
#   # Dry run (show what would be done):
#   bash rendering/onboard_dataset.sh --dataset tless --dry-run
#
# Supported datasets:
#   ycbv_gso, MI3DOR, housecat6d, shrec18  — existing OBJ-based galleries
#   tless, lmo, itodd                      — BOP PLY-based galleries (auto-prepared)
#
# Prerequisites:
#   - Blender 3.4+ installed at rendering/blender-*/blender (for rendering)
#   - Python 3 with trimesh (for partial PCs)
#   - Python 3 with transformers + LLaVA (for descriptions, inside Docker)
#   - BOP datasets extracted in eval/datasets/ (for tless/lmo/itodd)
#
# For rclone sync to Google Drive, run rclone_watch.sh in a separate WSL
# terminal while this script runs inside Docker.  See onboard_and_sync.sh
# for the full automated workflow (Docker + rclone).
#
# Environment variables:
#   NUM_VIEWS        — Number of icosphere views to render (default: 42)
#   NUM_POINTS       — Points per partial point cloud (default: 10000)
#   BLENDER_BIN      — Path to Blender binary (auto-detected if unset)
#   OSCAR_ROOT       — Project root (auto-detected from script location)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults & argument parsing
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OSCAR_ROOT="${OSCAR_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

DATASET=""
STEP="all"        # all | prepare | render | partial | describe
OVERWRITE=0
DRY_RUN=0
NUM_VIEWS="${NUM_VIEWS:-42}"
NUM_POINTS="${NUM_POINTS:-10000}"
BLENDER_BIN="${BLENDER_BIN:-}"
usage() {
    echo "Usage: $0 --dataset <name> [--step <step>] [--overwrite] [--dry-run]"
    echo ""
    echo "Datasets: ycbv_gso, MI3DOR, housecat6d, shrec18, tless, lmo, itodd"
    echo "Steps:    all, prepare, render, partial, describe"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2"; shift 2 ;;
        --step)           STEP="$2"; shift 2 ;;
        --overwrite)      OVERWRITE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --num-views)      NUM_VIEWS="$2"; shift 2 ;;
        --num-points)     NUM_POINTS="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *)                echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" ]] && { echo "ERROR: --dataset is required"; usage; }

# ---------------------------------------------------------------------------
# Auto-detect Blender
# ---------------------------------------------------------------------------
if [[ -z "$BLENDER_BIN" ]]; then
    # Search: Docker image path (/blender/), then local rendering/ dir
    for candidate in /blender/blender-*/blender "$SCRIPT_DIR"/blender-*/blender; do
        if [[ -x "$candidate" ]]; then
            BLENDER_BIN="$candidate"
            break
        fi
    done
fi

# ---------------------------------------------------------------------------
# Dataset-specific path configuration
# ---------------------------------------------------------------------------
# Each dataset defines:
#   CAD_DIR       — where the rendering script finds meshes
#   IMAGES_DIR    — where rendered images + partial PCs are stored
#   DESC_OUTPUT   — where the descriptions JSON goes
#   BOP_SOURCE    — (BOP datasets only) source PLY directory
#   MESH_GLOB     — (optional) glob pattern for generate_partial_pointclouds.py
#   IS_BOP        — whether BOP PLY preparation is needed

IS_BOP=0
BOP_SOURCE=""
MESH_GLOB=""

case "$DATASET" in
    ycbv_gso)
        CAD_DIR="$OSCAR_ROOT/object_database/ycbv_gso"
        IMAGES_DIR="$OSCAR_ROOT/object_images/ycbv_gso"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/ycbv_gso/descriptions_attributes.json"
        ;;
    MI3DOR)
        CAD_DIR="$OSCAR_ROOT/object_database/MI3DOR/model/test"
        IMAGES_DIR="$OSCAR_ROOT/object_images/MI3DOR"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/MI3DOR/descriptions_attributes.json"
        MESH_GLOB="$OSCAR_ROOT/object_database/MI3DOR/model/test/*/*.obj"
        ;;
    housecat6d)
        CAD_DIR="$OSCAR_ROOT/object_database/housecat6d"
        IMAGES_DIR="$OSCAR_ROOT/object_images/housecat6d"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/housecat6d/descriptions_attributes.json"
        MESH_GLOB="$OSCAR_ROOT/object_database/housecat6d/*/*.obj"
        ;;
    shrec18)
        CAD_DIR="$OSCAR_ROOT/eval/datasets/shrec18/shrec18_full/cad"
        IMAGES_DIR="$OSCAR_ROOT/object_images/shrec18"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/shrec18/descriptions_attributes.json"
        MESH_GLOB="$OSCAR_ROOT/eval/datasets/shrec18/shrec18_full/cad/*.obj"
        ;;
    tless)
        IS_BOP=1
        BOP_SOURCE="$OSCAR_ROOT/eval/datasets/tless/models_cad"
        CAD_DIR="$OSCAR_ROOT/object_database/tless"
        IMAGES_DIR="$OSCAR_ROOT/object_images/tless"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/tless/descriptions_attributes.json"
        ;;
    lmo)
        IS_BOP=1
        BOP_SOURCE="$OSCAR_ROOT/eval/datasets/lmo/models"
        CAD_DIR="$OSCAR_ROOT/object_database/lmo"
        IMAGES_DIR="$OSCAR_ROOT/object_images/lmo"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/lmo/descriptions_attributes.json"
        ;;
    itodd)
        IS_BOP=1
        BOP_SOURCE="$OSCAR_ROOT/eval/datasets/itodd/models"
        CAD_DIR="$OSCAR_ROOT/object_database/itodd"
        IMAGES_DIR="$OSCAR_ROOT/object_images/itodd"
        DESC_OUTPUT="$OSCAR_ROOT/object_database/itodd/descriptions_attributes.json"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "Supported: ycbv_gso, MI3DOR, housecat6d, shrec18, tless, lmo, itodd"
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[onboard] $*"; }

run_or_dry() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $*"
    else
        "$@"
    fi
}

# Count objects in a CAD directory (falls back to BOP_SOURCE for unbuilt BOP dirs)
count_objects() {
    local dir="$1"
    if [[ ! -d "$dir" ]] && [[ $IS_BOP -eq 1 ]] && [[ -d "$BOP_SOURCE" ]]; then
        # Count PLY files in BOP source (excluding models_info.json)
        find "$BOP_SOURCE" -maxdepth 1 -name "obj_*.ply" | wc -l
        return
    fi
    if [[ ! -d "$dir" ]]; then
        echo "0"
        return
    fi
    # Count unique objects by parent directory (for nested layouts like ycbv_gso)
    # or by file (for flat layouts like MI3DOR).  Rendering.py deduplicates
    # via infer_model_id + file_priority, so this is an approximation.
    local nested
    nested=$(find "$dir" -maxdepth 3 \( -name "*.obj" -o -name "*.glb" -o -name "*.ply" \) | \
        sed 's|/[^/]*$||' | sort -u | wc -l)
    local flat
    flat=$(find "$dir" -maxdepth 3 \( -name "*.obj" -o -name "*.glb" -o -name "*.ply" \) | wc -l)
    # Use the larger count (flat is more accurate for MI3DOR, nested for ycbv_gso)
    if [[ $flat -gt $nested ]]; then echo "$flat"; else echo "$nested"; fi
}

# ---------------------------------------------------------------------------
# Step 1: Prepare BOP PLY models
# ---------------------------------------------------------------------------
step_prepare() {
    if [[ $IS_BOP -eq 0 ]]; then
        log "Skip prepare: $DATASET is not a BOP dataset"
        return
    fi

    if [[ ! -d "$BOP_SOURCE" ]]; then
        echo "ERROR: BOP source directory not found: $BOP_SOURCE"
        echo "Download the dataset first (see eval/datasets/)"
        exit 1
    fi

    log "Preparing BOP models: $BOP_SOURCE → $CAD_DIR"

    # Create object_database/{dataset}/{obj_id}/ structure with symlinks to PLY
    local count=0
    for ply_file in "$BOP_SOURCE"/obj_*.ply; do
        [[ -f "$ply_file" ]] || continue
        local basename
        basename=$(basename "$ply_file" .ply)
        local obj_dir="$CAD_DIR/$basename"

        if [[ -d "$obj_dir" ]] && [[ $OVERWRITE -eq 0 ]]; then
            continue
        fi

        run_or_dry mkdir -p "$obj_dir"
        # Hard copy (not symlink) so the file is visible inside Docker containers
        run_or_dry cp -f "$ply_file" "$obj_dir/model.ply"
        count=$((count + 1))
    done

    log "Prepared $count BOP objects in $CAD_DIR"
}

# ---------------------------------------------------------------------------
# Step 2: Render reference images
# ---------------------------------------------------------------------------
step_render() {
    if [[ -z "$BLENDER_BIN" ]] || [[ ! -x "$BLENDER_BIN" ]]; then
        echo "ERROR: Blender not found. Set BLENDER_BIN or install Blender in rendering/"
        exit 1
    fi

    local obj_count
    obj_count=$(count_objects "$CAD_DIR")
    log "Rendering $obj_count objects from $CAD_DIR → $IMAGES_DIR ($NUM_VIEWS views each)"

    # Estimate time: ~3-5 sec per view on RTX 4050 (CYCLES, 16 samples)
    local est_seconds=$((obj_count * NUM_VIEWS * 4))
    local est_minutes=$((est_seconds / 60))
    log "Estimated time: ~${est_minutes} minutes (${est_seconds}s at ~4s/view)"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] Would run: OBJECT_FOLDER=$CAD_DIR OBJECT_IMAGES=$IMAGES_DIR NUM_VIEWS=$NUM_VIEWS $BLENDER_BIN -b -P rendering/rendering.py"
        return
    fi

    cd "$SCRIPT_DIR"
    OBJECT_FOLDER="$CAD_DIR" \
    OBJECT_IMAGES="$IMAGES_DIR" \
    NUM_VIEWS="$NUM_VIEWS" \
    OVERWRITE_EXISTING="$OVERWRITE" \
        "$BLENDER_BIN" --background --python rendering.py
    cd "$OSCAR_ROOT"
}

# ---------------------------------------------------------------------------
# Step 3: Generate partial point clouds
# ---------------------------------------------------------------------------
step_partial() {
    local obj_count
    obj_count=$(count_objects "$CAD_DIR")
    log "Generating partial point clouds for $obj_count objects ($NUM_VIEWS views × $NUM_POINTS points)"

    local est_seconds=$((obj_count * NUM_VIEWS / 5))  # ~0.2s per view
    local est_minutes=$((est_seconds / 60))
    log "Estimated time: ~${est_minutes} minutes"

    local args=(
        --images_dir "$IMAGES_DIR"
        --num_points "$NUM_POINTS"
    )

    if [[ -n "$MESH_GLOB" ]]; then
        args+=(--mesh-glob "$MESH_GLOB")
    else
        args+=(--cad_dir "$CAD_DIR")
    fi

    if [[ $OVERWRITE -eq 1 ]]; then
        args+=(--overwrite)
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] Would run: python3 rendering/generate_partial_pointclouds.py ${args[*]}"
        return
    fi

    python3 "$OSCAR_ROOT/rendering/generate_partial_pointclouds.py" "${args[@]}"
}

# ---------------------------------------------------------------------------
# Step 4: Generate text descriptions (LLaVA)
# ---------------------------------------------------------------------------
step_describe() {
    # Don't skip — generate_descriptions.py is idempotent and handles
    # resuming internally (skips already-captioned images per object).

    local obj_count
    obj_count=$(count_objects "$CAD_DIR")
    log "Generating LLaVA descriptions for $obj_count objects"

    # Estimate: ~2-3 sec per object on RTX 4050 (LLaVA 1.5-7B, float16)
    local est_seconds=$((obj_count * 3))
    local est_minutes=$((est_seconds / 60))
    log "Estimated time: ~${est_minutes} minutes"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] Would run: python3 rendering/generate_descriptions.py --images_dir $IMAGES_DIR --output $DESC_OUTPUT"
        return
    fi

    python3 "$OSCAR_ROOT/rendering/generate_descriptions.py" \
        --images_dir "$IMAGES_DIR" \
        --output "$DESC_OUTPUT"
}

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
log "=== Onboarding dataset: $DATASET ==="
log "  CAD source:    $CAD_DIR"
log "  Images output: $IMAGES_DIR"
log "  Descriptions:  $DESC_OUTPUT"
log "  Views:         $NUM_VIEWS"
log "  Points/PC:     $NUM_POINTS"
log "  Overwrite:     $OVERWRITE"
log "  BOP dataset:   $IS_BOP"
[[ $DRY_RUN -eq 1 ]] && log "  *** DRY RUN MODE ***"
echo ""

case "$STEP" in
    all)
        step_prepare
        step_render
        step_partial
        step_describe
        ;;
    prepare)  step_prepare ;;
    render)   step_render ;;
    partial)  step_partial ;;
    describe) step_describe ;;
    *)
        echo "ERROR: Unknown step '$STEP'. Use: all, prepare, render, partial, describe"
        exit 1
        ;;
esac

log "=== Done: $DATASET ($STEP) ==="
