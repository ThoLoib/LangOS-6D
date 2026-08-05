#!/usr/bin/env bash
# =============================================================================
# parallel_render.sh — render one dataset with N Blender workers on a single GPU
# =============================================================================
#
# During a normal single-worker render the GPU sits ~90% idle: the bottleneck is
# the per-model CPU work (importing the .obj, welding, recomputing normals,
# building the BVH), not the GPU sampling. Running N workers overlaps that CPU
# prep and keeps the GPU busy, cutting wall-clock roughly 2-3x.
#
# Each worker renders a disjoint 1/N shard of the models (via SHARD_INDEX /
# SHARD_TOTAL, which rendering.py understands) and they all write to the same
# OBJECT_IMAGES directory. Model ids are disjoint across shards, so there is no
# output collision.
#
# -----------------------------------------------------------------------------
# THIS SCRIPT RUNS INSIDE THE oscar CONTAINER (that is where Blender + the GPU
# are). Invoke it from the host through docker compose:
#
#   docker compose run --rm oscar bash -lc \
#     "cd /app && bash rendering/parallel_render.sh --dataset gso --workers 4"
#
# Or with explicit directories (repo-relative, i.e. /app-relative):
#
#   ... bash rendering/parallel_render.sh \
#         --object-folder object_database/gso \
#         --object-images object_images/gso --workers 4
#
# -----------------------------------------------------------------------------
# NOTES
#   * This replaces ONLY the render step. Run the partial-PC, description and
#     embedding steps afterwards as usual (e.g. via preprocess_gallery.sh, or
#     onboard_dataset.sh --step partial / --step describe).
#   * Resumable: already-rendered models are skipped, so re-running continues
#     where a previous run stopped.
#   * VRAM: each worker uses ~1 GB. 4 workers fit easily on a 24 GB GPU; going
#     much beyond ~6-8 rarely helps (CPU / memory-bandwidth bound) and raises
#     the chance of a GPU driver hang.
#   * SAFETY: several headless renderers still share one GPU driver. For a big
#     run, stop the desktop session first (host: `systemctl isolate
#     multi-user.target`) so the X server is not also contending for the GPU.
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OSCAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$OSCAR_ROOT" || { echo "ERROR: cannot cd to repo root $OSCAR_ROOT"; exit 3; }

# --- defaults (override on the command line or via env) -----------------------
BLENDER_BIN="${BLENDER_BIN:-/blender/blender-3.4.1-linux-x64/blender}"
WORKERS=4
NUM_VIEWS="${NUM_VIEWS:-42}"
OVERWRITE=0
OBJECT_FOLDER=""
OBJECT_IMAGES=""
DATASET=""
DRY_RUN=0

usage() {
    cat <<EOF
Usage (inside the oscar container):
  bash rendering/parallel_render.sh --dataset <name> [--workers N]
  bash rendering/parallel_render.sh --object-folder <dir> --object-images <dir> [--workers N]

Options:
  --dataset <name>        gso, ycbv, housecat6d, MI3DOR, tless, itodd, lmo
                          (fills --object-folder/--object-images from the known map)
  --object-folder <dir>   CAD source dir (repo-relative). Overrides --dataset.
  --object-images <dir>   Output dir for renders (repo-relative).
  --workers N             Number of parallel Blender workers (default: $WORKERS).
  --num-views N           Views per model (default: $NUM_VIEWS).
  --overwrite             Re-render models even if outputs already exist.
  --dry-run               Print the worker commands and the shard split; launch nothing.
  --help                  Show this help.

Env overrides: BLENDER_BIN, NUM_VIEWS
EOF
    exit "${1:-1}"
}

# Known dataset -> (folder, images), mirroring onboard_dataset.sh. BOP datasets
# (tless/itodd/lmo) must have been "prepared" first so their CAD dir exists.
resolve_dataset() {
    case "$1" in
        gso)        OBJECT_FOLDER="object_database/gso";              OBJECT_IMAGES="object_images/gso" ;;
        ycbv)       OBJECT_FOLDER="object_database/ycbv";             OBJECT_IMAGES="object_images/ycbv" ;;
        housecat6d) OBJECT_FOLDER="object_database/housecat6d";       OBJECT_IMAGES="object_images/housecat6d" ;;
        MI3DOR)     OBJECT_FOLDER="object_database/MI3DOR/model/test"; OBJECT_IMAGES="object_images/MI3DOR" ;;
        tless)      OBJECT_FOLDER="object_database/tless";            OBJECT_IMAGES="object_images/tless" ;;
        itodd)      OBJECT_FOLDER="object_database/itodd";            OBJECT_IMAGES="object_images/itodd" ;;
        lmo)        OBJECT_FOLDER="object_database/lmo";              OBJECT_IMAGES="object_images/lmo" ;;
        *) echo "ERROR: unknown --dataset '$1'. Pass --object-folder + --object-images instead."; exit 1 ;;
    esac
}

# --- parse args ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)        DATASET="$2"; shift 2 ;;
        --object-folder)  OBJECT_FOLDER="$2"; shift 2 ;;
        --object-images)  OBJECT_IMAGES="$2"; shift 2 ;;
        --workers)        WORKERS="$2"; shift 2 ;;
        --num-views)      NUM_VIEWS="$2"; shift 2 ;;
        --overwrite)      OVERWRITE=1; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        --help|-h)        usage 0 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# Resolve dataset shorthand if explicit dirs were not given.
if [[ -n "$DATASET" && ( -z "$OBJECT_FOLDER" || -z "$OBJECT_IMAGES" ) ]]; then
    resolve_dataset "$DATASET"
fi
[[ -z "$OBJECT_FOLDER" || -z "$OBJECT_IMAGES" ]] && { echo "ERROR: need --dataset OR --object-folder + --object-images"; usage; }
[[ "$WORKERS" =~ ^[0-9]+$ && "$WORKERS" -ge 1 ]] || { echo "ERROR: --workers must be a positive integer"; exit 1; }

echo "[parallel_render] folder=$OBJECT_FOLDER  images=$OBJECT_IMAGES  workers=$WORKERS  views=$NUM_VIEWS  overwrite=$OVERWRITE"

# --- checks (the blender check is skipped for --dry-run so it works on the host) -
if [[ ! -d "$OBJECT_FOLDER" ]]; then
    echo "ERROR: object folder not found: $OBJECT_FOLDER"
    echo "  (for BOP datasets run the 'prepare' step first: onboard_dataset.sh --dataset $DATASET --step prepare)"
    exit 1
fi
if [[ $DRY_RUN -eq 0 && ! -x "$BLENDER_BIN" ]]; then
    echo "ERROR: Blender not found/executable at '$BLENDER_BIN'. Run this inside the oscar container, or set BLENDER_BIN."
    exit 1
fi

mkdir -p "$OBJECT_IMAGES"
LOGDIR="$OBJECT_IMAGES/.parallel_render_logs"
mkdir -p "$LOGDIR"

# --- launch workers -----------------------------------------------------------
declare -a PIDS
for (( i=0; i<WORKERS; i++ )); do
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] worker $i/$WORKERS -> renders models where index % $WORKERS == $i"
        echo "          SHARD_INDEX=$i SHARD_TOTAL=$WORKERS OBJECT_FOLDER=$OBJECT_FOLDER OBJECT_IMAGES=$OBJECT_IMAGES \\"
        echo "          NUM_VIEWS=$NUM_VIEWS OVERWRITE_EXISTING=$OVERWRITE $BLENDER_BIN -b -P rendering/rendering.py"
        continue
    fi
    OBJECT_FOLDER="$OBJECT_FOLDER" OBJECT_IMAGES="$OBJECT_IMAGES" \
    NUM_VIEWS="$NUM_VIEWS" OVERWRITE_EXISTING="$OVERWRITE" \
    SHARD_INDEX="$i" SHARD_TOTAL="$WORKERS" \
        "$BLENDER_BIN" -b -P "rendering/rendering.py" > "$LOGDIR/worker_${i}.log" 2>&1 &
    wpid=$!
    PIDS+=("$wpid")
    echo "[parallel_render] launched worker $i (pid $wpid) -> $LOGDIR/worker_${i}.log"
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] nothing launched. Each worker renders a disjoint 1/$WORKERS of the models."
    exit 0
fi

# --- wait for all, aggregate exit codes --------------------------------------
echo "[parallel_render] waiting for $WORKERS workers…"
fail=0
for idx in "${!PIDS[@]}"; do
    if wait "${PIDS[$idx]}"; then
        echo "[parallel_render] worker $idx finished OK"
    else
        rc=$?
        echo "[parallel_render] worker $idx FAILED (rc=$rc) — see $LOGDIR/worker_${idx}.log"
        fail=1
    fi
done

if [[ $fail -ne 0 ]]; then
    echo "[parallel_render] ONE OR MORE WORKERS FAILED. Re-run to resume (finished models are skipped)."
    exit 1
fi
echo "[parallel_render] ALL $WORKERS workers done. Renders in $OBJECT_IMAGES."
