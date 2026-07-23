#!/usr/bin/env bash
# =============================================================================
# rendering/onboard_and_sync.sh — WSL launcher: Docker preprocessing + rclone
# =============================================================================
#
# Orchestrates the full onboarding workflow from WSL:
#   1. Starts Docker (oscar service) to run onboard_dataset.sh
#      → Blender rendering, partial point clouds, LLaVA descriptions
#   2. Starts rclone_watch.sh in the background to sync results to Google Drive
#   3. After Docker finishes, does a final sync
#   4. Optionally deletes local rendered files to free disk space
#
# This separation exists because:
#   - Docker has Python deps (numpy, trimesh, transformers) + GPU access
#   - WSL has rclone (not installed in Docker)
#   - Docker writes to mounted volume (.:/app), WSL sees the same files
#
# Usage:
#   # Full onboarding for LM-O (small test):
#   bash rendering/onboard_and_sync.sh --dataset lmo --remote gdrive:Masterthesis/OSCAR
#
#   # SHREC'18 with cleanup (delete local after sync):
#   bash rendering/onboard_and_sync.sh --dataset shrec18 --remote gdrive:Masterthesis/OSCAR --delete-after-sync
#
#   # Skip descriptions (run separately later):
#   bash rendering/onboard_and_sync.sh --dataset tless --remote gdrive:Masterthesis/OSCAR --skip-describe
#
#   # Only render (skip partial PCs and descriptions):
#   bash rendering/onboard_and_sync.sh --dataset lmo --remote gdrive:Masterthesis/OSCAR --step render
#
#   # Dry run:
#   bash rendering/onboard_and_sync.sh --dataset lmo --remote gdrive:Masterthesis/OSCAR --dry-run
#
# Prerequisites:
#   - Docker with nvidia runtime (oscar service in docker-compose.yml)
#   - rclone configured on WSL (rclone lsd gdrive: should work)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OSCAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET=""
REMOTE=""
STEP="all"
SYNC_INTERVAL=300  # seconds between rclone syncs
DELETE_AFTER_SYNC=0
SKIP_DESCRIBE=0
DRY_RUN=0
OVERWRITE=0

usage() {
    echo "Usage: $0 --dataset <name> --remote <rclone:path> [options]"
    echo ""
    echo "Options:"
    echo "  --step <step>         Step to run: all, prepare, render, partial, describe (default: all)"
    echo "  --delete-after-sync   Delete local rendered files after final sync"
    echo "  --skip-describe       Skip LLaVA descriptions"
    echo "  --sync-interval N     Seconds between rclone syncs (default: 300)"
    echo "  --overwrite           Overwrite existing outputs"
    echo "  --dry-run             Show what would be done"
    echo ""
    echo "Datasets: ycbv_gso, ycbv, gso, MI3DOR, housecat6d, shrec18, tless, lmo, itodd"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --remote)            REMOTE="$2"; shift 2 ;;
        --step)              STEP="$2"; shift 2 ;;
        --delete-after-sync) DELETE_AFTER_SYNC=1; shift ;;
        --skip-describe)     SKIP_DESCRIBE=1; shift ;;
        --sync-interval)     SYNC_INTERVAL="$2"; shift 2 ;;
        --overwrite)         OVERWRITE=1; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --help|-h)           usage ;;
        *)                   echo "Unknown option: $1"; usage ;;
    esac
done

[[ -z "$DATASET" ]] && { echo "ERROR: --dataset required"; usage; }
[[ -z "$REMOTE" ]] && { echo "ERROR: --remote required (e.g. gdrive:Masterthesis/OSCAR)"; usage; }

command -v rclone &>/dev/null || { echo "ERROR: rclone not found on WSL. Install it first."; exit 1; }
command -v docker &>/dev/null || { echo "ERROR: docker not found."; exit 1; }

# Queue halt guard: if this file exists, refuse to START a new dataset. Used to
# stop an already-running queue loop from advancing to its next dataset without
# killing the dataset currently in flight (which is already past this check).
if [[ -f "$OSCAR_ROOT/.halt_queue" ]]; then
    echo "[halt] $OSCAR_ROOT/.halt_queue present — refusing to start dataset '$DATASET'"
    exit 1
fi

# ---------------------------------------------------------------------------
# Dataset → images dir mapping (must match onboard_dataset.sh)
# ---------------------------------------------------------------------------
case "$DATASET" in
    ycbv_gso)  IMAGES_SUBDIR="object_images/ycbv_gso" ;;
    ycbv)      IMAGES_SUBDIR="object_images/ycbv" ;;
    gso)       IMAGES_SUBDIR="object_images/gso" ;;
    MI3DOR)    IMAGES_SUBDIR="object_images/MI3DOR" ;;
    housecat6d) IMAGES_SUBDIR="object_images/housecat6d" ;;
    shrec18)   IMAGES_SUBDIR="object_images/shrec18" ;;
    shrec18_fixed) IMAGES_SUBDIR="object_images/shrec18_fixed" ;;
    tless)     IMAGES_SUBDIR="object_images/tless" ;;
    lmo)       IMAGES_SUBDIR="object_images/lmo" ;;
    itodd)     IMAGES_SUBDIR="object_images/itodd" ;;
    *)         echo "ERROR: Unknown dataset '$DATASET'"; exit 1 ;;
esac

IMAGES_DIR="$OSCAR_ROOT/$IMAGES_SUBDIR"
DB_DIR="$OSCAR_ROOT/object_database/$DATASET"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Build the Docker command
# ---------------------------------------------------------------------------
DOCKER_STEP="$STEP"
if [[ "$SKIP_DESCRIBE" -eq 1 ]] && [[ "$STEP" == "all" ]]; then
    # Run prepare+render+partial, skip describe
    # onboard_dataset.sh doesn't have a "no describe" flag, so we run steps individually
    DOCKER_STEP="no-describe"
fi

DOCKER_ARGS=""
[[ "$OVERWRITE" -eq 1 ]] && DOCKER_ARGS="$DOCKER_ARGS --overwrite"

log "=== Onboard & Sync: $DATASET ==="
log "  Step:              $STEP"
log "  Remote:            $REMOTE"
log "  Sync interval:     ${SYNC_INTERVAL}s"
log "  Delete after sync: $DELETE_AFTER_SYNC"
log "  Skip describe:     $SKIP_DESCRIBE"
[[ $DRY_RUN -eq 1 ]] && log "  *** DRY RUN ***"
echo ""

# ---------------------------------------------------------------------------
# 1. Start rclone_watch.sh in background (syncs periodically from WSL)
# ---------------------------------------------------------------------------
RCLONE_PID=""
if [[ $DRY_RUN -eq 0 ]]; then
    log "Starting rclone_watch.sh in background (interval: ${SYNC_INTERVAL}s)..."
    bash "$SCRIPT_DIR/rclone_watch.sh" \
        --dataset "$DATASET" \
        --remote "$REMOTE" \
        --interval "$SYNC_INTERVAL" &
    RCLONE_PID=$!
    log "rclone_watch PID: $RCLONE_PID"
else
    log "[DRY-RUN] Would start rclone_watch.sh --dataset $DATASET --remote $REMOTE --interval $SYNC_INTERVAL"
fi

# Cleanup: kill rclone_watch on exit
cleanup() {
    if [[ -n "$RCLONE_PID" ]] && kill -0 "$RCLONE_PID" 2>/dev/null; then
        log "Stopping rclone_watch (PID $RCLONE_PID)..."
        kill "$RCLONE_PID" 2>/dev/null || true
        wait "$RCLONE_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Run onboard_dataset.sh inside Docker
# ---------------------------------------------------------------------------
log "Starting Docker preprocessing..."

if [[ "$DOCKER_STEP" == "no-describe" ]]; then
    # Run prepare, render, partial as separate steps (skip describe)
    STEPS_TO_RUN="prepare render partial"
else
    STEPS_TO_RUN="$DOCKER_STEP"
fi

for step in $STEPS_TO_RUN; do
    log "Running step: $step"
    if [[ $DRY_RUN -eq 0 ]]; then
        docker compose -f "$OSCAR_ROOT/docker-compose.yml" run --rm \
            oscar \
            bash -c "cd /app && bash rendering/onboard_dataset.sh --dataset $DATASET --step $step $DOCKER_ARGS" \
            2>&1
    else
        log "[DRY-RUN] docker compose run oscar bash -c 'cd /app && bash rendering/onboard_dataset.sh --dataset $DATASET --step $step $DOCKER_ARGS'"
    fi
done

log "Docker preprocessing finished."

# ---------------------------------------------------------------------------
# 3. Final rclone sync (catch anything rclone_watch might have missed)
# ---------------------------------------------------------------------------
log "Running final rclone sync..."
if [[ $DRY_RUN -eq 0 ]]; then
    # Sync images
    if [[ -d "$IMAGES_DIR" ]]; then
        rclone copy "$IMAGES_DIR" "$REMOTE/$IMAGES_SUBDIR" \
            --transfers 32 --checkers 32 \
            --stats-one-line --stats 5s \
            --log-level NOTICE 2>&1 || true
    fi

    # Sync object_database (descriptions, metadata)
    if [[ -d "$DB_DIR" ]]; then
        rclone copy "$DB_DIR" "$REMOTE/object_database/$DATASET" \
            --transfers 32 --checkers 32 \
            --stats-one-line --stats 0 \
            --log-level NOTICE 2>&1 || true
    fi
    log "Final sync complete."
else
    log "[DRY-RUN] rclone copy $IMAGES_DIR → $REMOTE/$IMAGES_SUBDIR"
    log "[DRY-RUN] rclone copy $DB_DIR → $REMOTE/object_database/$DATASET"
fi

# ---------------------------------------------------------------------------
# 4. Optional: delete local rendered files (keep CAD source)
# ---------------------------------------------------------------------------
if [[ $DELETE_AFTER_SYNC -eq 1 ]]; then
    log "Deleting local rendered files: $IMAGES_DIR"
    if [[ $DRY_RUN -eq 0 ]]; then
        if [[ -d "$IMAGES_DIR" ]]; then
            rm -rf "$IMAGES_DIR"
            log "Deleted $IMAGES_DIR"
        fi
    else
        log "[DRY-RUN] rm -rf $IMAGES_DIR"
    fi
fi

log "=== All done: $DATASET ==="
log "Results on: $REMOTE"
