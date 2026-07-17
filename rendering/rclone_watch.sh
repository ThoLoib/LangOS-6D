#!/usr/bin/env bash
# =============================================================================
# rendering/rclone_watch.sh — Background sync watcher for onboarding
# =============================================================================
#
# Run this in a separate WSL terminal while Docker renders inside the container.
# It watches the object_images/ directory and syncs to Google Drive periodically.
#
# Usage:
#   bash rendering/rclone_watch.sh --dataset lmo --remote gdrive:Masterthesis/OSCAR
#   bash rendering/rclone_watch.sh --dataset ycbv_gso --remote gdrive:Masterthesis/OSCAR --interval 600
#
# The script syncs both object_images/{dataset}/ and object_database/{dataset}/
# every --interval seconds (default: 300 = 5 minutes).
# It exits automatically when no new files appear for 2 consecutive intervals.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OSCAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET=""
REMOTE=""
INTERVAL=300  # seconds between syncs

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)   DATASET="$2"; shift 2 ;;
        --remote)    REMOTE="$2"; shift 2 ;;
        --interval)  INTERVAL="$2"; shift 2 ;;
        --help|-h)   echo "Usage: $0 --dataset <name> --remote <rclone:path> [--interval <secs>]"; exit 0 ;;
        *)           echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -z "$DATASET" ]] && { echo "ERROR: --dataset required"; exit 1; }
[[ -z "$REMOTE" ]] && { echo "ERROR: --remote required (e.g. gdrive:Masterthesis/OSCAR)"; exit 1; }

if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found. Install it first."
    exit 1
fi

IMAGES_DIR="$OSCAR_ROOT/object_images/$DATASET"
DB_DIR="$OSCAR_ROOT/object_database/$DATASET"

echo "[rclone-watch] Dataset:    $DATASET"
echo "[rclone-watch] Images dir: $IMAGES_DIR"
echo "[rclone-watch] DB dir:     $DB_DIR"
echo "[rclone-watch] Remote:     $REMOTE"
echo "[rclone-watch] Interval:   ${INTERVAL}s"
echo "[rclone-watch] Press Ctrl+C to stop"
echo ""

prev_count=0
idle_rounds=0

while true; do
    # Count total output files: PNGs (renders) + NPZs (partial PCs) + descriptions JSON
    cur_count=0
    if [[ -d "$IMAGES_DIR" ]]; then
        png_count=$(find "$IMAGES_DIR" -maxdepth 2 -name "*.png" | wc -l)
        npz_count=$(find "$IMAGES_DIR" -maxdepth 2 -name "*.npz" | wc -l)
        cur_count=$((png_count + npz_count))
    fi
    # Also count descriptions file (changes size as objects are added)
    if [[ -d "$DB_DIR" ]]; then
        desc_size=$(find "$DB_DIR" -name "descriptions_attributes.json" -printf '%s' 2>/dev/null || echo 0)
        cur_count=$((cur_count + ${desc_size:-0}))
    fi

    new=$((cur_count - prev_count))
    timestamp=$(date '+%H:%M:%S')

    if [[ $cur_count -gt 0 ]]; then
        echo "[$timestamp] ${png_count:-0} PNGs, ${npz_count:-0} NPZs (+$new new). Syncing..."

        # Sync images
        rclone copy "$IMAGES_DIR" "$REMOTE/object_images/$DATASET" \
            --transfers 8 --checkers 16 \
            --stats-one-line --stats 0 \
            2>&1 | grep -v "^$" | tail -1 || true

        # Sync object_database (descriptions JSON, CAD metadata)
        if [[ -d "$DB_DIR" ]]; then
            # Use --checksum so the growing descriptions JSON always gets
            # re-uploaded even if mtime doesn't change (Docker volume quirk)
            desc_file="$DB_DIR/descriptions_attributes.json"
            if [[ -f "$desc_file" ]]; then
                local_size=$(stat -c%s "$desc_file" 2>/dev/null || echo 0)
                echo "[$timestamp]   descriptions: $(( local_size / 1024 )) KB"
            fi
            rclone copy "$DB_DIR" "$REMOTE/object_database/$DATASET" \
                --transfers 4 --checkers 8 --checksum \
                --stats-one-line --stats 0 \
                2>&1 | grep -v "^$" | tail -1 || true
        fi

        echo "[$timestamp] Sync done."
    else
        echo "[$timestamp] No rendered objects yet. Waiting..."
    fi

    # Auto-exit after 2 idle rounds (no new objects)
    if [[ $new -eq 0 ]] && [[ $cur_count -gt 0 ]]; then
        idle_rounds=$((idle_rounds + 1))
        if [[ $idle_rounds -ge 2 ]]; then
            echo "[$timestamp] No new objects for $((INTERVAL * 2))s. Final sync and exit."
            # One last sync to be sure
            rclone copy "$IMAGES_DIR" "$REMOTE/object_images/$DATASET" \
                --transfers 8 --checkers 16 2>/dev/null || true
            [[ -d "$DB_DIR" ]] && rclone copy "$DB_DIR" "$REMOTE/object_database/$DATASET" \
                --transfers 4 --checkers 8 2>/dev/null || true
            echo "[$timestamp] All synced. Exiting."
            exit 0
        fi
    else
        idle_rounds=0
    fi

    prev_count=$cur_count
    sleep "$INTERVAL"
done
