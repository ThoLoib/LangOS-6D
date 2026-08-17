#!/usr/bin/env bash
# =============================================================================
# preprocess_gallery.sh — one-command gallery preprocessing for a dataset
# =============================================================================
#
# Runs the FULL gallery pipeline for a single dataset, end to end:
#
#     1. onboard   render 42 views  +  partial point clouds (HPR)  +  VLM
#                  descriptions, and sync the renders to Google Drive
#     2. embed     compute the gallery embeddings (CLIP-text, DINOv2 / SigLIP,
#                  ULIP-2, Uni3D, ...) — one cache per "pass"
#     3. sync      push the embedding caches to Google Drive
#     4. verify    rclone-check that everything is really on Drive
#     5. (optional) delete the local renders to free disk (only after verify)
#
# This is the plain, human-runnable version of the pipeline: you run it in a
# terminal, it prints each step, and it STOPS with a clear message the moment
# something fails. There is no systemd, no background daemon, no phone
# notification, and no auto-recovery magic — on failure you read the message,
# fix it, and re-run (every step is safe to re-run; finished work is cached).
#
# -----------------------------------------------------------------------------
# QUICK START
# -----------------------------------------------------------------------------
#   # Full run, keep local renders (needed if you will evaluate on this machine):
#   bash rendering/preprocess_gallery.sh --dataset shrec18_v2
#
#   # Full run, then delete local renders once verified on Drive (save disk):
#   bash rendering/preprocess_gallery.sh --dataset MI3DOR --delete-after-sync
#
#   # Only the appearance passes, no shape encoders:
#   bash rendering/preprocess_gallery.sh --dataset ycbv --passes base,siglip
#
# -----------------------------------------------------------------------------
# PREREQUISITES
# -----------------------------------------------------------------------------
#   * Docker with the NVIDIA runtime, and the `oscar` service defined in
#     docker-compose.yml (this is where Blender / PyTorch / the encoders live).
#   * rclone configured with a remote that can reach the Drive folder
#     (default remote below). Test with:  rclone lsd gdrive:Masterthesis/OSCAR
#     If your rclone binary is not on PATH, point RCLONE at it, e.g.:
#         RCLONE=/home/me/apps/rclone/rclone bash rendering/preprocess_gallery.sh ...
#   * A GPU with enough VRAM for Blender Cycles + the encoders (tested on a
#     24 GB RTX 4090). Tip: rendering a big dataset fights the desktop GPU
#     session — if the machine locks up, run the render on a headless target
#     (e.g. `systemctl isolate multi-user.target`).
#
# -----------------------------------------------------------------------------
# WHERE THE PER-DATASET SETTINGS LIVE
# -----------------------------------------------------------------------------
#   The CAD source directory, the mesh glob, and the partial-point-cloud knobs
#   (HPR radius param + jitter) are defined PER DATASET in the `case` block of
#   rendering/onboard_dataset.sh. To onboard a brand-new dataset you add one
#   case there (and, if you want the ulip_fullmesh pass, one line in the
#   MESH_GLOBS table below so this script can find its meshes).
#
#   Partial-PC knobs currently in onboard_dataset.sh:
#       default          HPR param 2.8, jitter 0.001 (corrected protocol — every
#                        current dataset: ycbv, tless, lmo, itodd, gso,
#                        housecat6d, MI3DOR, shrec18_v2)
#       legacy slots     HPR param 3.2, no jitter    (shrec18, shrec18_fixed,
#                        ycbv_gso — pinned for archived before/after comparison)
# =============================================================================

set -uo pipefail

# --- repo root (this script lives in <repo>/rendering/) -----------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OSCAR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$OSCAR_ROOT" || { echo "ERROR: cannot cd to repo root $OSCAR_ROOT"; exit 3; }

# --- defaults (override on the command line or via env) -----------------------
REMOTE="${REMOTE:-gdrive:Masterthesis/OSCAR}"   # rclone destination
RCLONE="${RCLONE:-rclone}"                       # rclone binary (PATH or full path)
PASSES="base,siglip,ulip_pc_rgb,ulip_pc_xyz,uni3d,ulip_fullmesh"  # full ablation set
DATASET=""
MESH_GLOB=""            # auto-filled for known datasets (see table below)
DELETE_AFTER_SYNC=0     # keep local renders by default (eval usually needs them)
SKIP_EMBED=0            # set to only render/partial/describe/sync, no embeddings

usage() {
    cat <<EOF
Usage: bash rendering/preprocess_gallery.sh --dataset <name> [options]

Options:
  --dataset <name>       REQUIRED. e.g. shrec18_v2, MI3DOR, ycbv, gso, housecat6d, tless, itodd
  --remote <rclone:path> Drive destination (default: $REMOTE)
  --passes <list>        Comma-separated embedding passes (default: all six)
                           base ulip_pc_rgb ulip_pc_xyz uni3d siglip ulip_fullmesh
                           (see: docker compose run --rm oscar \\
                                 python3 tools/precompute_embeddings.py --list)
  --mesh-glob <glob>     CAD mesh glob for the ulip_fullmesh pass. Auto-detected
                         for known datasets; required for unknown ones if that
                         pass is included.
  --delete-after-sync    Delete local renders AFTER they are verified on Drive
                         (frees disk; do NOT use if you evaluate on this machine).
  --skip-embed           Stop after onboard+sync (no embeddings).
  --help                 Show this help.

Environment overrides: REMOTE, RCLONE, PASSES
EOF
    exit "${1:-1}"
}

# --- CAD mesh globs for the ulip_fullmesh pass (mirror onboard_dataset.sh) -----
# Only datasets whose full-mesh glob is known are listed. Paths are relative to
# the repo root (they run inside the container at /app). Add a line here when
# you add a dataset in onboard_dataset.sh.
mesh_glob_for() {
    case "$1" in
        shrec18|shrec18_fixed|shrec18_v2) echo "eval/datasets/shrec18/shrec18_full/cad/*.obj" ;;
        MI3DOR)      echo "object_database/MI3DOR/model/test/*/*.obj" ;;
        housecat6d)  echo "object_database/housecat6d/*/*.obj" ;;
        *)           echo "" ;;   # unknown → caller must pass --mesh-glob
    esac
}

# --- parse args ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)           DATASET="$2"; shift 2 ;;
        --remote)            REMOTE="$2"; shift 2 ;;
        --passes)            PASSES="$2"; shift 2 ;;
        --mesh-glob)         MESH_GLOB="$2"; shift 2 ;;
        --delete-after-sync) DELETE_AFTER_SYNC=1; shift ;;
        --skip-embed)        SKIP_EMBED=1; shift ;;
        --help|-h)           usage 0 ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done
[[ -z "$DATASET" ]] && { echo "ERROR: --dataset is required"; usage; }

# --- logging + fail helpers ---------------------------------------------------
ts()   { date '+%F %H:%M:%S'; }
log()  { echo "[$(ts)] [$DATASET] $*"; }
fail() { echo "[$(ts)] [$DATASET] ERROR: $1" >&2; exit 1; }

# --- prerequisite checks (fail early, with a clear reason) --------------------
command -v docker >/dev/null 2>&1 || fail "docker not found on PATH."
command -v "$RCLONE" >/dev/null 2>&1 || fail "rclone not found ('$RCLONE'). Install it or set RCLONE=/path/to/rclone."

# Resolve the mesh glob only if the fullmesh pass is actually requested.
if [[ ",$PASSES," == *",ulip_fullmesh,"* ]]; then
    [[ -z "$MESH_GLOB" ]] && MESH_GLOB="$(mesh_glob_for "$DATASET")"
    [[ -z "$MESH_GLOB" ]] && fail "ulip_fullmesh is in --passes but no mesh glob is known for '$DATASET'. Pass --mesh-glob '<repo-relative>/*.obj' (see onboard_dataset.sh)."
fi

IMAGES_DIR="object_images/$DATASET"
DB_DIR="object_database/$DATASET"

log "=================== gallery preprocess: START ==================="
log "remote=$REMOTE  passes=$PASSES  delete_after_sync=$DELETE_AFTER_SYNC"
[[ -n "$MESH_GLOB" ]] && log "fullmesh mesh-glob=$MESH_GLOB"

# =============================================================================
# 1. ONBOARD — render + partial point clouds + descriptions, sync to Drive
#    (onboard_and_sync.sh runs onboard_dataset.sh inside Docker and starts a
#     background rclone sync of the renders. We do NOT pass --delete-after-sync
#     here because the embedding step still needs the local renders/partials.)
# =============================================================================
log "[1/4] onboarding (render + partial PCs + descriptions + sync)…"
bash rendering/onboard_and_sync.sh --dataset "$DATASET" --remote "$REMOTE" \
    || fail "onboarding failed (render / partial / describe). See the output above."

if [[ "$SKIP_EMBED" -eq 1 ]]; then
    log "--skip-embed set: stopping after onboard+sync."
    log "=================== gallery preprocess: DONE (no embeddings) ==================="
    exit 0
fi

# =============================================================================
# 2. EMBED — build the gallery embedding caches (one per pass)
#    Runs in the oscar container so it has the encoders + GPU. precompute
#    exits non-zero if ANY pass fails, so a partial gallery is never silently
#    declared "done".
# =============================================================================
log "[2/4] building embeddings ($PASSES)…"
MESH_ARG=""
[[ -n "$MESH_GLOB" ]] && MESH_ARG="--mesh-glob '$MESH_GLOB'"
docker compose run --rm oscar bash -lc "
    cd /app && python3 tools/precompute_embeddings.py \
        --dataset $DATASET \
        $MESH_ARG \
        --images-dir object_images/$DATASET \
        --desc-file object_database/$DATASET/descriptions_attributes.json \
        --results-root object_retrieval/results_${DATASET}_stage1 \
        --passes $PASSES" \
    || fail "embedding precompute failed. See the output above (per-pass errors are listed)."

# =============================================================================
# 3. SYNC — push the embedding caches to Drive (renders already went up in 1)
# =============================================================================
log "[3/4] syncing embedding caches to Drive…"
"$RCLONE" copy "$IMAGES_DIR" "$REMOTE/object_images/$DATASET" \
    --include "*cache*.pt" --include "precompute_manifest.json" \
    --transfers 8 --checkers 8 --retries 10 --low-level-retries 20 \
    --retries-sleep 15s --stats-one-line --stats 60s \
    || fail "cache sync (object_images) failed."
if [[ -d "$DB_DIR" ]]; then
    "$RCLONE" copy "$DB_DIR" "$REMOTE/object_database/$DATASET" \
        --include ".clip_text_cache_*.pt" --include "**/.ulip_cache_*.pt" \
        --include "descriptions_attributes.json" \
        --transfers 4 --checkers 4 --retries 10 --low-level-retries 20 \
        --retries-sleep 15s --stats-one-line --stats 60s \
        || fail "cache sync (object_database) failed."
fi

# =============================================================================
# 4. VERIFY — confirm every local render + cache is on Drive (one-way check).
#    --one-way: flags anything present locally but missing on the remote.
#    pipefail makes the check's exit code survive the `| tail`.
# =============================================================================
log "[4/4] verifying on Drive (one-way check)…"
"$RCLONE" check "$IMAGES_DIR" "$REMOTE/object_images/$DATASET" --one-way 2>&1 | tail -5 \
    || fail "Drive verify failed — some local files are NOT on Drive. Local kept; re-run to re-push."

log "verified: all of $IMAGES_DIR is on Drive."

# =============================================================================
# 5. OPTIONAL — delete local renders (only reached after a clean verify)
# =============================================================================
if [[ "$DELETE_AFTER_SYNC" -eq 1 ]]; then
    log "deleting local renders ($IMAGES_DIR) — verified on Drive."
    rm -rf "$IMAGES_DIR"
    log "local renders deleted."
fi

log "=================== gallery preprocess: DONE ==================="
log "results on: $REMOTE/object_images/$DATASET"
