#!/usr/bin/env python3
# =============================================================================
# tools/precompute_gedi_descriptors.py
# =============================================================================
# Precompute + cache GeDi local geometric descriptors (Poiesi & Boscaini,
# IEEE T-PAMI 2022) for a dataset's GALLERY (full CAD meshes) and its QUERIES
# (partial observation point clouds), so the Sub-step B2 geometry re-ranking
# does not recompute them at eval time.
#
# Runs INSIDE the oscar container and calls the GeDi HTTP service (the `gedi`
# docker-compose service, default http://gedi:5060). Start it first with:
#     docker compose up -d gedi
#
# Gallery: each full CAD mesh -> read_triangle_mesh + sample_points_uniformly(N)
#          -> GeDi.  N and the sampling match pipeline/step_b2 _load_cad_pointcloud
#          (10000 points, native mesh scale, no normalization) so the cached
#          descriptors are identical to what B2 would compute on the fly.
# Queries: each <id>.npz (key 'points', an (M,3) float32 partial cloud) -> GeDi.
#
# Descriptors are written per object as .npz {keypoints, descriptors} via
# GeDiDescriptorModule.compute_and_cache — resumable: an existing cache file is
# skipped (unless --overwrite).
#
# Typical use (inside the oscar container):
#   python3 tools/precompute_gedi_descriptors.py \
#       --mesh-glob 'eval/datasets/shrec18/shrec18_full/cad/*.obj' \
#       --queries-dir eval/datasets/shrec18/stage1/queries \
#       --gallery-cache-dir object_database/shrec18_v2/gedi_gallery \
#       --queries-cache-dir eval/datasets/shrec18/stage1/query_gedi_cache
# =============================================================================

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

# Make the `pipeline` package importable when run as tools/<this>.py from /app.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _log(msg):
    print(f"[gedi-precompute {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_open3d_loadable():
    """open3d needs libgomp.so.1, which is not installed system-wide in the
    oscar image but IS bundled with torch. Preload it so `import open3d`
    succeeds regardless of LD_LIBRARY_PATH."""
    try:
        import open3d  # noqa: F401 -- already loadable
        return
    except OSError:
        pass
    import ctypes
    for cand in (glob.glob("/usr/local/lib/python*/dist-packages/torch/lib/libgomp.so*")
                 + glob.glob("/usr/lib/*/libgomp.so*")):
        try:
            ctypes.CDLL(cand, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


def _model_id(path):
    """Object id from a mesh path (SHREC: the filename stem is the id)."""
    return os.path.splitext(os.path.basename(path))[0]


def _load_query_points(npz_path):
    """Load an (M,3) float32 point cloud from a query .npz (key 'points')."""
    d = np.load(npz_path)
    if "points" in d:
        return np.asarray(d["points"], dtype=np.float32)
    # fall back to the first (M,3) array present
    for k in d.files:
        a = np.asarray(d[k])
        if a.ndim == 2 and a.shape[1] == 3:
            return a.astype(np.float32)
    return None


def _run_group(name, items, cache_dir, make_pcd, gedi, num_keypoints,
               overwrite, o3d):
    """Compute+cache GeDi descriptors for a list of (obj_id, source) items.

    make_pcd(source) -> open3d.geometry.PointCloud (or None on load failure).
    Returns (n_ok, n_fail, n_skip).
    """
    os.makedirs(cache_dir, exist_ok=True)
    n_ok = n_fail = n_skip = 0
    total = len(items)
    _log(f"{name}: {total} objects -> {cache_dir}")
    for i, (obj_id, source) in enumerate(items, 1):
        cache_path = os.path.join(cache_dir, f"{obj_id}.npz")
        if not overwrite and os.path.isfile(cache_path):
            n_skip += 1
            continue
        pcd = make_pcd(source)
        if pcd is None or len(pcd.points) < 100:
            npts = 0 if pcd is None else len(pcd.points)
            _log(f"  ! {obj_id}: unusable point cloud ({npts} pts) — skipped")
            n_fail += 1
            continue
        # GeDi samples keypoints from the cloud; never ask for more than we have.
        n_kp = min(num_keypoints, len(pcd.points))
        res = gedi.compute_and_cache(pcd, cache_path, num_keypoints=n_kp,
                                     force=overwrite)
        if res.descriptors_np.size > 0:
            n_ok += 1
        else:
            _log(f"  ! {obj_id}: GeDi returned no descriptors")
            n_fail += 1
        if i % 100 == 0 or i == total:
            _log(f"  {name}: {i}/{total}  (ok={n_ok} fail={n_fail} skip={n_skip})")
    return n_ok, n_fail, n_skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh-glob", default="",
                    help="Glob for gallery full CAD meshes (e.g. '.../cad/*.obj').")
    ap.add_argument("--queries-dir", default="",
                    help="Directory of query <id>.npz partial point clouds.")
    ap.add_argument("--gallery-cache-dir", default="",
                    help="Output dir for gallery descriptor caches (.npz).")
    ap.add_argument("--queries-cache-dir", default="",
                    help="Output dir for query descriptor caches (.npz).")
    ap.add_argument("--num-points", type=int, default=10000,
                    help="Points sampled per CAD mesh (matches step_b2, default 10000).")
    ap.add_argument("--num-keypoints", type=int, default=0,
                    help="GeDi keypoints per cloud (0 = config default, capped at #points).")
    ap.add_argument("--gedi-url", default="",
                    help="Override GeDi service URL (default from config: http://gedi:5060).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N of each group (for testing).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if a cache file exists.")
    ap.add_argument("--skip-gallery", action="store_true")
    ap.add_argument("--skip-queries", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Enumerate + load/sample the first item of each group and "
                         "check GeDi health, but do NOT call GeDi (no GPU).")
    args = ap.parse_args(argv)

    # -- SUPERSEDED (2026-08-01) ------------------------------------------------
    # The Stage-1 eval (experiments/experiment1_shrec18_stage1.py --precompute-gedi)
    # now computes GeDi on UNIT-SPHERE-normalized + voxel-downsampled clouds and
    # caches them WITH a provenance fingerprint under
    # results_root/_cache/gedi_descriptors, which step_b2 consumes. This tool
    # samples native-scale clouds with NO normalization (it matched the OLD
    # step_b2 path), so its descriptors describe different points than the eval
    # registers — reusing them would corrupt geometry scores, not just miss the
    # cache. Computing them here would waste ~3-5h on unusable output, so this
    # tool no-ops by default and defers to the eval as the single GeDi source.
    # Set GEDI_PRECOMPUTE_FORCE=1 to run the original standalone behaviour.
    if os.environ.get("GEDI_PRECOMPUTE_FORCE", "0") != "1":
        _log("SUPERSEDED by experiment1_shrec18_stage1.py --precompute-gedi "
             "(unit-sphere clouds + provenance-fingerprinted cache). Skipping to "
             "avoid ~3-5h of unusable native-scale descriptors — the eval is now "
             "the single GeDi source. Set GEDI_PRECOMPUTE_FORCE=1 to override.")
        return 0

    _ensure_open3d_loadable()
    import open3d as o3d
    from pipeline.config import PipelineConfig
    from pipeline.gedi_descriptors import GeDiDescriptorModule

    cfg = PipelineConfig()
    if args.gedi_url:
        cfg.gedi_url = args.gedi_url
    num_keypoints = args.num_keypoints or cfg.gedi_num_keypoints

    gedi = GeDiDescriptorModule(cfg)
    _log(f"GeDi service: {cfg.gedi_url}  |  num_keypoints={num_keypoints}  "
         f"mesh_points={args.num_points}")
    if not gedi.available:
        _log("ERROR: GeDi service is NOT reachable. Start it: docker compose up -d gedi")
        return 2
    _log("GeDi service health: OK")

    # --- assemble work items --------------------------------------------------
    gallery_items = []
    if not args.skip_gallery and args.mesh_glob:
        meshes = sorted(glob.glob(args.mesh_glob))
        gallery_items = [(_model_id(p), p) for p in meshes]
        if args.limit:
            gallery_items = gallery_items[:args.limit]

    query_items = []
    if not args.skip_queries and args.queries_dir:
        npzs = sorted(glob.glob(os.path.join(args.queries_dir, "*.npz")))
        query_items = [(os.path.splitext(os.path.basename(p))[0], p) for p in npzs]
        if args.limit:
            query_items = query_items[:args.limit]

    _log(f"gallery meshes: {len(gallery_items)}  |  queries: {len(query_items)}")

    def mesh_to_pcd(path):
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) == 0:
            return None
        return mesh.sample_points_uniformly(number_of_points=args.num_points)

    def query_to_pcd(npz_path):
        pts = _load_query_points(npz_path)
        if pts is None or len(pts) == 0:
            return None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        return pcd

    # --- dry run: validate loading/sampling without touching the GPU ---------
    if args.dry_run:
        if gallery_items:
            gid, gp = gallery_items[0]
            p = mesh_to_pcd(gp)
            _log(f"[dry-run] gallery '{gid}': sampled {0 if p is None else len(p.points)} pts "
                 f"from {gp}")
        if query_items:
            qid, qp = query_items[0]
            p = query_to_pcd(qp)
            _log(f"[dry-run] query '{qid}': loaded {0 if p is None else len(p.points)} pts "
                 f"from {qp}")
        _log("[dry-run] health OK + loading/sampling OK — no GeDi calls made. Exiting.")
        return 0

    # --- compute --------------------------------------------------------------
    g_ok = g_fail = g_skip = 0
    q_ok = q_fail = q_skip = 0
    if gallery_items:
        g_ok, g_fail, g_skip = _run_group(
            "gallery", gallery_items, args.gallery_cache_dir, mesh_to_pcd,
            gedi, num_keypoints, args.overwrite, o3d)
    if query_items:
        q_ok, q_fail, q_skip = _run_group(
            "queries", query_items, args.queries_cache_dir, query_to_pcd,
            gedi, num_keypoints, args.overwrite, o3d)

    # --- manifest + summary ---------------------------------------------------
    summary = {
        "gedi_url": cfg.gedi_url, "num_keypoints": num_keypoints,
        "mesh_points": args.num_points,
        "gallery": {"ok": g_ok, "fail": g_fail, "skip": g_skip,
                    "cache_dir": args.gallery_cache_dir},
        "queries": {"ok": q_ok, "fail": q_fail, "skip": q_skip,
                    "cache_dir": args.queries_cache_dir},
    }
    for cd in (args.gallery_cache_dir, args.queries_cache_dir):
        if cd:
            try:
                os.makedirs(cd, exist_ok=True)
                with open(os.path.join(cd, "gedi_precompute_manifest.json"), "w") as f:
                    json.dump(summary, f, indent=2)
            except OSError:
                pass
    _log(f"DONE. gallery ok={g_ok} fail={g_fail} skip={g_skip} | "
         f"queries ok={q_ok} fail={q_fail} skip={q_skip}")

    # "It must work": hard-fail if the service produced nothing at all (the run
    # is broken), but tolerate a few individually-degenerate clouds.
    produced = g_ok + q_ok + g_skip + q_skip
    if produced == 0:
        _log("ERROR: no descriptors produced or cached — treating as failure.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
