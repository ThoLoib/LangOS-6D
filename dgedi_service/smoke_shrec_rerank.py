#!/usr/bin/env python3
"""Smoke test: dGeDi self-retrieval on the SHREC'18 gallery.

Runs the EXACT production rerank path (server.py internals, dGeDi repo config:
6000 kp / 10k RANSAC / +ICP) on a query cloud sampled from one gallery object's
own CAD. That object must come back as its own best geometric match — a clean
end-to-end check that the Stage-1 dGeDi integration works on SHREC data, with no
HTTP service and without touching the running dgedi service.
"""
import os
import sys
import json

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)                      # server, precompute_gallery
import numpy as np                             # noqa: E402
import server                                  # noqa: E402  (adds /dgedi to path itself)
from precompute_gallery import sample_cloud    # noqa: E402

CACHE = "/oscar/object_retrieval/.dgedi_gallery_shrec_smoke"
CFG = "/dgedi/config_dgedi.yaml"
DEV = "cuda"
# dGeDi repo config
RTH, TRIM, MAXIT, NKP, USE_ICP, ICP_THR = 0.03, 0.1, 10000, 6000, True, 0.05


def main():
    server._STATE["device"] = DEV
    server._STATE["model"] = server.load_model(CFG, "multi_scale", DEV)
    server._STATE["gallery"] = server.load_gallery(CACHE)
    server._STATE["diam"] = json.load(open(os.path.join(CACHE, "diameters.json")))
    manifest = json.load(open(os.path.join(CACHE, "manifest.json")))
    ids = sorted(server._STATE["gallery"].keys())
    print(f"[smoke] gallery={len(ids)} clouds, diameters={len(server._STATE['diam'])}")

    qid = ids[0]
    qpts = sample_cloud(os.path.join("/oscar", manifest[qid]), 10000).astype(np.float32)
    q_center = server.fps_center(qpts)
    print(f"[smoke] query = {qid} (self-match should rank #1)")

    results = {}
    for cid in ids:
        pcd_t, feats_t = server._STATE["gallery"][cid]
        diam = server._STATE["diam"].get(cid)
        if not diam:
            results[cid] = {"ok": False}
            continue
        try:
            q_norm = q_center / float(diam)
            pcd_q = server._cloud(q_norm)
            feats_q = server.extract_features(pcd_q, server._STATE["model"], server._STATE["device"])
            kp_q, kf_q = server._keypoints(pcd_q, feats_q, NKP)
            kp_t, kf_t = server._keypoints(pcd_t, feats_t, NKP)
            res = server.ransac_only(kp_q, kf_q, kp_t, kf_t, RTH, max_iter=MAXIT)
            if len(res.correspondence_set) == 0:
                results[cid] = {"ok": False}
                continue
            import open3d as o3d
            T = np.asarray(res.transformation)
            if USE_ICP:
                icp = o3d.pipelines.registration.registration_icp(
                    pcd_q, pcd_t, ICP_THR, T,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=2000))
                T = np.asarray(icp.transformation)
            q_aln = q_norm @ T[:3, :3].T + T[:3, 3]
            d_ransac = server.trimmed_chamfer(q_aln, np.asarray(pcd_t.points), TRIM)
            results[cid] = {"ok": True, "ransac_fitness": float(res.fitness),
                            "d_ransac": d_ransac}
        except Exception as exc:
            results[cid] = {"ok": False, "error": str(exc)}

    # Borda mean-rank (fitness desc, d_ransac asc) — mirrors eval_bop_pose._geo_rerank
    ok_ids = [c for c in ids if results[c].get("ok")]
    NEG = float("-inf")
    fit = np.array([results[c]["ransac_fitness"] if results[c].get("ok") else NEG for c in ids])
    dst = np.array([-results[c]["d_ransac"] if results[c].get("ok") else NEG for c in ids])

    def ranks(v):
        return np.argsort(np.argsort(-v, kind="stable"), kind="stable")
    borda = (ranks(fit) + ranks(dst)) / 2.0
    order = [ids[i] for i in np.argsort(borda, kind="stable")]

    print(f"[smoke] ok candidates: {len(ok_ids)}/{len(ids)}")
    print(f"[smoke] self fitness={results[qid].get('ransac_fitness'):.3f} "
          f"d_ransac={results[qid].get('d_ransac'):.4f}")
    print("[smoke] top-5 by Borda:")
    for r, c in enumerate(order[:5]):
        tag = " <-- SELF" if c == qid else ""
        print(f"    {r+1}. {c}  fit={results[c].get('ransac_fitness',0):.3f} "
              f"d={results[c].get('d_ransac',9):.4f}{tag}")
    self_rank = order.index(qid) + 1
    print(f"[smoke] SELF RANK = {self_rank}/{len(ids)}")
    assert results[qid].get("ok"), "self-match failed to register!"
    assert self_rank == 1, f"self did not rank #1 (rank {self_rank})"
    print("[smoke] PASS — dGeDi self-retrieval on SHREC works end-to-end.")


if __name__ == "__main__":
    main()
