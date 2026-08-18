#!/usr/bin/env python3
"""Diagnose why dGeDi self-match returned 0 correspondences on SHREC."""
import os, sys, json
_THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _THIS)
import numpy as np                              # noqa: E402
import open3d as o3d                            # noqa: E402
import server                                   # noqa: E402
from precompute_gallery import sample_cloud     # noqa: E402

CACHE = "/oscar/object_retrieval/.dgedi_gallery_shrec_smoke"
server._STATE["device"] = "cuda"
server._STATE["model"] = server.load_model("/dgedi/config_dgedi.yaml", "multi_scale", "cuda")
gallery = server.load_gallery(CACHE)
diam = json.load(open(os.path.join(CACHE, "diameters.json")))
manifest = json.load(open(os.path.join(CACHE, "manifest.json")))
qid = sorted(gallery)[0]

def scale(p):
    p = np.asarray(p); return dict(n=len(p), ext=(p.max(0)-p.min(0)).round(3).tolist(),
                                   rad=round(float(np.linalg.norm(p-p.mean(0),axis=1).max()),3))

# gallery (self) cloud + its stored feats
pcd_t, feats_t = gallery[qid]
print("DIAM[self] =", round(float(diam[qid]), 6))
print("gallery[self] cloud:", scale(pcd_t.points), "feats:", feats_t.shape)

# query built the production way
qpts = sample_cloud(os.path.join("/oscar", manifest[qid]), 10000).astype(np.float32)
print("raw query sample :", scale(qpts))
q_center = server.fps_center(qpts)
print("fps_center(query):", scale(q_center))
q_norm = q_center / float(diam[qid])
print("q_norm = /diam   :", scale(q_norm), "  <-- should match gallery[self] scale")

pcd_q = server._cloud(q_norm)
feats_q = server.extract_features(pcd_q, server._STATE["model"], server._STATE["device"])
print("query feats:", np.asarray(feats_q).shape)

# feature-space mutual match sanity: for 500 random query kps, nearest gallery feat,
# then the GEOMETRIC distance between the matched points (identity transform expected)
from scipy.spatial import cKDTree
fq = np.asarray(feats_q); ft = np.asarray(feats_t)
pq = np.asarray(pcd_q.points); pt = np.asarray(pcd_t.points)
ftree = cKDTree(ft)
idx = np.random.RandomState(0).choice(len(fq), min(500, len(fq)), replace=False)
_, nn = ftree.query(fq[idx], k=1)
gdist = np.linalg.norm(pq[idx] - pt[nn], axis=1)
print(f"feat-NN geometric dist: median={np.median(gdist):.4f} p90={np.percentile(gdist,90):.4f} "
      f"(<~0.03 means features map to the right place)")

# RANSAC at increasing thresholds (production nkp/maxit)
kp_q, kf_q = server._keypoints(pcd_q, fq, 6000)
kp_t, kf_t = server._keypoints(pcd_t, ft, 6000)
for thr in (0.03, 0.05, 0.1, 0.2, 0.4):
    res = server.ransac_only(kp_q, kf_q, kp_t, kf_t, thr, max_iter=10000)
    print(f"  RANSAC thr={thr:>4}: corr={len(res.correspondence_set):5d}  fitness={res.fitness:.3f}")
