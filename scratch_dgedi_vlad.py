"""
Aggregate dGeDi LOCAL descriptors into a GLOBAL signature for retrieval
(VLAD / mean / max / GeM), instead of RANSAC registration. Retrieval use,
occlusion-robust in principle (missing parts just drop features).

Codebook: MiniBatchKMeans on a sample of gallery descriptors (unsupervised VLAD,
no training). Gallery signatures built offline from .dgedi_gallery/*.npz (6000x64
each). Query descriptors via the service /features endpoint.

Reports, per dataset (ycbv/tless/lmo): fused (CLIP+DINO+ULIP) baseline; each
dGeDi-signature ALONE (full-gallery retrieval); and fused (+) VLAD Borda — i.e.
does the shape signature ADD to the fusion, and is it occlusion-robust (lmo)?

Run (oscar container, dgedi UP): docker compose run --rm oscar \
    python3 scratch_dgedi_vlad.py --datasets ycbv,tless,lmo --n 150
"""
import argparse, glob, os, sys
import numpy as np, httpx

_REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_REPO, "object_retrieval"))
sys.path.insert(0, os.path.join(_REPO, "object_retrieval")); sys.path.insert(0, _REPO)

from eval_common import run_query, fusion_ranking, crop_by_bbox
from query_cloud import backproject_masked
from stage3_gallery import assemble_gallery, TARGET_DATASETS
from eval_bop_pose import (_matching_instances, _bbox_of, _pad_bbox, _pose_inputs,
                           _cam_entry, load_bop_targets, DATASET_TEST, _THIS_DIR)
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

DGEDI_URL = "http://dgedi:5061"
GDIR = os.path.join(_REPO, "object_retrieval", ".dgedi_gallery")
K = 64            # VLAD codebook size


def l2(x, ax=-1):
    return x / (np.linalg.norm(x, axis=ax, keepdims=True) + 1e-9)


def sigs_from_feats(feats, centers):
    """feats (M,64) -> dict of global signatures."""
    mean = l2(feats.mean(0))
    mx = l2(feats.max(0))
    gem = l2((np.clip(feats, 1e-6, None) ** 3).mean(0) ** (1/3))
    # VLAD: hard-assign to nearest center, sum residuals per cluster
    d = ((feats[:, None, :] - centers[None]) ** 2).sum(-1)      # (M,K)
    a = d.argmin(1)
    V = np.zeros((len(centers), feats.shape[1]), np.float32)
    for k in range(len(centers)):
        m = a == k
        if m.any():
            V[k] = (feats[m] - centers[k]).sum(0)
    V = l2(V, ax=1)                       # intra-normalize
    vlad = l2(V.flatten())
    return {"mean": mean, "max": mx, "gem": gem, "vlad": vlad}


def query_feats(qc):
    r = httpx.post(f"{DGEDI_URL}/features",
                   json={"points": np.asarray(qc, np.float32).tolist()}, timeout=120)
    return np.asarray(r.json()["feats"], np.float32)


def rank_of(ids, scores, tgt):
    order = [ids[i] for i in np.argsort(-scores)]
    return order.index(tgt) + 1 if tgt in order else None


def borda(ids, fused_rank, sig_scores, tgt):
    sig_rank = {ids[i]: r for r, i in enumerate(np.argsort(-sig_scores))}
    mr = np.array([fused_rank.get(i, len(ids)) + sig_rank[i] for i in ids], float)
    order = [ids[i] for i in np.argsort(mr, kind="stable")]
    return order.index(tgt) + 1 if tgt in order else None


def M(rs):
    n = len(rs); return (sum(1 for r in rs if r == 1)/n, sum(1 for r in rs if r and r <= 5)/n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="ycbv"); ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    print("[setup] gallery ...", flush=True)
    G = assemble_gallery(TARGET_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = G.components()
    cfg = G.eval_cfg; top_k = len(G.gallery_ids) + 5; clip_rows = len(clip_retr._desc_labels)

    # --- codebook (kmeans on a sample of gallery descriptors) ---
    files = {os.path.basename(f)[:-4].replace("__", "/"): f
             for f in glob.glob(os.path.join(GDIR, "*.npz"))}
    ids = [i for i in G.gallery_ids if i in files]
    print(f"[vlad] {len(ids)} gallery objects with dGeDi feats; sampling codebook ...", flush=True)
    samp = []
    for i in ids[::max(1, len(ids)//60)]:
        samp.append(np.load(files[i])["feats"].astype(np.float32))
    samp = np.concatenate(samp)[::3]
    centers = MiniBatchKMeans(n_clusters=K, batch_size=4096, n_init=3,
                              random_state=0).fit(samp.astype(np.float32)).cluster_centers_.astype(np.float32)

    # --- gallery signatures ---
    print("[vlad] building gallery signatures ...", flush=True)
    GS = {m: [] for m in ("mean", "max", "gem", "vlad")}
    for i in ids:
        s = sigs_from_feats(np.load(files[i])["feats"].astype(np.float32), centers)
        for m in GS: GS[m].append(s[m])
    for m in GS: GS[m] = np.asarray(GS[m], np.float32)   # (Ngal, dim)
    print(f"[vlad] gallery sigs ready (vlad dim={GS['vlad'].shape[1]})", flush=True)

    for ds in args.datasets.split(","):
        ds_test = DATASET_TEST[ds]
        targets = load_bop_targets(os.path.join(_THIS_DIR, ds_test["targets"]))[:args.n]
        test_root = os.path.join(_THIS_DIR, ds_test["test_root"])
        R = {m: [] for m in ("fused", "mean", "max", "gem", "vlad", "fused+vlad")}
        for t in targets:
            sid, im, oid = t["scene_id"], t["im_id"], t["obj_id"]
            sdir = os.path.join(test_root, f"{sid:06d}")
            rgbp = os.path.join(sdir, "rgb", f"{im:06d}.png")
            if not os.path.isfile(rgbp): rgbp = rgbp[:-4] + ".jpg"
            if not os.path.isfile(rgbp): continue
            rgb = Image.open(rgbp).convert("RGB"); rgb_np = np.asarray(rgb, np.uint8)
            cam = _cam_entry(sdir, im); tgt = f"{ds}/obj_{oid:06d}"
            if tgt not in ids: continue
            for gi, gt, info in _matching_instances(sdir, im, oid):
                bb = _bbox_of(info)
                if bb is None: continue
                roi = crop_by_bbox(rgb, _pad_bbox(bb, rgb.width, rgb.height))
                depth_m, _dmm, mask, K_ = _pose_inputs(sdir, im, gi, cam)
                qc, qcol = backproject_masked(depth_m, mask, K_, rgb=rgb_np)
                if len(qc) < 64: continue
                qemb = shape_m.encode_pointcloud(qc, colors=qcol)
                out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m, roi, cfg,
                                ulip_query_emb=qemb, dino_full_top_k=top_k,
                                ulip_full_top_k=top_k, clip_full_top_k=clip_rows)
                fused = fusion_ranking(out["fused_full"])
                fr = {o: r for r, (o, _) in enumerate(fused)}
                R["fused"].append(next((r+1 for r, (o, _) in enumerate(fused) if o == tgt), None))
                try:
                    qs = sigs_from_feats(query_feats(qc), centers)
                except Exception:
                    continue
                for m in ("mean", "max", "gem", "vlad"):
                    R[m].append(rank_of(ids, GS[m] @ qs[m], tgt))
                R["fused+vlad"].append(borda(ids, fr, GS["vlad"] @ qs["vlad"], tgt))
        print(f"\n=== {ds} dGeDi-signature retrieval, N={len(R['fused'])} ===")
        print(f"{'variant':14s} {'R@1':>7} {'R@5':>7}")
        for m in ("fused", "mean", "max", "gem", "vlad", "fused+vlad"):
            r1, r5 = M(R[m]); print(f"{m:14s} {r1:7.3f} {r5:7.3f}")


if __name__ == "__main__":
    main()
