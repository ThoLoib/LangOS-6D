#!/usr/bin/env python3
"""Paired significance test for OSCAR+ Stage-1 arm deltas.

For each (arm_A, arm_B) pair, load results_per_query.json, match by query id,
and report the paired mean delta with a 95% bootstrap CI + Wilcoxon signed-rank
p on a metric (default nDCG). A delta is 'real' iff the 95% CI excludes 0. This
closes the "is this small delta noise?" question the 42v+k5 rerun opened up
(Uni3D≈ULIP, XYZ≈XYZ+RGB, config-change 0.5889->0.5868, etc.).

Env: SIG_FOLDER (default results_shrec18_v2_stage1_42v_k5), SIG_METRIC (nDCG),
     SIG_OUT.  Run from repo root (paths are object_retrieval/<folder>/<arm>/...).
"""
import os, json, csv
import numpy as np
try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

FOLDER = os.environ.get("SIG_FOLDER", "results_shrec18_v2_stage1_42v_k5")
OLD = "results_shrec18_v2_stage1_mean_mean_only"     # pre-fix (16v/k8) baseline
METRIC = os.environ.get("SIG_METRIC", "nDCG")
N_BOOT = 10000
OUT = os.environ.get("SIG_OUT",
                     f"object_retrieval/{FOLDER}/paired_significance_{METRIC}.csv")

# (label, (folderA|None, armA), (folderB|None, armB)); None -> FOLDER
PAIRS = [
    ("shape enc: ULIP-2 vs Uni3D (isolated)", (None, "E1_shape_only"),   (None, "E7_uni3d_shape_only")),
    ("shape enc: ULIP-2 vs Uni3D (fused)",    (None, "E1c_full_fusion"), (None, "E7_uni3d")),
    ("colour: XYZ+RGB vs XYZ (isolated)",     (None, "E1_shape_only"),   (None, "O5_xyz_shape_only")),
    ("colour: XYZ+RGB vs XYZ (fused)",        (None, "E1c_full_fusion"), (None, "O5_xyz_only")),
    ("ref: partial vs full-mesh (isolated)",  (None, "E1_shape_only"),   (None, "E2b_fullmesh_shape_only")),
    ("appearance: DINOv2 vs SigLIP (isolated)", (None, "E1_view_only"),  (None, "E4_siglip_only")),
    ("combiner: weighted vs RRF (fused)",     (None, "E1c_full_fusion"), (None, "E6_rrf")),
    ("shape views: V32 vs V42 (isolated)",    (None, "A7_shape_only_V32"), (None, "A7_shape_only_V42")),
    ("geometry: none vs GeDi+RANSAC (fused)", (None, "E1c_full_fusion"), (None, "E2_chamfer_ransac")),
    ("config: BASE 16v/k8 -> 42v/k5 (fused)", (OLD, "E1c_full_fusion"),  (None, "E1c_full_fusion")),
]


def load(folder, arm, metric):
    p = f"object_retrieval/{folder}/{arm}/results_per_query.json"
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    return {e["id"]: float(e[metric]) for e in d if metric in e}


def boot_ci(da, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(da), size=(n, len(da)))
    means = da[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    print(f"[sig] folder={FOLDER} metric={METRIC} boot={N_BOOT} "
          f"scipy={'yes' if HAVE_SCIPY else 'no (bootstrap only)'}")
    rows = []
    for label, (fa, aa), (fb, ab) in PAIRS:
        A = load(fa or FOLDER, aa, METRIC)
        B = load(fb or FOLDER, ab, METRIC)
        if A is None or B is None:
            print(f"[skip] {label}: missing ({aa} or {ab})")
            continue
        ids = sorted(set(A) & set(B))
        if len(ids) < 30:
            print(f"[skip] {label}: only {len(ids)} paired queries")
            continue
        da = np.array([A[i] - B[i] for i in ids])
        mean = float(da.mean())
        lo, hi = boot_ci(da)
        sig = (lo > 0 or hi < 0)
        p = (float(wilcoxon(da).pvalue) if (HAVE_SCIPY and np.any(da)) else float("nan"))
        rows.append((label, aa, ab, round(mean, 4), round(lo, 4), round(hi, 4),
                     (round(p, 4) if p == p else ""), "YES" if sig else "no", len(ids)))
        pstr = f"{p:.4f}" if p == p else "  n/a "
        print(f"{'SIG ' if sig else '    '}{label:42s} "
              f"Δ{METRIC}={mean:+.4f} 95%CI[{lo:+.4f},{hi:+.4f}] p={pstr} n={len(ids)}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "armA", "armB", f"mean_d_{METRIC}",
                    "ci_lo", "ci_hi", "wilcoxon_p", "significant", "n_paired"])
        w.writerows(rows)
    print(f"\n[sig] wrote {len(rows)} comparisons -> {OUT}")


if __name__ == "__main__":
    main()
