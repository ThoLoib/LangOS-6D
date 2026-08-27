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


def verdict(ci_sig: bool, w_sig: bool, wins_a: int, wins_b: int) -> str:
    """Combine the two tests honestly.

    The bootstrap CI tests the **mean** difference; Wilcoxon tests whether one arm
    wins *consistently* (signed ranks). They disagree when the per-query deltas are
    heavy-tailed: a handful of huge swings can move the mean while the win/loss
    split is ~50/50. For the near-ties this grid is full of, the **sign-consistency
    (Wilcoxon + the win counts) is the meaningful verdict** — a "win" carried by a
    few outliers is not a design argument.
    """
    if ci_sig and w_sig:
        return "REAL"                    # both agree
    if not ci_sig and not w_sig:
        return "tie"                     # both agree
    if w_sig and not ci_sig:
        return "CONSISTENT (small, systematic; mean noisy)"
    # ci_sig and not w_sig
    split = f"{wins_a}:{wins_b}"
    return f"outlier-driven ({split}) -> treat as tie"


def main():
    print(f"[sig] folder={FOLDER} metric={METRIC} boot={N_BOOT} "
          f"scipy={'yes' if HAVE_SCIPY else 'no (bootstrap only)'}")
    print("[sig] CI = mean test · Wilcoxon = sign-consistency test; for near-ties "
          "the sign split is authoritative.")
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
        mean, med = float(da.mean()), float(np.median(da))
        lo, hi = boot_ci(da)
        ci_sig = (lo > 0 or hi < 0)
        p = (float(wilcoxon(da).pvalue) if (HAVE_SCIPY and np.any(da)) else float("nan"))
        w_sig = (p == p and p < 0.05)
        wins_a, wins_b, tied = int((da > 0).sum()), int((da < 0).sum()), int((da == 0).sum())
        v = verdict(ci_sig, w_sig, wins_a, wins_b)
        rows.append((label, aa, ab, round(mean, 4), round(med, 4), round(lo, 4), round(hi, 4),
                     (round(p, 4) if p == p else ""), wins_a, wins_b, tied,
                     "YES" if ci_sig else "no", "YES" if w_sig else "no", v, len(ids)))
        pstr = f"{p:.4f}" if p == p else "  n/a "
        flag = "REAL" if v == "REAL" else ("~~~ " if v.startswith("outlier") else
                                           ("CONS" if v.startswith("CONSISTENT") else "    "))
        print(f"{flag} {label:42s} Δ={mean:+.4f} med={med:+.4f} "
              f"CI[{lo:+.4f},{hi:+.4f}] p={pstr} wins {wins_a}:{wins_b} -> {v}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "armA", "armB", f"mean_d_{METRIC}", f"median_d_{METRIC}",
                    "ci_lo", "ci_hi", "wilcoxon_p", "n_A_better", "n_B_better", "n_tied",
                    "ci_significant", "wilcoxon_significant", "verdict", "n_paired"])
        w.writerows(rows)
    print(f"\n[sig] wrote {len(rows)} comparisons -> {OUT}")


if __name__ == "__main__":
    main()
