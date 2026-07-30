#!/usr/bin/env python3
"""
experiment1_shrec18_stage1.py
=============================
Experiment 1 of the thesis — **Stage 1: RGB-D Retrieval Tuning** on
SHREC'18 ObjectNN+ [phamSHREC18RGBDObjecttoCAD2018].

Thesis references (chapters/evaluation.tex):
  * Stage 1 protocol .......... subsec:eval_stage1_retrieval
  * Ablation definitions ...... subsec:eval_baselines  (E1..E7, O1..O5)
  * Metrics ................... subsec:eval_retrieval_metrics and
                                background.tex subsec:retrieval_metrics
  * Result placeholders ....... tab:eval_stage1_ablation_grid,
                                fig:eval_stage1_ablation_bars

What this script does
---------------------
Runs the full *retrieval-side* Stage-1 ablation grid of the OSCAR+
pipeline [pulliOSCAROpenSetCAD2025] on SHREC'18 and selects the best
configuration by **highest DCG, tie-break mAP** (thesis
subsec:eval_stage1_retrieval).  The selected configuration is frozen and
reused unchanged in Stages 2, 3a, 3b and 5.

The grid covered here (pose-side ablations E3/E5 are deferred to Stage 3a):

  E1   channel set (each channel alone / S_text+S_view / full fusion /
       CLIP-pruned / faithful OSCAR cascade)
  E2   local geometry re-ranking: none / RANSAC fitness / unaligned trimmed
       distance (diagnostic) / RANSAC-aligned / RANSAC+ICP  [--with-geometry]
  E2b  shape reference: partial rendered views vs. full mesh
  E4   appearance encoder: DINOv2 vs. SigLIP  [zhaiSigmoidLossLanguage2023]
  E6   fusion: weighted sum vs. reciprocal rank fusion
       [cormackReciprocalRankFusion2009]
  E7   shape encoder: ULIP-2 vs. Uni3D
       [xueULIP2ScalableMultimodal2024, zhouUni3DExploringUnified2023]
  O1   S_shape redundancy vs. S_GeDi (4 configs)    [--with-geometry for c/d]
  O2   scope/ordering: full-database fusion vs. CLIP-pruned cascade
       (OSCAR) vs. visual-first (Stubborn Strawberries,
       [nguyenSHREC2025Retrieval2025])
  O4   number of reference views V ∈ {8, 16, 32, 42}
       [nguyenCNOSStrongBaseline2023]
  O5   query point cloud XYZ+RGB vs. XYZ-only

Execution model (cost control)
------------------------------
Two tiers:

  * Tier 1 — *channel-score passes*: the expensive encoder work.  Each
    pass computes, per query, a full score vector over the gallery for
    one (encoder x reference) combination and caches it under
    ``<results>/_cache/scores_<pass>.pt``.  The appearance channel stores
    one vector per view budget V, so ablation O4 needs **no** extra
    encoder work: all views are encoded once and re-aggregated over the
    FPS-ordered prefix (see pipeline/step4_dino_reranking.py).
  * Tier 2 — *derivations*: fusion weights/method, candidate scoping and
    geometry re-ranking are cheap post-processing of the cached vectors.
    Fusion semantics are NOT re-implemented — the cached vectors are
    wrapped into the pipeline's own result dataclasses and fused by
    ``pipeline.step6_fusion.ScoreFusion``, so every ablation exercises
    the exact production fusion code.

Dataset expectations (data is provided by the user, this script never
downloads anything):

  eval/datasets/shrec18/shrec18_full/    raw SHREC'18 distribution
      cad/*.obj            3,308 ShapeNetSem CAD models (gallery)
      rgbd/*.ply           2,101 query scans (colored triangle meshes)
      results/rgbd.*.txt   per-TRAINING-query relevance lists
      train.csv, test.csv  split files (category column is "unknown")
  object_images/shrec18/<cad_id>/        rendered gallery views
      <cad_id>_<v>.png                   42 views (FPS-ordered icosphere,
                                         rendering/rendering.py)
      <cad_id>_view<v>_partial.npz       per-view partial point clouds
  object_database/shrec18/descriptions_attributes.json
                                         LLaVA view descriptions
                                         (rendering/generate_descriptions.py)

Ground truth
------------
The public SHREC'18 dump ships *without* category labels (train.csv /
test.csv say "unknown"; the official GT was released separately by the
organisers and is not part of this distribution).  The labels are
reconstructed here by union-find over the ``results/`` relevance lists:
per the dataset README every listed result "is from the correct
category", so queries and CADs co-occurring in a list share a category,
and the transitive closure over all 1,452 training lists yields exactly
the 20 track categories.  Only the *training* queries are recoverable
this way — Stage 1 therefore tunes on the 1,452 labeled training
queries; test queries have no local GT.  (State this in the thesis; see
"Open risks" in the accompanying plan/docs.)

Usage
-----
  # everything except geometry ablations (resumable):
  python experiments/experiment1_shrec18_stage1.py --all --resume

  # include E2/O1c-e geometry ablations (GeDi service must be up:
  # ``docker compose up -d gedi``):
  python experiments/experiment1_shrec18_stage1.py --all --resume --with-geometry

  # selected groups or single cells:
  python experiments/experiment1_shrec18_stage1.py --ablations E1,E4,O4_V8

  # smoke test on a handful of queries against an incomplete gallery:
  python experiments/experiment1_shrec18_stage1.py --ablations E1 \
      --limit-queries 20 --allow-partial-gallery

  # list the registry without running anything:
  python experiments/experiment1_shrec18_stage1.py --list

Outputs
-------
  object_retrieval/results_shrec18_stage1/
      <ablation>/metrics_summary.json     all Stage-1 metrics + config
      <ablation>/results_per_query.json   per-query top-10 + ranks
      stage1_summary.csv                  aggregate over all ablations
      stage1_summary.tex                  booktabs table for the thesis
      best_config.json                    frozen best configuration
      _cache/                             score vectors, GeDi cache, ...

Bibliography keys used in comments (references.bib):
  phamSHREC18RGBDObjecttoCAD2018, pulliOSCAROpenSetCAD2025,
  nguyenCNOSStrongBaseline2023, zhaiSigmoidLossLanguage2023,
  voSAMURAIShapeAwareMultimodal2025, xueULIP2ScalableMultimodal2024,
  zhouUni3DExploringUnified2023, caraffaFreeZeTrainingfreeZeroshot2025,
  diUREDUnsupervised3D2023, nguyenSHREC2025Retrieval2025,
  shilanePrincetonShapeBenchmark2004, chuOPENOcclusionInvariantPerception2024a
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Repo-relative paths.  The script lives in OSCAR/experiments/; everything is
# resolved from the repo root so it can be launched from any CWD (host or the
# ``oscar`` docker container).  Heavy imports (torch, open3d, eval_common)
# are deferred into the functions that need them so that --list / --help and
# GT preparation also work in a bare Python environment.
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    "data_root":    os.path.join(ROOT, "eval", "datasets", "shrec18", "shrec18_full"),
    "stage1_root":  os.path.join(ROOT, "eval", "datasets", "shrec18", "stage1"),
    "images_dir":   os.path.join(ROOT, "object_images", "shrec18"),
    "desc_file":    os.path.join(ROOT, "object_database", "shrec18",
                                 "descriptions_attributes.json"),
    "results_root": os.path.join(ROOT, "object_retrieval",
                                 "results_shrec18_stage1"),
}

# SHREC'18 constants [phamSHREC18RGBDObjecttoCAD2018]
N_CATEGORIES = 20      # 20 manually grouped categories (dataset README)
N_CADS = 3308          # gallery size (ShapeNetSem indoor subset)
N_QUERIES_TOTAL = 2101 # all queries; only the ~70% train split has local GT
E_MEASURE_K = 32       # E-measure cut-off, Princeton Shape Benchmark
                       # convention [shilanePrincetonShapeBenchmark2004]

# OSCAR+ Stage-1 defaults (thesis Table 4.1 / methodology Step B1):
# full-database scoring, weighted-sum fusion with w = (0.3, 0.4, 0.3).
BASE_WEIGHTS = (0.3, 0.4, 0.3)
CLIP_PRUNE_K = 20      # OSCAR cascade shortlist size (O2 / E1d)
                       # [pulliOSCAROpenSetCAD2025]
VIEW_BUDGETS = (8, 16, 32, 42)  # O4; thesis lists {8,16,32}, 42 added as
                       # the CNOS/repo default (full icosphere)
                       # [nguyenCNOSStrongBaseline2023]
GEOM_SHORTLIST = 5     # geometry re-ranking shortlist (config default)
# Full-database S_GeDi would need |queries| x |gallery| RANSAC fits.
GEDI_FULL_DB_PAIRS = 1452 * N_CADS
# Number of (FPS-ordered) partial views the shape channel AGGREGATES over at
# retrieval time.  The reference cache still stores all encoded views (whole
# dataset); this only trims the top-k_v pooling input.  16 is sufficient for
# the first experiment and matches the DINO/CNOS view budget; set to None to
# aggregate over every cached view.
SHAPE_AGG_VIEWS = 16

ULIP_CKPT_DEFAULT = "/ulip/checkpoints/ulip2_pointbert_10k.pt"
# O5 XYZ-only arm uses the released ULIP-2 *xyz* PointBERT (8,192 pts, no RGB,
# input_dim=3, SLIP ViT-B tower) — the colored 10k checkpoint has a 6-channel
# input conv and cannot encode xyz-only clouds. Its basename is part of the
# cache fingerprint (step5 _get_partial_cache_path), so the eval PC must stage
# the same file at the same path/name to reuse these caches.
ULIP_CKPT_XYZ = "/ulip/checkpoints/ulip2_pointbert_8k_xyz.pt"

# Official SHREC'18 evaluation kit (git clone of hkust-vgd/shrec18): the
# real category+subcategory GT (rgbd.csv/cad.csv, all 2,101 queries and
# 3,308 CADs) and the exact metric implementations (metrics.py) used to
# score the published track participants.  Using these makes every OSCAR+
# number leaderboard-comparable by construction, for tuning and final runs
# alike [phamSHREC18RGBDObjecttoCAD2018].
OFFICIAL_DIR = os.path.join(ROOT, "eval", "shrec18_official")


# ===========================================================================
# 1. Ground-truth reconstruction (union-find over results/ lists)
# ===========================================================================

def build_gt(data_root: str, stage1_root: str, force: bool = False) -> dict:
    """Reconstruct category labels from the ``results/`` relevance lists.

    The SHREC'18 distribution withholds category labels, but ships, for
    every *training* query, a (non-exhaustive) retrieval list whose
    entries are guaranteed to be from the correct category (dataset
    README).  Treating queries and CADs as nodes and every
    (query, listed CAD) pair as an edge, the connected components of
    this co-occurrence graph are exactly the track categories: 20
    components emerge, covering all 1,452 training queries and 3,305 of
    the 3,308 CADs.  The 3 CADs never listed for any query stay
    unlabeled and act as gallery distractors.

    Category names are synthetic (``cat_00``.. by descending CAD count)
    since the official names are not in the dump.  Category-level
    metrics only need the partition, not the names.  If the official GT
    is ever obtained, drop a file with the same JSON schema at
    ``<stage1>/gt/category_labels.json`` and delete the reconstructed
    one — the rest of the script is agnostic to the label source.
    """
    gt_dir = os.path.join(stage1_root, "gt")
    out_path = os.path.join(gt_dir, "category_labels.json")
    if os.path.isfile(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)

    results_dir = os.path.join(data_root, "results")
    files = sorted(f for f in os.listdir(results_dir) if f.endswith(".txt"))
    if not files:
        raise SystemExit(f"[gt] no relevance lists in {results_dir}")

    # --- Union-find --------------------------------------------------------
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]     # path halving
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    query_ids: List[str] = []
    for fname in files:
        # file name:  rgbd.<hash>.txt ;  lines:  cad.<hash> <distance>
        qid = fname[:-4]                       # strip ".txt"
        query_ids.append(qid)
        with open(os.path.join(results_dir, fname)) as f:
            for line in f:
                tok = line.split()
                if tok:
                    union(qid, tok[0])

    # --- Components -> categories -----------------------------------------
    comps: Dict[str, dict] = defaultdict(lambda: {"cad": [], "rgbd": []})
    for node in parent:
        kind = "cad" if node.startswith("cad.") else "rgbd"
        comps[find(node)][kind].append(node.split(".", 1)[1])

    comps_list = sorted(comps.values(),
                        key=lambda c: (-len(c["cad"]), min(c["cad"] or [""])))
    if len(comps_list) != N_CATEGORIES:
        raise SystemExit(
            f"[gt] union-find produced {len(comps_list)} components, "
            f"expected {N_CATEGORIES}.  The results/ folder is incomplete "
            f"or corrupted — refusing to fabricate ground truth.")

    cad_labels: Dict[str, str] = {}
    query_labels: Dict[str, str] = {}
    sizes: Dict[str, dict] = {}
    for i, comp in enumerate(comps_list):
        cat = f"cat_{i:02d}"
        for h in comp["cad"]:
            cad_labels[h] = cat
        for h in comp["rgbd"]:
            query_labels[h] = cat
        sizes[cat] = {"cad": len(comp["cad"]), "queries": len(comp["rgbd"])}

    all_cads = {os.path.splitext(f)[0]
                for f in os.listdir(os.path.join(data_root, "cad"))
                if f.endswith(".obj")}
    unlabeled = sorted(all_cads - set(cad_labels))

    gt = {
        "provenance": (
            "Reconstructed via union-find over results/*.txt "
            "(SHREC'18 README: every listed result is from the correct "
            "category).  Covers the training split only; official "
            "category names are not available in this dump."),
        "cad": cad_labels,
        "queries": query_labels,
        "unlabeled_cads": unlabeled,
        "component_sizes": sizes,
    }
    os.makedirs(gt_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"[gt] {len(comps_list)} categories, {len(query_labels)} labeled "
          f"queries, {len(cad_labels)}/{len(all_cads)} labeled CADs "
          f"({len(unlabeled)} distractors) -> {out_path}")
    return gt


def _read_shrec_csv(path: str) -> Dict[str, tuple]:
    """Parse an official rgbd.csv / cad.csv: 'rgbd.<hash>,cat,subcat' rows.

    The leading 'rgbd.' / 'cad.' prefix is stripped so the keys match our
    query stems (ply basename) and gallery ids (obj basename).
    """
    out: Dict[str, tuple] = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            key = row[0].split(".", 1)[1] if "." in row[0] else row[0]
            out[key] = (row[1], row[2])
    return out


def load_official_gt(data_root: str, stage1_root: str) -> dict:
    """Load the official SHREC'18 category+subcategory ground truth.

    Uses the track's own rgbd.csv / cad.csv (see :data:`OFFICIAL_DIR`),
    which cover all 2,101 queries and 3,308 CADs with real category and
    subcategory labels — the same GT the published participants were
    scored against.  Falls back to the union-find reconstruction only if
    the official kit is absent.

    Returns::

        {"queries": {qid: (cat, subcat)},   # 2,101
         "cad":     {cadid: (cat, subcat)},  # 3,308
         "freqs":   {cat: n_cads_in_cat},    # official recall denominators
         "provenance": "official hkust-vgd/shrec18 rgbd.csv + cad.csv"}
    """
    rgbd_csv = os.path.join(OFFICIAL_DIR, "rgbd.csv")
    cad_csv = os.path.join(OFFICIAL_DIR, "cad.csv")
    if not (os.path.isfile(rgbd_csv) and os.path.isfile(cad_csv)):
        raise SystemExit(
            f"[gt] official GT not found at {OFFICIAL_DIR}. Clone it with:\n"
            f"  git clone https://github.com/hkust-vgd/shrec18 "
            f"{OFFICIAL_DIR}\n"
            f"(or fall back to the union-find reconstruction via build_gt).")

    queries = _read_shrec_csv(rgbd_csv)
    cad = _read_shrec_csv(cad_csv)
    freqs: Dict[str, int] = {}
    for cat, _sub in cad.values():
        freqs[cat] = freqs.get(cat, 0) + 1

    gt = {"queries": queries, "cad": cad, "freqs": freqs,
          "provenance": "official hkust-vgd/shrec18 rgbd.csv + cad.csv"}
    gt_dir = os.path.join(stage1_root, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    with open(os.path.join(gt_dir, "official_labels.json"), "w") as f:
        json.dump({"queries": {k: list(v) for k, v in queries.items()},
                   "cad": {k: list(v) for k, v in cad.items()},
                   "freqs": freqs, "provenance": gt["provenance"]}, f)
    print(f"[gt] official: {len(queries)} queries, {len(cad)} CADs, "
          f"{len(freqs)} categories "
          f"({sum(1 for v in cad.values() if v[1] and v[1] != v[0])} have "
          f"subcategories) -> {gt_dir}/official_labels.json")
    return gt


def score_official(ranking_cad_ids: Sequence[str], q_label: tuple,
                   cad_labels: Dict[str, tuple], freqs: Dict[str, int]
                   ) -> Optional[Dict[str, float]]:
    """Score one query's ranking with the OFFICIAL SHREC'18 metrics.

    Faithful replication of evaluate.py's per-query loop, reusing the
    unchanged official ``metrics.py`` so the numbers are bug-for-bug
    identical to the published leaderboard:
      * graded relevance — 2 if (category, subcategory) match, 1 if only
        category, else 0 (``categories_to_rel``);
      * every metric computed on the top-f results, f = category size in
        cad.csv (``freqs``);
      * precision/recall/F1/AP/NNT1/NNT2 count any category match
        (np.count_nonzero); nDCG uses the graded gains (subcategory=2).
    """
    if OFFICIAL_DIR not in sys.path:
        sys.path.insert(0, OFFICIAL_DIR)
    import metrics as MM  # official, unchanged

    qc, qs = q_label
    f = freqs.get(qc, 0)
    if f == 0:
        return None
    rel = []
    for cid in ranking_cad_ids:
        lab = cad_labels.get(cid)
        if lab is None:
            rel.append(0.0)
        elif lab[0] == qc and lab[1] == qs:
            rel.append(2.0)
        elif lab[0] == qc:
            rel.append(1.0)
        else:
            rel.append(0.0)
    x = rel[:f]
    if not x:
        return None
    return {
        "nDCG":      float(MM.ndcg(x)),
        "precision": float(MM.precision(x)),
        "recall":    float(MM.recall(x, f)),
        "F1":        float(MM.f1score(x, f)),
        "AP":        float(MM.average_precision(x, f)),
        "NNT1":      float(MM.nnt1(x, f)),
        "NNT2":      float(MM.nnt2(x, f)),
    }


# ===========================================================================
# 2. Query preparation (PLY -> RGB crop + raw point cloud)
# ===========================================================================
# Each SHREC'18 query is a colored triangle mesh reconstructed from a
# SceneNN RGB-D scan.  The OSCAR+ image channels (S_text via CLIP, S_view
# via DINOv2/SigLIP, S_shape in cross-modal mode via the ULIP-2 image
# branch) expect an RGB object crop on neutral grey — the Step-A ROI
# convention (pipeline/step1_localization.py, thesis Step A4).  Since the
# original camera pose is not part of the dump, the crop is re-rendered
# from the mean-vertex-normal direction (scan meshes are one-sided, so
# this approximates the original viewpoint).  The point-cloud channels
# (S_shape pc-mode for O5/E7, S_GeDi/S_chamfer for E2/O1) use the raw
# vertices + colors stored as .npz; normalization to the encoder input
# convention happens inside the pipeline (step5 normalize_pointcloud).

_GREY = 205  # neutral background grey, matches eval_common.crop_with_mask


def _offscreen_available() -> bool:
    """Whether Open3D's GL OffscreenRenderer can be used safely.

    OffscreenRenderer needs EGL/GL; on a headless CPU container it does
    not merely raise — it *segfaults* the interpreter (uncatchable). It is
    therefore auto-enabled only when libEGL is confirmed loadable (true on
    the GPU compose run, which gives clean solid mesh crops), and skipped
    otherwise in favour of the CPU-only, headless-safe point-splat
    renderer. Set OSCAR_QUERY_OFFSCREEN=0 to force the splat even with a
    GPU present.
    """
    if os.environ.get("OSCAR_QUERY_OFFSCREEN") == "0":
        return False
    try:
        import ctypes
        ctypes.CDLL("libEGL.so.1")
        return True
    except OSError:
        return False


def _view_basis(points: np.ndarray, normals: Optional[np.ndarray]):
    """Camera basis from the mean vertex normal (fallback: PCA)."""
    n = None
    if normals is not None and len(normals):
        n = normals.mean(axis=0)
        if np.linalg.norm(n) < 1e-6:
            n = None
    if n is None:
        # Degenerate normals: view along the least-extent PCA axis.
        c = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 5000)], full_matrices=False)
        n = vt[-1]
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, n)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    u = np.cross(up, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def _render_query_offscreen(mesh, n, size: int):
    """Preferred path: Open3D OffscreenRenderer (needs EGL/GPU)."""
    import open3d as o3d
    r = o3d.visualization.rendering.OffscreenRenderer(size, size)
    try:
        g = _GREY / 255.0
        r.scene.set_background([g, g, g, 1.0])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"           # show raw vertex colors
        r.scene.add_geometry("query", mesh, mat)
        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        radius = np.linalg.norm(bbox.get_extent()) / 2.0
        pts = np.asarray(mesh.vertices)
        nrm = np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None
        n_dir, _, v = _view_basis(pts, nrm)
        eye = center + n_dir * (2.5 * max(radius, 1e-3))
        r.setup_camera(60.0, center.astype(np.float32),
                       eye.astype(np.float32), v.astype(np.float32))
        img = np.asarray(r.render_to_image())
        return img
    finally:
        del r


def _render_query_splat(points, colors, n, u, v, size: int):
    """Headless fallback: orthographic z-buffered point splatting.

    Purely numpy — no GL required.  The splat radius adapts to point
    density (sparse scans get larger dots) so the object reads as a solid
    surface rather than a dotted texture; nearer points overwrite farther
    ones (painter's algorithm).  Coarser than the mesh render but
    adequate for category-level encoder crops; verify with --viz-check.
    """
    x = points @ u
    y = points @ v
    depth = points @ n                          # camera on +n side
    pad = size // 12
    draw = size - 2 * pad
    span = max(x.max() - x.min(), y.max() - y.min(), 1e-9)
    px = ((x - x.min()) / span * draw + pad).astype(np.int32)
    py = (size - 1 - ((y - y.min()) / span * draw + pad)).astype(np.int32)

    # Adaptive disk radius ≈ mean inter-point pixel spacing, so gaps close
    # for sparse clouds without over-blurring dense ones.
    n_pts = max(len(points), 1)
    spacing = draw / max(np.sqrt(n_pts), 1.0)
    radius = int(np.clip(round(0.7 * spacing), 1, 8))

    order = np.argsort(depth)                   # far first, near overwrites
    xs0, ys0, c8 = px[order], py[order], (np.clip(colors, 0, 1) * 255).astype(np.uint8)[order]
    img = np.full((size, size, 3), _GREY, dtype=np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:   # disk, not square
                continue
            xs = np.clip(xs0 + dx, 0, size - 1)
            ys = np.clip(ys0 + dy, 0, size - 1)
            img[ys, xs] = c8
    return img


def _render_query_surface(mesh, n, u, v, size: int,
                          n_samples: int = 200000, seed: int = 0):
    """Headless SOLID-surface render — pure numpy, no GL.

    The raw SHREC'18 query scans carry only ~10^3 vertices, so splatting the
    vertices alone leaves a dotted, holey cloud that does not read as a rigid
    segmented object (bad crop for the DINOv2/CLIP/SigLIP appearance
    channels).  Here we area-weight-sample points *on the triangle faces*
    (barycentric), so the whole surface is covered, then reuse the z-buffered
    point splatter.  The result is a filled, shaded object crop.

    Deterministic (fixed ``seed``) so each query renders identically on every
    run/machine.  Returns ``None`` if the mesh has no usable faces (caller
    then falls back to vertex splatting).
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0 or len(verts) == 0:
        return None
    vcols = (np.asarray(mesh.vertex_colors, dtype=np.float32)
             if mesh.has_vertex_colors()
             else np.full((len(verts), 3), 0.5, dtype=np.float32))
    tri = verts[faces]                                   # (F, 3, 3)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(areas.sum())
    if total <= 0:
        return None
    rng = np.random.RandomState(seed)
    fidx = rng.choice(len(faces), size=n_samples, p=areas / total)
    r1 = np.sqrt(rng.random_sample(n_samples).astype(np.float32))
    r2 = rng.random_sample(n_samples).astype(np.float32)
    bary = np.stack([1.0 - r1, r1 * (1.0 - r2), r1 * r2], axis=1)  # (S, 3)
    fv = faces[fidx]                                     # (S, 3)
    s_pts = (bary[:, :, None] * verts[fv]).sum(axis=1)   # (S, 3)
    s_cols = (bary[:, :, None] * vcols[fv]).sum(axis=1)  # (S, 3)
    # Keep the raw vertices too so silhouette edges stay crisp.
    pts = np.concatenate([verts, s_pts], axis=0)
    cols = np.concatenate([vcols, s_cols], axis=0)
    return _render_query_splat(pts, cols, n, u, v, size)


def _crop_to_content(img: np.ndarray, margin: int = 8) -> np.ndarray:
    mask = np.any(np.abs(img.astype(np.int16) - _GREY) > 6, axis=2)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, img.shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def prepare_queries(data_root: str, stage1_root: str, gt: dict,
                    size: int = 448, force: bool = False) -> List[dict]:
    """Produce <hash>.png + <hash>.npz for every labeled query (cached).

    Returns the query index [{id, category, png, npz}, ...] and writes it
    to <stage1>/gt/queries_index.json.
    """
    import open3d as o3d
    from PIL import Image

    qdir = os.path.join(stage1_root, "queries")
    os.makedirs(qdir, exist_ok=True)
    index_path = os.path.join(stage1_root, "gt", "queries_index.json")
    if os.path.isfile(index_path) and not force:
        with open(index_path) as f:
            index = json.load(f)
        missing = [q for q in index
                   if not (os.path.isfile(q["png"]) and os.path.isfile(q["npz"]))]
        # Only trust a cached index if it covers exactly the current GT
        # query set — a stale/partial index (e.g. from an earlier limited
        # test) must trigger a full regenerate, not silently shrink the run.
        covers = {q["id"] for q in index} == set(gt["queries"])
        if not missing and covers:
            return index
        print(f"[prepare] cached index covers {len(index)}/"
              f"{len(gt['queries'])} queries, {len(missing)} missing files "
              f"— regenerating.")

    offscreen_ok = _offscreen_available()  # default False -> numpy splat
    print(f"[prepare] query renderer: "
          f"{'Open3D OffscreenRenderer (GL)' if offscreen_ok else 'numpy point-splat (headless)'}")
    index = []
    qids = sorted(gt["queries"])
    print(f"[prepare] rendering {len(qids)} query crops -> {qdir}")
    for i, qid in enumerate(qids):
        png = os.path.join(qdir, f"{qid}.png")
        npz = os.path.join(qdir, f"{qid}.npz")
        entry = {"id": qid, "category": gt["queries"][qid],
                 "png": png, "npz": npz}
        index.append(entry)
        if os.path.isfile(png) and os.path.isfile(npz) and not force:
            continue

        ply = os.path.join(data_root, "rgbd", f"{qid}.ply")
        mesh = o3d.io.read_triangle_mesh(ply)
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        cols = (np.asarray(mesh.vertex_colors, dtype=np.float32)
                if mesh.has_vertex_colors() else
                np.full((len(pts), 3), 0.5, dtype=np.float32))
        if len(pts) == 0:
            print(f"[prepare] WARNING: empty mesh {ply}, skipped")
            index.pop()
            continue

        # Raw (unnormalized, metric) points + colors: the shape channel
        # normalizes internally (step5 normalize_pointcloud, thesis
        # Sec. 3.3); geometry re-ranking normalizes to the unit sphere in
        # this script; the scale gate needs the raw metric extents.
        np.savez_compressed(npz, points=pts, colors=cols)

        if not mesh.has_vertex_normals():
            try:
                mesh.compute_vertex_normals()
            except Exception:
                pass
        nrm = (np.asarray(mesh.vertex_normals, dtype=np.float32)
               if mesh.has_vertex_normals() else None)
        n, u, v = _view_basis(pts, nrm)

        img = None
        # Primary: pure-numpy solid-surface render from the mesh faces (no GL,
        # deterministic) — gives a filled rigid-object crop for the appearance
        # encoders instead of the sparse dotted vertex splat.
        if mesh.has_triangles():
            try:
                img = _render_query_surface(mesh, n, u, v, size)
            except Exception as exc:
                print(f"[prepare] surface render failed ({exc}); "
                      f"trying other renderers.")
        # Optional GL path (only if libEGL is present and forced on).
        if img is None and offscreen_ok and mesh.has_triangles():
            try:
                img = _render_query_offscreen(mesh, n, size)
            except Exception as exc:
                print(f"[prepare] OffscreenRenderer unavailable ({exc}); "
                      f"falling back to point splatting for all queries.")
                offscreen_ok = False
        # Fallback: vertex point-splat (face-less clouds).
        if img is None:
            img = _render_query_splat(pts, cols, n, u, v, size)
        Image.fromarray(_crop_to_content(img)).save(png)

        if (i + 1) % 100 == 0:
            print(f"[prepare]   {i + 1}/{len(qids)}")

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[prepare] done: {len(index)} queries indexed -> {index_path}")
    return index


def viz_check(index: List[dict], stage1_root: str, n: int) -> None:
    """Contact sheet of n random query crops for visual sanity."""
    from PIL import Image
    rng = np.random.RandomState(42)
    picks = [index[i] for i in rng.choice(len(index), min(n, len(index)),
                                          replace=False)]
    cell = 224
    cols = int(np.ceil(np.sqrt(len(picks))))
    rows = int(np.ceil(len(picks) / cols))
    sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i, q in enumerate(picks):
        img = Image.open(q["png"]).convert("RGB")
        img.thumbnail((cell, cell))
        sheet.paste(img, ((i % cols) * cell, (i // cols) * cell))
    out = os.path.join(stage1_root, "viz_check.png")
    sheet.save(out)
    print(f"[viz] contact sheet of {len(picks)} crops -> {out}")


# ===========================================================================
# 3. Retrieval metrics (SHREC'18 / Princeton Shape Benchmark conventions)
# ===========================================================================
# NN, FT, ST, E-measure and DCG follow the PSB definitions
# [shilanePrincetonShapeBenchmark2004] used by the SHREC'18 track
# [phamSHREC18RGBDObjecttoCAD2018]; Recall@k and mAP are added per thesis
# subsec:eval_retrieval_metrics.  All metrics are category-level: a
# retrieved CAD is relevant iff it belongs to the query's category.
# Implementation reuses object_retrieval/eval_common.py helpers where
# they exist (dcg_at_k, ideal_dcg_at_k, average_precision_from_binary).

def evaluate_ranking(rels: np.ndarray, num_rel: int) -> Dict[str, float]:
    """All Stage-1 metrics for one query.

    Args:
        rels:    binary relevance in rank order over the FULL gallery.
        num_rel: |C| — number of relevant models in the scored gallery.
    """
    from eval_common import (dcg_at_k, ideal_dcg_at_k,
                             average_precision_from_binary)
    n = len(rels)
    ft_k = num_rel
    st_k = min(2 * num_rel, n)
    r32 = int(rels[:E_MEASURE_K].sum())
    p = r32 / min(E_MEASURE_K, n)
    r = r32 / num_rel
    idcg = ideal_dcg_at_k(num_rel, n)
    return {
        "NN":  float(rels[0]),
        "FT":  float(rels[:ft_k].sum()) / num_rel,
        "ST":  float(rels[:st_k].sum()) / num_rel,
        # E-measure = F-score at the fixed cut-off K=32 (PSB convention).
        "E":   (2 * p * r / (p + r)) if (p + r) > 0 else 0.0,
        # DCG over the full ranked list, normalized by the ideal DCG —
        # the Stage-1 model-selection metric (subsec:eval_stage1_retrieval).
        "DCG": dcg_at_k(rels.tolist(), n) / idcg if idcg > 0 else 0.0,
        "R@1": float(rels[0]),
        "R@5": float(rels[:5].max()) if n else 0.0,
        "AP":  average_precision_from_binary(rels.tolist()),
    }


# ===========================================================================
# 4. Tier 1 — channel-score passes
# ===========================================================================
# Each pass builds the pipeline via eval_common.build_pipeline (which owns
# all on-disk embedding caches) and produces per-query score vectors over
# the canonical gallery list.  Pass definitions:
#
#   base          CLIP + DINOv2 (all views, per-V aggregation) — the
#                 appearance/semantic half of the OSCAR+ channel stack.
#                 S_shape is NOT part of this pass: Stage-1 scores the shape
#                 channel in pc-mode (see ulip_pc_rgb / _BASE_CH).
#   siglip        appearance channel re-scored with SigLIP (E4).
#   ulip_pc_fullmesh  S_shape with full-mesh reference embeddings (E2b),
#                 pc-mode query.
#   ulip_pc_rgb   S_shape in pc-mode, query = XYZ+RGB point cloud
#                 (E7 ULIP-2 arm / O5 RGB arm).  pc-mode is forced for
#                 these ablations so that the *point cloud* is what varies
#                 (the default cross-modal mode encodes the query image,
#                 which would make O5 a no-op).
#   ulip_pc_xyz   as above with colors disabled on both sides (O5).
#   uni3d         S_shape scored by Uni3D (pc-mode only; E7).

PASS_DEFS: "OrderedDict[str, dict]" = OrderedDict([
    ("base",          dict(channels=("clip", "dino"),
                           ulip2_mode="cross", partial=True, no_shape=True,
                           overrides={})),
    ("siglip",        dict(channels=("dino",), ulip2_mode="cross",
                           partial=True, no_shape=True,
                           overrides={"appearance_encoder": "siglip"})),
    # Full-mesh S_shape reference in pc-mode (E2b).  Mode only affects the
    # QUERY side, so this reuses both existing caches unchanged: the
    # full-mesh gallery cache (cad/.ulip_cache_*.pt) and the pc query cache
    # (keyed on encoder config, not on `partial` — see _pc_query_cache_path).
    ("ulip_pc_fullmesh", dict(channels=("shape",), ulip2_mode="pc",
                           partial=False, overrides={})),
    ("ulip_pc_rgb",   dict(channels=("shape",), ulip2_mode="pc",
                           partial=True, overrides={})),
    ("ulip_pc_xyz",   dict(channels=("shape",), ulip2_mode="pc",
                           partial=True,
                           overrides={"ulip2_use_colors": False,
                                      "ulip2_backbone": "pointbert",
                                      "ulip2_checkpoint": ULIP_CKPT_XYZ,
                                      "ulip2_num_points": 8192,
                                      # SLIP ViT-B tower → 512-d space (the
                                      # colored 10k arm is ViT-g / 1280-d)
                                      "ulip2_embed_dim": 512})),
    ("uni3d",         dict(channels=("shape",), ulip2_mode="pc",
                           partial=True,
                           overrides={"shape_encoder": "uni3d"})),
])


def _stack_view_embeddings(emb_map: Dict[str, list], object_ids: List[str]):
    """Stack {obj: [(emb, path)...] | tensor} into one (M, D) matrix.

    Returns (big, ranges) with ranges[i] = (start, end) row span of
    object_ids[i].  Objects without embeddings get an empty span.
    """
    import torch
    chunks, ranges, pos = [], [], 0
    for oid in object_ids:
        v = emb_map.get(oid)
        if v is None:
            ranges.append((pos, pos))
            continue
        if isinstance(v, list):                      # DINO: [(emb, path)]
            t = torch.stack([e for e, _ in v], dim=0)
        else:                                        # ULIP: tensor
            t = v if v.dim() == 2 else v.unsqueeze(0)
        chunks.append(t.float().cpu())
        ranges.append((pos, pos + t.shape[0]))
        pos += t.shape[0]
    big = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 1)
    return big, ranges


def _aggregate_groups(sims, ranges, prefix: Optional[int],
                      method: str, top_k: int, tau: float) -> np.ndarray:
    """Per-object aggregation of per-view similarities.

    Vectorized re-implementation of step4/_aggregate_view_scores
    ("topk_softmax": softmax-weighted top-k_v pooling, the OPEN-style
    query-conditioned aggregation [chuOPENOcclusionInvariantPerception2024a]
    with k_v = 5, tau = 0.5 per CNOS [nguyenCNOSStrongBaseline2023]).
    Exact for uniform aggregation params; falls back to the pipeline's
    own scalar helper for non-default methods.

    ``prefix`` keeps only the first N views per object — valid because
    views are FPS-ordered (rendering/rendering.py; O4 relies on this and
    on the natural-sort fix in step4_dino_reranking).
    """
    import torch
    from pipeline.step4_dino_reranking import _aggregate_view_scores

    out = np.full(len(ranges), -np.inf, dtype=np.float32)
    # group object indices by effective view count for batched pooling
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, (s, e) in enumerate(ranges):
        cnt = e - s
        if prefix is not None:
            cnt = min(cnt, prefix)
        if cnt > 0:
            groups[cnt].append(i)

    for cnt, idxs in groups.items():
        rows = torch.stack([sims[ranges[i][0]:ranges[i][0] + cnt]
                            for i in idxs])                    # (G, cnt)
        if method == "topk_softmax":
            k = min(top_k, cnt)
            vals, _ = rows.topk(k, dim=1)
            w = torch.softmax(vals / tau, dim=1)
            agg = (w * vals).sum(dim=1)
        elif method == "mean":
            agg = rows.mean(dim=1)
        elif method == "max":
            agg = rows.max(dim=1).values
        else:                                   # exact but slow fallback
            agg = torch.tensor([_aggregate_view_scores(
                r_, method=method, top_k=top_k, temperature=tau)[0]
                for r_ in rows])
        out[np.asarray(idxs)] = agg.cpu().numpy()
    return out


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _pass_provenance(pass_key: str, pipe_cfg, need) -> dict:
    """Encoder identity of a precomputed pass, so the eval PC can confirm the
    shipped caches were built with a matching encoder (same weights + code)."""
    return {
        "pass": pass_key,
        "channels": sorted(need),
        "appearance_encoder": pipe_cfg.appearance_encoder,
        "shape_encoder": pipe_cfg.shape_encoder,
        "dino_model": pipe_cfg.dino_model_name,
        "siglip_model": pipe_cfg.siglip_model_name,
        "ulip2_checkpoint": os.path.basename(pipe_cfg.ulip2_checkpoint or ""),
        "ulip2_backbone": pipe_cfg.ulip2_backbone,
        "ulip2_use_colors": bool(pipe_cfg.ulip2_use_colors),
        "ulip2_mode": pipe_cfg.ulip2_mode,
        "ulip2_num_points": pipe_cfg.ulip2_num_points,
        "ulip2_embed_dim": pipe_cfg.ulip2_embed_dim,
        "code_commit": _git_commit(),
    }


def _pc_query_cache_path(paths: dict, pipe_cfg) -> str:
    """Cache path for pc-mode query point-cloud embeddings.

    Same fingerprint fields as step5_shape_matching's gallery partial-view
    cache (encoder identity + checkpoint + dims) — content-based, not
    path/mtime-based, so it stays valid across --results-root and
    --limit-queries changes and never collides between ulip_pc_rgb,
    ulip_pc_xyz, and uni3d (different checkpoints/dims each).
    """
    import hashlib
    encoder = getattr(pipe_cfg, "shape_encoder", "ulip2")
    if encoder == "uni3d":
        meta = (f"encoder=uni3d|model={pipe_cfg.uni3d_model_name}|"
               f"npts={pipe_cfg.uni3d_num_points}|"
               f"ckpt={os.path.basename(pipe_cfg.uni3d_checkpoint or '')}")
    else:
        meta = (f"encoder=ulip2|backbone={pipe_cfg.ulip2_backbone}|"
               f"npts={pipe_cfg.ulip2_num_points}|"
               f"colors={int(pipe_cfg.ulip2_use_colors)}|"
               f"edim={pipe_cfg.ulip2_embed_dim}|"
               f"ckpt={os.path.basename(pipe_cfg.ulip2_checkpoint or '')}")
    digest = hashlib.sha256(meta.encode("utf-8")).hexdigest()[:16]
    # Lives under stage1_root (dataset-scoped, same as queries_index.json),
    # not results_root, so it survives switching --results-root between runs.
    cache_dir = os.path.join(paths["stage1_root"], "query_pc_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pc_query_cache_{digest}.pt")


def _load_or_build_pc_query_cache(paths: dict, pipe_cfg, index: List[dict],
                                  shape_m) -> Dict[str, "torch.Tensor"]:
    """Load (or build+save) query point-cloud embeddings for a pc-mode pass.

    Mirrors eval_common.pre_encode_ulip_queries' cache-then-encode pattern
    for the image branch, extended to the point-cloud branch, which
    previously had no cache at all — this is the expensive step (one
    point-cloud forward pass per query, not batched), so this is worth far
    more than the image-branch cache in wall-clock time saved on reruns.
    """
    import torch

    cache_path = _pc_query_cache_path(paths, pipe_cfg)
    if os.path.isfile(cache_path):
        print(f"[pc-query-cache] loading {cache_path}...")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        missing = [q for q in index if q["id"] not in cache]
        if not missing:
            print(f"[pc-query-cache] {len(cache)} embeddings loaded.")
            return cache
        print(f"[pc-query-cache] {len(missing)}/{len(index)} queries "
              f"missing from cache — encoding those.")
    else:
        cache = {}
        missing = index

    from tqdm import tqdm
    for q in tqdm(missing, desc="pc-query pre-enc", unit="query"):
        data = np.load(q["npz"])
        emb = shape_m.encode_pointcloud(data["points"], colors=data["colors"])
        cache[q["id"]] = emb.detach().cpu()

    torch.save(cache, cache_path)
    print(f"[pc-query-cache] saved {len(cache)} embeddings -> {cache_path}")
    return cache


def precompute_gallery(paths: dict, object_ids: List[str],
                       resume: bool) -> None:
    """Build every gallery *reference* cache with no query scoring.

    This is the precompute entry point meant to run on the
    gallery-generating PC (see --precompute).  build_pipeline writes the
    DINO/SigLIP/ULIP/Uni3D caches into object_images/shrec18 and the cad/
    dir; here we just drive all six pass configs and record a provenance
    manifest that the eval PC checks before trusting the shipped caches.
    """
    manifest = {"code_commit": _git_commit(),
                "gallery_size": len(object_ids), "passes": []}
    for pkey in PASS_DEFS:
        try:
            prov = run_pass(pkey, paths, index=None, object_ids=object_ids,
                            limit=None, resume=resume, build_only=True)
            manifest["passes"].append(prov)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[precompute:{pkey}] FAILED ({exc})")
            manifest["passes"].append({"pass": pkey, "error": str(exc)})
    out = os.path.join(paths["images_dir"], "precompute_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    n_ok = len([p for p in manifest["passes"] if "error" not in p])
    print(f"\n[precompute] done — {n_ok}/{len(PASS_DEFS)} passes built.\n"
          f"[precompute] manifest -> {out}\n"
          f"[precompute] ship to the eval PC (same rclone path as the "
          f"gallery): object_images/shrec18/*cache*.pt + "
          f"precompute_manifest.json, and "
          f"eval/datasets/shrec18/shrec18_full/cad/.ulip_cache_*.pt")


def verify_precompute_provenance(paths: dict) -> None:
    """Warn (loudly, non-fatally) if shipped caches were built at a different
    code commit than this PC — the divergence guard for cross-machine caches."""
    man = os.path.join(paths["images_dir"], "precompute_manifest.json")
    if not os.path.isfile(man):
        return
    with open(man) as f:
        m = json.load(f)
    remote, local = m.get("code_commit", "unknown"), _git_commit()
    if remote != local and "unknown" not in (remote, local):
        print(f"[provenance] WARNING: gallery caches were precomputed at "
              f"commit {remote}, but this PC is at {local}. If the encoder "
              f"path (pipeline/step3-5, config, eval_common) differs between "
              f"those commits, gallery and query embeddings may be "
              f"inconsistent — verify those files match before trusting "
              f"results.")
    else:
        print(f"[provenance] precompute manifest present (commit {remote}) — OK.")


def run_pass(pass_key: str, paths: dict, index: List[dict],
             object_ids: List[str], limit: Optional[int],
             resume: bool, build_only: bool = False) -> dict:
    """Compute (or load) the score store for one channel pass.

    Store layout::

        {"pass": key, "object_ids": [...],
         "queries": {qid: {"clip": (N,), "dino": {V: (N,)}, "shape": (N,)}}}

    With ``build_only=True`` only the gallery *reference* caches are built
    (DINO/SigLIP/ULIP/Uni3D embeddings, written by build_pipeline) and a
    provenance record is returned — no queries are scored.  This is the
    precompute path meant to run on the gallery-generating PC.
    """
    import torch

    tag = f"_n{limit}" if limit else ""
    cache = os.path.join(paths["results_root"], "_cache",
                         f"scores_{pass_key}{tag}.pt")
    qids = ([q["id"] for q in (index or [])][:limit] if limit
            else [q["id"] for q in (index or [])])
    if resume and not build_only and os.path.isfile(cache):
        store = torch.load(cache, map_location="cpu", weights_only=False)
        if store.get("object_ids") == object_ids and \
                all(q in store["queries"] for q in qids):
            print(f"[pass:{pass_key}] loaded cache ({len(store['queries'])} "
                  f"queries) <- {cache}")
            return store

    import eval_common as ec
    from PIL import Image

    pdef = PASS_DEFS[pass_key]
    need = set(pdef["channels"])
    print(f"[pass:{pass_key}] computing channels {sorted(need)} "
          f"for {len(qids)} queries ...")

    cfg = ec.EvalConfig(
        ref_dir=paths["images_dir"],
        desc_file=paths["desc_file"],
        # shape passes need CAD meshes (id source / full-mesh encoding);
        # appearance-only passes skip ULIP entirely.
        cad_mesh_glob=("" if pdef.get("no_shape")
                       else os.path.join(paths["data_root"], "cad", "*.obj")),
        result_folder=os.path.join(paths["results_root"], "_cache"),
        clip_top_k=10 ** 6, dino_top_k=10 ** 6,
        ulip2_top_k=10 ** 6, fusion_top_k=10 ** 6,
        weight_clip=BASE_WEIGHTS[0], weight_dino=BASE_WEIGHTS[1],
        weight_ulip=BASE_WEIGHTS[2],
        ulip2_mode=pdef["ulip2_mode"],
        ulip2_use_partial_views=pdef["partial"],
        ulip2_checkpoint=ULIP_CKPT_DEFAULT,
        # Encode/keep ALL views; O4 trimming happens at derivation time.
        pipeline_overrides={"num_views": None, **pdef["overrides"]},
    )
    if "shape" not in need:
        cfg.cad_mesh_glob = ""

    pipe_cfg, clip_retr, dino_rer, _fusion, shape_m = ec.build_pipeline(cfg)
    if "shape" in need and shape_m is None:
        raise SystemExit(f"[pass:{pass_key}] shape encoder failed to load "
                         f"(checkpoint/repo missing?) — cannot compute "
                         f"the S_shape channel.")

    if build_only:
        # Gallery reference caches are now written by build_pipeline; return
        # provenance and stop (no query scoring in the precompute path).
        prov = _pass_provenance(pass_key, pipe_cfg, need)
        del clip_retr, dino_rer, shape_m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[precompute:{pass_key}] reference caches built.")
        return prov

    # --- pre-stack reference embeddings into one matrix per channel -------
    dino_big = dino_ranges = None
    if "dino" in need:
        dino_big, dino_ranges = _stack_view_embeddings(
            dino_rer._ref_embeddings, object_ids)
        dino_big = dino_big.to(pipe_cfg.device)
    shape_big = shape_ranges = None
    if "shape" in need:
        shape_big, shape_ranges = _stack_view_embeddings(
            shape_m._cad_embeddings, object_ids)
        shape_big = shape_big.to(pipe_cfg.device)

    # --- ULIP image-branch query cache (cross-modal passes) ---------------
    ulip_img_cache = None
    if "shape" in need and pdef["ulip2_mode"] == "cross":
        cache_img = os.path.join(paths["results_root"], "_cache",
                                 "ulip_query_img_cache.pt")
        ulip_img_cache = ec.load_ulip_query_cache(cache_img)
        if ulip_img_cache is None:
            pngs = [q["png"] for q in index]
            ulip_img_cache = ec.pre_encode_ulip_queries(pngs, shape_m)
            os.makedirs(os.path.dirname(cache_img), exist_ok=True)
            torch.save(ulip_img_cache, cache_img)

    # --- point-cloud query cache (pc-mode passes) --------------------------
    # Unlike the cross-modal image branch above, this had NO cache at all:
    # every pc-mode pass (ulip_pc_rgb, ulip_pc_xyz, uni3d) re-encoded every
    # query's point cloud from scratch, every single run — the single most
    # expensive step in the whole ablation grid (~1-2s/query, one at a time,
    # no batching). Keyed by encoder config (not path/mtime), same fields as
    # the gallery partial-view cache, so ulip_pc_rgb/xyz/uni3d never collide
    # and stay valid across --results-root / --limit-queries changes.
    pc_query_cache = None
    if "shape" in need and pdef["ulip2_mode"] != "cross":
        pc_query_cache = _load_or_build_pc_query_cache(paths, pipe_cfg, index, shape_m)

    by_id = {q["id"]: q for q in index}
    store = {"pass": pass_key, "object_ids": object_ids, "queries": {}}
    for qi, qid in enumerate(qids):
        q = by_id[qid]
        rec: dict = {}
        roi = None
        if "clip" in need or "dino" in need:
            roi = Image.open(q["png"]).convert("RGB")

        if "clip" in need:
            res = clip_retr.retrieve(roi, top_k=10 ** 6)
            smap = {c.object_id: c.score for c in res.candidates}
            rec["clip"] = np.array([smap.get(o, -np.inf) for o in object_ids],
                                   dtype=np.float32)

        if "dino" in need:
            with torch.no_grad():
                qe = dino_rer.encode_image(roi)                  # (1, D)
                sims = (qe @ dino_big.T).squeeze(0).float().cpu()
            rec["dino"] = {
                V: _aggregate_groups(sims, dino_ranges, V,
                                     pipe_cfg.dino_view_aggregation,
                                     pipe_cfg.dino_view_topk,
                                     pipe_cfg.dino_view_temperature)
                for V in VIEW_BUDGETS}

        if "shape" in need:
            if pdef["ulip2_mode"] == "cross":
                qe = ulip_img_cache.get(q["png"])
                if qe is None:                       # encoded on the fly
                    qe = shape_m.encode_image(roi or Image.open(q["png"]))
                qe = qe.to(pipe_cfg.device)
            else:
                qe = pc_query_cache[qid].to(pipe_cfg.device)
            with torch.no_grad():
                sims = (qe.float() @ shape_big.T).squeeze(0).float().cpu()
            # partial refs: multi-view pooling over the first SHAPE_AGG_VIEWS
            # (FPS-ordered) of the cached views — the cache still holds all 42
            # encoded; full-mesh refs have a single embedding per object.
            rec["shape"] = _aggregate_groups(
                sims, shape_ranges,
                SHAPE_AGG_VIEWS if pdef["partial"] else None,
                pipe_cfg.ulip_view_aggregation if pdef["partial"] else "max",
                pipe_cfg.ulip_view_topk, pipe_cfg.ulip_view_temperature)

        store["queries"][qid] = rec
        if (qi + 1) % 50 == 0 or qi + 1 == len(qids):
            print(f"[pass:{pass_key}]   {qi + 1}/{len(qids)}")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(store, cache)
    print(f"[pass:{pass_key}] cached -> {cache}")

    # Free GPU memory before the next pass (encoders + big matrices).
    del clip_retr, dino_rer, shape_m, dino_big, shape_big
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store


# ===========================================================================
# 5. Ablation registry
# ===========================================================================

@dataclass(frozen=True)
class AblationSpec:
    """One cell of the Stage-1 grid (thesis tab:eval_stage1_ablation_grid).

    ``channels`` maps channel name -> (pass_key, view_budget|None).  A
    channel that is absent is excluded from fusion.  ``alias_of`` marks
    grid cells that are numerically identical to another cell (e.g. the
    "default" arm of most ablations is the BASE config E1c) — they are
    not recomputed, only cross-referenced in the summary.
    """
    name: str
    group: str
    question: str
    thesis_ref: str = "subsec:eval_baselines"
    bib: str = ""
    channels: Dict[str, tuple] = field(default_factory=dict)
    weights: Tuple[float, float, float] = BASE_WEIGHTS
    fusion_method: str = "weighted_sum"
    scope: str = "full"          # "full" | "clip_topk" | "dino_topk"
    geometry: Optional[str] = None  # None | fitness | chamfer_unaligned |
                                    # chamfer_ransac | chamfer_icp | scale_gate
    alias_of: Optional[str] = None
    notes: str = ""

    @property
    def passes(self) -> Tuple[str, ...]:
        return tuple(sorted({p for p, _ in self.channels.values()}))

    @property
    def needs_gedi(self) -> bool:
        # Every signal except the unaligned diagnostic needs GeDi, since
        # alignment is what GeDi correspondences are for.
        return self.geometry in ("fitness", "chamfer_ransac", "chamfer_icp")


# Stage-1 BASE.  S_shape is scored in **pc-mode**: SHREC'18 queries are real
# RGB-D scans, so the shape channel encodes the query *point cloud* rather
# than the query image.  (Stage 2 / MI3DOR keeps cross-mode because it has no
# depth — thesis subsec:eval_datasets_mi3dor_restricted_role.)
_BASE_CH = {"clip": ("base", None), "dino": ("base", 42),
            "shape": ("ulip_pc_rgb", None)}
_TV_CH = {"clip": ("base", None), "dino": ("base", 42)}   # text+view only


def _spec(name, group, question, **kw) -> Tuple[str, AblationSpec]:
    return name, AblationSpec(name=name, group=group, question=question, **kw)


ABLATIONS: "OrderedDict[str, AblationSpec]" = OrderedDict([
    # --- E1: does the global shape channel help? (thesis E1) -------------
    # "S_text alone; (S_text,S_view) OSCAR-equivalent; full fusion; plus a
    #  CLIP-pruned variant — all variants produced in a single pass."
    _spec("E1a_text_only", "E1", "S_text alone",
          bib="pulliOSCAROpenSetCAD2025",
          channels={"clip": ("base", None)}, weights=(1.0, 0.0, 0.0)),
    _spec("E1_view_only", "E1", "S_view alone (appearance channel)",
          bib="nguyenCNOSStrongBaseline2023",
          channels={"dino": ("base", 42)}, weights=(0.0, 1.0, 0.0)),
    _spec("E1_shape_only", "E1", "S_shape alone (pc-mode)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={"shape": ("ulip_pc_rgb", None)}, weights=(0.0, 0.0, 1.0)),
    _spec("E1b_text_view", "E1",
          "(S_text, S_view) — OSCAR's channel set, full-database fusion",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0)),
    # Faithful OSCAR reproduction: CLIP selects the top-k shortlist, DINOv2
    # arg-maxes *within* it, and there is no shape channel at all
    # (thesis background.tex, "OSCAR" paragraph).  Weight (0,1,0) + scope
    # clip_topk makes derive_ranking take the single-channel shortcut: the
    # pool is ordered by S_view alone, the tail by S_text.  This differs from
    # E1d/O2_clip_cascade, which score S_view AND S_shape on the shortlist.
    _spec("E1_oscar_cascade", "E1",
          "OSCAR cascade: CLIP top-20 shortlist, DINOv2 arg-max within it",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_TV_CH), weights=(0.0, 1.0, 0.0), scope="clip_topk"),
    _spec("E1c_full_fusion", "E1",
          "full fusion (S_text, S_view, S_shape) — OSCAR+ BASE config",
          bib="pulliOSCAROpenSetCAD2025, zhouCrossModal3DRepresentation",
          channels=dict(_BASE_CH)),
    _spec("E1d_clip_pruned", "E1",
          "S_view/S_shape scored only on the CLIP top-20 shortlist",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_topk"),
    # --- E2: local geometry re-ranking (thesis E2) -----------------------
    # The five configurations of subsec:eval_baselines.  Alignment requires
    # correspondences, so every *aligned* variant necessarily runs GeDi ->
    # RANSAC first; a distance without GeDi is by construction unaligned and
    # serves only as the diagnostic control.
    _spec("E2_none", "E2", "no geometry re-ranking (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E2_fitness", "E2", "RANSAC fitness only (no surface distance)",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels=dict(_BASE_CH), geometry="fitness"),
    _spec("E2_chamfer_unaligned", "E2",
          "unaligned trimmed distance — diagnostic control, not a method",
          bib="diUREDUnsupervised3D2023",
          channels=dict(_BASE_CH), geometry="chamfer_unaligned",
          notes=("Expected to underperform E2_chamfer_ransac; it exists to "
                 "verify that the gain comes from evaluating the distance "
                 "AFTER alignment (thesis subsec:eval_baselines).")),
    _spec("E2_chamfer_ransac", "E2",
          "GeDi-RANSAC alignment, then trimmed surface distance",
          bib="caraffaFreeZeTrainingfreeZeroshot2025, diUREDUnsupervised3D2023",
          channels=dict(_BASE_CH), geometry="chamfer_ransac"),
    _spec("E2_chamfer_icp", "E2",
          "GeDi-RANSAC + ICP refinement, then trimmed surface distance",
          bib="caraffaFreeZeTrainingfreeZeroshot2025",
          channels=dict(_BASE_CH), geometry="chamfer_icp"),
    # --- E2b: partial-view vs full-mesh shape reference ------------------
    _spec("E2b_partial", "E2b", "partial rendered views as S_shape "
          "reference (= BASE)", bib="linSAM6DSegmentAnything2024",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E2b_fullmesh", "E2b", "full-mesh S_shape reference",
          bib="diUREDUnsupervised3D2023",
          channels={**_TV_CH, "shape": ("ulip_pc_fullmesh", None)}),
    # --- E4: appearance encoder ------------------------------------------
    _spec("E4_dinov2", "E4", "DINOv2 appearance channel (= BASE)",
          bib="nguyenCNOSStrongBaseline2023",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E4_siglip", "E4", "SigLIP appearance channel",
          bib="zhaiSigmoidLossLanguage2023, nguyenSHREC2025Retrieval2025",
          channels={**_BASE_CH, "dino": ("siglip", 42)}),
    # --- E6: fusion strategy ---------------------------------------------
    _spec("E6_weighted", "E6", "weighted-sum fusion (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    # Rank-based fusion = Reciprocal Rank Fusion (k=60), the standard
    # scale-free rank combiner.  SAMURAI/COMPASS motivate *why* rank-based
    # fusion is worth testing, but SAMURAI's own vote runs over text/
    # silhouette retrieval strategies, not over our three channels — so RRF
    # is cited on its own terms rather than as a SAMURAI reproduction.
    _spec("E6_rrf", "E6", "reciprocal rank fusion (RRF, k=60)",
          bib="cormackReciprocalRankFusion2009",
          channels=dict(_BASE_CH), fusion_method="rank_fusion"),
    # --- E7: 3D foundation model in S_shape ------------------------------
    # Both arms run in pc-mode so encoder choice is the only difference
    # (Uni3D has no ULIP-style image branch; cross-modal would confound).
    # Since Stage-1 BASE is itself pc-mode, the ULIP-2 arm *is* the BASE
    # config — cross-referenced rather than recomputed.
    _spec("E7_ulip2_pc", "E7", "ULIP-2 shape encoder (pc-mode) (= BASE)",
          bib="xueULIP2ScalableMultimodal2024",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("E7_uni3d", "E7", "Uni3D shape encoder (pc-mode)",
          bib="zhouUni3DExploringUnified2023, "
              "vandenherrewegenFinetuning3DFoundation2024",
          channels={**_TV_CH, "shape": ("uni3d", None)}),
    # --- O1: is S_shape redundant once S_GeDi exists? --------------------
    _spec("O1a_no_geometry", "O1", "neither S_shape nor S_GeDi",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0),
          alias_of="E1b_text_view"),
    _spec("O1b_shape_in_fusion", "O1", "S_shape in fusion (= BASE)",
          channels=dict(_BASE_CH), alias_of="E1c_full_fusion"),
    _spec("O1c_gedi_post_fusion", "O1",
          "S_GeDi as post-fusion re-ranker (no S_shape)",
          channels=dict(_TV_CH), weights=(0.43, 0.57, 0.0),
          geometry="fitness"),
    _spec("O1d_shape_plus_gedi", "O1", "S_shape in fusion + S_GeDi rerank",
          channels=dict(_BASE_CH), geometry="fitness",
          alias_of="E2_fitness"),
    # NOTE: "S_GeDi replaces S_shape inside the fusion score" (the former
    # O1e) is intentionally absent.  It requires a GeDi score for EVERY
    # gallery entry, i.e. |queries| x |gallery| = 1,452 x 3,308 ~ 4.8M RANSAC
    # fits — see --bench-gedi for the measured per-fit cost.  S_GeDi is
    # therefore only evaluable as a post-fusion re-ranker (O1c/O1d).
    # --- O2: scope and ordering ------------------------------------------
    _spec("O2_full_database", "O2", "simultaneous full-database fusion "
          "(= BASE)", channels=dict(_BASE_CH),
          alias_of="E1c_full_fusion"),
    _spec("O2_clip_cascade", "O2", "OSCAR cascade: CLIP top-20 shortlist",
          bib="pulliOSCAROpenSetCAD2025",
          channels=dict(_BASE_CH), scope="clip_topk",
          alias_of="E1d_clip_pruned"),
    _spec("O2_visual_first", "O2", "visual-first: S_view prunes to top-20",
          bib="nguyenSHREC2025Retrieval2025",
          channels=dict(_BASE_CH), scope="dino_topk"),
    # --- O4: number of reference views (appearance channel) --------------
    *[_spec(f"O4_V{v}", "O4", f"V = {v} FPS-ordered reference views",
            bib="nguyenCNOSStrongBaseline2023",
            channels={**_BASE_CH, "dino": ("base", v)},
            **({"alias_of": "E1c_full_fusion"} if v == 42 else {}))
      for v in VIEW_BUDGETS],
    # --- O5: query point cloud colors (pc-mode, see E7 note) -------------
    _spec("O5_xyzrgb", "O5", "XYZ+RGB query point cloud (pc-mode)",
          bib="xueULIP2ScalableMultimodal2024",
          channels={**_TV_CH, "shape": ("ulip_pc_rgb", None)},
          alias_of="E7_ulip2_pc"),
    _spec("O5_xyz_only", "O5", "XYZ-only query point cloud (colors zeroed)",
          channels={**_TV_CH, "shape": ("ulip_pc_xyz", None)}),
])


def select_ablations(arg: Optional[str], run_all: bool,
                     with_geometry: bool) -> List[AblationSpec]:
    """Resolve --ablations/--all into an ordered spec list."""
    if run_all:
        picked = list(ABLATIONS.values())
    else:
        picked, wanted = [], [t.strip() for t in (arg or "").split(",") if t.strip()]
        if not wanted:
            raise SystemExit("Nothing selected — use --all, --ablations or --list.")
        for tok in wanted:
            hits = [s for s in ABLATIONS.values()
                    if s.name == tok or s.group == tok]
            if not hits:
                raise SystemExit(f"Unknown ablation '{tok}' (see --list).")
            picked.extend(h for h in hits if h not in picked)
    if not with_geometry:
        dropped = [s.name for s in picked if s.geometry]
        picked = [s for s in picked if not s.geometry]
        if dropped:
            print(f"[select] geometry ablations skipped (add "
                  f"--with-geometry): {', '.join(dropped)}")
    return picked


# ===========================================================================
# 6. Tier 2 — derivations (fusion / scoping / geometry on cached vectors)
# ===========================================================================

def _synthetic_results(spec: AblationSpec, vecs: Dict[str, np.ndarray],
                       pool: np.ndarray, object_ids: List[str],
                       cad_dir: str):
    """Wrap cached score vectors into the pipeline's result dataclasses.

    This is what makes tier 2 faithful: the production ScoreFusion
    consumes CLIPRetrievalResult / DINOReRankingResult /
    ShapeMatchingResult exactly as in the live pipeline (step6_fusion).
    Candidates are sorted per channel because majority voting (E6)
    consumes rank order.
    """
    from pipeline.step3_clip_retrieval import (CLIPCandidate,
                                               CLIPRetrievalResult)
    from pipeline.step4_dino_reranking import (DINOCandidate,
                                               DINOReRankingResult)
    from pipeline.step5_shape_matching import (ShapeCandidate,
                                               ShapeMatchingResult)

    dummy = np.zeros(1, dtype=np.float32)

    def ordered(ch):
        v = vecs[ch]
        return pool[np.argsort(-v[pool], kind="stable")]

    clip_r = dino_r = shape_r = None
    if "clip" in vecs:
        clip_r = CLIPRetrievalResult(
            candidates=[CLIPCandidate(object_id=object_ids[i],
                                      score=float(vecs["clip"][i]))
                        for i in ordered("clip")],
            query_embedding=dummy)
    if "dino" in vecs:
        dino_r = DINOReRankingResult(
            candidates=[DINOCandidate(object_id=object_ids[i],
                                      dino_score=float(vecs["dino"][i]),
                                      clip_score=0.0)
                        for i in ordered("dino")],
            query_embedding=dummy)
    if "shape" in vecs:
        shape_r = ShapeMatchingResult(
            candidates=[ShapeCandidate(
                object_id=object_ids[i],
                shape_score=float(vecs["shape"][i]),
                cad_model_path=os.path.join(cad_dir, object_ids[i] + ".obj"))
                for i in ordered("shape")],
            query_embedding=dummy)
    return clip_r, dino_r, shape_r


def derive_ranking(spec: AblationSpec, qid: str, stores: Dict[str, dict],
                   object_ids: List[str], fusion_mod, cad_dir: str
                   ) -> List[int]:
    """Rank the gallery for one query under one ablation config.

    Returns gallery indices in rank order (full list — the tail beyond a
    pruned shortlist is ordered by the pruning channel, mirroring how a
    cascade would leave non-shortlisted items ranked by its first stage).
    """
    n = len(object_ids)
    vecs: Dict[str, np.ndarray] = {}
    for ch, (pkey, budget) in spec.channels.items():
        rec = stores[pkey]["queries"][qid][ch]
        v = rec[budget] if isinstance(rec, dict) else rec
        # -inf marks gallery objects a channel could not score (missing
        # views/embeddings).  Clamp to the channel minimum so min-max
        # normalization in ScoreFusion stays well-defined; such objects
        # simply rank last in that channel.
        if not np.isfinite(v).all():
            finite = v[np.isfinite(v)]
            v = np.nan_to_num(v, nan=0.0, posinf=0.0,
                              neginf=float(finite.min()) if finite.size else 0.0)
        vecs[ch] = v

    # --- scope: candidate pool + tail ordering (thesis O2) ---------------
    if spec.scope == "clip_topk":
        prune = vecs["clip"]
    elif spec.scope == "dino_topk":
        prune = vecs["dino"]
    else:
        prune = None
    if prune is not None:
        order = np.argsort(-prune, kind="stable")
        pool, tail = order[:CLIP_PRUNE_K], order[CLIP_PRUNE_K:]
    else:
        pool, tail = np.arange(n), np.array([], dtype=int)

    # --- single-channel shortcut (E1a) ------------------------------------
    active = {ch for ch, w in zip(("clip", "dino", "shape"), spec.weights)
              if ch in vecs and w > 0}
    if spec.fusion_method == "weighted_sum" and len(active) == 1:
        ch = active.pop()
        ranked = pool[np.argsort(-vecs[ch][pool], kind="stable")]
        return list(ranked) + list(tail)

    # --- production fusion (step6_fusion.ScoreFusion) ---------------------
    sub = {ch: v for ch, v in vecs.items() if ch in active}
    clip_r, dino_r, shape_r = _synthetic_results(
        spec, sub, pool, object_ids, cad_dir)
    fused = fusion_mod.fuse(clip_r, dino_r, shape_r,
                            method=spec.fusion_method, top_k=n)
    idx = {oid: i for i, oid in enumerate(object_ids)}
    ranked = [idx[c.object_id] for c in fused.candidates]
    seen = set(ranked)
    ranked += [i for i in pool if i not in seen]     # fusion dropped none, but be safe
    return ranked + list(tail)


def make_fusion_module(spec: AblationSpec):
    """ScoreFusion configured with this spec's weights/method.

    Reuses pipeline/step6_fusion.py — weights follow thesis Step B1
    (w_text, w_view, w_shape) = (0.3, 0.4, 0.3) unless the ablation
    renormalizes after dropping a channel (E1b/O1a: (0.43, 0.57, 0)).
    """
    from pipeline.config import PipelineConfig
    from pipeline.step6_fusion import ScoreFusion
    cfg = PipelineConfig(
        weight_clip=spec.weights[0], weight_dino=spec.weights[1],
        weight_ulip=spec.weights[2], fusion_method=spec.fusion_method)
    return ScoreFusion(cfg)


# ===========================================================================
# 7. Geometry re-ranking (E2, O1c-e) — Sub-step B2 on the SHREC'18 pairs
# ===========================================================================
# SHREC'18 queries are metric SceneNN crops while ShapeNetSem CADs use
# arbitrary model units, so both clouds are normalized to the unit sphere
# before GeDi/Chamfer — scale is not a retrieval cue here (it is ablated
# separately by the scale gate).  voxel/correspondence radii are set for
# unit-sphere scale.  GeDi = RANSAC inlier count [FreeZe:
# caraffaFreeZeTrainingfreeZeroshot2025]; Chamfer = trimmed one-sided
# distance (trim 10%, U-RED-style [diUREDUnsupervised3D2023]).

GEOM_VOXEL = 0.02   # unit-sphere-scale voxel size for B2 (repo default
                    # 0.002 assumes metric tabletop scenes)
GEDI_RETRIES = 4    # retries when the GeDi service drops out mid-run
GEDI_WAIT_S = 300   # how long to wait for it to come back before aborting
QUERY_MAX_PTS = 500_000   # cap on query cloud size fed to GeDi/RANSAC
                          # (larger clouds crash the service; see _query_cloud)


class _GeometryEngine:
    """Lazy wrapper around pipeline.step_b2_geometry_reranking with
    per-(query, cad) score caching (append-only jsonl, resumable)."""

    def __init__(self, paths: dict):
        self.paths = paths
        self.cache_path = os.path.join(paths["results_root"], "_cache",
                                       "geometry_scores.jsonl")
        self.cache: Dict[Tuple[str, str], dict] = {}
        if os.path.isfile(self.cache_path):
            with open(self.cache_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.cache[(r["qid"], r["cad"])] = r
                    except (json.JSONDecodeError, KeyError):
                        continue
        self._reranker = None
        self._gedi_available = None
        self._cad_clouds: "OrderedDict[str, object]" = OrderedDict()
        self._extents_path = os.path.join(paths["results_root"], "_cache",
                                          "cad_extents.json")
        self._extents: Dict[str, list] = {}
        if os.path.isfile(self._extents_path):
            with open(self._extents_path) as f:
                self._extents = json.load(f)

    # -- infrastructure ----------------------------------------------------
    def _get_reranker(self):
        if self._reranker is None:
            from pipeline.config import PipelineConfig
            from pipeline.step_b2_geometry_reranking import GeometryReRanker

            cad_dir = os.path.join(self.paths["data_root"], "cad")
            cloud_cache = self._cad_clouds

            class UnitSphereReRanker(GeometryReRanker):
                """B2 re-ranker with unit-sphere CAD normalization.

                Overrides the parent's metric-scale CAD loader: SHREC'18
                CADs (ShapeNetSem) have arbitrary units, so every cloud
                is centered and scaled to the unit sphere (same frame as
                the query cloud built in _query_cloud)."""

                def _load_cad_pointcloud(self, cad_path, n_points=10000):
                    import open3d as o3d
                    oid = os.path.splitext(os.path.basename(cad_path))[0]
                    if oid in cloud_cache:
                        cloud_cache.move_to_end(oid)
                        return cloud_cache[oid]
                    mesh = o3d.io.read_triangle_mesh(
                        os.path.join(cad_dir, oid + ".obj"))
                    if mesh.is_empty():
                        return None
                    pcd = mesh.sample_points_uniformly(n_points)
                    pts = np.asarray(pcd.points)
                    pts -= pts.mean(axis=0)
                    r = np.linalg.norm(pts, axis=1).max()
                    if r > 0:
                        pts /= r
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(
                            radius=GEOM_VOXEL * 4, max_nn=30))
                    cloud_cache[oid] = pcd
                    while len(cloud_cache) > 400:     # ~50 MB ceiling
                        cloud_cache.popitem(last=False)
                    return pcd

            cfg = PipelineConfig(voxel_size=GEOM_VOXEL,
                                 geometry_reranking_top_k=GEOM_SHORTLIST)
            self._reranker = UnitSphereReRanker(cfg)
        return self._reranker

    def gedi_available(self) -> bool:
        if self._gedi_available is None:
            # A SINGLE 5s probe is not enough to decide whether to skip four
            # ablations: on 2026-07-28 it timed out against a service that was
            # healthy throughout (the GeDi Flask server is single-threaded, so
            # /health blocks behind any in-flight descriptor call, and a
            # cold container's first DNS lookup adds to that).  The cost of a
            # false negative here is silently dropping E2_fitness /
            # chamfer_ransac / chamfer_icp / O1c from the grid, so retry.
            self._gedi_available = self._wait_for_gedi(GEDI_WAIT_S)
            if not self._gedi_available:
                print("[geometry] GeDi service unreachable "
                      "(docker compose up -d gedi) — GeDi-signal "
                      "ablations will be skipped.")
        return self._gedi_available

    def _gedi_healthy(self) -> bool:
        """LIVE probe of the GeDi service (unlike the cached gedi_available).

        Needed to tell a genuine registration failure apart from the service
        having died mid-run.  The two are indistinguishable at the call site
        — both surface as ``registration_failed`` — but only the first may be
        written to the resumable cache.
        """
        try:
            import requests
            from pipeline.config import PipelineConfig
            url = getattr(PipelineConfig(), "gedi_url", "http://gedi:5060")
            # Generous timeout: the service is a single-threaded Flask dev
            # server, so /health queues behind any in-flight descriptor
            # computation (~5s each) rather than answering immediately.
            return requests.get(f"{url}/health", timeout=30).status_code == 200
        except Exception:
            return False

    def _wait_for_gedi(self, timeout_s: int) -> bool:
        """Block until the GeDi service answers again, or give up.

        Covers the compose auto-restart window (model reload ~60-90s).
        """
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._gedi_healthy():
                print("[geometry] GeDi is back — resuming.", flush=True)
                time.sleep(5)          # let the model finish warming up
                return True
            time.sleep(10)
        return False

    def _query_cloud(self, npz_path: str):
        import open3d as o3d
        data = np.load(npz_path)
        pts = data["points"].astype(np.float64)
        pts -= pts.mean(axis=0)
        r = np.linalg.norm(pts, axis=1).max()
        if r > 0:
            pts /= r
        # Cap only the pathological clouds.  SHREC'18 scans run from 540 to
        # 5.7M points (median 27.5k); 145 of the 2,101 exceed 1M, and handing
        # one of those to GeDi kills the service outright (silent `exit=0`,
        # observed 5x on query 0d2ff0fffe... / 4.45M points).
        #
        # Deliberately NOT voxel_down_sample(GEOM_VOXEL): that was tried on
        # 2026-07-28 and, while it did stop the crashes, it decimated every
        # query to 0.5-14k points and RANSAC then ran to its iteration limit
        # on the sparse correspondences — 40 min/query vs 26 s (~58 days for
        # the grid).  Dense queries are what makes registration converge fast.
        # 999k points is the largest cloud observed to work; 500k keeps a 2x
        # margin under that and 9x under the smallest crash.
        if len(pts) > QUERY_MAX_PTS:
            sel = np.random.default_rng(0).choice(      # deterministic
                len(pts), QUERY_MAX_PTS, replace=False)
            pts = pts[np.sort(sel)]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=GEOM_VOXEL * 4, max_nn=30))
        return pcd

    def _append_cache(self, rec: dict) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # -- scoring -----------------------------------------------------------
    def pair_scores(self, qid: str, npz_path: str, cad_ids: Sequence[str],
                    signal: str) -> Dict[str, dict]:
        """Geometry scores for (qid x cad_ids); cached pairs are free.

        Cache records are keyed per SIGNAL, because D_trim under different
        alignments is a different quantity: the unaligned diagnostic, the
        RANSAC-aligned distance and the ICP-refined distance must never be
        read from one another's entries.
        """
        # Field names are signal-scoped; "fitness" is shared by every signal
        # that performs a registration (it is the same RANSAC either way).
        #
        # The three ALIGNED signals all require the same GeDi + RANSAC step,
        # which at ~35 s/query dominates everything else (the trimmed distance
        # on its own — the chamfer_unaligned ablation — is ~0.5 s/query, and
        # ICP is local refinement).  So they are computed together in one pass
        # (rerank(all_aligned=True)) and cached together: whichever aligned
        # ablation runs first pays the registration, the other two are free
        # cache hits.  Scoring them separately cost ~60 h instead of ~26 h
        # over the 2,101 official queries.
        _ALIGNED_FIELDS = ("fitness", "d_ransac", "d_icp")
        need_fields = {
            "fitness": _ALIGNED_FIELDS,
            "chamfer_unaligned": ("d_unaligned",),
            "chamfer_ransac": _ALIGNED_FIELDS,
            "chamfer_icp": _ALIGNED_FIELDS,
        }[signal]
        all_aligned = need_fields is _ALIGNED_FIELDS
        missing = [c for c in cad_ids
                   if any(self.cache.get((qid, c), {}).get(f) is None
                          for f in need_fields)]
        if missing:
            from pipeline.step6_fusion import FusedCandidate
            rr = self._get_reranker()
            obs = self._query_cloud(npz_path)
            cands = [FusedCandidate(object_id=c, fused_score=0.0,
                                    cad_model_path=c) for c in missing]
            rr.config.geometry_reranking_top_k = len(cands)
            # A dead GeDi service fails every fit, and those failures must NOT
            # be cached as if they were real (the B2 policy then ranks the
            # candidate last, permanently).  On 2026-07-27 this poisoned 2,845
            # pairs across 10h before anyone noticed.
            #
            # The service also restarts on its own (compose `restart:
            # unless-stopped`) and needs ~60-90s to reload the model, so a
            # blanket abort would end a 15h run over a 90s blip.  Wait for it
            # to come back and retry; abort only if it stays down.
            for attempt in range(1, GEDI_RETRIES + 1):
                res = rr.rerank(cands, obs, signal=signal,
                                all_aligned=all_aligned)
                n_bad = sum(1 for gc in res.candidates
                            if gc.registration_failed)
                if (signal == "chamfer_unaligned" or not n_bad
                        or self._gedi_healthy()):
                    break                      # genuine result — cache it
                print(f"[geometry] GeDi unreachable ({n_bad}/"
                      f"{len(res.candidates)} fits failed for query {qid}) — "
                      f"waiting for it to come back "
                      f"(attempt {attempt}/{GEDI_RETRIES})", flush=True)
                if not self._wait_for_gedi(GEDI_WAIT_S):
                    raise SystemExit(
                        f"[geometry] GeDi service still down after "
                        f"{GEDI_WAIT_S}s — aborting rather than caching bogus "
                        f"failures.\n"
                        f"  Restart it:  docker compose up -d gedi\n"
                        f"  Then re-run; completed pairs are already cached.")
            else:
                raise SystemExit(
                    f"[geometry] GeDi kept failing across {GEDI_RETRIES} "
                    f"attempts on query {qid} — aborting rather than caching "
                    f"bogus failures.")
            dist_field = {"chamfer_unaligned": "d_unaligned",
                          "chamfer_ransac": "d_ransac",
                          "chamfer_icp": "d_icp"}.get(signal)

            def _num(x):
                return None if x is None or np.isinf(x) else float(x)

            for gc in res.candidates:
                rec = dict(self.cache.get((qid, gc.object_id),
                                          {"qid": qid, "cad": gc.object_id}))
                rec["failed"] = bool(gc.registration_failed)
                if "fitness" in need_fields:
                    rec["fitness"] = float(gc.ransac_fitness)
                if all_aligned:
                    # One registration, all three aligned readouts.
                    rec["d_ransac"] = _num(gc.d_ransac)
                    rec["d_icp"] = _num(gc.d_icp)
                    rec["icp_fitness"] = float(gc.icp_fitness)
                    rec["icp_rmse"] = float(gc.icp_inlier_rmse)
                elif dist_field:
                    rec[dist_field] = _num(gc.chamfer_score)
                self.cache[(qid, gc.object_id)] = rec
                self._append_cache(rec)
        return {c: self.cache.get((qid, c), {}) for c in cad_ids}

    def geometry_score(self, rec: dict, signal: str) -> Tuple[float, float]:
        """Sort key mirroring GeometryReRanker's ranking rule.

        Returns (primary, tiebreak) with higher = better: failed
        registrations sink to -inf, distance signals rank by -D_trim, and
        RANSAC fitness breaks ties (thesis Sec. 3.3 — no arbitrary scale
        factor combining an inlier statistic with a distance).
        """
        fitness = rec.get("fitness") or 0.0
        if rec.get("failed"):
            return (float("-inf"), fitness)
        if signal == "fitness":
            return (fitness, fitness)
        field = {"chamfer_unaligned": "d_unaligned",
                 "chamfer_ransac": "d_ransac",
                 "chamfer_icp": "d_icp"}[signal]
        d = rec.get(field)
        if d is None:
            return (float("-inf"), fitness)
        return (-float(d), fitness)

    # -- scale gate (legacy E2 variant, thesis "Scale gate (legacy)") ------
    def cad_extent(self, cad_id: str) -> Optional[np.ndarray]:
        if cad_id not in self._extents:
            try:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(os.path.join(
                    self.paths["data_root"], "cad", cad_id + ".obj"))
                ext = mesh.get_axis_aligned_bounding_box().get_extent()
                self._extents[cad_id] = [float(x) for x in ext]
            except Exception:
                self._extents[cad_id] = None
            with open(self._extents_path, "w") as f:
                json.dump(self._extents, f)
        e = self._extents.get(cad_id)
        return None if e is None else np.array(e)

    def scale_gate(self, npz_path: str, cad_ids: List[str]) -> List[str]:
        """Promote the first candidate whose sorted-bbox scale ratio lies
        in [scale_gate_min, scale_gate_max] (rotation-invariant estimate,
        cf. pipeline/config.py scale_gate_*).  reject_policy =
        fallback_best: order unchanged when nothing passes.  ShapeNetSem
        units are heterogeneous, so frequent rejection is an *expected*
        outcome of this ablation, not a failure."""
        q_ext = np.sort(np.ptp(np.load(npz_path)["points"], axis=0))[::-1]
        for i, cid in enumerate(cad_ids):
            c_ext = self.cad_extent(cid)
            if c_ext is None or (c_ext <= 0).any():
                continue
            scale = float(np.median(q_ext / np.sort(c_ext)[::-1]))
            if 0.8 <= scale <= 1.2:
                return [cid] + [c for j, c in enumerate(cad_ids) if j != i]
        return cad_ids


def bench_gedi(paths: dict, index: List[dict], object_ids: List[str],
               n_pairs: int) -> None:
    """Measure the per-pair GeDi+RANSAC cost and extrapolate to full-database.

    Backs the thesis claim that using S_GeDi *inside* the fusion score (rather
    than as a post-fusion re-ranker) is computationally infeasible: it would
    need one registration per (query, gallery entry) pair.  Descriptor
    extraction on the gallery side is amortisable; the RANSAC fit is not, so
    that is what is timed here.
    """
    import time

    geom = _GeometryEngine(paths)
    if not geom.gedi_available():
        raise SystemExit("[bench] GeDi service unavailable — start it with "
                         "`docker compose up -d gedi` and retry.")

    rng = np.random.RandomState(42)
    q = index[int(rng.randint(len(index)))]
    cads = [object_ids[i] for i in rng.choice(len(object_ids),
                                              min(n_pairs, len(object_ids)),
                                              replace=False)]
    print(f"[bench] timing {len(cads)} GeDi+RANSAC fits "
          f"(query {q['id']}) ...")

    t0 = time.perf_counter()
    for cad in cads:
        # One pair at a time, bypassing the cache, so the timing reflects a
        # single registration rather than a batched shortlist.
        geom.cache.pop((q["id"], cad), None)
        geom.pair_scores(q["id"], q["npz"], [cad], "chamfer_ransac")
    elapsed = time.perf_counter() - t0

    per_fit = elapsed / max(len(cads), 1)
    total_s = per_fit * GEDI_FULL_DB_PAIRS
    print(f"\n[bench] {len(cads)} fits in {elapsed:.1f}s "
          f"-> {per_fit * 1000:.0f} ms per fit")
    print(f"[bench] full-database S_GeDi = 1,452 queries x {N_CADS} CADs "
          f"= {GEDI_FULL_DB_PAIRS:,} fits")
    print(f"[bench] extrapolated: {total_s / 3600:.0f} h "
          f"({total_s / 86400:.1f} days) of RANSAC alone, per ablation cell.")
    print("[bench] -> S_GeDi is reported as a post-fusion re-ranker only "
          "(thesis O1c/O1d).")


def apply_geometry(spec: AblationSpec, qid: str, npz_path: str,
                   ranking: List[int], object_ids: List[str],
                   geom: _GeometryEngine) -> List[int]:
    """Apply the spec's geometry stage to a derived base ranking."""
    idx = {oid: i for i, oid in enumerate(object_ids)}

    if spec.geometry == "scale_gate":
        top = [object_ids[i] for i in ranking[:GEOM_SHORTLIST]]
        new = geom.scale_gate(npz_path, top)
        return [idx[o] for o in new] + ranking[GEOM_SHORTLIST:]

    # Sub-step B2: re-order the fusion top-k shortlist.
    top = [object_ids[i] for i in ranking[:GEOM_SHORTLIST]]
    recs = geom.pair_scores(qid, npz_path, top, spec.geometry)
    scored = sorted(top, key=lambda o: geom.geometry_score(
        recs[o], spec.geometry), reverse=True)
    return [idx[o] for o in scored] + ranking[GEOM_SHORTLIST:]


# ===========================================================================
# 8. Per-ablation evaluation loop
# ===========================================================================

def run_ablation(spec: AblationSpec, paths: dict, index: List[dict],
                 stores: Dict[str, dict], object_ids: List[str],
                 cad_labels: Dict[str, tuple], freqs: Dict[str, int],
                 limit: Optional[int],
                 geom: Optional[_GeometryEngine]) -> dict:
    out_dir = os.path.join(paths["results_root"], spec.name)
    os.makedirs(out_dir, exist_ok=True)

    fusion_mod = make_fusion_module(spec)
    cad_dir = os.path.join(paths["data_root"], "cad")
    # Category of every scored gallery model (for the per-query record /
    # first-relevant rank); official scoring itself uses cad_labels+freqs.
    obj_cats = np.array([cad_labels.get(o, (None, None))[0]
                         for o in object_ids], dtype=object)

    qlist = index[:limit] if limit else index
    sums = defaultdict(float)
    per_query = []
    for q in qlist:
        qid = q["id"]
        q_label = tuple(q["category"])          # (category, subcategory)
        qc = q_label[0]
        if freqs.get(qc, 0) == 0:
            continue        # query category unknown to the official GT
        ranking = derive_ranking(spec, qid, stores, object_ids,
                                 fusion_mod, cad_dir)
        if spec.geometry:
            ranking = apply_geometry(spec, qid, q["npz"], ranking,
                                     object_ids, geom)
        ranked_ids = [object_ids[i] for i in ranking]
        m = score_official(ranked_ids, q_label, cad_labels, freqs)
        if m is None:
            continue
        for k, v in m.items():
            sums[k] += v
        rels_cat = obj_cats[np.asarray(ranking)] == qc
        first_rel = int(np.argmax(rels_cat)) + 1 if rels_cat.any() else -1
        per_query.append({
            "id": qid, "category": list(q_label),
            "top10": [[object_ids[i],
                       list(cad_labels.get(object_ids[i], (None, None)))]
                      for i in ranking[:10]],
            "first_relevant_rank": first_rel,
            "AP": round(m["AP"], 4), "nDCG": round(m["nDCG"], 4),
        })

    nq = len(per_query)
    metrics = {k: (sums[k] / nq if nq else float("nan"))
               for k in ("nDCG", "precision", "recall", "F1", "AP",
                         "NNT1", "NNT2")}
    summary = {
        "ablation": spec.name, "group": spec.group,
        "question": spec.question, "thesis_ref": spec.thesis_ref,
        "bib": spec.bib, "notes": spec.notes,
        "num_queries": nq, "gallery_size": len(object_ids),
        "metrics": metrics,
        "config": {
            "channels": {c: list(v) for c, v in spec.channels.items()},
            "weights": list(spec.weights),
            "fusion_method": spec.fusion_method,
            "scope": spec.scope, "geometry": spec.geometry,
            "clip_prune_k": CLIP_PRUNE_K,
            "geometry_shortlist": GEOM_SHORTLIST,
        },
        "limit_queries": limit,
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "results_per_query.json"), "w") as f:
        json.dump(per_query, f)
    print(f"[run:{spec.name}] n={nq}  nDCG={metrics['nDCG']:.4f}  "
          f"P={metrics['precision']:.4f}  mAP={metrics['AP']:.4f}")
    return summary


def _resolve_alias_root(name: str) -> str:
    """Follow an alias chain to the cell that is actually computed.

    Aliases may chain (O5_xyzrgb -> E7_ulip2_pc -> E1c_full_fusion once BASE
    is pc-mode).  Resolving to the root means a cell can be materialized even
    when the intermediate alias is not part of the selected run.
    """
    seen = set()
    while name in ABLATIONS and ABLATIONS[name].alias_of:
        if name in seen:            # defensive: never loop on a cyclic chain
            break
        seen.add(name)
        name = ABLATIONS[name].alias_of
    return name


def write_alias(spec: AblationSpec, paths: dict) -> Optional[dict]:
    """Materialize an alias cell by copying its canonical metrics."""
    root = _resolve_alias_root(spec.alias_of)
    src = os.path.join(paths["results_root"], root,
                       "metrics_summary.json")
    if not os.path.isfile(src):
        return None
    with open(src) as f:
        summary = json.load(f)
    summary.update({"ablation": spec.name, "group": spec.group,
                    "question": spec.question, "alias_of": spec.alias_of})
    out_dir = os.path.join(paths["results_root"], spec.name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ===========================================================================
# 9. Aggregation: CSV + LaTeX + best-config selection
# ===========================================================================

# Official SHREC'18 metric set (metrics.py, all at top-f).  nDCG is the
# graded model-selection metric; AP averaged over queries is the mAP.
METRIC_COLS = ("nDCG", "precision", "recall", "F1", "AP", "NNT1", "NNT2")


def aggregate(paths: dict) -> None:
    """Collect every metrics_summary.json into the Stage-1 deliverables:

      * stage1_summary.csv — machine-readable grid
      * stage1_summary.tex — booktabs rows for tab:eval_stage1_ablation_grid
                             / data for fig:eval_stage1_ablation_bars
      * best_config.json   — argmax DCG (tie-break mAP,
                             subsec:eval_stage1_retrieval), frozen for
                             Stages 2/3a/3b/5.
    """
    rows = []
    for name, spec in ABLATIONS.items():
        p = os.path.join(paths["results_root"], name, "metrics_summary.json")
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            s = json.load(f)
        rows.append((spec, s))
    if not rows:
        print("[aggregate] no results yet.")
        return

    csv_path = os.path.join(paths["results_root"], "stage1_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", "group", "alias_of", "num_queries",
                    *METRIC_COLS])
        for spec, s in rows:
            w.writerow([spec.name, spec.group, s.get("alias_of", ""),
                        s["num_queries"],
                        *[f"{s['metrics'][m]:.4f}" for m in METRIC_COLS]])
    print(f"[aggregate] {len(rows)} configs -> {csv_path}")

    # --- best config: computed (non-alias) cells only ---------------------
    real = [(spec, s) for spec, s in rows if not s.get("alias_of")]
    if not real:
        print("[aggregate] only alias entries present — no best config.")
        return
    best_spec, best_sum = max(
        real, key=lambda t: (t[1]["metrics"]["nDCG"], t[1]["metrics"]["AP"]))
    best = {
        "selection_rule": ("highest (graded) nDCG on SHREC'18 ObjectNN+ "
                           "under the official evaluation, mAP as "
                           "tie-breaker (thesis subsec:eval_stage1_retrieval)"),
        "name": best_spec.name,
        "metrics": best_sum["metrics"],
        "config": best_sum["config"],
        "frozen_for": ["Stage 2 (MI3DOR)", "Stage 3a/3b (BOP pose)",
                       "Stage 5 (efficiency)"],
    }
    best_path = os.path.join(paths["results_root"], "best_config.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"[aggregate] best: {best_spec.name} "
          f"(nDCG={best_sum['metrics']['nDCG']:.4f}) -> {best_path}")

    # --- LaTeX (booktabs, grouped; \_ escaping for names) -----------------
    def tex(s):
        return str(s).replace("_", r"\_")

    lines = [
        "% Auto-generated by experiments/experiment1_shrec18_stage1.py",
        "% Stage 1 results (thesis tab:eval_stage1_ablation_grid /",
        "% fig:eval_stage1_ablation_bars).  Official SHREC'18 metrics",
        "% (hkust-vgd/shrec18 metrics.py); nDCG is the selection metric.",
        r"\begin{tabular}{ll" + "r" * len(METRIC_COLS) + "}",
        r"\toprule",
        "Config & Ablation & " + " & ".join(
            c.replace("_", r"\_") for c in METRIC_COLS) + r" \\",
        r"\midrule",
    ]
    last_group = None
    for spec, s in rows:
        if last_group is not None and spec.group != last_group:
            lines.append(r"\addlinespace")
        last_group = spec.group
        name = tex(spec.name)
        if spec.name == best_spec.name:
            name = r"\textbf{" + name + "}"
        vals = " & ".join(f"{s['metrics'][m]:.3f}" for m in METRIC_COLS)
        lines.append(f"{name} & {tex(spec.group)} & {vals} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    tex_path = os.path.join(paths["results_root"], "stage1_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[aggregate] LaTeX table -> {tex_path}")


# ===========================================================================
# 10. Input validation (data is provided by the user — never downloaded)
# ===========================================================================

def validate_inputs(paths: dict, allow_partial: bool) -> List[str]:
    """Check that the user-provided data is in place; return the canonical
    gallery id list (intersection of CADs, rendered views, descriptions)."""
    problems = []
    data_root = paths["data_root"]
    for sub in ("cad", "rgbd", "results"):
        if not os.path.isdir(os.path.join(data_root, sub)):
            problems.append(f"missing {os.path.join(data_root, sub)} "
                            f"(raw SHREC'18 distribution)")
    if problems:
        raise SystemExit("[validate] " + "; ".join(problems))

    cads = {os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(data_root, "cad"))
            if f.endswith(".obj")}

    imgs_dir = paths["images_dir"]
    rendered = set()
    if os.path.isdir(imgs_dir):
        for d in os.listdir(imgs_dir):
            full = os.path.join(imgs_dir, d)
            if os.path.isdir(full) and any(
                    f.endswith(".png") and not f.endswith("_bg.png")
                    for f in os.listdir(full)):
                rendered.add(d)

    desc_ids = set()
    if os.path.isfile(paths["desc_file"]):
        with open(paths["desc_file"]) as f:
            desc_ids = set(json.load(f).keys())

    object_ids = sorted(cads & rendered & desc_ids)
    print(f"[validate] CADs: {len(cads)}  rendered: {len(rendered)}  "
          f"described: {len(desc_ids)}  -> scored gallery: "
          f"{len(object_ids)}")

    # Partial-view point clouds: without them the S_shape channel falls
    # back to full-mesh encoding (eval_common.build_pipeline), which
    # would silently collapse ablation E2b (partial vs. full mesh).
    with_partial = sum(
        1 for d in object_ids
        if any(f.endswith("_partial.npz")
               for f in os.listdir(os.path.join(imgs_dir, d))))
    if with_partial < len(object_ids):
        print(f"[validate] WARNING: only {with_partial}/{len(object_ids)} "
              f"rendered models have *_partial.npz partial point clouds "
              f"(rendering/generate_partial_pointclouds.py).  E2b's "
              f"partial-view arm needs them; without them the base pass "
              f"degrades to full-mesh references.")

    if len(object_ids) < len(cads):
        msg = (f"[validate] gallery incomplete: {len(object_ids)}/{len(cads)} "
               f"models have renders+descriptions.\n"
               f"  renders:      bash rendering/onboard_and_sync.sh "
               f"--dataset shrec18 ...  (rendering/onboard_dataset.sh)\n"
               f"  descriptions: bash rendering/onboard_dataset.sh "
               f"--dataset shrec18 --step describe")
        if allow_partial and object_ids:
            print(msg + "\n[validate] continuing (--allow-partial-gallery); "
                  "metrics are NOT final Stage-1 numbers.")
        else:
            raise SystemExit(msg + "\nProvide the missing data or pass "
                             "--allow-partial-gallery for a smoke test.")
    return object_ids


# ===========================================================================
# 11. CLI
# ===========================================================================

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Thesis Experiment 1 — Stage 1 RGB-D retrieval tuning "
                    "on SHREC'18 ObjectNN+ (ablation grid; see module "
                    "docstring).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ablations", help="comma list of groups/cells, e.g. "
                                        "'E1,E4,O4_V8'")
    ap.add_argument("--all", action="store_true",
                    help="run the whole registered grid")
    ap.add_argument("--list", action="store_true",
                    help="print the ablation registry and exit")
    ap.add_argument("--resume", action="store_true",
                    help="skip ablations/passes whose outputs already exist")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute even if results exist")
    ap.add_argument("--with-geometry", action="store_true",
                    help="include E2/O1c-e (GeDi service: docker compose "
                         "up -d gedi; chamfer/scale-gate run without it)")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="smoke test on the first N queries")
    ap.add_argument("--allow-partial-gallery", action="store_true",
                    help="run even if renders/descriptions are incomplete")
    ap.add_argument("--precompute", action="store_true",
                    help="gallery PC: build all reference embedding caches "
                         "(no queries/ablations) + a provenance manifest to "
                         "ship to the eval PC")
    ap.add_argument("--bench-gedi", type=int, default=0, metavar="N",
                    help="time N GeDi+RANSAC fits and extrapolate the cost of "
                         "full-database S_GeDi, then exit (see thesis O1)")
    ap.add_argument("--viz-check", type=int, default=0, metavar="N",
                    help="save a contact sheet of N query crops")
    ap.add_argument("--data-root", default=DEFAULTS["data_root"])
    ap.add_argument("--images-dir", default=DEFAULTS["images_dir"])
    ap.add_argument("--desc-file", default=DEFAULTS["desc_file"])
    ap.add_argument("--results-root", default=DEFAULTS["results_root"])
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'name':<24} {'group':<4} {'passes':<28} geometry  question")
        for s in ABLATIONS.values():
            extra = f"alias of {s.alias_of}" if s.alias_of else \
                    (s.geometry or "")
            print(f"{s.name:<24} {s.group:<4} {','.join(s.passes):<28} "
                  f"{extra:<18} {s.question}")
        return

    paths = {"data_root": args.data_root, "images_dir": args.images_dir,
             "desc_file": args.desc_file, "results_root": args.results_root,
             "stage1_root": DEFAULTS["stage1_root"]}
    os.makedirs(os.path.join(paths["results_root"], "_cache"), exist_ok=True)

    # Make repo modules importable regardless of CWD.
    for p in (ROOT, os.path.join(ROOT, "object_retrieval")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Make repo modules importable regardless of CWD (before the selection,
    # which is skipped in --precompute mode).
    # torch bundles libgomp.so.1 that open3d also needs — import it first.
    import torch  # noqa: F401

    # ---- gallery PC: precompute all reference caches and exit ------------
    if args.precompute:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        precompute_gallery(paths, object_ids,
                           resume=args.resume and not args.overwrite)
        return

    # ---- GeDi cost benchmark (thesis O1 feasibility claim) and exit ------
    if args.bench_gedi:
        object_ids = validate_inputs(paths, args.allow_partial_gallery)
        gt = load_official_gt(paths["data_root"], paths["stage1_root"])
        index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
        bench_gedi(paths, index, object_ids, args.bench_gedi)
        return

    specs = select_ablations(args.ablations, args.all, args.with_geometry)
    if not specs:
        raise SystemExit("Nothing to run after selection.")

    # ---- inputs, GT, queries (lazy + cached) -----------------------------
    # Official SHREC'18 GT (rgbd.csv/cad.csv): real category+subcategory for
    # all 2,101 queries and 3,308 CADs, scored with the track's own
    # metrics.py so results are leaderboard-comparable (tuning + final).
    object_ids = validate_inputs(paths, args.allow_partial_gallery)
    verify_precompute_provenance(paths)
    gt = load_official_gt(paths["data_root"], paths["stage1_root"])
    cad_labels = gt["cad"]          # {cadid: (category, subcategory)}
    freqs = gt["freqs"]             # {category: n_cads}  (official denominators)
    index = prepare_queries(paths["data_root"], paths["stage1_root"], gt)
    if args.viz_check:
        viz_check(index, paths["stage1_root"], args.viz_check)

    # ---- tier 1: passes --------------------------------------------------
    resume = args.resume and not args.overwrite
    todo = [s for s in specs if not s.alias_of]
    if resume:
        todo = [s for s in todo if not os.path.isfile(os.path.join(
            paths["results_root"], s.name, "metrics_summary.json"))]
        done = len([s for s in specs if not s.alias_of]) - len(todo)
        if done:
            print(f"[resume] {done} ablations already computed.")

    needed_passes: List[str] = [
        p for p in PASS_DEFS
        if any(p in s.passes for s in todo)]
    stores: Dict[str, dict] = {}
    for pkey in needed_passes:      # PASS_DEFS order: 'base' always first
        try:
            stores[pkey] = run_pass(pkey, paths, index, object_ids,
                                    args.limit_queries, resume=True)
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[pass:{pkey}] FAILED ({exc}) — dependent ablations "
                  f"will be skipped.")

    # ---- tier 2: ablations ----------------------------------------------
    geom = _GeometryEngine(paths) if any(s.geometry for s in todo) else None
    if geom and any(s.needs_gedi for s in todo) and not geom.gedi_available():
        skipped = [s.name for s in todo if s.needs_gedi]
        print(f"[run] skipping GeDi-signal ablations: {', '.join(skipped)}")
        todo = [s for s in todo if not s.needs_gedi]

    for spec in todo:
        if any(p not in stores for p in spec.passes):
            print(f"[run:{spec.name}] skipped (missing pass "
                  f"{[p for p in spec.passes if p not in stores]}).")
            continue
        run_ablation(spec, paths, index, stores, object_ids, cad_labels,
                     freqs, args.limit_queries, geom)

    # ---- aliases + aggregate (always refreshed) --------------------------
    for spec in specs:
        if spec.alias_of:
            write_alias(spec, paths)
    aggregate(paths)


if __name__ == "__main__":
    main()
