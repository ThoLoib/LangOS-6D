"""
stage3_gallery.py
=================
Assemble a multi-dataset *union* gallery for the Stage-3 BOP evaluation.

Stage 3 retrieves against a gallery that spans several preprocessed datasets:

    3a (exact CAD available):  G_proxy  ∪  G_target,d
    3b (proxy only):           G_proxy

where ``G_proxy = GSO ∪ HouseCat6D ∪ ITODD`` (no curation, per
``docs/BOP_PREPROCESSING_HANDOFF.md`` §3) and ``G_target,d`` is the set of
evaluated target CADs of BOP dataset ``d`` (ycbv | tless | lmo).

``eval_common.build_pipeline`` only loads a *single* ``ref_dir``; the gallery
ids come from ``DINOReRanker._ref_embeddings`` (one dataset).  This module reuses
the already-loaded encoders and the per-dataset embedding caches written by
``tools/precompute_embeddings.py`` — it loads each dataset in turn and merges the
three reference stores under **namespaced instance ids** ``"<ds>/<obj_id>"`` so
ids never collide across datasets.

Nothing is re-encoded: every dataset's DINO / CLIP-text / ULIP-partial embeddings
are pulled straight from disk cache (the loaders fall back to encoding only if a
cache is missing, which should not happen post-preprocessing).

Base retrieval channel only (``base`` pass = CLIP-text + DINOv2@42 + ULIP-2
partial), matching the frozen config; the ``uni3d`` / ``ulip_fullmesh`` ablation
channels are not merged here.
"""

import copy
import os

import torch

try:
    from eval_common import EvalConfig, build_pipeline
except ImportError:  # pragma: no cover
    from .eval_common import EvalConfig, build_pipeline


_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, ".."))


# ---------------------------------------------------------------------------
# Per-dataset layout (paths are relative to object_retrieval/, matching the
# existing eval scripts which run from that dir).
# ---------------------------------------------------------------------------
# - ref_dir     : object_images/<ds>        (renders + partial .npz + caches)
# - desc_file   : object_database/<ds>/descriptions_attributes.json
# - mesh_glob   : the ulip_fullmesh glob (only used for _cad_paths fallback)
# - id_mode     : how the gallery obj_id maps from the mesh path (see
#                 tools/precompute_embeddings.build_mesh_items)
# - pose_mesh   : native-scale mesh used at POSE time (Phase B/C):
#                   targets -> BOP models_eval (mm, BOP frame)
#                   proxies -> the original CAD (native units; see units_m)
# - units_m     : True if pose_mesh is in METRES (needs *1000 -> mm for BOP)
DATASET_LAYOUT = {
    "ycbv": dict(
        ref_dir="../object_images/ycbv",
        desc_file="../object_database/ycbv/descriptions_attributes.json",
        mesh_glob="../object_database/ycbv/*/textured_simple.obj",
        id_mode="parent",
        pose_mesh_dir="../eval/datasets/ycbv/models_eval",   # obj_0000NN.ply, mm
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "tless": dict(
        ref_dir="../object_images/tless",
        desc_file="../object_database/tless/descriptions_attributes.json",
        mesh_glob="../object_database/tless/*/model.ply",
        id_mode="stem",
        pose_mesh_dir="../eval/datasets/tless/models_eval",
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "lmo": dict(
        ref_dir="../object_images/lmo",
        desc_file="../object_database/lmo/descriptions_attributes.json",
        mesh_glob="../object_database/lmo/*/model.ply",
        id_mode="stem",
        pose_mesh_dir="../eval/datasets/lmo/models_eval",
        pose_mesh_pat="obj_{id6}.ply",
        units_m=False,
    ),
    "gso": dict(
        ref_dir="../object_images/gso",
        desc_file="../object_database/gso/descriptions_attributes.json",
        mesh_glob="../object_database/gso/*/meshes/model.obj",
        id_mode="grandparent",
        pose_mesh_dir="../object_database/gso",
        pose_mesh_pat="{oid}/meshes/model.obj",
        units_m=True,       # GSO meshes are in metres
    ),
    "housecat6d": dict(
        ref_dir="../object_images/housecat6d",
        desc_file="../object_database/housecat6d/descriptions_attributes.json",
        mesh_glob="../object_database/housecat6d/*/*.obj",
        id_mode="stem",
        pose_mesh_dir="../object_database/housecat6d",
        pose_mesh_pat=None,   # resolved by glob (category subdir unknown from id)
        units_m=True,         # HouseCat6D meshes are in metres
    ),
    "itodd": dict(
        ref_dir="../object_images/itodd",
        desc_file="../object_database/itodd/descriptions_attributes.json",
        mesh_glob="../object_database/itodd/*/model.ply",
        id_mode="stem",
        pose_mesh_dir="../object_database/itodd",
        pose_mesh_pat="{oid}/model.ply",
        units_m=False,        # ITODD (BOP) meshes are in millimetres
    ),
}

PROXY_DATASETS = ("gso", "housecat6d", "itodd")
TARGET_DATASETS = ("ycbv", "tless", "lmo")


def namespaced_id(ds: str, obj_id: str) -> str:
    return f"{ds}/{obj_id}"


def split_id(nsid: str):
    ds, _, obj_id = nsid.partition("/")
    return ds, obj_id


# ---------------------------------------------------------------------------
# Gallery assembly
# ---------------------------------------------------------------------------

class UnionGallery:
    """Holds the assembled pipeline components plus id bookkeeping."""

    def __init__(self, config, clip_retr, dino_rer, fusion_mod, shape_m,
                 gallery_ids, id_to_pose_mesh, target_ds, proxy_ds,
                 eval_cfg=None):
        self.config = config          # PipelineConfig
        self.eval_cfg = eval_cfg      # EvalConfig (run_query needs it for S')
        self.clip_retr = clip_retr
        self.dino_rer = dino_rer
        self.fusion_mod = fusion_mod
        self.shape_m = shape_m
        self.gallery_ids = gallery_ids            # set of namespaced ids
        self.id_to_pose_mesh = id_to_pose_mesh    # nsid -> (path, units_m)
        self.target_ds = target_ds                # str or None
        self.proxy_ds = proxy_ds                  # tuple

    def components(self):
        """Return the 5-tuple that run_evaluation / run_query expect."""
        return (self.config, self.clip_retr, self.dino_rer,
                self.fusion_mod, self.shape_m)


def _pose_mesh_path(ds: str, obj_id: str):
    """Resolve the native-scale mesh used at pose time for a gallery id."""
    import glob as _glob
    lay = DATASET_LAYOUT[ds]
    base = os.path.join(_THIS, lay["pose_mesh_dir"])
    pat = lay["pose_mesh_pat"]
    if pat is None:
        # HouseCat6D: <cat>/<oid>.obj — category unknown from id, glob for it.
        hits = _glob.glob(os.path.join(base, "*", f"{obj_id}.obj"))
        path = hits[0] if hits else ""
    else:
        # obj ids like obj_000001 -> id6 = "000001"
        id6 = obj_id.replace("obj_", "") if obj_id.startswith("obj_") else obj_id
        path = os.path.join(base, pat.format(id6=id6, oid=obj_id))
    return (os.path.abspath(path), lay["units_m"])


def _base_cfg(ds: str, extra_overrides=None) -> EvalConfig:
    """EvalConfig seeded to one dataset, with the frozen base-pass overrides."""
    lay = DATASET_LAYOUT[ds]
    overrides = {
        "num_views": 42,
        "dino_view_aggregation": "topk_softmax",
        "dino_view_topk": 5,
        "dino_view_temperature": 0.5,
        "ulip_view_aggregation": "topk_softmax",
        "ulip_view_topk": 5,
        "ulip_view_temperature": 0.5,
        # Mean-patch DINO pooling (Pulli), the MI3DOR-proven default — user
        # decision for Stage-3 too. NOTE the preprocessed gallery .dino_cache_*
        # are CLS-pooled (precompute used the global "cls" default; the "mean"
        # default was MI3DOR-driver scoped), so the FIRST assembly cache-misses
        # and re-encodes DINO from the renders, writing new *_mean caches that
        # then persist. Query DINO uses mean too, so gallery/query stay
        # consistent. (One-time ~15-20 min GPU encode, gso dominating.)
        "dino_pooling": "mean",
    }
    if extra_overrides:
        overrides.update(extra_overrides)
    return EvalConfig(
        ref_dir=lay["ref_dir"],
        desc_file=lay["desc_file"],
        cad_mesh_glob=lay["mesh_glob"],
        result_folder="results_stage3_tmp",
        clip_top_k=10 ** 6, dino_top_k=10 ** 6,
        ulip2_top_k=10 ** 6, fusion_top_k=10 ** 6,
        ulip2_use_partial_views=True,   # base pass = ULIP-2 partial-view shape
        pipeline_overrides=overrides,
    )


def _mesh_items(ds: str):
    """(obj_id, mesh_path) list for a dataset, using the correct id mode."""
    import glob as _glob
    lay = DATASET_LAYOUT[ds]
    paths = sorted(_glob.glob(os.path.join(_THIS, lay["mesh_glob"])))
    mode = lay["id_mode"]
    items = []
    for p in paths:
        if mode == "stem":
            oid = os.path.splitext(os.path.basename(p))[0]
        elif mode == "parent":
            oid = os.path.basename(os.path.dirname(p))
        elif mode == "grandparent":
            oid = os.path.basename(os.path.dirname(os.path.dirname(p)))
        else:
            raise ValueError(mode)
        items.append((oid, p))
    return items


def _absorb_dataset(ds, clip_retr, dino_rer, shape_m, master):
    """Load one dataset's cached stores and merge into the master dicts
    under namespaced ids. Reuses the already-loaded encoders."""
    lay = DATASET_LAYOUT[ds]
    ref_dir = os.path.join(_THIS, lay["ref_dir"])
    desc_file = os.path.join(_THIS, lay["desc_file"])

    # --- DINO (per-object per-view embeddings) ---
    dino_rer.load_reference_images(ref_dir=ref_dir)   # model reused, cache hit
    ds_gallery_ids = list(dino_rer._ref_embeddings.keys())
    for oid, views in dino_rer._ref_embeddings.items():
        master["dino"][namespaced_id(ds, oid)] = views

    # --- CLIP text (per-view description rows) ---
    clip_retr.load_descriptions(desc_file=desc_file)  # no id_to_label -> obj ids
    embs = clip_retr._desc_embeddings                 # (M, D) on device
    for i, (txt, lbl) in enumerate(zip(clip_retr._desc_texts,
                                       clip_retr._desc_labels)):
        master["clip_emb"].append(embs[i].detach().cpu())
        master["clip_txt"].append(txt)
        master["clip_lbl"].append(namespaced_id(ds, lbl))

    # --- ULIP-2 partial-view shape embeddings ---
    if shape_m is not None:
        partial_items = shape_m._collect_partial_items(ref_dir)
        if partial_items:
            shape_m._partial_view_paths = dict(partial_items)
            cache_path = shape_m._get_partial_cache_path(ref_dir, partial_items)
            if shape_m._try_load_partial_cache(cache_path):
                for oid, emb in shape_m._cad_embeddings.items():
                    nsid = namespaced_id(ds, oid)
                    master["cad_emb"][nsid] = emb.detach().cpu()
                    master["cad_path"][nsid] = shape_m._cad_paths.get(oid, "")

    # --- pose-mesh map (native scale) keyed by the GALLERY ids (render-dir
    # names = obj_XXXXXX / gso ids), NOT the fullmesh glob stem (which is
    # "model" for BOP */model.ply layouts). ---
    for oid in ds_gallery_ids:
        nsid = namespaced_id(ds, oid)
        master["pose_mesh"][nsid] = _pose_mesh_path(ds, oid)


def assemble_gallery(target_datasets=(), proxy_ds=PROXY_DATASETS,
                     extra_overrides=None):
    """Build the union gallery = G_proxy ∪ (exact CADs of each target dataset).

    3a (one big DB): assemble_gallery(TARGET_DATASETS) -> proxies + ALL targets;
        every query dataset retrieves against this single combined gallery.
    3b:              assemble_gallery(())              -> G_proxy only.
    """
    target_datasets = tuple(d for d in target_datasets)
    # targets first so the encoders/models are constructed once against a target
    datasets = list(target_datasets) + [d for d in proxy_ds
                                        if d not in target_datasets]

    seed_ds = datasets[0]
    cfg = _base_cfg(seed_ds, extra_overrides)
    # Build components ONCE (loads CLIP/DINO/ULIP models + seed dataset caches).
    config, clip_retr, dino_rer, fusion_mod, shape_m = build_pipeline(
        cfg, cad_mesh_items=_mesh_items(seed_ds))

    master = {"dino": {}, "clip_emb": [], "clip_txt": [], "clip_lbl": [],
              "cad_emb": {}, "cad_path": {}, "pose_mesh": {}}
    for ds in datasets:
        _absorb_dataset(ds, clip_retr, dino_rer, shape_m, master)

    # --- write merged stores back into the (single) component objects ---
    dino_rer._ref_embeddings = master["dino"]
    dino_rer._apply_view_limit()

    if master["clip_emb"]:
        clip_retr._desc_embeddings = torch.stack(master["clip_emb"]).to(
            clip_retr.device)
        clip_retr._desc_texts = master["clip_txt"]
        clip_retr._desc_labels = master["clip_lbl"]

    if shape_m is not None and master["cad_emb"]:
        shape_m._cad_embeddings = master["cad_emb"]
        shape_m._cad_paths = master["cad_path"]
        shape_m._partial_mode = True

    gallery_ids = set(master["dino"].keys())
    return UnionGallery(
        config, clip_retr, dino_rer, fusion_mod, shape_m,
        gallery_ids=gallery_ids,
        id_to_pose_mesh=master["pose_mesh"],
        target_ds=target_datasets,          # tuple of included target datasets
        proxy_ds=tuple(proxy_ds),
        eval_cfg=cfg,
    )


if __name__ == "__main__":
    # Quick self-test: assemble a 3b (proxy-only) gallery and print sizes.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="",
                    help="comma-separated target datasets to include "
                         "(empty = 3b proxies only; 'all' = 3a big DB)")
    ap.add_argument("--proxy", default=",".join(PROXY_DATASETS),
                    help="comma-separated proxy datasets (subset for quick tests)")
    args = ap.parse_args()
    _proxy = tuple(p.strip() for p in args.proxy.split(",") if p.strip())
    if args.targets == "all":
        _targets = TARGET_DATASETS
    else:
        _targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())
    g = assemble_gallery(_targets, proxy_ds=_proxy)
    from collections import Counter
    by_ds = Counter(split_id(i)[0] for i in g.gallery_ids)
    print(f"[stage3_gallery] union |gallery| = {len(g.gallery_ids)}  by-dataset={dict(by_ds)}")
    print(f"[stage3_gallery] CLIP rows={len(g.clip_retr._desc_labels)}  "
          f"DINO objs={len(g.dino_rer._ref_embeddings)}  "
          f"ULIP objs={len(g.shape_m._cad_embeddings) if g.shape_m else 0}")
    missing = [i for i, (p, _) in g.id_to_pose_mesh.items() if not os.path.isfile(p)]
    print(f"[stage3_gallery] pose-mesh resolved for "
          f"{len(g.id_to_pose_mesh)-len(missing)}/{len(g.id_to_pose_mesh)}; "
          f"missing={len(missing)}")
    if missing[:5]:
        print("  e.g. missing:", missing[:5])
