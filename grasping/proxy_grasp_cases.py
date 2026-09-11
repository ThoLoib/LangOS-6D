#!/usr/bin/env python3
"""Object set of the Stage-5 proxy-grasp study — FIXED before any grasp trial.

Source: the Stage-3 3b run `results_bop_stage3_v2/3b_cross` (OSCAR+, cross ×
partial, the frozen configuration) and Thomas' ranking of 2026-09-10
(OBJEKTAUSWAHL): for every BOP target the proxy CAD that was retrieved at
rank 1 most consistently, ranked by size deviation (dimDev), pose-median D_sym
(pMed) and how many instances carried that proxy (n, share).

"Proxy (fixiert)" means: a trial uses exactly the instances (scene, image,
gt_idx) in which THIS proxy was the 3b top-1 — otherwise the proxy lottery
across instances would dilute a small sample. The instance lists are built by
`build_grasp_instances.py` into `proxy_grasp_instances.json`.

Tiers (read the ranking top-down):
    1  ranks 1–10  — the success-rate set
    2  ranks 11–20 — pulled in only when the sample size needs it (border-line
                     dimDev, small proxy shares, n = 9)
    x  exhibits    — mustard bottle and tless 16: shown next to the rate as
                     mechanism exhibits, not counted in it
EXCLUDED objects are documented with the reason and can only be run by naming
them explicitly (`--cases ycbv1`).

This module is pure Python (no numpy) so the instance builder can run on a
machine without the sim stack.
"""
from __future__ import annotations

import glob
import os
import zlib
from typing import Dict, List, Optional, Tuple

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)


def _c(rank, key, dataset, obj_id, name, proxy, tier, dim_dev, p_med, n, share, note=""):
    return dict(rank=rank, key=key, dataset=dataset, obj_id=obj_id, name=name,
                proxy=proxy, tier=tier, dim_dev=dim_dev, p_med=p_med, n=n,
                share=share, note=note)


NESQUIK = "gso/Nestle_Nesquik_Chocolate_Powder_Flavored_Milk_Additive_109_Oz_Canister"

CASES: List[dict] = [
    # rank key       ds       id  name             proxy (3b top-1, fixed)                           tier dimDev pMed  n   share
    _c(1,  "ycbv14",  "ycbv",  14, "mug",           "housecat6d/cup-red_heart",                        1, 0.07, 4.4,  78, 0.52, "empirisch bester Fall, p95 8 mm"),
    _c(2,  "tless30", "tless", 30, "obj_30",        "gso/BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup", 1, 0.14, 5.0, 47, 0.33, "bester Neuzugang"),
    _c(3,  "tless19", "tless", 19, "obj_19",        "itodd/obj_000018",                                1, 0.13, 6.6,  78, 0.41),
    _c(4,  "ycbv2",   "ycbv",  2,  "cracker box",   "gso/Nestle_Carnation_Cinnamon_Coffeecake_Kit_1913OZ", 1, 0.09, 5.0, 25, 0.11, "Karton→Karton"),
    _c(5,  "ycbv17",  "ycbv",  17, "scissors",      "gso/Diamond_Visions_Scissors_Red",                1, 0.20, 3.4,  16, 0.21, "Schere→Schere, bester pMed"),
    _c(6,  "tless22", "tless", 22, "obj_22",        "itodd/obj_000026",                                1, 0.46, 8.2, 121, 0.63, "empirisch bestätigt"),
    _c(7,  "lmo9",    "lmo",   9,  "duck",          "gso/CHICKEN_RACER",                               1, 0.14, 11.7, 52, 0.29, "3 mm Greifer-Luft"),
    _c(8,  "tless10", "tless", 10, "obj_10",        "itodd/obj_000018",                                1, 0.16, 8.1,  32, 0.22),
    _c(9,  "tless28", "tless", 28, "obj_28",        "itodd/obj_000018",                                1, 0.26, 6.8,  59, 0.30),
    _c(10, "tless18", "tless", 18, "obj_18",        "itodd/obj_000013",                                1, 0.26, 5.8,  42, 0.29),
    _c(11, "tless9",  "tless", 9,  "obj_09",        "itodd/obj_000018",                                2, 0.29, 8.7, 109, 0.44, "dev grenzwertig, größtes n"),
    _c(12, "tless25", "tless", 25, "obj_25",        "itodd/obj_000018",                                2, 0.27, 5.0,  14, 0.15),
    _c(13, "ycbv3",   "ycbv",  3,  "sugar box",     NESQUIK,                                           2, 0.26, 7.0,   9, 0.02, "nur mit Szenenliste"),
    _c(14, "tless4",  "tless", 4,  "obj_04",        "gso/Germanium_GE132",                             2, 0.10, 5.5,  26, 0.04, "nur mit Szenenliste"),
    _c(15, "tless21", "tless", 21, "obj_21",        "gso/Android_Figure_Chrome",                       2, 0.10, 7.6,  10, 0.05, "Proxy skurril, Geometrie passt"),
    _c(16, "tless5",  "tless", 5,  "obj_05",        "itodd/obj_000018",                                2, 0.25, 4.9,  17, 0.09),
    _c(17, "tless7",  "tless", 7,  "obj_07",        "itodd/obj_000013",                                2, 0.24, 9.9,  42, 0.17),
    _c(18, "tless24", "tless", 24, "obj_24",        "housecat6d/bottle-sanitizer_small_white",         2, 0.26, 8.3,  33, 0.17),
    _c(19, "lmo10",   "lmo",   10, "eggbox",        "gso/MINI_ROLLER",                                 2, 0.27, 10.4,  9, 0.05, "n=9"),
    _c(20, "lmo1",    "lmo",   1,  "ape",           "gso/Nintendo_Mario_Action_Figure",                2, 0.28, 12.9,  9, 0.05, "n=9"),
    # mechanism exhibits — reported next to the rate, never inside it
    _c(0,  "ycbv5",   "ycbv",  5,  "mustard bottle", NESQUIK,                                         "x", None, 6.8, 114, 0.76, "Exhibit: Proxy-Kandidaten blockiert (Pilot 2026-09-06)"),
    _c(0,  "tless16", "tless", 16, "obj_16",        "itodd/obj_000027",                                "x", None, 5.2,  92, 0.48, "Exhibit: halbhoher Proxy (Pilot 2026-09-06)"),
]

# Objects Thomas excluded up-front, with the reason. Runnable only by name.
EXCLUDED: List[dict] = [
    _c(0, "ycbv1",   "ycbv",  1,  "master chef can", "gso/Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g", "-", None, 2.1, 278, 0.93,
       "Anwendbarkeit: 102 mm Durchmesser > 80 mm Greiferöffnung (vorab, aus der Geometrie)"),
    _c(0, "tless20", "tless", 20, "obj_20",          "itodd/obj_000018", "-", None, 5.0, 97, 0.39,
       "FoundationPose scheitert im Sim schon mit dem eigenen CAD (Pilot 2026-09-06)"),
    _c(0, "lmo11",   "lmo",   11, "glue",            "housecat6d/bottle-cleansing_lotion_small", "-", None, 14.7, 31, 0.22,
       "Bbox-Abweichung 0.05 täuscht — Pose-Median 64 mm"),
]

ALL_CASES = CASES + EXCLUDED


def case_by_key(key: str) -> dict:
    for c in ALL_CASES:
        if c["key"] == key:
            return c
    raise KeyError(f"unknown case {key!r}; known: {[c['key'] for c in ALL_CASES]}")


def select_cases(ranks: str = "1-10", cases: Optional[str] = None,
                 exhibits: bool = False) -> List[dict]:
    """Cases to run: explicit `--cases` keys (any tier, order kept) or the rank
    range `--ranks a-b` (+ exhibits on request)."""
    if cases:
        return [case_by_key(k.strip()) for k in cases.split(",") if k.strip()]
    lo, hi = (int(x) for x in ranks.split("-")) if "-" in ranks else (int(ranks), int(ranks))
    out = [c for c in CASES if c["tier"] in (1, 2) and lo <= c["rank"] <= hi]
    if exhibits:
        out += [c for c in CASES if c["tier"] == "x"]
    return out


# ---------------------------------------------------------------------------
# Proxy CAD resolution — mirrors stage3_gallery.DATASET_LAYOUT / _pose_mesh_path
# (GSO + HouseCat6D in metres, ITODD in mm) with the one local fallback that
# this machine needs: ITODD CADs live under eval/datasets/itodd/models here.
# ---------------------------------------------------------------------------
def proxy_mesh(nsid: str) -> Tuple[Optional[str], bool]:
    """namespaced gallery id -> (mesh path or None, units_m)."""
    ds, oid = nsid.split("/", 1)
    if ds == "gso":
        return _exists(os.path.join(_ROOT, "object_database/gso", oid, "meshes/model.obj")), True
    if ds == "housecat6d":
        hits = glob.glob(os.path.join(_ROOT, "object_database/housecat6d", "*", oid + ".obj"))
        return (hits[0] if hits else None), True
    if ds == "itodd":
        p = os.path.join(_ROOT, "object_database/itodd", oid, "model.ply")
        if not os.path.isfile(p):
            p = os.path.join(_ROOT, "eval/datasets/itodd/models", oid + ".ply")
        return _exists(p), False
    raise ValueError(f"not a proxy dataset: {nsid}")


def _exists(p: str) -> Optional[str]:
    return p if os.path.isfile(p) else None


def proxy_pool() -> List[str]:
    """G_proxy as used in Stage 3b — GSO (1030) ∪ HouseCat6D (199) ∪ ITODD (28)
    = 1257 gallery ids, enumerated from the CAD files present. Sorted, so the
    random-proxy draw below is stable across machines with the same data."""
    ids = []
    for d in sorted(glob.glob(os.path.join(_ROOT, "object_database/gso/*/meshes/model.obj"))):
        oid = os.path.basename(os.path.dirname(os.path.dirname(d)))
        if oid != "models_orig":
            ids.append("gso/" + oid)
    for f in sorted(glob.glob(os.path.join(_ROOT, "object_database/housecat6d/*/*.obj"))):
        ids.append("housecat6d/" + os.path.splitext(os.path.basename(f))[0])
    itodd = sorted(glob.glob(os.path.join(_ROOT, "object_database/itodd/*/model.ply")))
    if itodd:
        ids += ["itodd/" + os.path.basename(os.path.dirname(f)) for f in itodd]
    else:
        ids += ["itodd/" + os.path.splitext(os.path.basename(f))[0] for f in
                sorted(glob.glob(os.path.join(_ROOT, "eval/datasets/itodd/models/obj_*.ply")))]
    return sorted(set(ids))


def random_proxy(dataset: str, obj_id: int, pool: List[str], salt: str = "random-proxy") -> str:
    """Predeclared baseline: ONE uniformly random gallery CAD per target object,
    drawn by a hash of the object's identity — no test outcome, no category
    label (none is available consistently across the three proxy sources), no
    RNG state to get wrong. The same object always gets the same proxy."""
    h = zlib.crc32(f"{salt}/{dataset}/{obj_id}".encode("utf-8"))
    return pool[h % len(pool)]
