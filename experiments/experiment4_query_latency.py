#!/usr/bin/env python3
"""
Stage 4b — Query-Latenz: wie lange dauert eine Anfrage bis zur Pose?

Frage
-----
Ein Nutzer nennt ein Objekt ("die Mayonnaisetube"), die Kamera liefert RGB-D.
Wie lange dauert es bis zur einsetzbaren 6D-Pose, und welcher Schritt kostet?

Gemessene Kette — jeder Schritt einzeln
    io_load        RGB + Tiefe von der Platte lesen, dekodieren
    segment_box    GroundingDINO: Box aus dem Sprachprompt
    segment_mask   SAM2.1: Maske aus der Box (inkl. Nachbearbeitung)
    pointcloud     Tiefe ruecksprojizieren und mit der Maske schneiden
    encode_query   ULIP-2 ueber die Query-Punktwolke
    clip           S_text  gegen die Gallery-Beschreibungen
    dino           S_view  gegen die Renderings je Gallery-Objekt
    ulip           S_shape gegen die Teilwolken
    fusion         gewichtete Summe der drei Kanaele
    geometry       GeDi-Deskriptoren + RANSAC ueber Top-K   (nur --geometry)
    pose           FoundationPose auf dem Top-1-CAD          (ausser --no-pose)

Kalt und warm werden GETRENNT berichtet. Die Modelle einmal zu laden kostet ein
Vielfaches einer Query; eine Zahl, die beides vermischt, haengt nur davon ab,
wie viele Queries man gemittelt hat, und sagt ueber das System nichts aus.

View-Zahl
---------
``--views 16,42`` misst dieselbe Kette bei unterschiedlich vielen Views je
Gallery-Objekt. Das ist billig, weil die Embeddings immer fuer alle 42 Views
im Cache liegen und ``_apply_view_limit()`` nur filtert — es wird nichts neu
encodiert. Stage 1 (SHREC'18, O4) misst dazu die Qualitaetsseite:
V8 0.5714 | V16 0.5820 | V32 0.5800 | V42 0.5868 nDCG.

Gallery
-------
Wie in Stage 3: G_proxy + Ziel-CADs = 1316 Objekte. Mit ``--proxy-only``
laeuft es gegen die reine 3b-Datenbank (1257) — der Fall, in dem das exakte
Modell fehlt und ein Proxy gefunden werden muss.

Beispiele
---------
    # Schnelltest ohne Pose
    python3 experiments/experiment4_query_latency.py --dataset ycbv \\
        --n-queries 5 --no-pose

    # Vollstaendig, 16 gegen 42 Views
    python3 experiments/experiment4_query_latency.py --dataset ycbv \\
        --n-queries 50 --views 16,42 \\
        --out results_stage4/query_latency_ycbv.json

    # Mit geometrischem Re-Ranking bei K=5
    python3 experiments/experiment4_query_latency.py --dataset lmo \\
        --n-queries 30 --geometry --geo-k 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_ROOT, os.path.join(_ROOT, "object_retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from stage4_common import (Timings, aggregate, host_provenance,  # noqa: E402
                           print_table, summarize, write_results)

STAGE1_QUALITY = {8: 0.5714, 16: 0.5820, 32: 0.5800, 42: 0.5868}


def wrap_timed(obj, method_name: str, label: str, sink: dict) -> bool:
    """Eine Methode umhuellen, sodass ihre Dauer in `sink['timings']` landet.

    Bewusst hier statt im Pipeline-Code: ``run_query`` fuehrt alle Kanaele in
    einem Durchlauf aus, und die von allen Stages gemeinsam genutzten Module
    sollen fuer ein Messexperiment nicht angefasst werden.
    """
    fn = getattr(obj, method_name, None)
    if fn is None or getattr(fn, "_stage4_wrapped", False):
        return False

    def wrapper(*a, **kw):
        with sink["timings"].measure(label):
            return fn(*a, **kw)

    wrapper._stage4_wrapped = True                    # type: ignore[attr-defined]
    setattr(obj, method_name, wrapper)
    return True


def scene_camera(test_root, scene, im_id):
    """K und depth_scale je Aufnahme — BOP legt beides pro Szene ab."""
    with open(os.path.join(test_root, scene, "scene_camera.json")) as fh:
        cams = json.load(fh)
    cam = cams[str(int(im_id))]
    import numpy as np
    return (np.array(cam["cam_K"], float).reshape(3, 3),
            float(cam.get("depth_scale", 1.0)))


def prompt_for(dataset, obj_id, _cache={}):
    """Sprachprompt = erste Beschreibung des Zielobjekts.

    Bewusst aus derselben Quelle wie der Textkanal: ein handgeschriebener
    Prompt wuerde die Segmentierung besser oder schlechter machen, als das
    System es im Betrieb koennte, und wuerde die Latenzmessung mit einer
    Qualitaetsfrage vermischen.
    """
    if dataset not in _cache:
        f = os.path.join(_ROOT, "object_database", dataset,
                         "descriptions_attributes.json")
        _cache[dataset] = json.load(open(f)) if os.path.isfile(f) else {}
    entry = _cache[dataset].get(obj_id) or {}
    if isinstance(entry, dict):
        for k in ("description", "descriptions", "caption", "text"):
            v = entry.get(k)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, list) and v:
                return str(v[0])
    if isinstance(entry, list) and entry:
        return str(entry[0])
    return f"object {obj_id}"


def run_one(tgt, dataset, test_root, gallery, localizer, pose_est,
            args, sink) -> dict:
    """Eine Query von der Platte bis zur Pose. Wirft nichts — Fehler landen
    im Datensatz, damit ein einzelner Ausfall die Messreihe nicht beendet."""
    import numpy as np
    from PIL import Image

    from eval_bop_pose import backproject_masked
    from eval_common import run_query

    t = sink["timings"]
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = gallery.components()
    rec = {"scene": tgt["scene_id"], "im": tgt["im_id"], "obj": tgt["obj_id"]}

    scene, im = f"{tgt['scene_id']:06d}", f"{tgt['im_id']:06d}"
    with t.measure("io_load"):
        rgb_p = os.path.join(test_root, scene, "rgb", im + ".png")
        if not os.path.isfile(rgb_p):
            rgb_p = os.path.join(test_root, scene, "rgb", im + ".jpg")
        rgb = Image.open(rgb_p).convert("RGB")
        depth_raw = np.array(Image.open(os.path.join(
            test_root, scene, "depth", im + ".png")), dtype=np.float32)
        K, dscale = scene_camera(test_root, scene, im)

    prompt = prompt_for(dataset, f"obj_{tgt['obj_id']:06d}")
    rec["prompt"] = prompt[:80]

    with t.measure("segment"):
        loc = localizer.localize(rgb, prompt, top_k=1)
    if loc is None:
        rec["error"] = "keine Detektion"
        return rec

    with t.measure("pointcloud"):
        depth_m = depth_raw * dscale / 1000.0
        cloud, colors = backproject_masked(
            depth_m, np.asarray(loc.mask), K, rgb=np.asarray(rgb))

    ulip_q = None
    if cloud is not None and len(cloud):
        with t.measure("encode_query"):
            ulip_q = shape_m.encode_pointcloud(cloud, colors=colors)

    with t.measure("retrieval_total"):
        out = run_query(pcfg, clip_retr, dino_rer, fusion_mod, shape_m,
                        loc.roi, gallery.eval_cfg, ulip_query_emb=ulip_q)

    if args.geometry and cloud is not None:
        from pipeline.step_b2_geometry_reranking import GeometryReRanker
        with t.measure("geometry"):
            GeometryReRanker(pcfg).rerank(out, observed_pc=cloud,
                                          top_k=args.geo_k)

    if pose_est is not None:
        top1 = _top1_id(out)
        mesh = (gallery.id_to_pose_mesh or {}).get(top1)
        rec["top1"] = top1
        if mesh:
            path = mesh[0] if isinstance(mesh, (tuple, list)) else mesh
            with t.measure("pose"):
                pose_est.estimate(np.asarray(rgb), depth_m,
                                  np.asarray(loc.mask), cad_path=path,
                                  fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                  method="foundationpose")
        else:
            rec["pose_skipped"] = f"kein Pose-Mesh fuer {top1}"
    return rec


def _top1_id(out):
    """Top-1 aus dem run_query-Ergebnis, unabhaengig von der Arm-Benennung."""
    for key in ("fused", "clip_dino_ulip_full", "ranking", "candidates"):
        v = out.get(key) if isinstance(out, dict) else None
        if v:
            first = v[0]
            for attr in ("object_id", "id", "label"):
                if hasattr(first, attr):
                    return getattr(first, attr)
            if isinstance(first, (tuple, list)) and first:
                return first[0]
            return first
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ycbv", choices=["ycbv", "tless", "lmo"])
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2,
                    help="Nicht gewertete Vorlaeufe (CUDA-Kernel, Allokator).")
    ap.add_argument("--views", default="42",
                    help="Komma-Liste, z.B. '16,42'. Filtert nur den Cache, "
                         "encodiert nichts neu.")
    ap.add_argument("--proxy-only", action="store_true",
                    help="Gallery ohne die Ziel-CADs (3b-Fall, Proxy noetig).")
    ap.add_argument("--geometry", action="store_true",
                    help="Geometrisches Re-Ranking mitmessen.")
    ap.add_argument("--geo-k", type=int, default=5)
    ap.add_argument("--no-pose", action="store_true",
                    help="Ohne FoundationPose (nur Retrieval-Latenz).")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results_stage4",
                                                  "query_latency.json"))
    args = ap.parse_args(argv)

    import random

    from eval_bop_pose import DATASET_TEST, load_bop_targets
    from stage3_gallery import PROXY_DATASETS, assemble_gallery

    views = [int(v) for v in args.views.split(",") if v.strip()]
    cold = Timings()

    print("[stage4] Gallery und Modelle laden ...")
    with cold.measure("gallery_assembly"):
        gallery = assemble_gallery(
            target_datasets=() if args.proxy_only else (args.dataset,),
            proxy_ds=PROXY_DATASETS)
    pcfg, clip_retr, dino_rer, fusion_mod, shape_m = gallery.components()

    with cold.measure("load_groundingdino_sam"):
        from pipeline.step1_localization import ObjectLocalizer
        localizer = ObjectLocalizer(pcfg)

    pose_est = None
    if not args.no_pose:
        with cold.measure("load_foundationpose"):
            from pipeline.step8_pose_estimation import PoseEstimator
            pose_est = PoseEstimator(pcfg)

    print(f"[stage4] |Gallery| = {len(gallery.gallery_ids)}, Views: {views}")

    sink = {"timings": Timings()}
    instrumented = [lbl for obj, meth, lbl in [
        (clip_retr, "retrieve", "clip"), (dino_rer, "rerank", "dino"),
        (shape_m, "match", "ulip"), (fusion_mod, "fuse", "fusion")]
        if wrap_timed(obj, meth, lbl, sink)]
    print(f"[stage4] instrumentierte Kanaele: {instrumented}")

    ds_cfg = DATASET_TEST[args.dataset]
    test_root = os.path.join(_ROOT, "object_retrieval", ds_cfg["test_root"])
    targets = load_bop_targets(os.path.join(_ROOT, "object_retrieval",
                                            ds_cfg["targets"]))
    random.Random(args.seed).shuffle(targets)
    targets = targets[:args.n_queries + args.warmup]

    by_views, records = {}, []
    for V in views:
        print(f"\n[stage4] --- {V} Views ---")
        pcfg.num_views = V
        if hasattr(dino_rer, "_apply_view_limit"):
            dino_rer.config.num_views = V
            dino_rer._apply_view_limit()
        per_query = []
        for i, tgt in enumerate(targets):
            is_warmup = i < args.warmup
            sink["timings"] = Timings()
            try:
                rec = run_one(tgt, args.dataset, test_root, gallery, localizer,
                              pose_est, args, sink)
            except Exception as exc:
                rec = {"error": f"{type(exc).__name__}: {exc}"}
            rec.update(num_views=V, warmup=is_warmup,
                       timings=sink["timings"].as_dict(),
                       total_s=sink["timings"].total())
            records.append(rec)
            if not is_warmup and "error" not in rec:
                per_query.append(rec["timings"])
            note = "  (warmup)" if is_warmup else ""
            note += "  " + rec["error"] if "error" in rec else ""
            print(f"  [{i + 1}/{len(targets)}] {rec['total_s']:6.2f} s{note}")
        by_views[V] = {
            "per_step": aggregate(per_query),
            "per_query_total_s": summarize(
                [sum(sum(v) for v in q.values()) for q in per_query]),
        }

    payload = {
        "experiment": "stage4b_query_latency",
        "dataset": args.dataset,
        "gallery_size": len(gallery.gallery_ids),
        "proxy_only": args.proxy_only,
        "geometry": args.geometry, "geo_k": args.geo_k,
        "pose": not args.no_pose,
        "views": views, "n_warmup": args.warmup,
        "provenance": host_provenance(),
        "cold_start_s": {k: v[0] for k, v in cold.as_dict().items()},
        "by_views": by_views,
        "stage1_quality_ndcg": {str(k): v for k, v in STAGE1_QUALITY.items()},
        "records": records,
    }

    for V in views:
        print_table(f"Query-Latenz (warm) — {V} Views", by_views[V]["per_step"],
                    total_key="retrieval_total")
        tot = by_views[V]["per_query_total_s"]
        if tot.get("n"):
            print(f"\n  Ende zu Ende: Median {tot['median']:.2f} s, "
                  f"IQR {tot['iqr']:.2f} s, p95 {tot['p95']:.2f} s  (n={tot['n']})")

    if len(views) > 1:
        ref = max(views)
        ref_c = by_views[ref]["per_query_total_s"].get("median", 0.0)
        print("\n=== Kosten-Nutzen: Views (Query-Seite) ===")
        print(f"  {'Views':>6}{'Latenz (Median)':>19}{'Kosten':>10}"
              f"{'nDCG (Stage 1)':>16}")
        for V in sorted(views):
            c = by_views[V]["per_query_total_s"].get("median", 0.0)
            q = STAGE1_QUALITY.get(V)
            cs = f"{100 * c / ref_c:6.0f}%" if ref_c else "     —"
            print(f"  {V:>6}{c:>16.3f} s{cs:>10}"
                  f"{(f'{q:.4f}' if q else '—'):>16}")

    print("\n  Kaltstart, einmalig (NICHT Teil der Query-Latenz):")
    for k, v in payload["cold_start_s"].items():
        print(f"    {k:<28}{v:8.2f} s")
    write_results(args.out, payload)


if __name__ == "__main__":
    main()
