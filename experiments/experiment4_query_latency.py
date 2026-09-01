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
sys.path.insert(0, _THIS)
for p in (_ROOT, os.path.join(_ROOT, "object_retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

# DATASET_LAYOUT und EvalConfig fuehren relative Datenpfade ("../object_database/…"),
# die gegen das ARBEITSVERZEICHNIS aufgeloest werden, nicht gegen den Modulpfad.
# Alle bestehenden Treiber laufen aus object_retrieval/; ohne diesen Wechsel
# scheitert schon load_descriptions. Vor jedem Import von Pipeline-Code.
os.chdir(os.path.join(_ROOT, "object_retrieval"))

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


# LLaVA beginnt praktisch jede Beschreibung mit derselben Floskel. Fuer CLIP ist
# das harmlos, fuer eine Detektion nicht: GroundingDINO sucht nach dem Substantiv
# im Prompt, und "image" ist im Bild nun mal nicht zu finden.
_LLAVA_OPENERS = (
    "the image features a ", "the image features an ", "the image features ",
    "the image shows a ", "the image shows an ", "the image shows ",
    "the image depicts a ", "the image depicts an ", "the image depicts ",
    "this image features ", "in the image, there is a ", "in the image, there is ",
    "the object in the image is a ", "the object in the image is an ",
    "the object in the image is ",
)


def prompt_for(dataset, obj_id, mode="phrase", _cache={}):
    """Sprachprompt aus der gespeicherten Beschreibung des Zielobjekts.

    Bewusst dieselbe Quelle wie der Textkanal: ein handgeschriebener Prompt
    wuerde die Segmentierung besser oder schlechter machen, als das System es im
    Betrieb koennte, und wuerde die Latenzmessung mit einer Qualitaetsfrage
    vermischen.

    Die Datei liegt als {obj_id: {"image_descriptions": {bild: text}}} vor.
    ``mode="phrase"`` nimmt den ERSTEN SATZ ohne die LLaVA-Floskel — GroundingDINO
    erwartet eine kurze Nominalphrase, ein 300-Zeichen-Absatz laeuft in die
    Token-Grenze des Textencoders und detektiert schlechter. ``mode="full"``
    reicht die ganze Beschreibung durch, als Kontrolle.
    """
    if dataset not in _cache:
        f = os.path.join(_ROOT, "object_database", dataset,
                         "descriptions_attributes.json")
        _cache[dataset] = json.load(open(f)) if os.path.isfile(f) else {}
    entry = _cache[dataset].get(obj_id) or {}

    text = ""
    if isinstance(entry, dict):
        imgs = entry.get("image_descriptions")
        if isinstance(imgs, dict) and imgs:
            text = str(next(iter(imgs.values())))
        else:
            for k in ("description", "descriptions", "caption", "text"):
                v = entry.get(k)
                if isinstance(v, str) and v:
                    text = v
                    break
                if isinstance(v, list) and v:
                    text = str(v[0])
                    break
    elif isinstance(entry, list) and entry:
        text = str(entry[0])

    if not text:
        return f"object {obj_id}"
    if mode == "full":
        return text

    sentence = text.split(".")[0].strip()
    low = sentence.lower()
    for opener in _LLAVA_OPENERS:
        if low.startswith(opener):
            sentence = sentence[len(opener):].strip()
            break
    return sentence or text


def run_one(tgt, dataset, test_root, gallery, localizer, args, sink) -> dict:
    """Eine Query von der Platte bis zur Pose. Wirft nichts — Fehler landen
    im Datensatz, damit ein einzelner Ausfall die Messreihe nicht beendet."""
    import numpy as np
    from PIL import Image

    from eval_bop_pose import FP_URL, backproject_masked
    from eval_common import run_query
    from pipeline.foundationpose_bridge import call_foundationpose

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
                        loc.roi_image, gallery.eval_cfg, ulip_query_emb=ulip_q)

    if args.geometry and cloud is not None and len(cloud):
        # GENAU der Pfad aus eval_bop_pose (Zeile ~594): der dGeDi-DIENST ueber
        # dgedi_rerank, mit denselben Repo-Parametern. Der erste Entwurf rief
        # GeometryReRanker(pcfg).rerank(out, ...) — eine andere Implementierung
        # (lokales GeDi), mit falschen Argumenten (erwartet List[FusedCandidate]
        # und eine Open3D-Wolke) und pro Query neu konstruiert. Das waere sofort
        # gescheitert und haette, wenn nicht, eine Latenz fuer einen Pfad
        # gemessen, den die Evaluation nie benutzt.
        from dgedi_bridge import dgedi_rerank
        from eval_common import fusion_ranking as _fr
        cand_ids = [oid for oid, _ in _fr(out["fused_full"])[:args.geo_k]]
        with t.measure("geometry"):
            geo = dgedi_rerank(cloud, cand_ids, ransac_keypoints=6000,
                               ransac_max_iter=10000, use_icp=True)
        rec["geo_ok"] = sum(1 for v in (geo or {}).values() if v.get("ok"))
        rec["geo_requested"] = len(cand_ids)

    if not args.no_pose:
        top1 = _top1_id(out)
        rec["top1"] = top1
        entry = (gallery.id_to_pose_mesh or {}).get(top1)
        if entry:
            path, units_m = (entry if isinstance(entry, (tuple, list))
                             else (entry, False))
            # Genau wie Stage 3: FoundationPose rechnet in Metern. Meshes in
            # Millimetern (BOP, ITODD) bekommen scale=0.001, Meshes in Metern
            # (GSO, HouseCat6D) scale=1.0.
            with t.measure("pose"):
                call_foundationpose(FP_URL, rgb=np.asarray(rgb), depth=depth_m,
                                    mask=np.asarray(loc.mask), K=K,
                                    cad_path=path,
                                    scale=1.0 if units_m else 0.001,
                                    refine_iter=args.refine_iter)
        else:
            rec["pose_skipped"] = f"kein Pose-Mesh fuer {top1}"
    return rec


# `retrieval_total` umschliesst clip/dino/ulip/fusion — es ist eine Klammer um
# bereits gemessene Schritte, keine eigene Arbeit. Waere es in der Summe, zaehlte
# die Retrieval-Zeit doppelt (im Smoke-Test 1.10 s echte Arbeit vs 1.90 s Summe).
# Es bleibt als eigene Zeile stehen: die Differenz zu den vier Kanaelen ist der
# Overhead von run_query selbst (~1 ms, also praktisch keiner).
_CONTAINER_STEPS = {"retrieval_total"}


def _wall(timings: dict) -> float:
    """Echte Ende-zu-Ende-Zeit: Klammermessungen nicht mitzaehlen."""
    return sum(sum(v) for k, v in timings.items() if k not in _CONTAINER_STEPS)


def _top1_id(out):
    """Top-1 der vollen Fusion — derselbe Pfad wie eval_bop_pose.

    `run_query` liefert kein flaches Ranking, sondern die Rohergebnisse aller
    Arme; Stage 3 zieht daraus `fusion_ranking(out["fused_full"])`. Der erste
    Entwurf hier riet Schluesselnamen und bekam durchweg None, worauf die
    Pose-Stufe stillschweigend uebersprungen wurde.
    """
    from eval_common import fusion_ranking
    ranking = fusion_ranking(out["fused_full"])
    return ranking[0][0] if ranking else None


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
    ap.add_argument("--refine-iter", type=int, default=5,
                    help="FoundationPose-Verfeinerungsschritte (Stage-3-Default: 5).")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results_stage4",
                                                  "query_latency.json"))
    args = ap.parse_args(argv)
    # Relative --out gegen die Repo-Wurzel aufloesen, nicht gegen das durch
    # os.chdir() gesetzte object_retrieval/ (sonst landen die Ergebnisse dort).
    if not os.path.isabs(args.out):
        args.out = os.path.join(_ROOT, args.out)

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
        # ObjectLocalizer laedt LAZY (erst in localize()). Ohne diesen Aufruf
        # landen ~25 s Modell-Ladezeit in der ERSTEN Query und verfaelschen
        # sowohl den Kaltstart- als auch den Warm-Median.
        localizer._load_model()

    if args.geometry:
        # dGeDi laeuft ebenfalls als Dienst. Erreichbarkeit und Gallery-Groesse
        # VOR der Messung pruefen: eine falsche Gallery hat am 2026-08-28 einen
        # 17-Stunden-Leerlauf verursacht, in dem keine Registrierung gelang.
        with cold.measure("check_dgedi"):
            from dgedi_bridge import dgedi_health
            h = dgedi_health()
            # dgedi_health() liefert das Health-Dict ODER None bei
            # Unerreichbarkeit — es gibt KEINEN "ok"-Schluessel. Ein
            # h.get("ok") schlaegt deshalb auch dann Alarm, wenn der Dienst
            # laeuft (beobachtet 2026-09-01). Massgeblich ist n_gallery.
            n_gal = (h or {}).get("n_gallery", 0)
            print(f"[stage4] dGeDi n_gallery={n_gal}")
            if not n_gal:
                print("[stage4] WARNUNG: dGeDi nicht erreichbar oder leere "
                      "Gallery — die geometry-Stufe wird nichts messen.")

    if not args.no_pose:
        # FoundationPose laeuft als eigener Dienst; "laden" heisst hier
        # Erreichbarkeit pruefen. Ohne den Test faellt ein toter Dienst erst in
        # der ersten Query auf, und zwar als Latenz statt als Fehler.
        with cold.measure("check_foundationpose"):
            import urllib.request
            try:
                urllib.request.urlopen("http://foundationpose:5050/health",
                                       timeout=10).read()
            except Exception as exc:
                print(f"[stage4] WARNUNG: FoundationPose nicht erreichbar "
                      f"({exc}) — Pose-Stufe wird scheitern.")

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
    # ABSTEIGEND. _apply_view_limit() ERSETZT self._ref_embeddings durch die
    # getrimmte Fassung — der Schnitt ist destruktiv und nicht umkehrbar. In
    # aufsteigender Reihenfolge liefe der 42er-Durchgang auf den 16 Views, die
    # der vorige uebrig gelassen hat, und beide Zeilen waeren dieselbe Messung
    # (im ersten Lauf: 0.866 s gegen 0.856 s, also scheinbar kein Unterschied).
    for V in sorted(views, reverse=True):
        print(f"\n[stage4] --- {V} Views ---")
        pcfg.num_views = V
        if hasattr(dino_rer, "_apply_view_limit"):
            dino_rer.config.num_views = V
            dino_rer._apply_view_limit()
        # Der Shape-Kanal hat kein Gegenstueck zu _apply_view_limit: seine
        # Gallery-Embeddings sind (42, D) je Objekt und die top-k-softmax laeuft
        # ueber alle. Ohne diesen Schnitt bliebe ULIP bei 42 Views, waehrend DINO
        # auf V faellt — der Vergleich waere nur halb durchgefuehrt.
        if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
            shape_m._cad_embeddings = {
                oid: (emb[:V] if getattr(emb, "ndim", 1) == 2 else emb)
                for oid, emb in shape_m._cad_embeddings.items()}
        per_query = []
        for i, tgt in enumerate(targets):
            is_warmup = i < args.warmup
            sink["timings"] = Timings()
            try:
                rec = run_one(tgt, args.dataset, test_root, gallery, localizer,
                              args, sink)
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
            "per_query_total_s": summarize([_wall(q) for q in per_query]),
            "n_no_detection": sum(1 for r in records
                                  if r.get("num_views") == V
                                  and not r.get("warmup")
                                  and r.get("error") == "keine Detektion"),
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
