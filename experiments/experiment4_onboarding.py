#!/usr/bin/env python3
"""
Stage 4a — Onboarding-Latenz: was kostet EIN neues CAD-Modell?

Frage
-----
Ein Nutzer hat ein CAD-File und will das Objekt auffindbar machen. Wie lange
dauert das, und welcher Schritt dominiert?

Aufbau
------
Basis ist die **3b-Datenbank** (G_proxy = gso + housecat6d + itodd, 1257
Objekte). Jedes Ziel-CAD aus ycbv/tless/lmo (59 Stueck) wird EINZELN
onboardet und gemessen; ausgewertet wird die Verteilung ueber diese 59
Faelle. Reale CADs unterschiedlicher Komplexitaet zu nehmen ist der Grund
fuer die Streuung — Vertexzahl und Dateigroesse werden mitgeschrieben,
damit die Varianz erklaerbar bleibt.

Gemessene Schritte (in Ausfuehrungsreihenfolge)
    mesh        Mesh laden, Vertices verschweissen, Normalen, Durchmesser
    render      Blender, 42 Views von Ikosphaeren-Vertices, FPS-geordnet
    partial     partielle Punktwolken je View (Hidden Point Removal)
    describe    LLaVA-Beschreibung je View
    embed_dino  DINOv2 ueber die 42 Renderings
    embed_clip  CLIP-Text ueber die 42 Beschreibungen
    embed_ulip  ULIP-2 ueber die 42 Teilwolken
    dgedi       GeDi-Deskriptoren (nur mit --dgedi; nur noetig fuer Geometrie)

Die Embed-Schritte messen die **inkrementellen** Kosten: nur die Views des
neuen Objekts, mit bereits geladenen Modellen. Das ist die Arbeit, die ein
anhaengender Cache leisten wuerde — nicht simuliert, sondern direkt gemessen.

Wichtig: der aktuelle Cache kann das NICHT
------------------------------------------
Der Cache-Schluessel ist ein Fingerprint ueber das gesamte Inventar
(step5_shape_matching.py, `_get_partial_cache_path`: je Objekt je View eine
Zeile). Ein neues Objekt aendert den Hash und invalidiert alles — Onboarding
kostet real O(Gallery), nicht O(1). `--measure-invalidation` misst diesen
Aufschlag einmal, damit beide Zahlen nebeneinander stehen.

Beispiele
---------
    # Schnelltest: 3 Objekte, ohne Blender und LLaVA
    python3 experiments/experiment4_onboarding.py --max-objects 3 \\
        --stages mesh,embed

    # Vollstaendig, alle 59 Ziel-CADs
    python3 experiments/experiment4_onboarding.py --stages all \\
        --out results_stage4/onboarding.json

    # Aufschlag der Cache-Invalidierung dazu
    python3 experiments/experiment4_onboarding.py --stages embed \\
        --measure-invalidation
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
for p in (_ROOT, os.path.join(_ROOT, "object_retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Wie in experiment4_query_latency.py: relative Datenpfade ("../object_database/…")
# loesen gegen das Arbeitsverzeichnis auf. Alle Pfade dieses Skripts sind gegen
# _ROOT absolut gebildet, die Subprozesse bekommen ihr cwd explizit — der Wechsel
# betrifft also nur den Pipeline-Code.
os.chdir(os.path.join(_ROOT, "object_retrieval"))

from stage4_common import (Timings, aggregate, host_provenance,  # noqa: E402
                           print_table, summarize, write_results)

ALL_STAGES = ["mesh", "render", "partial", "describe", "embed", "dgedi"]

# Ziel-CADs: genau die Objekte, die in der 3b-Datenbank FEHLEN und deshalb
# onboardet werden muessten. Layout je Datensatz wie in stage3_gallery.
TARGET_LAYOUT = {
    "ycbv":  ("object_database/ycbv/*/textured_simple.obj", "parent"),
    "tless": ("object_database/tless/*/model.ply", "parent"),
    "lmo":   ("object_database/lmo/*/model.ply", "parent"),
}


def target_meshes(datasets):
    """(dataset, obj_id, mesh_path) fuer alle zu onboardenden CADs."""
    out = []
    for ds in datasets:
        pattern, mode = TARGET_LAYOUT[ds]
        for p in sorted(glob.glob(os.path.join(_ROOT, pattern))):
            oid = (os.path.basename(os.path.dirname(p)) if mode == "parent"
                   else os.path.splitext(os.path.basename(p))[0])
            out.append((ds, oid, p))
    return out


# --------------------------------------------------------------------------
# Einzelschritte. Jeder gibt zusaetzlich Kennzahlen zurueck, die die Streuung
# erklaeren (Vertexzahl, Dateigroesse) — ohne die ist die Verteilung nicht
# interpretierbar.
# --------------------------------------------------------------------------
def stage_mesh(mesh_path, t: Timings) -> dict:
    import trimesh
    with t.measure("mesh"):
        m = trimesh.load(mesh_path, force="mesh", process=True)
        m.merge_vertices()
        m.fix_normals()
        extents = m.bounding_box.extents
        diameter = float((extents ** 2).sum() ** 0.5)
    return {"vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
            "diameter": diameter,
            "file_mb": round(os.path.getsize(mesh_path) / 1e6, 3)}


def stage_render(mesh_path, out_dir, obj_id, num_views, blender, t: Timings) -> dict:
    """Blender ist ein Fremdprozess; RENDER_ONLY beschraenkt ihn auf ein Objekt.

    Ausgabe geht bewusst in ein Arbeitsverzeichnis, NICHT in object_images/ —
    das Experiment darf die bestehende Gallery nicht ueberschreiben.
    """
    env = dict(os.environ,
               OBJECT_FOLDER=os.path.dirname(os.path.dirname(mesh_path)),
               OBJECT_IMAGES=out_dir + "/",
               RENDER_ONLY=obj_id,
               NUM_VIEWS=str(num_views),
               OVERWRITE_EXISTING="1")
    cmd = [blender, "-b", "-P", os.path.join(_ROOT, "rendering", "rendering.py")]
    with t.measure("render"):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                           cwd=os.path.join(_ROOT, "rendering"))
    n = len(glob.glob(os.path.join(out_dir, obj_id, "*.png")))
    # Blender beendet sich bei einem Python-Fehler mit rc=0 (beobachtet
    # 2026-08-31: fehlendes PIL -> Traceback, rc=0, 250 ms, null Bilder).
    # Der Rueckgabewert taugt hier also nicht als Erfolgspruefung; die Zahl der
    # erzeugten Bilder tut es.
    out = {"render_rc": r.returncode, "render_views": n}
    if n == 0:
        out["render_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


def stage_partial(mesh_path, out_dir, num_points, obj_id, t: Timings) -> dict:
    """Teilwolken je View aus Mesh + gespeicherter Kameramatrix (HPR)."""
    # KEIN --mesh-glob: das ist ein eigenstaendiges Muster mit obj_id =
    # Dateistamm, was fuer ycbv jedes Objekt auf "textured_simple" abbilden
    # wuerde (dieselbe ID-Kollision wie im Full-Mesh-Pfad). Stattdessen die
    # <cad_dir>/<obj_id>/-Konvention: das Skript entdeckt die Objekte ueber
    # --images_dir, und obj_root enthaelt genau eines.
    cad_dir = os.path.dirname(os.path.dirname(mesh_path))
    cmd = [sys.executable, os.path.join(_ROOT, "rendering",
                                        "generate_partial_pointclouds.py"),
           "--cad_dir", cad_dir,
           "--images_dir", out_dir,
           "--num_points", str(num_points), "--overwrite"]
    with t.measure("partial"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    n = len(glob.glob(os.path.join(out_dir, obj_id, "*_partial.npz")))
    out = {"partial_rc": r.returncode, "partial_clouds": n}
    if n == 0:
        out["partial_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


def reuse_renders(ds, oid, obj_dir, num_views, gen_partial=False) -> dict:
    """Vorhandene Renderings ins Arbeitsverzeichnis kopieren.

    Blender ist auf dieser Maschine nicht installiert (die Gallery wurde auf dem
    zweiten PC gerendert), damit faellt die render-Stufe aus — und mit ihr die
    Eingabe fuer describe und embed. Die Kosten von LLaVA und den Encodern
    haengen aber nicht davon ab, WOHER ein Bild kommt, nur davon, wie viele es
    sind. Die ersten V Renderings zu kopieren macht diese Stufen also ehrlich
    messbar; nur die Renderzeit selbst fehlt und wird als solche berichtet.
    """
    import shutil
    src = os.path.join(_ROOT, "object_images", ds, oid)
    if not os.path.isdir(src):
        return {"reuse_renders": f"keine Renderings unter {src}"}
    imgs = [p for p in sorted(glob.glob(os.path.join(src, "*.png")))
            if not p.endswith("_bg.png")][:num_views]
    # Kameramatrizen MUESSEN mit: generate_partial_pointclouds.py entdeckt seine
    # Views ueber <obj>_viewN_CamMatrix.npy. Ohne sie findet es null Views und
    # die partial-Stufe laesst sich gar nicht messen — der Grund, warum sie in
    # allen Laeufen bis 2026-09-01 fehlte.
    cams = sorted(glob.glob(os.path.join(src, "*_CamMatrix.npy")))
    cams = _first_n_views(cams, num_views)
    files = imgs + cams
    npz = []
    if not gen_partial:
        # Wolken nur uebernehmen, wenn sie NICHT erzeugt werden sollen; sonst
        # wuerde die Messung ueberschriebene Dateien zaehlen statt neue.
        npz = sorted(glob.glob(os.path.join(src, "*_partial.npz")))[:num_views]
        files += npz
    for p in files:
        dst = os.path.join(obj_dir, os.path.basename(p))
        if not os.path.exists(dst):
            shutil.copy2(p, dst)
    return {"reused_images": len(imgs), "reused_cams": len(cams),
            "reused_clouds": len(npz)}


def _first_n_views(paths, n):
    """Die ersten n Views in FPS-Reihenfolge, nach View-Index sortiert.

    Lexikographisch waere view10 < view2 — der Schnitt traefe dann eine andere
    Teilmenge als die, die Stage 1 als V16 ausgewertet hat.
    """
    import re
    keyed = []
    for p in paths:
        m = re.search(r"_view(\d+)_", os.path.basename(p))
        keyed.append((int(m.group(1)) if m else 10 ** 9, p))
    return [p for _, p in sorted(keyed)[:n]]


def stage_describe(obj_root, t: Timings) -> dict:
    """LLaVA-Beschreibung je View.

    ``--images_dir`` muss der Ordner sein, der die OBJEKTORDNER enthaelt, nicht
    der Objektordner selbst — sonst meldet das Skript "Total objects: 0" und
    kehrt in Millisekunden mit rc=0 zurueck (Fehler vom 2026-08-31). Deshalb
    liegt je Objekt ein eigener Wurzelordner mit genau einem Unterordner darin.
    """
    out_json = os.path.join(obj_root, "descriptions.json")
    cmd = [sys.executable, os.path.join(_ROOT, "rendering",
                                        "generate_descriptions.py"),
           "--images_dir", obj_root, "--output", out_json, "--overwrite"]
    with t.measure("describe"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    info = {"describe_rc": r.returncode,
            "describe_json": os.path.isfile(out_json)}
    if not info["describe_json"]:
        info["describe_error"] = (r.stderr or r.stdout or "")[-300:]
    return info


class Encoders:
    """Modelle EINMAL laden, danach nur noch encodieren.

    Genau diese Trennung macht die Zahl aussagekraeftig: die Ladezeit ist eine
    Systemstartkosten, keine Onboarding-Kosten. Sie wird separat berichtet.
    """

    def __init__(self, t: Timings):
        # Ueber build_pipeline, NICHT ueber ein blankes PipelineConfig(): dessen
        # ulip_repo_path ist "" und der Encoder bricht ab. Wichtiger noch — nur
        # so sind Backbone, Checkpoint, Punktzahl und Farbmodus identisch mit der
        # Pipeline, die in Stage 1–3 gemessen wurde. Eine Latenzzahl aus einer
        # anders konfigurierten Encoder-Instanz waere nicht vergleichbar.
        from eval_common import build_pipeline
        from stage3_gallery import _base_cfg
        with t.measure("load_encoders"):
            cfg = _base_cfg("ycbv")
            _, self.clip, self.dino, _, self.ulip = build_pipeline(cfg)
        if self.ulip is None:
            raise RuntimeError("ShapeMatcher nicht verfuegbar — "
                               "ULIP-Repo/Checkpoint pruefen.")

    def embed_object(self, obj_dir, num_views, t: Timings) -> dict:
        """Inkrementelle Kosten: nur die Views DIESES Objekts.

        Auf die ersten ``num_views`` beschraenkt. Das ist zulaessig, weil die
        Renderings FPS-geordnet abgelegt sind — die ersten V sind genau das
        V-View-Set, das die O4-Ablation in Stage 1 ausgewertet hat.

        Jeder Kanal wird EINZELN gemessen, und innerhalb der Kanaele wird das
        Laden von der Rechnung getrennt: sonst steckt in `embed_dino` die
        JPEG-Dekodierung, die mit dem Encoder nichts zu tun hat und auf einer
        anderen Hardware ganz anders skaliert.
        """
        import json as _json

        import numpy as np
        from PIL import Image
        info = {}

        imgs = [p for p in sorted(glob.glob(os.path.join(obj_dir, "*.png")))
                if not p.endswith("_bg.png")][:num_views]
        if imgs:
            loaded = []
            with t.measure("io_load_images"):
                for p in imgs:
                    loaded.append(Image.open(p).convert("RGB"))
            with t.measure("embed_dino"):
                for im in loaded:
                    self.dino.encode_image(im)
            info["n_views_dino"] = len(imgs)

        desc = os.path.join(os.path.dirname(obj_dir), "descriptions.json")
        if os.path.isfile(desc):
            try:
                texts = _json.load(open(desc))
                texts = (list(texts.values()) if isinstance(texts, dict)
                         else list(texts))[:num_views]
                # _encode_texts_batch ist der Pfad, den load_descriptions selbst
                # nimmt (step3_clip_retrieval.py:212) — gebatcht, nicht je String.
                # Ein oeffentliches encode_text gibt es nicht und wird auch nicht
                # gebraucht; eine Schleife ueber Einzelstrings haette zudem etwas
                # anderes gemessen als die Pipeline tut.
                with t.measure("embed_clip"):
                    self.clip._encode_texts_batch([str(s) for s in texts])
                info["n_texts"] = len(texts)
            except Exception as exc:            # nicht abbrechen, nur vermerken
                info["embed_clip_skipped"] = str(exc)

        npz = sorted(glob.glob(os.path.join(obj_dir, "*_partial.npz")))[:num_views]
        if npz:
            clouds = []
            with t.measure("io_load_clouds"):
                for p in npz:
                    d = np.load(p)
                    clouds.append((d["points"], d.get("colors")))
            with t.measure("embed_ulip"):
                for pts, col in clouds:
                    self.ulip.encode_pointcloud(pts, colors=col)
            info["n_clouds"] = len(npz)

        info.update(self.append_to_cache(num_views, t))
        return info

    @staticmethod
    def append_to_cache(num_views, t, work=None):
        """Das simulierte INKREMENTELLE Anhaengen — echt, nicht als Attrappe.

        Genau das waere die Arbeit eines anhaengenden Caches: bestehende Datei
        laden, den einen neuen Objekteintrag einfuegen, zurueckschreiben. Als
        Stellvertreter dient der groesste reale Gallery-Cache (gso, 1030 der
        1257 Proxy-Objekte).

        Die Messung getrennt nach load / insert / save zu fuehren ist der Punkt:
        das Einfuegen ist O(1), aber Laden und Schreiben einer monolithischen
        .pt-Datei sind O(Gallery). Selbst ein "anhaengender" Cache in dieser
        Ablageform zahlt also pro neuem Objekt fuer die ganze Gallery — nur
        eben Serialisierung statt Encoding.
        """
        import torch
        cands = sorted(glob.glob(os.path.join(
            _ROOT, "object_images", "gso", ".ulip_partial_cache_*.pt")))
        if not cands:
            return {"cache_append_skipped": "kein gso-Partial-Cache gefunden"}
        with t.measure("cache_load"):
            blob = torch.load(cands[0], map_location="cpu", weights_only=False)
        emb = blob.get("embeddings", blob) if isinstance(blob, dict) else blob
        n_before = len(emb)
        with t.measure("cache_insert"):
            any_v = next(iter(emb.values()))
            emb["__stage4_probe__"] = torch.zeros(
                num_views, any_v.shape[-1], dtype=any_v.dtype)
        tmp = os.path.join(work or "/tmp", ".stage4_cache_append.pt")
        with t.measure("cache_save"):
            torch.save(blob, tmp)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return {"cache_entries": n_before}


def stage_dgedi(mesh_path, obj_root, obj_id, t: Timings) -> dict:
    """GeDi-Deskriptoren fuer das neue Objekt, damit es geometrisch suchbar ist.

    Ueber tools/precompute_gedi_descriptors.py — den Weg, den die Preprocessing-
    Kette geht. `dgedi_bridge` bietet nur dgedi_health und dgedi_rerank; ein
    `compute_descriptors` gibt es dort NICHT (der erste Entwurf importierte es
    und waere sofort gescheitert).

    Nur relevant, wenn geometrisches Re-Ranking benutzt wird — was Stage 3 fuer
    BOP widerlegt hat. Deshalb optional, nicht Default.
    """
    cache = os.path.join(obj_root, "gedi_cache")
    cmd = [sys.executable, os.path.join(_ROOT, "tools",
                                        "precompute_gedi_descriptors.py"),
           "--mesh-glob", mesh_path,
           "--gallery-cache-dir", cache,
           "--skip-queries", "--overwrite"]
    with t.measure("dgedi"):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    n = len(glob.glob(os.path.join(cache, "*.npz")))
    out = {"dgedi_rc": r.returncode, "dgedi_descriptors": n}
    if n == 0:
        out["dgedi_error"] = (r.stderr or r.stdout or "")[-300:]
    return out


# --------------------------------------------------------------------------
def measure_invalidation(enc, num_views, sample, t: Timings) -> dict:
    """Der Aufschlag, den der AKTUELLE Cache erzwingt — gemessen, nicht geraten.

    Kommt ein Objekt dazu, aendert sich der Inventar-Fingerprint und der
    gesamte Cache verfaellt: jedes Gallery-Objekt muss neu encodiert werden.
    Gemessen wird deshalb das Encoding einer Stichprobe echter Gallery-Objekte
    OHNE Cache, hochgerechnet auf die Gallery-Groesse.

    Der erste Versuch (2026-09-01) hat hier `assemble_gallery` gestoppt und
    6,1 s gemeldet — das war eine Assembly mit WARMEN Caches, also gerade nicht
    die Groesse, um die es geht. Deshalb jetzt ueber die Stueckkosten.
    """
    GALLERY = 1257                      # G_proxy, die 3b-Datenbank
    src_root = os.path.join(_ROOT, "object_images", "gso")
    objs = [d for d in sorted(os.listdir(src_root))
            if os.path.isdir(os.path.join(src_root, d))][:sample]
    per = Timings()
    for oid in objs:
        enc.embed_object(os.path.join(src_root, oid), num_views, per)
    enc_keys = [k for k in per.as_dict() if k.startswith("embed_")]
    per_obj = sum(sum(per.runs[k]) for k in enc_keys) / max(len(objs), 1)
    t.add("invalidation_total_extrapolated", per_obj * GALLERY)
    return {"gallery_size": GALLERY, "sampled_objects": len(objs),
            "per_object_encode_s": round(per_obj, 3),
            "extrapolated_full_reencode_s": round(per_obj * GALLERY, 1),
            "extrapolated_full_reencode_min": round(per_obj * GALLERY / 60, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="ycbv,tless,lmo",
                    help="Datensaetze, deren CADs onboardet werden (Default: alle 59).")
    ap.add_argument("--stages", default="mesh,partial,describe,embed",
                    help=f"Komma-Liste aus {ALL_STAGES} oder 'all'. Default ist "
                         "die vollstaendige Kette OHNE render — Blender liegt "
                         "ausserhalb des Containers und laeuft ueber "
                         "scripts/stage4_onboarding.sh auf dem Host.")
    ap.add_argument("--max-objects", type=int, default=0,
                    help="Nur die ersten N CADs (0 = alle).")
    ap.add_argument("--num-views", default="42",
                    help="Komma-Liste, z.B. '16,42'. Jede Zahl wird als eigener "
                         "Durchgang gemessen. Die Views sind FPS-geordnet, die "
                         "ersten 16 von 42 sind also ein gueltiges 16-View-Set "
                         "(Stage-1 O4: 0.5820 bei V16 vs 0.5868 bei V42).")
    ap.add_argument("--num-points", type=int, default=8192)
    ap.add_argument("--blender", default=os.environ.get(
                        "BLENDER_BIN",
                        "/home/tessa/Cap3D/captioning_pipeline/"
                        "blender-3.4.1-linux-x64/blender"),
                    help="Blender-Binary. Die 3.3.1-Installation hat KEIN PIL "
                         "und scheitert still; 3.4.1 hat es.")
    ap.add_argument("--work-dir", default=os.path.join(_ROOT, ".stage4_work"),
                    help="Renders/Wolken landen HIER, nicht in object_images/.")
    ap.add_argument("--reuse-renders", action="store_true",
                    help="Vorhandene Renderings/Wolken ins Arbeitsverzeichnis "
                         "kopieren, statt sie zu erzeugen. Noetig, wo kein "
                         "Blender installiert ist — describe und embed werden "
                         "dann echt gemessen, nur die Renderzeit fehlt.")
    ap.add_argument("--inv-sample", type=int, default=15,
                    help="Wie viele echte Gallery-Objekte fuer die "
                         "Invalidierungs-Stueckkosten encodiert werden.")
    ap.add_argument("--measure-invalidation", action="store_true",
                    help="Zusaetzlich den Aufschlag der Cache-Invalidierung messen.")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results_stage4",
                                                  "onboarding.json"))
    args = ap.parse_args(argv)
    if not os.path.isabs(args.out):
        args.out = os.path.join(_ROOT, args.out)

    stages = (ALL_STAGES if args.stages == "all"
              else [s.strip() for s in args.stages.split(",") if s.strip()])
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        ap.error(f"unbekannte Stufe(n): {unknown}; erlaubt: {ALL_STAGES}")

    if "render" in stages:
        from shutil import which
        if which(args.blender) is None:
            print(f"[stage4] WARNUNG: '{args.blender}' nicht gefunden — "
                  f"Stufe 'render' wird uebersprungen (--blender setzen).")
            stages = [s for s in stages if s != "render"]

    view_counts = [int(v) for v in str(args.num_views).split(",") if v.strip()]
    meshes = target_meshes([d.strip() for d in args.targets.split(",")])
    if args.max_objects:
        meshes = meshes[:args.max_objects]
    print(f"[stage4] {len(meshes)} Ziel-CADs, Stufen: {stages}, "
          f"View-Zahlen: {view_counts}")
    os.makedirs(args.work_dir, exist_ok=True)

    setup = Timings()
    enc = Encoders(setup) if "embed" in stages else None

    by_views, records = {}, []
    for V in view_counts:
        per_object = []
        print(f"\n[stage4] --- {V} Views ---")
        for i, (ds, oid, mesh) in enumerate(meshes, 1):
            t = Timings()
            rec = {"dataset": ds, "object_id": oid, "num_views": V,
                   "mesh": os.path.relpath(mesh, _ROOT)}
            # Zwei Ebenen: obj_root enthaelt genau EINEN Objektordner, weil
            # generate_descriptions.py ueber Unterordner iteriert. Ein flaches
            # Layout wuerde bei jedem Objekt alle vorherigen mitbeschreiben.
            obj_root = os.path.join(args.work_dir, f"v{V}", ds, oid)
            obj_dir = os.path.join(obj_root, oid)
            os.makedirs(obj_dir, exist_ok=True)
            try:
                if "mesh" in stages:
                    rec.update(stage_mesh(mesh, t))
                if args.reuse_renders:
                    rec.update(reuse_renders(ds, oid, obj_dir, V,
                                             gen_partial="partial" in stages))
                if "render" in stages:
                    rec.update(stage_render(mesh, obj_root, oid, V,
                                            args.blender, t))
                if "partial" in stages:
                    rec.update(stage_partial(mesh, obj_root, args.num_points,
                                             oid, t))
                if "describe" in stages:
                    rec.update(stage_describe(obj_root, t))
                if "embed" in stages and enc is not None:
                    src = obj_dir if os.path.isdir(obj_dir) else None
                    # Ohne render/describe liegen die Views in der bestehenden
                    # Gallery — von dort lesen, statt die Stufe stillschweigend
                    # zu ueberspringen.
                    if not glob.glob(os.path.join(obj_dir, "*.png")):
                        src = os.path.join(_ROOT, "object_images", ds, oid)
                    rec.update(enc.embed_object(src, V, t))
                if "dgedi" in stages:
                    rec.update(stage_dgedi(mesh, obj_root, oid, t))
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [{i}/{len(meshes)}] {ds}/{oid}  FEHLER: {rec['error']}")

            rec["timings"] = t.as_dict()
            rec["total_s"] = t.total()
            per_object.append(t.as_dict())
            records.append(rec)
            print(f"  [{i}/{len(meshes)}] {ds}/{oid:<16} {t.total():7.2f} s")
        by_views[V] = {
            "per_step": aggregate(per_object),
            "per_object_total_s": summarize(
                [r["total_s"] for r in records if r["num_views"] == V]),
        }

    payload = {
        "experiment": "stage4a_onboarding",
        "base_gallery": "G_proxy (3b) = gso + housecat6d + itodd = 1257",
        "stages": stages,
        "view_counts": view_counts,
        "n_objects": len(meshes),
        "provenance": host_provenance(),
        "model_load_once_s": {k: v[0] for k, v in setup.as_dict().items()},
        "by_views": by_views,
        "records": records,
        # Qualitaetsseite aus Stage 1 (SHREC'18, nDCG), damit die Kostenzahlen
        # unmittelbar gegen den Nutzen gestellt werden koennen.
        "stage1_quality_ndcg": {"8": 0.5714, "16": 0.5820,
                                "32": 0.5800, "42": 0.5868},
    }

    if args.measure_invalidation:
        if enc is None:
            payload["invalidation"] = {"skipped": "braucht --stages embed"}
        else:
            inv = Timings()
            payload["invalidation"] = {
                **measure_invalidation(enc, max(view_counts), args.inv_sample, inv),
                **{k: round(sum(v), 2) for k, v in inv.as_dict().items()}}

    for V in view_counts:
        print_table(f"Onboarding pro Objekt — {V} Views", by_views[V]["per_step"])
        tot = by_views[V]["per_object_total_s"]
        if tot.get("n"):
            print(f"\n  Gesamt je Objekt: Median {tot['median']:.2f} s, "
                  f"IQR {tot['iqr']:.2f} s, p95 {tot['p95']:.2f} s  (n={tot['n']})")

    if len(view_counts) > 1:
        # Nur view-abhaengige Stufen duerfen in den Vergleich: `mesh` kostet bei
        # 16 und 42 Views dasselbe, die Differenz waere reines Messrauschen und
        # wuerde als Ergebnis missverstanden.
        view_dependent = {"render", "partial", "describe", "embed", "dgedi"}
        print_view_tradeoff(by_views, payload["stage1_quality_ndcg"],
                            has_view_stage=bool(view_dependent & set(stages)))

    if payload["model_load_once_s"]:
        print("\n  Einmalige Modell-Ladezeit (KEINE Onboarding-Kosten):")
        for k, v in payload["model_load_once_s"].items():
            print(f"    {k:<24}{v:8.2f} s")
    write_results(args.out, payload)


def print_view_tradeoff(by_views, quality, has_view_stage=True):
    """Kosten gegen Nutzen — der eigentliche Zweck des View-Sweeps.

    Stage 1 zeigt, dass die Qualitaet ab 16 Views flach laeuft (V32 liegt sogar
    unter V16). Wenn die Onboarding-Kosten linear mit der View-Zahl steigen,
    ist 42 nicht zu rechtfertigen — diese Tabelle stellt beides nebeneinander.
    """
    if not has_view_stage:
        print("\n[stage4] View-Vergleich uebersprungen: keine der aktiven "
              "Stufen haengt von der View-Zahl ab (mesh kostet bei 16 und 42 "
              "dasselbe). Mit --stages render,partial,describe,embed messen.")
        return
    ref = max(by_views)
    ref_cost = by_views[ref]["per_object_total_s"].get("median", 0.0)
    ref_q = quality.get(str(ref))
    print("\n=== Kosten-Nutzen: Views ===")
    print(f"  {'Views':>6}{'Onboarding (Median)':>22}{'Kosten':>10}"
          f"{'nDCG (Stage 1)':>16}{'Nutzen':>10}")
    for V in sorted(by_views):
        c = by_views[V]["per_object_total_s"].get("median", 0.0)
        q = quality.get(str(V))
        cs = f"{100 * c / ref_cost:6.0f}%" if ref_cost else "     —"
        qs = f"{q - ref_q:+.4f}" if (q is not None and ref_q is not None) else "    —"
        print(f"  {V:>6}{c:>19.2f} s{cs:>10}"
              f"{(f'{q:.4f}' if q is not None else '—'):>16}{qs:>10}")
    print(f"  (Kosten und Nutzen jeweils relativ zu {ref} Views)")


if __name__ == "__main__":
    main()
