#!/usr/bin/env python3
"""
repro_preprocess.py — generisches Preprocessing fuer alle OSCAR+-Galerien.

Eine Stufe pro Aufruf, flach und flag-basiert. Die Defaults sind exakt die
Werte, mit denen die Galerien der Evaluation gebaut wurden (docs/REPRO_SPEC.md,
Schicht P1–P5). Fremde Datensaetze lassen sich ueber --cad-dir/--images-dir/
--id-mode ergaenzen, ohne dieses Skript zu aendern.

Das Skript laedt NICHTS herunter (Beschaffung: docs/DATASETS.md). Es prueft
vor jeder Stufe, ob die Eingaben da sind, und NACH jeder Stufe, ob das
Ergebnis plausibel ist — Rueckgabewerte allein sind hier nachweislich
unzuverlaessig (Blender rc=0 bei Fehler, "0 objects" bei falschem Pfad).

Stufen und wo sie laufen:
    render    Host  (Blender 3.4.1 + CUDA; --blender bzw. $BLENDER)
    partial   Container (docker compose run oscar) — wird automatisch gewrappt
    describe  Container — automatisch gewrappt
    embed     Container — automatisch gewrappt
    dgedi     Host (startet den dgedi-Compose-Dienst selbst)
    check     ueberall — prueft nur Bestand, rechnet nichts

Beispiele (jeweils EINE Terminalzeile, vom Repo-Root des Hosts):
    python3 repro_preprocess.py --dataset shrec18_v2 --step check
    python3 repro_preprocess.py --dataset ycbv --step render --views 42
    python3 repro_preprocess.py --dataset ycbv --step partial
    python3 repro_preprocess.py --dataset ycbv --step describe
    python3 repro_preprocess.py --dataset ycbv --step embed --passes base
    python3 repro_preprocess.py --dataset MI3DOR --step partial --hpr-param 3.2 --jitter-std 0
    python3 repro_preprocess.py --dataset fremd --cad-dir pfad/zu/cads --id-mode stem --step render
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
IN_CONTAINER = os.path.exists("/.dockerenv")

# ---------------------------------------------------------------------------
# Datensatz-Tabelle (docs/REPRO_SPEC.md, P-Tabelle). "id_mode" bestimmt, wie
# aus dem Mesh-Pfad die Objekt-ID wird; "mesh_glob" nur, wo das Layout vom
# Standard <cad_dir>/<obj_id>/ abweicht.
# ---------------------------------------------------------------------------
DATASETS = {
    "shrec18_v2":  dict(cad_dir="eval/datasets/shrec18/shrec18_full/cad",
                        mesh_glob="*.obj", id_mode="stem", n_objects=3308,
                        note="Farbe: Textur (~70 % auslesbar)"),
    "MI3DOR":      dict(cad_dir="object_database/MI3DOR/model/test",
                        mesh_glob="*/*.obj", id_mode="stem", n_objects=3848,
                        note="KEINE Mesh-Farbe (uniform 0.4). Partialwolken der "
                             "Evaluation: --hpr-param 3.2 --jitter-std 0"),
    "ycbv":        dict(cad_dir="object_database/ycbv",
                        mesh_glob="*/textured_simple.obj", id_mode="parent",
                        n_objects=21, note="Textur; mm"),
    "tless":       dict(cad_dir="object_database/tless",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=30, note="keine Farbe; mm"),
    "lmo":         dict(cad_dir="object_database/lmo",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=8, note="Vertexfarben; mm"),
    "gso":         dict(cad_dir="object_database/gso",
                        mesh_glob="*/meshes/model.obj", id_mode="grandparent",
                        n_objects=1030, note="Textur; Einheit METER"),
    "housecat6d":  dict(cad_dir="object_database/housecat6d",
                        mesh_glob="*/*.obj", id_mode="stem",
                        n_objects=199, note="Kategorie-Ordner, ID = Dateistamm"),
    "itodd":       dict(cad_dir="object_database/itodd",
                        mesh_glob="*/model.ply", id_mode="parent",
                        n_objects=28, note="keine Farbe; mm"),
}

EMBED_PASSES = ["base", "siglip", "ulip_fullmesh", "ulip_pc_rgb",
                "ulip_pc_xyz", "uni3d", "all"]


def log(msg: str) -> None:
    print(f"[repro_preprocess] {msg}", flush=True)


def die(msg: str) -> None:
    sys.exit(f"[repro_preprocess] ABBRUCH: {msg}")


def run(cmd, env_extra=None, cwd=None) -> None:
    env = dict(os.environ)
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    log("$ " + " ".join(shlex.quote(str(c)) for c in cmd)
        + ("" if not env_extra else "   [env: "
           + " ".join(f"{k}={v}" for k, v in env_extra.items()) + "]"))
    rc = subprocess.call([str(c) for c in cmd], env=env, cwd=cwd or ROOT)
    if rc != 0:
        die(f"Unterprozess endete mit rc={rc}")


def reexec_in_container(argv) -> None:
    """describe/partial/embed brauchen den oscar-Container — selbst wrappen."""
    cmd = ["docker", "compose", "run", "--rm", "--no-deps", "oscar",
           "python3", "/app/repro_preprocess.py"] + argv
    log("Stufe laeuft im oscar-Container — wrappe automatisch.")
    os.execvp("docker", cmd)


def mesh_list(ds: dict):
    return sorted(glob.glob(os.path.join(ROOT, ds["cad_dir"], ds["mesh_glob"])))


def images_dir(name: str) -> str:
    return os.path.join(ROOT, "object_images", name)


def desc_file(name: str) -> str:
    return os.path.join(ROOT, "object_database", name, "descriptions_attributes.json")


# ---------------------------------------------------------------------------
# Verifikation je Stufe (Anforderung 2 aus REPRO_SPEC: Ergebnis pruefen,
# nicht den Rueckgabewert)
# ---------------------------------------------------------------------------
def verify_render(name, ds, views) -> None:
    pngs = glob.glob(os.path.join(images_dir(name), "*", "*_[0-9]*.png"))
    mats = glob.glob(os.path.join(images_dir(name), "*", "*_CamMatrix.npy"))
    want = ds["n_objects"] * views
    log(f"render: {len(pngs)} Views, {len(mats)} Kameramatrizen "
        f"(erwartet je ~{want})")
    if len(mats) < want * 0.98:
        die(f"nur {len(mats)}/{want} Kameramatrizen — Blender ist vermutlich "
            "still gescheitert (bekannt bei Blender != 3.4.1: rc=0 ohne PIL).")


def verify_partial(name, ds, views) -> None:
    npz = glob.glob(os.path.join(images_dir(name), "*", "*_partial.npz"))
    want = ds["n_objects"] * views
    log(f"partial: {len(npz)} Teilwolken (erwartet ~{want})")
    if len(npz) < want * 0.98:
        die(f"nur {len(npz)}/{want} Teilwolken.")


def verify_describe(name, ds) -> None:
    p = desc_file(name)
    if not os.path.isfile(p):
        die(f"{p} fehlt.")
    d = json.load(open(p))
    n_caps = sum(len(v.get("image_descriptions", {})) for v in d.values())
    empty = [k for k, v in d.items() if not v.get("image_descriptions")]
    log(f"describe: {len(d)} Objekte, {n_caps} Bildbeschreibungen "
        f"(erwartet {ds['n_objects']} Objekte)")
    if len(d) < ds["n_objects"]:
        die("unvollstaendig — haeufigste Ursache: --images_dir zeigte auf den "
            "Objektordner statt auf den Ordner MIT Objektunterordnern.")
    if empty:
        die(f"{len(empty)} Objekte OHNE Beschreibungen (z.B. {empty[:3]}). "
            "Haeufigste Ursache: CUDA OOM je Batch (laufende Dienste belegen "
            "Speicher) — erneut mit --batch-size 2 ausfuehren.")


def verify_embed(name) -> None:
    caches = [os.path.basename(p) for p in
              glob.glob(os.path.join(images_dir(name), ".*cache*.pt"))]
    log(f"embed: {len(caches)} Cache-Dateien in object_images/{name}: "
        + (", ".join(sorted(caches)) or "KEINE"))
    if not caches:
        die("kein Embedding-Cache entstanden.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="Name aus der eingebauten Tabelle ODER frei fuer fremde "
                         "Datensaetze (dann --cad-dir/--id-mode angeben). "
                         "Bekannt: " + ", ".join(DATASETS))
    ap.add_argument("--step", required=True,
                    choices=["render", "partial", "describe", "embed", "dgedi",
                             "check", "all"],
                    help="'all' = render -> partial -> describe -> embed in "
                         "einem Rutsch (vom Host starten; genau das braucht "
                         "eine neue Gallery fuer pipeline.run_pipeline)")
    # Render
    ap.add_argument("--views", type=int, default=42,
                    help="Anzahl Views (Ikosphaere, FPS-geordnet). Default 42 = Eval.")
    ap.add_argument("--blender", default=os.environ.get(
        "BLENDER", "/home/tessa/Cap3D/captioning_pipeline/blender-3.4.1-linux-x64/blender"),
        help="Blender-3.4.1-Binary (Host). 3.3.x scheitert STILL.")
    ap.add_argument("--shard", default="0/1",
                    help="i/n zur Parallelisierung des Renderns, ergebnisneutral.")
    ap.add_argument("--overwrite", action="store_true")
    # Partial
    ap.add_argument("--num-points", type=int, default=10000)
    ap.add_argument("--hpr-param", type=float, default=2.8,
                    help="Hidden-Point-Removal-Parameter. Eval: 2.8 — AUSSER "
                         "MI3DOR (3.2, mit --jitter-std 0).")
    ap.add_argument("--jitter-std", type=float, default=0.001)
    # Describe
    ap.add_argument("--batch-size", type=int, default=8)
    # Embed
    ap.add_argument("--passes", default="base",
                    help="Kommaliste aus " + "|".join(EMBED_PASSES) +
                         ". Eval nutzte: base, siglip, ulip_fullmesh, "
                         "ulip_pc_rgb, ulip_pc_xyz, uni3d.")
    # dGeDi
    ap.add_argument("--dgedi-out", default="",
                    help="Zielordner der Deskriptoren (Default .dgedi_gallery_<dataset>).")
    # Fremde Datensaetze
    ap.add_argument("--cad-dir", default="", help="CAD-Wurzel (fremder Datensatz)")
    ap.add_argument("--mesh-glob", default="", help="Glob relativ zu --cad-dir")
    ap.add_argument("--id-mode", default="stem",
                    choices=["stem", "parent", "grandparent"])
    ap.add_argument("--n-objects", type=int, default=0,
                    help="Erwartete Objektzahl (fuer die Verifikation, fremder Datensatz)")
    args = ap.parse_args()

    name = args.dataset
    if name in DATASETS:
        ds = dict(DATASETS[name])
    else:
        if not args.cad_dir:
            die(f"unbekannter Datensatz '{name}' — fuer fremde Datensaetze "
                "--cad-dir (und ggf. --mesh-glob/--id-mode/--n-objects) angeben.")
        ds = dict(cad_dir=args.cad_dir, mesh_glob=args.mesh_glob or "*/*.obj",
                  id_mode=args.id_mode, n_objects=args.n_objects or 0, note="fremd")
    if args.cad_dir:
        ds["cad_dir"] = args.cad_dir
    if args.mesh_glob:
        ds["mesh_glob"] = args.mesh_glob

    meshes = mesh_list(ds)
    if ds["n_objects"] == 0:
        ds["n_objects"] = len(meshes)
    log(f"Datensatz {name}: {len(meshes)} Meshes unter {ds['cad_dir']}/{ds['mesh_glob']} "
        f"(erwartet {ds['n_objects']}) — {ds['note']}")
    if args.step != "check" and len(meshes) == 0:
        die("keine Meshes gefunden. Beschaffung: docs/DATASETS.md")
    if ds["n_objects"] and abs(len(meshes) - ds["n_objects"]) > 0 and args.step != "check":
        die(f"Meshzahl {len(meshes)} != erwartet {ds['n_objects']} — "
            "Datensatz unvollstaendig? (docs/DATASETS.md)")

    if args.step == "all":
        if IN_CONTAINER:
            die("--step all vom HOST starten (render braucht Blender).")
        argv = []
        skip_next = False
        for a in sys.argv[1:]:
            if skip_next:
                skip_next = False
                continue
            if a == "--step":
                skip_next = True
                continue
            argv.append(a)
        for st in ["render", "partial", "describe", "embed"]:
            log(f"===== Stufe {st} =====")
            rc = subprocess.call([sys.executable, os.path.abspath(__file__)]
                                 + argv + ["--step", st])
            if rc != 0:
                die(f"Stufe {st} endete mit rc={rc}")
        log("alle Stufen fertig — Gallery ist einsatzbereit "
            "(pipeline.run_pipeline --gallery <name>).")
        return

    if args.step == "check":
        for label, fn in [("render (PNGs)", lambda: len(glob.glob(os.path.join(
                              images_dir(name), "*", "*_[0-9]*.png")))),
                          ("cam-matrizen", lambda: len(glob.glob(os.path.join(
                              images_dir(name), "*", "*_CamMatrix.npy")))),
                          ("partial (npz)", lambda: len(glob.glob(os.path.join(
                              images_dir(name), "*", "*_partial.npz")))),
                          ("describe", lambda: len(json.load(open(desc_file(name))))
                              if os.path.isfile(desc_file(name)) else 0),
                          ("embed-caches", lambda: len(glob.glob(os.path.join(
                              images_dir(name), ".*cache*.pt"))))]:
            try:
                log(f"  {label:14s}: {fn()}")
            except Exception as e:                                    # noqa: BLE001
                log(f"  {label:14s}: FEHLER {e}")
        log("Hinweis: partial(npz)=0 und cam-matrizen=0 sind in Ordnung, wenn "
            "ein .ulip_partial_cache_*.pt existiert — der Embedding-Cache "
            "ersetzt die Rohdateien (Erzwingung uebernimmt repro_experiment).")
        return

    # ---- render (Host, Blender) -------------------------------------------
    if args.step == "render":
        if IN_CONTAINER:
            die("render laeuft auf dem HOST (Blender), nicht im Container.")
        if not os.path.isfile(args.blender):
            die(f"Blender nicht gefunden: {args.blender} (--blender setzen). "
                "Es MUSS 3.4.1 sein — 3.3.x scheitert still mit rc=0.")
        si, st = args.shard.split("/")
        run([args.blender, "-b", "-P", "rendering/rendering.py"],
            env_extra={"OBJECT_FOLDER": os.path.join(ROOT, ds["cad_dir"]),
                       "OBJECT_IMAGES": images_dir(name) + "/",
                       "NUM_VIEWS": args.views,
                       "OVERWRITE_EXISTING": "1" if args.overwrite else "0",
                       "SHARD_INDEX": si, "SHARD_TOTAL": st})
        verify_render(name, ds, args.views)
        return

    # ---- Container-Stufen: selbst wrappen ---------------------------------
    if args.step in ("partial", "describe", "embed") and not IN_CONTAINER:
        reexec_in_container(sys.argv[1:])

    if args.step == "partial":
        cmd = ["python3", "rendering/generate_partial_pointclouds.py",
               "--cad_dir", ds["cad_dir"], "--images_dir", f"object_images/{name}",
               "--num_points", args.num_points, "--hpr-param", args.hpr_param,
               "--jitter-std", args.jitter_std]
        if ds["mesh_glob"] and ds["id_mode"] == "stem":
            cmd += ["--mesh-glob", ds["mesh_glob"]]
        if args.overwrite:
            cmd += ["--overwrite"]
        run(cmd)
        verify_partial(name, ds, args.views)
        return

    if args.step == "describe":
        cmd = ["python3", "rendering/generate_descriptions.py",
               "--images_dir", f"object_images/{name}",
               "--output", f"object_database/{name}/descriptions_attributes.json",
               "--batch-size", args.batch_size]
        if args.overwrite:
            cmd += ["--overwrite"]
        run(cmd)
        verify_describe(name, ds)
        return

    if args.step == "embed":
        bad = [p for p in args.passes.split(",") if p not in EMBED_PASSES]
        if bad:
            die(f"unbekannte Passes: {bad}")
        run(["python3", "tools/precompute_embeddings.py",
             "--dataset", name,
             "--data-root", ds["cad_dir"],
             # precompute_embeddings erwartet den VOLLEN Glob, nicht relativ
             "--mesh-glob", os.path.join(ds["cad_dir"], ds["mesh_glob"]),
             "--mesh-id-mode", ds["id_mode"],
             "--images-dir", f"object_images/{name}",
             "--desc-file", f"object_database/{name}/descriptions_attributes.json",
             "--results-root", f"object_retrieval/results_prep_{name}",
             "--passes", args.passes])
        verify_embed(name)
        return

    # ---- dgedi (Host, eigener Compose-Dienst) -----------------------------
    if args.step == "dgedi":
        out = args.dgedi_out or f".dgedi_gallery_{name}"
        manifest = os.path.join(ROOT, f"dgedi_manifest_{name}.json")
        json.dump({(os.path.splitext(os.path.basename(m))[0]
                    if ds["id_mode"] == "stem" else
                    os.path.basename(os.path.dirname(m))
                    if ds["id_mode"] == "parent" else
                    os.path.basename(os.path.dirname(os.path.dirname(m)))):
                   os.path.relpath(m, ROOT) for m in meshes},
                  open(manifest, "w"), indent=1)
        log(f"Manifest: {manifest} ({len(meshes)} Objekte)")
        run(["docker", "compose", "run", "--rm", "--no-deps", "dgedi",
             "python3", "/oscar/dgedi_service/precompute_gallery.py",
             "--manifest", f"/oscar/{os.path.basename(manifest)}",
             "--out", f"/oscar/{out}", "--n-points", 10000,
             "--mode", "multi_scale"])
        n = len(glob.glob(os.path.join(ROOT, out, "*")))
        log(f"dgedi: {n} Deskriptor-Eintraege in {out}")
        if n < len(meshes):
            die(f"nur {n}/{len(meshes)} Deskriptoren.")
        return


if __name__ == "__main__":
    main()
