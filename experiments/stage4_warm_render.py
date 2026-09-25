#!/usr/bin/env python3
"""Warme Blender-Renderzeit je Objekt — eine Sitzung je Datensatz x View-Zahl.

stage_render in experiment4_onboarding.py startet je Objekt einen eigenen
Blender-Prozess (RENDER_ONLY); die Zeiten in onboarding_render_n59.json
enthalten damit den Blender-Start je Objekt. rendering/rendering.py loopt
aber ohnehin ueber alle Objekte eines OBJECT_FOLDER in EINER Sitzung — hier
wird genau das genutzt: RENDER_TIMING_JSON liefert die Wandzeit je Objekt
(Import + alle View-Renders + Kameramatrizen), die Differenz aus aeusserer
Wandzeit und script_total_s ist der Blender-Binaerstart (inkl. bpy-Import).

Ausgabe geht in ein Arbeitsverzeichnis, NICHT nach object_images/ — die
bestehende Gallery bleibt unangetastet (gleiche Regel wie stage_render).

Aufruf (Host, Blender noetig):
    python3 experiments/stage4_warm_render.py \
        --work .stage4_warm_rd/render --out .stage4_warm_rd/render_timings.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER_DEFAULT = ("/home/tessa/Cap3D/captioning_pipeline/"
                   "blender-3.4.1-linux-x64/blender")
TARGETS = {"ycbv": 21, "tless": 30, "lmo": 8}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", default="ycbv,tless,lmo")
    ap.add_argument("--views", default="16,42")
    ap.add_argument("--blender",
                    default=os.environ.get("BLENDER_BIN", BLENDER_DEFAULT))
    ap.add_argument("--work", required=True,
                    help="Render-Arbeitsordner (im Repo)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert os.path.isfile(args.blender), f"Blender fehlt: {args.blender}"
    targets = [t for t in args.targets.split(",") if t]
    views = [int(v) for v in args.views.split(",") if v]
    for ds in targets:
        assert os.path.isdir(os.path.join(_ROOT, "object_database", ds)), ds

    out = {"blender": args.blender, "sessions": {}}
    for V in views:
        for ds in targets:
            render_dir = os.path.join(_ROOT, args.work, f"v{V}", ds)
            timing = os.path.join(_ROOT, args.work, f"v{V}_{ds}_timing.json")
            os.makedirs(render_dir, exist_ok=True)
            env = dict(os.environ,
                       OBJECT_FOLDER=os.path.join(_ROOT, "object_database", ds),
                       OBJECT_IMAGES=render_dir + "/",
                       NUM_VIEWS=str(V),
                       OVERWRITE_EXISTING="1",
                       RENDER_TIMING_JSON=timing)
            cmd = [args.blender, "-b", "-P",
                   os.path.join(_ROOT, "rendering", "rendering.py")]
            print(f"[warm-render] v{V}/{ds} ...", flush=True)
            t0 = time.perf_counter()
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               cwd=os.path.join(_ROOT, "rendering"))
            wall = time.perf_counter() - t0
            if not os.path.isfile(timing):
                print(f"[warm-render] FEHLGESCHLAGEN v{V}/{ds} (rc={r.returncode}):")
                print((r.stderr or r.stdout or "")[-500:])
                raise SystemExit(1)
            tj = json.load(open(timing))
            n = len(tj["per_object_s"])
            assert n == TARGETS[ds], f"v{V}/{ds}: {n} statt {TARGETS[ds]} Objekte"
            sess = {"wall_s": wall, "script_total_s": tj["script_total_s"],
                    "blender_start_s": wall - tj["script_total_s"],
                    "per_object_s": tj["per_object_s"]}
            out["sessions"][f"v{V}_{ds}"] = sess
            print(f"[warm-render] v{V}/{ds}: {n} Objekte, wall {wall:.1f} s, "
                  f"Start {sess['blender_start_s']:.2f} s", flush=True)
            # Persistenz nach jeder Sitzung
            with open(os.path.join(_ROOT, args.out), "w") as f:
                json.dump(out, f, indent=1)
    print(f"[warm-render] fertig -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
