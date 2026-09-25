#!/usr/bin/env python3
"""Warme LLaVA-Beschreibungszeit je Objekt — Modell EINMAL geladen.

stage_describe in experiment4_onboarding.py startet je Objekt einen eigenen
Prozess von rendering/generate_descriptions.py und misst damit Python-Start
UND das Laden des 7B-Modells je Objekt mit. Hier: Modell einmal laden
(Ladezeit getrennt), dann je Objekt nur das Beschreiben seiner Views messen
(time.perf_counter um Bild-Laden + Batches). Die Batching-Funktion ist die
PRODUKTIONSFUNKTION generate_captions aus generate_descriptions.py — kein
Nachbau. Batch 8 = bisheriger Default; Batch 1 = Ein-Bild-Aufrufe wie die
Original-OSCAR-Skripte (description_genertor/*.py), um den Batching-Effekt
je View zu beziffern.

Waechter: die Batch-8-Beschreibungen werden gegen die bestehenden
object_database/<ds>/descriptions_attributes.json verglichen (LLaVA laeuft
greedy); der Anteil identischer Texte wird berichtet, ersetzt wird nichts.

Aufruf (im oscar-Container):
    python3 experiments/stage4_warm_describe.py \
        --out .stage4_warm_rd/describe_timings.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "rendering"))

TARGETS = {"ycbv": 21, "tless": 30, "lmo": 8}


def object_ids(ds):
    root = os.path.join(_ROOT, "object_images", ds)
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d))
                  and not d.startswith("."))


def view_files(ds, oid, num_views):
    """Die ersten V Views (FPS-Prefix = Indizes 0..V-1), dann lexikographisch
    sortiert — exakt die pending-Reihenfolge von generate_descriptions.py."""
    files = [f"{oid}_{i}.png" for i in range(num_views)]
    root = os.path.join(_ROOT, "object_images", ds, oid)
    missing = [f for f in files if not os.path.isfile(os.path.join(root, f))]
    if missing:
        raise FileNotFoundError(f"{ds}/{oid}: {len(missing)} Views fehlen "
                                f"(z. B. {missing[0]})")
    return sorted(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", default="ycbv,tless,lmo")
    ap.add_argument("--views", default="16,42")
    ap.add_argument("--batch-sizes", default="8,1")
    ap.add_argument("--prompt", default="Extract visual attributes of the "
                    "object in the image: object type, brand name, color, "
                    "material, and label text.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    targets = [t for t in args.targets.split(",") if t]
    views = [int(v) for v in args.views.split(",") if v]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b]

    objs = [(ds, oid) for ds in targets for oid in object_ids(ds)]
    for ds in targets:
        n = sum(1 for d, _ in objs if d == ds)
        assert n == TARGETS[ds], f"{ds}: {n} Objekte statt {TARGETS[ds]}"
    print(f"[warm-describe] {len(objs)} Objekte, Views {views}, "
          f"Batches {batch_sizes}", flush=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from generate_descriptions import generate_captions

    t0 = time.perf_counter()
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16,
        device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model_load_s = time.perf_counter() - t0
    print(f"[warm-describe] Modell geladen in {model_load_s:.2f} s", flush=True)

    # Ein Warm-up-Batch (2 Bilder), damit cuDNN-Autotuning nicht die erste
    # Objektmessung verfaelscht; nicht gewertet.
    ds0, oid0 = objs[0]
    root0 = os.path.join(_ROOT, "object_images", ds0, oid0)
    warm_imgs = [Image.open(os.path.join(root0, f)).convert("RGB")
                 for f in view_files(ds0, oid0, 16)[:2]]
    generate_captions(model, processor, warm_imgs, args.prompt)
    print("[warm-describe] Warm-up fertig", flush=True)

    out = {"model_load_s": model_load_s, "prompt": args.prompt,
           "runs": {}, "captions": {}}
    for V in views:
        for bs in batch_sizes:
            key = f"v{V}_b{bs}"
            per_obj, caps = {}, {}
            for i, (ds, oid) in enumerate(objs):
                root = os.path.join(_ROOT, "object_images", ds, oid)
                files = view_files(ds, oid, V)
                t1 = time.perf_counter()
                images = [Image.open(os.path.join(root, f)).convert("RGB")
                          for f in files]
                captions = []
                for k in range(0, len(images), bs):
                    captions += generate_captions(
                        model, processor, images[k:k + bs], args.prompt)
                per_obj[f"{ds}/{oid}"] = time.perf_counter() - t1
                caps[f"{ds}/{oid}"] = dict(zip(files, captions))
                if (i + 1) % 10 == 0 or i == len(objs) - 1:
                    print(f"[warm-describe] {key}: {i + 1}/{len(objs)} "
                          f"(zuletzt {per_obj[f'{ds}/{oid}']:.2f} s)",
                          flush=True)
                # Persistenz VOR jedem weiteren Schritt — ein Abbruch darf
                # keine gemessenen Stunden kosten (Lektion Sweep v2).
                out["runs"][key] = per_obj
                out["captions"][key] = caps
                with open(args.out, "w") as f:
                    json.dump(out, f)

    # Waechter: Batch-8-Texte gegen die bestehenden Beschreibungen
    guard = {}
    for V in views:
        key = f"v{V}_b8"
        if key not in out["captions"]:
            continue
        same = total = 0
        for ds in targets:
            ref = json.load(open(os.path.join(
                _ROOT, "object_database", ds, "descriptions_attributes.json")))
            for (dsoid, caps) in out["captions"][key].items():
                if not dsoid.startswith(ds + "/"):
                    continue
                oid = dsoid.split("/", 1)[1]
                old = ref.get(oid, {}).get("image_descriptions", {})
                for fn, cap in caps.items():
                    if fn in old:
                        total += 1
                        same += (old[fn] == cap)
        guard[key] = {"identisch": same, "verglichen": total,
                      "anteil": round(same / total, 4) if total else None}
        print(f"[warm-describe] Waechter {key}: {same}/{total} identisch",
              flush=True)
    out["guard_identical_to_production"] = guard
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[warm-describe] fertig -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
