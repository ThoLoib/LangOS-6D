#!/usr/bin/env python3
# =============================================================================
# tools/stage_ycbv_bop_ids.py — file the TEXTURED YCB-Video meshes under BOP
#                               object ids, so the YCB-V gallery keys on
#                               obj_000001..obj_000021.
# =============================================================================
#
# WHY BOP IDS
# -----------
# T-LESS, LM-O and ITODD galleries key on obj_0000NN (their BOP model filenames
# already are the ids). YCB-V was the odd one out, keyed on names like
# 002_master_chef_can, which then needed id_to_label.json to reach the obj_id
# used by test_targets_bop19.json. Filing YCB-V the same way removes that
# mapping step and makes all four BOP galleries uniform.
#
# WHY NOT JUST RENDER THE BOP PLYs
# --------------------------------
# BOP ships eval/datasets/ycbv/models/obj_0000NN.ply, which would give the ids
# for free — but those PLYs reference their texture through a
# `comment TextureFile obj_0000NN.png` header, and rendering.py imports PLY via
# Blender's legacy `bpy.ops.import_mesh.ply`, which does not read that comment.
# The PLYs carry texture_u/texture_v but NO per-vertex colour, so there is no
# fallback: every YCB-V object would render flat grey, and DINOv2 / SigLIP /
# LLaVA would all see unlabelled blobs. For a dataset whose objects are
# distinguished largely by their packaging, that is a severe loss.
#
# The YCB-Video meshes carry a proper .mtl -> map_Kd -> texture_map.png chain
# that Blender loads correctly, and their geometry is IDENTICAL to BOP's (bbox
# sizes agree to 0.00 mm on all 21 objects; only the origin differs, and both
# rendering.py and generate_partial_pointclouds.py normalise that away by
# centring on the bbox and scaling max-dim to 1.0). So the textured .obj is
# strictly better for the gallery.
#
# Pose evaluation is unaffected and must keep using eval/datasets/ycbv/models —
# GT poses live in BOP's bbox-centred frame, which differs from the YCB-Video
# frame by up to 28.23 mm.
#
# WHAT IT DOES
# ------------
# For each BOP id N -> name (from eval/datasets/ycbv/id_to_label.json):
#
#     object_database/ycbv/obj_0000NN/textured_simple.obj
#                                     textured_simple.obj.mtl
#                                     texture_map.png
#
# copied verbatim from object_database/ycbv_ycbvideo/<name>/. No rewriting is
# needed: the .obj references `./textured_simple.obj.mtl` and the .mtl
# references `texture_map.png`, both resolved relative to the file, and each
# object keeps its own directory — so the names never collide.
#
# `infer_model_id` in rendering.py treats textured_simple.obj as a generic model
# name and takes the PARENT directory as the id, which is exactly obj_0000NN.
#
# Idempotent. Run inside the oscar container:
#     docker compose run --rm --no-deps oscar bash -lc \
#         "cd /app && python3 -u tools/stage_ycbv_bop_ids.py"
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("textured_simple.obj", "textured_simple.obj.mtl", "texture_map.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=os.path.join(ROOT, "object_database", "ycbv_ycbvideo"),
                    help="YCB-Video package, name-keyed (default: %(default)s)")
    ap.add_argument("--dst", default=os.path.join(ROOT, "object_database", "ycbv"),
                    help="BOP-id-keyed output (default: %(default)s)")
    ap.add_argument("--id-map", default=os.path.join(ROOT, "eval", "datasets", "ycbv",
                                                     "id_to_label.json"),
                    help="BOP id -> YCB name (default: %(default)s)")
    ap.add_argument("--n-objects", type=int, default=21,
                    help="BOP ids 1..N to stage (default: %(default)s)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print(f"[stage] ERROR: source not found: {args.src}\n"
              f"        Expected the YCB-Video package (moved aside 2026-08-08).",
              file=sys.stderr)
        return 1
    with open(args.id_map) as f:
        id_map = json.load(f)

    os.makedirs(args.dst, exist_ok=True)
    staged = skipped = 0
    for n in range(1, args.n_objects + 1):
        name = id_map.get(str(n)) or id_map.get(n)
        if not name:
            print(f"[stage] ERROR: no name for BOP id {n} in {args.id_map}", file=sys.stderr)
            return 2
        src_dir = os.path.join(args.src, name)
        obj_id = f"obj_{n:06d}"
        dst_dir = os.path.join(args.dst, obj_id)

        missing = [f for f in FILES if not os.path.isfile(os.path.join(src_dir, f))]
        if missing:
            print(f"[stage] ERROR: {name} is missing {missing} in {src_dir}", file=sys.stderr)
            return 3

        if (not args.overwrite
                and all(os.path.isfile(os.path.join(dst_dir, f)) for f in FILES)):
            skipped += 1
            continue

        os.makedirs(dst_dir, exist_ok=True)
        for f in FILES:
            shutil.copyfile(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        staged += 1
        print(f"[stage]   {obj_id} <- {name}")

    print(f"[stage] done — {staged} staged, {skipped} already present, "
          f"{args.n_objects} total in {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
