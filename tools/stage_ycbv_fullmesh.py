#!/usr/bin/env python3
# =============================================================================
# tools/stage_ycbv_fullmesh.py — give every YCB-V mesh a unique filename so the
#                                ulip_fullmesh pass can key on it.
# =============================================================================
#
# SUPERSEDED 2026-08-08 — nothing calls this any more.
# precompute_embeddings.py gained `--mesh-id-mode {stem,parent,grandparent}`,
# which fixes the same id collision by handing build_pipeline an explicit
# cad_mesh_items list instead of copying meshes into a flat directory
# (`--mesh-id-mode parent` for YCB-V, `grandparent` for GSO). That scales to
# GSO's 1030 objects, where staging would have duplicated 17 GB. Kept only in
# case a flat physical copy is ever wanted for another reason.
#
# WHY
# ---
# eval_common.build_pipeline derives the full-mesh object id from the FILENAME
# STEM:
#
#     cad_mesh_items = [(os.path.splitext(os.path.basename(p))[0], p) ...]
#
# YCB-V's layout is object_database/ycbv/<id>/textured_simple.obj, so a glob of
# `object_database/ycbv/*/textured_simple.obj` maps all 21 objects to the single
# id "textured_simple" — 20 of them are silently overwritten and the resulting
# cache is garbage. (LM-O and T-LESS avoid this by globbing the flat BOP model
# dirs, where filenames already are the ids.)
#
# WHAT IT DOES
# ------------
# For each object <id>, writes into object_database/ycbv/_fullmesh/:
#
#     <id>.obj           copy of textured_simple.obj, `mtllib` rewritten
#     <id>.mtl           copy of textured_simple.obj.mtl, `map_Kd` rewritten
#     <id>_texture.png   copy of texture_map.png
#
# Both renames are necessary: the material file and the texture have the same
# name in every object directory, so a flat staging dir needs all three
# uniquified or objects would share the first-copied texture.
#
# Geometry is copied byte-for-byte; only the two reference lines change. The
# script re-loads each staged mesh with trimesh afterwards and fails loudly if
# the material/texture did not come back — a silently texture-less mesh would
# produce colored-ULIP embeddings of a grey object, which is worse than no
# fullmesh pass at all.
#
# Idempotent: an object whose staged .obj is newer than its source is skipped.
#
# Run inside the oscar container:
#     docker compose run --rm --no-deps oscar bash -lc \
#         "cd /app && python3 -u tools/stage_ycbv_fullmesh.py"
# =============================================================================

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_OBJ = "textured_simple.obj"
SRC_MTL = "textured_simple.obj.mtl"
SRC_TEX = "texture_map.png"


def stage_one(obj_id: str, src_dir: str, out_dir: str, overwrite: bool) -> str | None:
    """Stage one object. Returns the staged .obj path, or None if skipped."""
    src_obj = os.path.join(src_dir, SRC_OBJ)
    src_mtl = os.path.join(src_dir, SRC_MTL)
    src_tex = os.path.join(src_dir, SRC_TEX)
    for p in (src_obj, src_mtl, src_tex):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{obj_id}: missing {p}")

    dst_obj = os.path.join(out_dir, f"{obj_id}.obj")
    dst_mtl = os.path.join(out_dir, f"{obj_id}.mtl")
    dst_tex = os.path.join(out_dir, f"{obj_id}_texture.png")

    if (not overwrite and os.path.isfile(dst_obj)
            and os.path.getmtime(dst_obj) >= os.path.getmtime(src_obj)):
        return None

    # --- .obj: rewrite the mtllib line -------------------------------------
    with open(src_obj, "r", errors="replace") as f:
        obj_text = f.read()
    obj_text, n = re.subn(r"(?m)^\s*mtllib\s+.*$", f"mtllib {obj_id}.mtl", obj_text)
    if n != 1:
        raise ValueError(f"{obj_id}: expected exactly 1 mtllib line in {SRC_OBJ}, found {n}")
    with open(dst_obj, "w") as f:
        f.write(obj_text)

    # --- .mtl: rewrite the map_* texture references ------------------------
    with open(src_mtl, "r", errors="replace") as f:
        mtl_text = f.read()
    # NB: a function replacement, not r"\1" + name. Every YCB-V id starts with a
    # digit (002_master_chef_can), so the string form "\1002_master_chef_can..."
    # is read by re as backreference \100, not group 1 followed by "002".
    mtl_text, n = re.subn(
        r"(?m)^(\s*map_\w+\s+)(?:\S+[/\\])?" + re.escape(SRC_TEX) + r"\s*$",
        lambda m: m.group(1) + f"{obj_id}_texture.png",
        mtl_text,
    )
    if n < 1:
        raise ValueError(f"{obj_id}: no map_Kd reference to {SRC_TEX} in {SRC_MTL}")
    with open(dst_mtl, "w") as f:
        f.write(mtl_text)

    shutil.copyfile(src_tex, dst_tex)
    return dst_obj


def verify(path: str) -> None:
    """Load the staged mesh and assert it still carries its texture."""
    import trimesh
    mesh = trimesh.load(path, process=False, force="mesh")
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"{path}: loaded empty mesh")
    # trimesh hands back either a SimpleMaterial (.image, from map_Kd) or a
    # PBRMaterial (.baseColorTexture) depending on version — accept both.
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    if img is None:
        raise ValueError(
            f"{path}: trimesh loaded the mesh but found no texture image — the "
            f"mtl/texture rewrite did not survive. Refusing to stage a mesh the "
            f"colored ULIP-2 pass would read as grey.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-dir", default=os.path.join(ROOT, "object_database", "ycbv"),
                    help="object_database/ycbv (default: %(default)s)")
    ap.add_argument("--out-dir", default="",
                    help="staging dir (default: <db-dir>/_fullmesh)")
    ap.add_argument("--overwrite", action="store_true",
                    help="restage objects that are already up to date")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the trimesh texture check (not recommended)")
    args = ap.parse_args()

    db_dir = os.path.abspath(args.db_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(db_dir, "_fullmesh"))
    os.makedirs(out_dir, exist_ok=True)

    obj_ids = sorted(
        d for d in os.listdir(db_dir)
        if not d.startswith("_")
        and os.path.isfile(os.path.join(db_dir, d, SRC_OBJ))
    )
    if not obj_ids:
        print(f"[stage] ERROR: no <id>/{SRC_OBJ} under {db_dir}", file=sys.stderr)
        return 1

    print(f"[stage] {len(obj_ids)} objects: {db_dir} → {out_dir}")
    staged = skipped = 0
    for obj_id in obj_ids:
        try:
            path = stage_one(obj_id, os.path.join(db_dir, obj_id), out_dir, args.overwrite)
        except Exception as exc:
            print(f"[stage] FAILED {obj_id}: {exc}", file=sys.stderr)
            return 2
        if path is None:
            skipped += 1
            continue
        if not args.no_verify:
            try:
                verify(path)
            except Exception as exc:
                print(f"[stage] FAILED {obj_id}: {exc}", file=sys.stderr)
                return 3
        staged += 1
        print(f"[stage]   {obj_id} ok")

    print(f"[stage] done — {staged} staged, {skipped} already up to date.")
    print(f"[stage] use --mesh-glob 'object_database/ycbv/_fullmesh/*.obj'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
