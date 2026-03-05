#!/usr/bin/env python3
"""Check consistency across all 4 datasets: descriptions vs models vs renderings."""

import json
import os
from pathlib import Path

BASE = Path("/home/tholoi/thesis/OSCAR")

def load_json_keys(path):
    with open(path) as f:
        data = json.load(f)
    return set(data.keys())

def list_subdirs(path):
    if not path.exists():
        return set()
    return {d.name for d in path.iterdir() if d.is_dir()}

def list_files(path, extensions=None):
    if not path.exists():
        return set()
    results = set()
    for f in path.iterdir():
        if f.is_file():
            if extensions is None or f.suffix.lower() in extensions:
                results.add(f.stem)
    return results

def count_rendering_contents(render_dir):
    """Count PNGs and NPYs in a rendering subfolder."""
    if not render_dir.exists():
        return 0, 0
    pngs = len([f for f in render_dir.iterdir() if f.suffix.lower() == '.png'])
    npys = len([f for f in render_dir.iterdir() if f.suffix.lower() == '.npy'])
    return pngs, npys

def print_list(items, label, max_show=50):
    if len(items) == 0:
        print(f"  {label}: 0")
    elif len(items) <= max_show:
        print(f"  {label}: {len(items)}")
        for item in sorted(items):
            print(f"    - {item}")
    else:
        print(f"  {label}: {len(items)} (too many to list)")

def report(name, desc_keys, model_keys, render_keys):
    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"{'='*70}")
    print(f"  Description keys:    {len(desc_keys)}")
    print(f"  Model entries:       {len(model_keys)}")
    print(f"  Rendering folders:   {len(render_keys)}")
    
    missing_models = desc_keys - model_keys
    missing_renders = desc_keys - render_keys
    extra_renders = render_keys - desc_keys
    extra_models = model_keys - desc_keys
    
    print()
    print_list(missing_models, "Descriptions WITHOUT models")
    print_list(missing_renders, "Descriptions WITHOUT renderings")
    print(f"  Renderings WITHOUT descriptions: {len(extra_renders)}")
    if extra_renders and len(extra_renders) <= 50:
        for item in sorted(extra_renders):
            print(f"    - {item}")
    print(f"  Models WITHOUT descriptions: {len(extra_models)}")
    if extra_models and len(extra_models) <= 50:
        for item in sorted(extra_models):
            print(f"    - {item}")


# ─── 1. YCBV_GSO ─────────────────────────────────────────────────────────────
desc_path = BASE / "object_database/descriptions_tessa/ycbv_gso/descriptions_attributes.json"
model_dir = BASE / "object_database/ycbv_gso"
render_dir = BASE / "object_images/ycbv_gso"

desc_keys = load_json_keys(desc_path)
model_keys = list_subdirs(model_dir)
render_keys = list_subdirs(render_dir)

report("YCBV_GSO (all)", desc_keys, model_keys, render_keys)

# Check rendering completeness for YCBV_GSO
incomplete = []
for obj in sorted(desc_keys & render_keys):
    pngs, npys = count_rendering_contents(render_dir / obj)
    if pngs < 9 or npys < 8:
        incomplete.append((obj, pngs, npys))
if incomplete:
    print(f"\n  Incomplete renderings (expected 9 PNG + 8 NPY): {len(incomplete)}")
    for obj, p, n in incomplete[:30]:
        print(f"    - {obj}: {p} PNGs, {n} NPYs")
    if len(incomplete) > 30:
        print(f"    ... and {len(incomplete)-30} more")
else:
    print(f"\n  All matched renderings have complete files (9 PNG + 8 NPY).")

# ─── 2. YCBV (subset) ────────────────────────────────────────────────────────
desc_path_ycbv = BASE / "object_database/descriptions_tessa/ycbv/descriptions_attributes.json"
# models and renderings are in the same ycbv_gso folders

desc_keys_ycbv = load_json_keys(desc_path_ycbv)
# model_keys and render_keys are same as above

report("YCBV (subset)", desc_keys_ycbv, model_keys, render_keys)

# ─── 3. HouseCat6D ───────────────────────────────────────────────────────────
desc_path_hcat = BASE / "object_database/descriptions_tessa/housecat6d/descriptions_attributes.json"
model_dir_hcat = BASE / "object_database/housecat6d"
render_dir_hcat = BASE / "object_images/housecat6d"

desc_keys_hcat = load_json_keys(desc_path_hcat)

# Models are in category subfolders, files like bottle-85_alcool.obj
# We need to collect all model file stems across all category subdirs
hcat_model_keys = set()
if model_dir_hcat.exists():
    for cat_dir in model_dir_hcat.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ('.obj', '.glb', '.ply', '.stl'):
                    hcat_model_keys.add(f.stem)

hcat_render_keys = list_subdirs(render_dir_hcat)

report("HouseCat6D", desc_keys_hcat, hcat_model_keys, hcat_render_keys)

# Check rendering completeness for HouseCat6D
incomplete_hcat = []
for obj in sorted(desc_keys_hcat & hcat_render_keys):
    pngs, npys = count_rendering_contents(render_dir_hcat / obj)
    if pngs < 9 or npys < 8:
        incomplete_hcat.append((obj, pngs, npys))
if incomplete_hcat:
    print(f"\n  Incomplete renderings (expected 9 PNG + 8 NPY): {len(incomplete_hcat)}")
    for obj, p, n in incomplete_hcat[:30]:
        print(f"    - {obj}: {p} PNGs, {n} NPYs")
    if len(incomplete_hcat) > 30:
        print(f"    ... and {len(incomplete_hcat)-30} more")
else:
    print(f"\n  All matched renderings have complete files (9 PNG + 8 NPY).")

# ─── 4. MI3DOR ───────────────────────────────────────────────────────────────
desc_path_mi = BASE / "object_database/descriptions_tessa/MI3DOR/descriptions_attributes.json"
model_dir_mi = BASE / "object_database/MI3DOR/model/test"
render_dir_mi = BASE / "object_images/MI3DOR"

desc_keys_mi = load_json_keys(desc_path_mi)

# Models are in category subfolders, files like knife_0001.glb
mi_model_keys = set()
if model_dir_mi.exists():
    for cat_dir in model_dir_mi.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.iterdir():
                if f.is_file() and f.suffix.lower() in ('.obj', '.glb', '.ply', '.stl', '.off'):
                    mi_model_keys.add(f.stem)

mi_render_keys = list_subdirs(render_dir_mi)

report("MI3DOR", desc_keys_mi, mi_model_keys, mi_render_keys)

# Check rendering completeness for MI3DOR
incomplete_mi = []
for obj in sorted(desc_keys_mi & mi_render_keys):
    pngs, npys = count_rendering_contents(render_dir_mi / obj)
    if pngs < 9 or npys < 8:
        incomplete_mi.append((obj, pngs, npys))
if incomplete_mi:
    print(f"\n  Incomplete renderings (expected 9 PNG + 8 NPY): {len(incomplete_mi)}")
    for obj, p, n in incomplete_mi[:30]:
        print(f"    - {obj}: {p} PNGs, {n} NPYs")
    if len(incomplete_mi) > 30:
        print(f"    ... and {len(incomplete_mi)-30} more")
else:
    print(f"\n  All matched renderings have complete files (9 PNG + 8 NPY).")

print(f"\n{'='*70}")
print("  DONE")
print(f"{'='*70}")
