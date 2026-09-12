#!/usr/bin/env python3
"""Vollplan: alle GREIFBAREN BOP-Zielobjekte, 3b-Proxy UND 3c-Substitut je Objekt.

Einziges Gate ist die Greifbarkeit (kleinste Abmessung 20-78 mm; Mug als
dokumentierte 81-mm-Ausnahme). Die Proxy-Qualitaet ist KEIN Kriterium: je
Objekt der HAEUFIGSTE Rang-1-Proxy des eingefrorenen 3b-Laufs (das
tatsaechliche Retrieval-Ergebnis) und das haeufigste 3c-Substitut (nb_id;
kann ein Geschwisterobjekt desselben Datensatzes sein). Instanzen: bis zu 10
aus den 3b-Records, in denen der 3b-Proxy Rang 1 war (Ziehregel wie gehabt;
Pool < 10 wird gekappt und geloggt).

    python3 -m grasping.stage_5_full        (baut den Plan und startet den Lauf)
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from grasping.build_grasp_instances import _visib          # noqa: E402
from grasping.build_solo_plan import PER_OBJECT, MIN_VISIB, draw  # noqa: E402
from grasping.sim_scene import object_name                 # noqa: E402

DS_OBJS = {"ycbv": range(1, 22), "tless": range(1, 31), "lmo": [1, 5, 6, 8, 9, 10, 11, 12]}


def build_plan():
    plan, warns, rank = [], [], 0
    for ds, objs in DS_OBJS.items():
        r3b = json.load(open(os.path.join(_ROOT, "object_retrieval",
                                          "results_bop_stage3_v2", "3b_cross",
                                          f"{ds}_stage3b", "records.json")))
        r3c = json.load(open(os.path.join(_ROOT, "object_retrieval",
                                          "results_bop_stage3_v2", "3c_cross",
                                          f"{ds}_stage3c", "records.json")))
        mi = json.load(open(os.path.join(_ROOT, "eval", "datasets", ds,
                                         "models_eval", "models_info.json")))
        for oid in objs:
            # Gate: GREIFBAR (20-78 mm kleinste Abmessung; Mug als dokumentierte
            # 81-mm-Ausnahme) — Proxy-Qualitaet ist ausdruecklich KEIN Kriterium.
            info = mi.get(str(oid), {})
            dims = sorted([info.get("size_x", 0), info.get("size_y", 0),
                           info.get("size_z", 0)])
            if not (20 <= dims[0] <= 78 or (ds == "ycbv" and oid == 14)):
                warns.append(f"{ds}{oid}: nicht greifbar (minDim {dims[0]:.0f} mm) "
                             f"— ausgeschlossen")
                continue
            recs = [r for r in r3b if r["obj_id"] == oid and r.get("top1")]
            if not recs:
                warns.append(f"{ds}{oid}: keine 3b-Records — uebersprungen")
                continue
            proxy3b = collections.Counter(r["top1"] for r in recs).most_common(1)[0][0]
            c3 = collections.Counter(r["nb_id"] for r in r3c
                                     if r["obj_id"] == oid and r.get("nb_id"))
            proxy3c = c3.most_common(1)[0][0] if c3 else ""
            insts = []
            for r in recs:
                if r["top1"] != proxy3b:
                    continue
                v = _visib(ds, r["scene_id"], r["im_id"], r["gt_idx"])
                insts.append(dict(scene=r["scene_id"], im=r["im_id"],
                                  gt_idx=r["gt_idx"], visib=v,
                                  diameter=r.get("diameter")))
            want = min(PER_OBJECT, len(insts))
            good = [i for i in insts if (i["visib"] or 0) >= MIN_VISIB]
            pick = draw(good, want)
            if len(pick) < want:
                rest = sorted((i for i in insts if i not in pick),
                              key=lambda i: -(i["visib"] or 0))
                pick += rest[:want - len(pick)]
            if want < PER_OBJECT:
                warns.append(f"{ds}{oid}: Pool nur {len(insts)} -> {want} Instanzen")
            rank += 1
            for i in pick:
                plan.append(dict(case=f"{ds}{oid}", rank=rank, tier=0, dataset=ds,
                                 obj_id=oid, name=object_name(ds, oid),
                                 proxy=proxy3b, proxy3c=proxy3c,
                                 pool=len(insts), n_top1=len(insts), **i))
            print(f"#{rank:>2} {ds}{oid:<3} 3b={proxy3b.split('/', 1)[1][:28]:<30} "
                  f"3c={(proxy3c.split('/', 1)[1][:28] if proxy3c else '—'):<30} "
                  f"n={len(pick)}")
    out = os.path.join(_ROOT, "_s5_out", "solo_full")
    os.makedirs(out, exist_ok=True)
    json.dump(dict(ts=_dt.datetime.now().isoformat(timespec="seconds"),
                   scenario="solo-full: alle greifbaren BOP-Objekte (20-78 mm, Mug-"
                            "Ausnahme), 3b-Proxy + 3c-Substitut, haeufigster Rang-1, "
                            "keine Proxy-Kuration", per_object=PER_OBJECT,
                   warns=warns, plan=plan),
              open(os.path.join(out, "plan.json"), "w"), indent=1)
    for w in warns:
        print("WARNUNG:", w)
    print(f"[plan] {rank} Objekte, {len(plan)} Instanzen -> {out}/plan.json")


def main():
    """Kompletter Stage-5-Lauf: Plan bauen (falls noetig), dann alle Objekte
    mit gt + 3b-Proxy + 3c-Substitut durchlaufen (Resume aus der CSV)."""
    import argparse
    import subprocess
    ap = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    ap.add_argument("--rebuild-plan", action="store_true",
                    help="Plan neu bauen, auch wenn er schon existiert")
    ap.add_argument("--conditions", default="gt,proxy,proxy3c")
    ap.add_argument("--csv",
                    default=os.path.join(_ROOT, "_s5_out", "solo_full", "trials.csv"))
    args, extra = ap.parse_known_args()
    plan_p = os.path.join(_ROOT, "_s5_out", "solo_full", "plan.json")
    if args.rebuild_plan or not os.path.exists(plan_p):
        build_plan()
    cmd = [sys.executable, "-m", "grasping.stage_5", "--all", "--canonical",
           "--plan", plan_p, "--conditions", args.conditions,
           "--csv", args.csv] + extra
    print("[stage5-full]", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
