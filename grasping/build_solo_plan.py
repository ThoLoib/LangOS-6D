#!/usr/bin/env python3
"""Plan fuer das Solo-Szenario (OBJEKTAUSWAHL_SOLO.md): 20 Objekte, je bis zu 10 Instanzen.

Je Objekt ist der Proxy FIX der beste haeufige Stage-3-Proxy (bester dimDev unter
allen Proxys mit >= 8 Rang-1-Instanzen; Herleitung in OBJEKTAUSWAHL_SOLO.md).
Instanzen werden NUR aus den 3b-Records gezogen, in denen genau dieser Proxy
Rang 1 war — die Verbindung zum echten Retrieval bleibt damit erhalten; die
Instanz liefert im Solo-Szenario nur noch Kamera + Ausgangspose.

Ziehung wie im Stage-5-Protokoll: bevorzugt visib >= 0.5, reihum ueber die
Szenen (sortiert), innerhalb einer Szene gleichmaessig ueber die sortierten
Frames. Reichen die visib->=0.5-Instanzen nicht (Minimum n_top1 = 9 bei drei
Faellen), wird mit den sichtbarsten restlichen aufgefuellt und das geloggt.

    python3 grasping/build_solo_plan.py          # -> _s5_out/solo_run/plan.json
"""
from __future__ import annotations

import collections
import datetime as _dt
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from grasping.build_grasp_instances import _visib  # noqa: E402  (BOP scene_gt_info)

# Format je Zeile: (Fallname, Datensatz, obj_id, Anzeigename, Berichtsgruppe,
# Proxy als "quelle/namensprefix" mit Quelle in {gso, housecat6d, itodd}; der
# Prefix muss genau einen Rang-1-Proxy dieses Objekts im 3b-Lauf treffen).
# Zusammensetzung dieser Liste: grasping/OBJEKTAUSWAHL_SOLO.md.
# Ein EINZELNES Objekt/Proxy freier Wahl braucht keinen Plan:
#   stage_5.py --object <ds>:<id> --proxy <quelle>/<name>   (siehe dort)
SOLO_CASES = [
    ("tless30", "tless", 30, "obj_30",      1, "gso/BIA_Porcelain_Ramekin"),
    ("ycbv2",   "ycbv",   2, "cracker box", 1, "gso/Nestle_Carnation_Cinnamon"),
    ("tless1",  "tless",  1, "obj_1",       1, "itodd/obj_000024"),
    ("tless4",  "tless",  4, "obj_4",       1, "gso/Germanium_GE132"),
    ("tless19", "tless", 19, "obj_19",      1, "itodd/obj_000018"),
    ("tless21", "tless", 21, "obj_21",      1, "gso/Android_Figure_Chrome"),
    ("tless10", "tless", 10, "obj_10",      1, "itodd/obj_000018"),
    ("tless22", "tless", 22, "obj_22",      1, "gso/Android_Figure_Chrome"),
    ("lmo9",    "lmo",    9, "duck",        1, "gso/CHICKEN_RACER"),
    ("ycbv14",  "ycbv",  14, "mug",         1, "housecat6d/cup-red_heart"),
    ("tless5",  "tless",  5, "obj_5",       2, "itodd/obj_000018"),
    ("tless25", "tless", 25, "obj_25",      2, "itodd/obj_000018"),
    ("tless18", "tless", 18, "obj_18",      2, "itodd/obj_000013"),
    ("ycbv5",   "ycbv",   5, "mustard",     2, "gso/Nestle_Nesquik_Chocolate"),
    ("tless28", "tless", 28, "obj_28",      2, "itodd/obj_000018"),
    ("ycbv3",   "ycbv",   3, "sugar box",   2, "gso/Nestle_Nesquik_Chocolate"),
    ("tless24", "tless", 24, "obj_24",      2, "housecat6d/bottle-sanitizer_small_white"),
    ("tless9",  "tless",  9, "obj_9",       2, "itodd/obj_000018"),
    ("tless2",  "tless",  2, "obj_2",       2, "gso/Germanium_GE132"),
    ("tless7",  "tless",  7, "obj_7",       2, "itodd/obj_000013"),
    ("lmo10",   "lmo",   10, "eggbox",      2, "gso/MINI_ROLLER"),
]
PER_OBJECT = 10          # 2026-09-11 von 6 auf 10 erhoeht; Faelle mit kleinerem
                         # Rang-1-Pool werden beim Pool-Maximum gekappt (geloggt)
MIN_VISIB = 0.5


def draw(insts: list, k: int) -> list:
    """Reihum ueber Szenen (sortiert), innerhalb der Szene gleichmaessig ueber
    die sortierten Frames — die Ziehregel des Stage-5-Protokolls."""
    by_scene = collections.defaultdict(list)
    for r in insts:
        by_scene[r["scene"]].append(r)
    for s in by_scene:
        by_scene[s].sort(key=lambda r: r["im"])
        n = len(by_scene[s])
        order = sorted(range(n), key=lambda i: (i * n) % n)  # stabil
        # gleichmaessig: nimm Indizes in Reihenfolge maximaler Spreizung
        spread = []
        step = max(1, n // max(1, min(k, n)))
        seen = set()
        for start in range(step):
            for i in range(start, n, step):
                if i not in seen:
                    seen.add(i)
                    spread.append(by_scene[s][i])
        by_scene[s] = spread
    out, scenes = [], sorted(by_scene)
    while len(out) < k and any(by_scene[s] for s in scenes):
        for s in scenes:
            if by_scene[s] and len(out) < k:
                out.append(by_scene[s].pop(0))
    return out


def main():
    # Bereits gezogene Instanzen eines frueheren Plans bleiben ERHALTEN
    # (Aufstockung statt Neuziehung — fertige Trials behalten ihre Gueltigkeit).
    old = {}
    old_path = os.path.join(_ROOT, "_s5_out", "solo_v2", "plan.json")
    if os.path.exists(old_path):
        for t in json.load(open(old_path))["plan"]:
            old.setdefault(t["case"], []).append(t)
        print(f"[plan] Aufstockung: {sum(map(len, old.values()))} bestehende "
              f"Instanzen aus {old_path} bleiben erhalten")

    plan, warns = [], []
    for rank, (case, ds, oid, name, tier, proxy_pref) in enumerate(SOLO_CASES, 1):
        recs = json.load(open(os.path.join(
            _ROOT, "object_retrieval", "results_bop_stage3_v2", "3b_cross",
            f"{ds}_stage3b", "records.json")))
        src, pref = proxy_pref.split("/", 1)
        tops = sorted({r["top1"] for r in recs
                       if r["obj_id"] == oid and r.get("top1", "").startswith(f"{src}/{pref}")})
        assert len(tops) == 1, f"{case}: Proxy-Prefix {proxy_pref} nicht eindeutig: {tops}"
        proxy = tops[0]
        insts = []
        for r in recs:
            if r["obj_id"] != oid or r["top1"] != proxy:
                continue
            v = _visib(ds, r["scene_id"], r["im_id"], r["gt_idx"])
            insts.append(dict(scene=r["scene_id"], im=r["im_id"], gt_idx=r["gt_idx"],
                              visib=v, diameter=r.get("diameter")))
        want = min(PER_OBJECT, len(insts))
        if want < PER_OBJECT:
            warns.append(f"{case}: Rang-1-Pool hat nur {len(insts)} Instanzen "
                         f"— gekappt auf {want}/{PER_OBJECT}")
        keep = [i for t in old.get(case, [])
                for i in [dict(scene=t["scene"], im=t["im"], gt_idx=t["gt_idx"],
                               visib=t["visib"], diameter=t["diameter"])]]
        kept_keys = {(i["scene"], i["im"], i["gt_idx"]) for i in keep}
        good = [i for i in insts
                if (i["visib"] or 0) >= MIN_VISIB
                and (i["scene"], i["im"], i["gt_idx"]) not in kept_keys]
        pick = keep + draw(good, want - len(keep))
        if len(pick) < want:
            rest = sorted((i for i in insts if i not in pick
                           and (i["scene"], i["im"], i["gt_idx"]) not in kept_keys),
                          key=lambda i: -(i["visib"] or 0))
            fill = rest[:want - len(pick)]
            warns.append(f"{case}: {len(fill)} Instanzen mit visib<{MIN_VISIB} "
                         f"aufgefuellt (visib {[i['visib'] for i in fill]})")
            pick += fill
        assert len(pick) == want, f"{case}: nur {len(pick)} Instanzen im Pool"
        for i in pick:
            plan.append(dict(case=case, rank=rank, tier=tier, dataset=ds, obj_id=oid,
                             name=name, proxy=proxy, pool=len(insts), n_top1=len(insts),
                             **i))
        print(f"#{rank:>2} {case:<8} Proxy {proxy.split('/')[-1][:36]:<38} "
              f"Pool {len(insts):>3}  gezogen {len(pick)}")
    out = os.path.join(_ROOT, "_s5_out", "solo_v2")
    os.makedirs(out, exist_ok=True)
    doc = dict(ts=_dt.datetime.now().isoformat(timespec="seconds"),
               scenario="solo (Objekt allein auf dem Tisch; OBJEKTAUSWAHL_SOLO.md)",
               per_object=PER_OBJECT, min_visib=MIN_VISIB, warns=warns, plan=plan)
    with open(os.path.join(out, "plan.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    for w in warns:
        print("WARNUNG:", w)
    print(f"[plan] {len(SOLO_CASES)} Faelle x {PER_OBJECT} = {len(plan)} Instanzen "
          f"-> {out}/plan.json")


if __name__ == "__main__":
    main()
