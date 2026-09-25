#!/usr/bin/env python3
"""Stage-1-Gewichts-Sweep v2 — Referenzkonfiguration, Produktionsfusion.

Ersetzt die 16v/k8-Sweeps vom 25./26.08. (weightmap_pc/cross.csv, nur nDCG).
Baugleich zum Stage-2-Sweep v2: 66 Simplex-Punkte (Schritt 0.1) auf der
FINALEN 42v/k5-Konfiguration, je Punkt via AblationSpec + make_fusion_module +
derive_ranking + score_official/score_depth_matched — byte-identisch zu den
regulaeren Armen (Kanaele mit Gewicht 0 werden wie in run_weight_sweep aus dem
Spec entfernt, Ecken == isolierte Arme). Waechter: BASE (0.3,0.4,0.3) muss die
Produktions-Arme treffen (pc: E1c_full_fusion nDCG 0.5868 / hit@1 0.3413;
cross: E7_ulip2_cross nDCG 0.5588 / hit@1 0.3289), sonst kein CSV.

    docker compose run --rm oscar python3 experiments/stage1_weight_sweep_v2.py
    # Smoke: S1SWEEP_LIMIT=25 S1SWEEP_POINTS=3
"""
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "object_retrieval"))
sys.path.insert(0, os.path.join(_ROOT, "experiments"))
os.environ.setdefault("SHREC_FORCE_PARTIAL_CACHE", "1")

import numpy as np                                                  # noqa: E402
import experiment1_shrec18_stage1 as E                              # noqa: E402
from experiment1_shrec18_stage1 import (                            # noqa: E402
    AblationSpec, DEFAULTS, make_fusion_module, derive_ranking,
    run_pass, validate_inputs, load_official_gt, prepare_queries,
    score_official, score_depth_matched)

E.GEOM_K = 5                     # Tiefe der Tabelle-B-Metriken (42v/k5-Config)
STEP = 0.1
GUARDS = {"pc":    {"pass": "ulip_pc_rgb",    "nDCG": 0.5868, "hit1": 0.3413,
                    "arm": "E1c_full_fusion"},
          "cross": {"pass": "ulip_cross_rgb", "nDCG": 0.5588, "hit1": 0.3289,
                    "arm": "E7_ulip2_cross"}}
TOL = 2e-4
OUT_DIR = os.path.join(_ROOT, "final_results", "stage1")
LIMIT = int(os.environ.get("S1SWEEP_LIMIT", "0") or "0") or None
MAX_PTS = int(os.environ.get("S1SWEEP_POINTS", "0") or "0") or None

PATHS = {"data_root": os.path.join(_ROOT, "eval/datasets/shrec18/shrec18_full"),
         "images_dir": os.path.join(_ROOT, "object_images/shrec18_v2"),
         "desc_file": os.path.join(_ROOT, "object_database/shrec18_v2/descriptions_attributes.json"),
         "results_root": os.path.join(_ROOT, "object_retrieval/results_shrec18_v2_stage1_42v_k5"),
         "stage1_root": DEFAULTS["stage1_root"]}


def simplex(step):
    n = int(round(1.0 / step))
    return [(round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))
            for i in range(n + 1) for j in range(n + 1 - i)]


def main():
    object_ids = validate_inputs(PATHS, True)
    gt = load_official_gt(PATHS["data_root"], PATHS["stage1_root"])
    cad_labels, freqs = gt["cad"], gt["freqs"]
    index = prepare_queries(PATHS["data_root"], PATHS["stage1_root"], gt)
    if LIMIT:
        index = index[:LIMIT]
    cad_dir = os.path.join(PATHS["data_root"], "cad")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=_ROOT).stdout.strip()

    for mode, g in GUARDS.items():
        stores = {pk: run_pass(pk, PATHS, index, object_ids, LIMIT, resume=True)
                  for pk in ("base", g["pass"])}
        full = {"clip": ("base", None), "dino": ("base", 42),
                "shape": (g["pass"], None)}

        def eval_point(wt, wv, ws):
            chan = {ch: full[ch] for ch, w in
                    zip(("clip", "dino", "shape"), (wt, wv, ws)) if w > 0}
            spec = AblationSpec(name=f"W_{wt}_{wv}_{ws}", group="WSWEEP2",
                                question="weight sensitivity v2",
                                channels=chan, weights=(wt, wv, ws))
            fm = make_fusion_module(spec)
            s = dict(nDCG=0.0, hit1=0.0, NN_cat=0.0, MRR=0.0)
            nq = 0
            for q in index:
                ql = tuple(q["category"])
                if freqs.get(ql[0], 0) == 0:
                    continue
                ranking = derive_ranking(spec, q["id"], stores, object_ids,
                                         fm, cad_dir, None)
                ranked = [object_ids[i] for i in ranking]
                mo = score_official(ranked, ql, cad_labels, freqs)
                if mo is None:
                    continue
                mb = score_depth_matched(ranked, ql, cad_labels, E.GEOM_K)
                s["nDCG"] += mo["nDCG"]
                s["hit1"] += mb["NN_sub"]
                s["NN_cat"] += mb["NN_cat"]
                s["MRR"] += mb["MRR"]
                nq += 1
            return {k: v / nq for k, v in s.items()}, nq

        # WAECHTER zuerst
        base, nq = eval_point(0.3, 0.4, 0.3)
        print(f"[s1sweep2:{mode}] BASE: nDCG={base['nDCG']:.4f} "
              f"hit@1={base['hit1']:.4f} (Produktion {g['arm']}: "
              f"{g['nDCG']}/{g['hit1']}; n={nq})", flush=True)
        if LIMIT:
            print(f"[s1sweep2:{mode}] Smoke — Waechter uebersprungen")
        elif abs(base["nDCG"] - g["nDCG"]) > TOL or abs(base["hit1"] - g["hit1"]) > TOL:
            sys.exit(f"[s1sweep2:{mode}] ABBRUCH: BASE verfehlt {g['arm']}")

        rows = []
        pts = simplex(STEP)[:MAX_PTS] if MAX_PTS else simplex(STEP)
        for (wt, wv, ws) in pts:
            m, _ = eval_point(wt, wv, ws)
            rows.append({"w_text": wt, "w_view": wv, "w_shape": ws,
                         **{k: round(v, 4) for k, v in m.items()}})
            print(f"[s1sweep2:{mode}] w=({wt},{wv},{ws}) nDCG={m['nDCG']:.4f} "
                  f"hit@1={m['hit1']:.4f}", flush=True)
        out_csv = os.path.join(OUT_DIR, f"weight_sweep_66_{mode}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["w_text", "w_view", "w_shape",
                                              "nDCG", "hit1", "NN_cat", "MRR"])
            w.writeheader()
            w.writerows(rows)
        best_h = max(rows, key=lambda r: r["hit1"])
        best_n = max(rows, key=lambda r: r["nDCG"])
        noshape = [r for r in rows if r["w_shape"] == 0]
        manifest = dict(ts=_dt.datetime.now().isoformat(timespec="seconds"),
                        git=git, mode=mode, grid=f"{len(rows)} Punkte, Schritt {STEP}",
                        config="42v/k5 (SHAPE_AGG_VIEWS=42, ulip_view_topk=5, "
                               "DINO 42v, partielle Referenzen), GEOM_K=5",
                        guard=dict(arm=g["arm"], base=base,
                                   produktion=dict(nDCG=g["nDCG"], hit1=g["hit1"])),
                        best_hit1=best_h, best_nDCG=best_n,
                        best_ohne_shape_hit1=max(noshape, key=lambda r: r["hit1"]) if noshape else None,
                        best_ohne_shape_nDCG=max(noshape, key=lambda r: r["nDCG"]) if noshape else None)
        json.dump(manifest, open(os.path.join(
            OUT_DIR, f"weight_sweep_66_{mode}_manifest.json"), "w"), indent=1)
        print(f"[s1sweep2:{mode}] {len(rows)} Punkte -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
