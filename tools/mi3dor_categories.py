#!/usr/bin/env python3
"""
mi3dor_categories.py — Kategorientabelle zu Stage 2 (RESULTS.md 2.6).

Per-Kategorie-NN der drei isolierten Kanaele plus Fusion, und die Zaehlung
"Fusion schlechter als bester Einzelkanal" (erwartet: 9 von 21, vase −0.132).
Die Fusionsspalte ist ``clip_dino_ulip_full`` des angegebenen Laufs — fuer die
publizierte Tabelle der Full-Mesh-Lauf.

    python3 tools/mi3dor_categories.py
    python3 tools/mi3dor_categories.py --results object_retrieval/results_repro_stage2/fullmesh --csv tabelle.csv
"""
import argparse
import collections
import csv
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ("object_retrieval/results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix"
           "/fullmesh")
ARMS = ["clip_only", "dino_only_full", "ulip_only_full", "clip_dino_ulip_full"]
LBL = dict(zip(ARMS, ["text", "view", "shape", "fusion"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--results", default=DEFAULT,
                    help="Lauf-Ordner mit results_topk_15.json (Default: "
                         "publizierter Full-Mesh-Lauf)")
    ap.add_argument("--csv", help="Tabelle zusaetzlich als CSV")
    args = ap.parse_args()

    p = os.path.join(_ROOT, args.results, "results_topk_15.json")
    if not os.path.isfile(p):
        sys.exit(f"fehlt: {p}")
    data = json.load(open(p))

    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in data:
        for a in ARMS:
            rp = r["eval_trace"]["arms"][a]["rel_positions"]
            acc[r["gt"]][a].append(1.0 if (rp and rp[0] == 1) else 0.0)

    rows = []
    for cat, per in acc.items():
        vals = {a: sum(per[a]) / len(per[a]) for a in ARMS}
        best_v, best_n = max((vals[a], LBL[a]) for a in ARMS[:3])
        rows.append((cat, len(per[ARMS[0]]), vals["clip_only"],
                     vals["dino_only_full"], vals["ulip_only_full"],
                     best_n, best_v, vals["clip_dino_ulip_full"],
                     vals["clip_dino_ulip_full"] - best_v))
    rows.sort(key=lambda r: r[8])

    print(f"{'Kategorie':<12}{'n':>5}{'text':>8}{'view':>8}{'shape':>8}"
          f"   {'bester':<8}{'Fusion':>8}{'Delta':>9}")
    for r in rows:
        print(f"{r[0]:<12}{r[1]:>5}{r[2]:>8.3f}{r[3]:>8.3f}{r[4]:>8.3f}"
              f"   {r[5]:<2} {r[6]:<5.3f}{r[7]:>8.3f}{r[8]:>+9.3f}")
    neg = [r for r in rows if r[8] < -1e-9]
    print(f"\nFusion schlechter als bester Einzelkanal: {len(neg)} von {len(rows)}"
          f" (groesster Verlust {min(r[8] for r in rows):+.3f})")
    win = collections.Counter(r[5] for r in rows)
    print("Bester Einzelkanal je Kategorie:", dict(win))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["category", "n", "NN_text", "NN_view", "NN_shape",
                        "best_single", "best_single_NN", "NN_fusion", "delta"])
            w.writerows(rows)
        print(f"geschrieben: {args.csv}")


if __name__ == "__main__":
    main()
