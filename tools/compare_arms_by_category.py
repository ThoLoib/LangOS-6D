#!/usr/bin/env python3
"""
compare_arms_by_category.py — welcher Arm gewinnt in WELCHER Kategorie?

    # die 2x2-Matrix Query-Modus x Gallery-Repraesentation
    python3 tools/compare_arms_by_category.py --preset shape-matrix

    # zwei beliebige Arme, nach groesstem Abstand sortiert
    python3 tools/compare_arms_by_category.py \\
        E1_shape_only E2b_fullmesh_shape_only --metric nDCG

    # nach Kategorie sortiert statt nach Abstand, als CSV
    python3 tools/compare_arms_by_category.py --preset shape-matrix \\
        --sort category --csv docs/shape_matrix_by_category.csv

Liest ``results_per_query.json`` je Arm — dort steht neben den Metriken die
GT-Kategorie jeder Query. Ein Aggregat sagt nur, WELCHER Arm besser ist; erst
die Aufschluesselung sagt, WO und damit warum.

Gepaart ueber die Query-ID: beide Arme sehen dieselben 2101 Queries, also wird
je Kategorie die mittlere Differenz derselben Queries gebildet, nicht die
Differenz zweier unabhaengiger Mittelwerte.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1 = os.path.join(_ROOT, "object_retrieval", "results_shrec18_v2_stage1_42v_k5")

PRESETS = {
    # Die vier Zellen der Matrix, isolierter Shape-Kanal.
    "shape-matrix": [
        ("E1_shape_only", "pc x partial"),
        ("E2b_fullmesh_shape_only", "pc x full-mesh"),
        ("E7_ulip2_cross_shape_only", "cross x partial"),
        ("E7_ulip2_cross_fullmesh_shape_only", "cross x full-mesh"),
    ],
    "fusion": [
        ("E1c_full_fusion", "BASE (pc, partial)"),
        ("E2b_fullmesh", "pc x full-mesh"),
        ("E7_ulip2_cross_fullmesh", "cross x full-mesh"),
    ],
}


def load(arm: str) -> Dict[str, dict]:
    f = os.path.join(S1, arm, "results_per_query.json")
    if not os.path.isfile(f):
        return {}
    d = json.load(open(f))
    rows = d if isinstance(d, list) else list(d.values())
    return {r["id"]: r for r in rows if "id" in r}


def category_of(rec) -> str:
    """Die Kategorie steht als String einer Liste drin ("['keyboard', ...]").

    Erster Eintrag = Kategorie, zweiter = Unterkategorie. Wir gruppieren auf
    der Kategorie; ein literal_eval waere fragil, deshalb wird gesaeubert.
    """
    c = rec.get("category")
    if isinstance(c, (list, tuple)):
        return str(c[0])
    s = str(c or "?").strip("[]")
    return s.split(",")[0].strip().strip("'\"") or "?"


def compare(a: str, b: str, metric: str):
    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return None, []
    per: Dict[str, List[float]] = {}
    for q in common:
        va, vb = ra[q].get(metric), rb[q].get(metric)
        if va is None or vb is None:
            continue
        per.setdefault(category_of(ra[q]), []).append(float(va) - float(vb))
    rows = [(c, len(v), statistics.fmean(v),
             statistics.fmean([1.0 if d > 0 else 0.0 for d in v]))
            for c, v in per.items()]
    overall = statistics.fmean([d for v in per.values() for d in v])
    return overall, sorted(rows, key=lambda r: -abs(r[2]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", help="genau zwei Arme, oder --preset")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--metric", default="nDCG",
                    help="nDCG | NN_sub | NN_cat | MRR | AP | nDCG_K")
    ap.add_argument("--sort", choices=["delta", "category"], default="delta")
    ap.add_argument("--top", type=int, default=0, help="nur die N groessten Abstaende")
    ap.add_argument("--csv", help="zusaetzlich als CSV schreiben")
    args = ap.parse_args()

    if args.preset:
        arms = PRESETS[args.preset]
        missing = [a for a, _ in arms
                   if not os.path.isfile(os.path.join(S1, a, "results_per_query.json"))]
        if missing:
            print(f"[warn] fehlende Arme (noch nicht gelaufen?): {missing}\n")
        arms = [x for x in arms if x[0] not in missing]
        pairs = [(arms[0], b) for b in arms[1:]]
    elif len(args.arms) == 2:
        pairs = [((args.arms[0], args.arms[0]), (args.arms[1], args.arms[1]))]
        pairs = [(pairs[0][0], pairs[0][1])]
    else:
        ap.error("entweder --preset oder genau zwei Arme angeben")

    out_rows = []
    for (a, la), (b, lb) in pairs:
        overall, rows = compare(a, b, args.metric)
        if overall is None:
            print(f"!! keine gemeinsamen Queries: {a} vs {b}")
            continue
        if args.sort == "category":
            rows.sort(key=lambda r: r[0])
        if args.top:
            rows = rows[:args.top]
        print(f"\n=== {la}  minus  {lb}   ({args.metric}) ===")
        print(f"  Gesamt: {overall:+.4f}\n")
        print(f"  {'Kategorie':<22}{'n':>6}{'Δ ' + args.metric:>12}{'Anteil gewonnen':>18}")
        for c, n, d, w in rows:
            bar = "+" * min(int(abs(d) * 40), 18)
            print(f"  {c:<22}{n:>6}{d:>+12.4f}{w:>17.0%}  {bar if d > 0 else ''}")
            out_rows.append((la, lb, args.metric, c, n, round(d, 5), round(w, 4)))
        print(f"\n  Positiv = '{la}' ist dort besser.")

    if args.csv and out_rows:
        import csv
        p = args.csv if os.path.isabs(args.csv) else os.path.join(_ROOT, args.csv)
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["arm_a", "arm_b", "metric", "category", "n",
                        "delta_mean", "share_a_wins"])
            w.writerows(out_rows)
        print(f"\n  CSV: {p}")


if __name__ == "__main__":
    main()
