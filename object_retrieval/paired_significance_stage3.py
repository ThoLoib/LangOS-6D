#!/usr/bin/env python3
"""
paired_significance_stage3.py — gepaarte Signifikanz fuer die BOP-Arme.

    python3 object_retrieval/paired_significance_stage3.py                 # Standardpaare
    python3 object_retrieval/paired_significance_stage3.py 3a_cross_v2 3a_cross_fullmesh_v2
    python3 object_retrieval/paired_significance_stage3.py --metric mrr --per-dataset

Gegenstueck zu `paired_significance.py` (Stage 1). Gepaart wird ueber den
Instanzschluessel (dataset, scene, image, object, gt_idx) — beide Arme sehen
dieselben 12.284 BOP-Instanzen, also wird die Differenz DERSELBEN Instanzen
gebildet und nicht die zweier unabhaengiger Mittelwerte.

Metriken aus `target_rank`:
  hit1  1 wenn der Zielrang 1 ist, sonst 0   -> entspricht Recall@1
  mrr   1/Rang                               -> entspricht MRR

Berichtet werden **beide** Tests, weil sie Verschiedenes messen:
  * das 95%-Bootstrap-KI prueft den MITTELWERT (empfindlich fuer Ausreisser),
  * Wilcoxon prueft die KONSISTENZ des Vorzeichens.
Bei knappen Faellen ist die Gewinnbilanz massgeblich — genau daran hat sich der
vermeintliche Uni3D-Vorsprung in Stage 1 als Rauschen erwiesen (p=0.54 bei
1009:1027), waehrend ein aehnlich kleiner Abstand anderswo real war.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(_THIS, "results_bop_stage3_v2")

DEFAULT_PAIRS = [
    ("Gallery: partial vs full-mesh (cross)", "3a_cross_v2", "3a_cross_fullmesh_v2"),
    ("Gallery: partial vs full-mesh (pc)", "3a_pc_v2", "3a_pc_fullmesh_v2"),
    ("Query: cross vs pc (partial)", "3a_cross_v2", "3a_pc_v2"),
    ("Shape-Kanal: OSCAR+ vs OSCAR-Kaskade", "3a_cross", "3a_oscar"),
    ("Geometrie: ohne vs Distanz (cross)", "3a_cross", "3a_cross_geo_distance"),
    ("Geometrie: ohne vs Fitness (cross)", "3a_cross", "3a_cross_geo_fitness"),
]


def load(arm):
    """{instanzschluessel: target_rank} fuer einen 3a-Arm."""
    out = {}
    for f in glob.glob(os.path.join(B, arm, "*_stage3a", "records.json")):
        for r in json.load(open(f)):
            rk = r.get("target_rank")
            if rk is None:
                continue
            key = (r.get("dataset"), r.get("scene_id"), r.get("im_id"),
                   r.get("obj_id"), r.get("gt_idx"))
            out[key] = rk
    return out


def score(rank, metric):
    return (1.0 if rank == 1 else 0.0) if metric == "hit1" else 1.0 / rank


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(lo, hi, p, wa, wb):
    if lo > 0 or hi < 0:
        return "REAL"
    if p is not None and p < 0.05:
        return "KONSISTENT (klein, systematisch; Mittelwert verrauscht)"
    if abs(wa - wb) <= 0.02 * max(wa + wb, 1):
        return "unentschieden"
    return "nicht belegt"


def compare(a, b, metric, per_dataset=False):
    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return None
    groups = {"ALLE": common}
    if per_dataset:
        for k in common:
            groups.setdefault(k[0], []).append(k)
    out = []
    for g, keys in groups.items():
        d = np.array([score(ra[k], metric) - score(rb[k], metric) for k in keys])
        lo, hi = boot_ci(d)
        try:
            from scipy.stats import wilcoxon
            nz = d[d != 0]
            p = float(wilcoxon(nz).pvalue) if len(nz) else None
        except Exception:
            p = None
        wa, wb = int((d > 0).sum()), int((d < 0).sum())
        out.append((g, len(keys), float(d.mean()), float(statistics.median(d)),
                    lo, hi, p, wa, wb, verdict(lo, hi, p, wa, wb)))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="*", help="genau zwei Arme, sonst Standardpaare")
    ap.add_argument("--metric", default="hit1", choices=["hit1", "mrr"])
    ap.add_argument("--per-dataset", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()

    pairs = ([("benutzerdefiniert", a.arms[0], a.arms[1])] if len(a.arms) == 2
             else DEFAULT_PAIRS)
    print(f"[sig3] metric={a.metric} · KI = Mittelwerttest · "
          f"Wilcoxon = Konsistenztest · positiv = erster Arm besser\n")
    rows = []
    for label, x, y in pairs:
        res = compare(x, y, a.metric, a.per_dataset)
        if res is None:
            print(f"[skip] {label}: keine gemeinsamen Instanzen ({x} / {y})")
            continue
        print(f"=== {label}\n    {x}  minus  {y}")
        for g, n, mean, med, lo, hi, p, wa, wb, v in res:
            ps = f"p={p:.4g}" if p is not None else "p=—"
            print(f"    {g:<8} n={n:<6} Δ={mean:+.4f} med={med:+.4f} "
                  f"KI[{lo:+.4f},{hi:+.4f}] {ps:<12} {wa}:{wb} -> {v}")
            rows.append((label, x, y, a.metric, g, n, round(mean, 5),
                         round(lo, 5), round(hi, 5), p, wa, wb, v))
        print()
    if a.csv and rows:
        import csv
        p = a.csv if os.path.isabs(a.csv) else os.path.join(
            os.path.dirname(_THIS), a.csv)
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["vergleich", "arm_a", "arm_b", "metrik", "gruppe", "n",
                        "delta", "ci_lo", "ci_hi", "wilcoxon_p",
                        "wins_a", "wins_b", "urteil"])
            w.writerows(rows)
        print(f"CSV: {p}")


if __name__ == "__main__":
    main()
