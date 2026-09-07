#!/usr/bin/env python3
"""
occlusion_analysis.py — was macht Verdeckung mit Retrieval und Pose?

Frage
-----
Stage 1-3 vergleichen Entwurfsentscheidungen. Diese Auswertung fragt etwas
anderes: wie stark haengt das Ergebnis davon ab, wie viel vom Objekt ueberhaupt
zu sehen ist? BOP liefert die Antwort mit — es braucht keinen neuen Lauf.

Datenquelle
-----------
* ``eval/datasets/<ds>/test*/<scene>/scene_gt_info.json`` — BOPs eigene
  Annotation. ``visib_fract = px_count_visib / px_count_valid``, also der Anteil
  der projizierten Objektflaeche, der im Bild sichtbar ist.
* ``object_retrieval/results_bop_stage3_v2/<run>/<ds>_stage3{a,b}/records.json``
  — unsere Per-Instanz-Ergebnisse.

Verknuepft wird ueber ``(scene_id, im_id, gt_idx)``. ACHTUNG: die Felder sind in
3a Strings und in 3b Integers — beide Seiten werden mit ``int()`` gecastet.

Selbstpruefung
--------------
R@1 wird aus ``target_rank == 1`` SELBST gebildet und gegen den publizierten
Wert in ``combined_stage3a.json`` geprueft. Weicht er ab, bricht das Skript ab:
eine falsche Verknuepfung faellt sonst nicht auf.

Beispiele
---------
    # Standardauswertung (3a-Retrieval + 3b-Pose, gepoolt und je Datensatz)
    python3 tools/occlusion_analysis.py

    # nur das Retrieval, anderer Lauf
    python3 tools/occlusion_analysis.py --mode 3a --run 3a_cross_fullmesh_v2

    # eigene Klassengrenzen
    python3 tools/occlusion_analysis.py --bins 0.25,0.5,0.75

    # als CSV fuer die Weiterverarbeitung
    python3 tools/occlusion_analysis.py --csv docs/occlusion_by_visibility.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_ROOTS = {
    "ycbv":  "eval/datasets/ycbv/test",
    "tless": "eval/datasets/tless/test_primesense",
    "lmo":   "eval/datasets/lmo/test",
}
STAGE3 = "object_retrieval/results_bop_stage3_v2"


def visibility_map(ds: str) -> dict:
    """(scene, im, gt_idx) -> visib_fract, ueber alle Testszenen eines Datensatzes."""
    out = {}
    pattern = os.path.join(_ROOT, TEST_ROOTS[ds], "*", "scene_gt_info.json")
    for f in sorted(glob.glob(pattern)):
        scene = int(os.path.basename(os.path.dirname(f)))
        for im, entries in json.load(open(f)).items():
            for gt_idx, e in enumerate(entries):
                if "visib_fract" in e:
                    out[(scene, int(im), gt_idx)] = e["visib_fract"]
    if not out:
        sys.exit(f"keine scene_gt_info.json unter {pattern}")
    return out


def load_records(run: str, ds: str, mode: str) -> list:
    p = os.path.join(_ROOT, STAGE3, run, f"{ds}_stage3{mode[-1]}", "records.json")
    if not os.path.isfile(p):
        return []
    return json.load(open(p))


def value_of(rec: dict, mode: str):
    """Die auszuwertende Groesse je Instanz."""
    if mode == "3a":
        tr = rec.get("target_rank")
        if str(tr) in ("None", ""):
            return 0.0                      # Ziel gar nicht im Ranking
        return 1.0 if int(tr) == 1 else 0.0
    return float(rec["d_posed"])            # 3b: D_sym in mm


def collect(run: str, mode: str, datasets) -> dict:
    """dataset -> [(visib_fract, wert, d_sym_norm|None), ...]"""
    out = {}
    for ds in datasets:
        recs = load_records(run, ds, mode)
        if not recs:
            continue
        vm = visibility_map(ds)
        rows, missed = [], 0
        for r in recs:
            key = (int(r["scene_id"]), int(r["im_id"]), int(r["gt_idx"]))
            v = vm.get(key)
            if v is None:
                missed += 1
                continue
            norm = float(r["d_sym_norm"]) if "d_sym_norm" in r else None
            rows.append((v, value_of(r, mode), norm))
        if missed:
            print(f"  WARNUNG {ds}: {missed} von {len(recs)} Instanzen ohne "
                  f"Sichtbarkeitsannotation")
        out[ds] = rows
    return out


def selfcheck(run: str, data: dict, mode: str) -> None:
    """R@1 selbst gerechnet gegen die publizierte Zahl. Bricht bei Abweichung ab."""
    if mode != "3a":
        return
    combined = os.path.join(_ROOT, STAGE3, run, "combined_stage3a.json")
    if not os.path.isfile(combined):
        print("  (kein combined_stage3a.json — Selbstpruefung uebersprungen)")
        return
    published = json.load(open(combined)).get("recall@1")
    allrows = [x for rows in data.values() for x in rows]
    mine = sum(x[1] for x in allrows) / len(allrows)
    print(f"  Selbstpruefung R@1: eigene Rechnung {mine:.6f} gegen "
          f"publiziert {published:.6f}")
    if abs(mine - published) > 1e-4:
        sys.exit("  ABBRUCH: Abweichung zu gross — die Verknuepfung stimmt nicht.")
    print("  -> stimmt ueberein, die Verknuepfung ist korrekt.")


def pearson(rows) -> float:
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    n = len(rows)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def bin_label(lo, hi):
    return f"{lo*100:.0f}-{hi*100:.0f} %"


def summarise(rows, edges, mode):
    """Je Klasse: (Label, n, Wert). 3a mittelt (Trefferquote), 3b nimmt den Median."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r[1] for r in rows if lo <= r[0] < hi]
        if not sel:
            out.append((bin_label(lo, hi), 0, float("nan")))
            continue
        val = (sum(sel) / len(sel)) if mode == "3a" else statistics.median(sel)
        out.append((bin_label(lo, hi), len(sel), val))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["3a", "3b", "both"], default="both",
                    help="3a = Retrieval (R@1), 3b = Pose (D_sym). Default beide.")
    ap.add_argument("--run", default=None,
                    help="Ergebnisordner unter results_bop_stage3_v2 "
                         "(Default: 3a_cross bzw. 3b_cross je nach --mode).")
    ap.add_argument("--datasets", default="ycbv,tless,lmo")
    ap.add_argument("--bins", default="0.5,0.8,0.95",
                    help="Innere Klassengrenzen, kommasepariert (Default 0.5,0.8,0.95).")
    ap.add_argument("--csv", default=None, help="Ergebnis zusaetzlich als CSV.")
    args = ap.parse_args()

    edges = [0.0] + [float(x) for x in args.bins.split(",")] + [1.0001]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = ["3a", "3b"] if args.mode == "both" else [args.mode]
    csv_rows = []

    for mode in modes:
        run = args.run or f"{mode}_cross"
        title = ("Retrieval — findet es das exakte CAD? (R@1)" if mode == "3a"
                 else "Pose mit Ersatzmodell — D_sym Median in mm")
        print(f"\n=== {mode}  {title}")
        print(f"    Lauf: {run}")
        data = collect(run, mode, datasets)
        if not data:
            print("    keine Records gefunden — uebersprungen.")
            continue
        selfcheck(run, data, mode)

        allrows = [x for rows in data.values() for x in rows]
        print(f"\n  {'Sichtbarkeit':<16}{'n':>8}{'Anteil':>9}{'Wert':>10}")
        for label, n, val in summarise(allrows, edges, mode):
            share = 100 * n / len(allrows) if allrows else 0
            print(f"  {label:<16}{n:>8}{share:>8.1f} %{val:>10.3f}")
            csv_rows.append({"mode": mode, "run": run, "dataset": "ALLE",
                             "bin": label, "n": n, "value": round(val, 4)})
        print(f"  {'gesamt':<16}{len(allrows):>8}")
        print(f"  Pearson r (Sichtbarkeit vs Metrik): {pearson(allrows):+.3f}")

        # Kontrolle: haelt der Effekt INNERHALB jedes Datensatzes? Die Klassen sind
        # unterschiedlich zusammengesetzt (T-LESS stellt den Grossteil der stark
        # verdeckten Instanzen), deshalb ist diese Aufschluesselung Pflicht.
        print(f"\n  je Datensatz — haelt der Effekt einzeln?")
        head = "".join(f"{bin_label(lo, hi):>13}"
                       for lo, hi in zip(edges[:-1], edges[1:]))
        print(f"  {'':<8}{head}{'Spanne':>10}")
        for ds, rows in data.items():
            cells, vals = "", []
            for label, n, val in summarise(rows, edges, mode):
                cells += f"{val:>8.3f} ({n:>3})" if n else f"{'—':>13}"
                if n:
                    vals.append(val)
                    csv_rows.append({"mode": mode, "run": run, "dataset": ds,
                                     "bin": label, "n": n, "value": round(val, 4)})
            span = (max(vals) - min(vals)) if vals else float("nan")
            print(f"  {ds:<8}{cells}{span:>10.3f}")

    if args.csv and csv_rows:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\ngeschrieben: {args.csv} ({len(csv_rows)} Zeilen)")


if __name__ == "__main__":
    main()
