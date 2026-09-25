#!/usr/bin/env python3
"""Offizielle MI3DOR-Masse fuer alle sieben Arme aus den eval_trace-Positionen.

Seit Commit ea84ffb8 (2026-08-07) schreibt eval_common._make_per_query_record
in jeden Query-Record ein Feld ``eval_trace`` mit ``num_rel_true`` und je Arm
``len`` + ``rel_positions`` (1-basierte Positionen ALLER relevanten Modelle
ueber die volle Rangliste). Daraus laesst sich der binaere Relevanzvektor
verlustfrei rekonstruieren — die offiziellen Masse (cross_performance.m,
Port: mi3dor_official_metrics.py) sind damit OHNE Capture- oder GPU-Lauf
fuer alle Arme nachrechenbar.

Waechter:
  W1  NN/FT je Arm exakt (1e-9) gegen die metrics_summary_topk_15.json des
      SELBEN Laufs — sonst Abbruch ohne Ablage.
  W2  Die vier partial-Zeilen von metrics_official.csv muessen auf vier
      Nachkommastellen getroffen werden (inkl. AUC) — sonst Abbruch.
  W3  Die zwei full-mesh-Zeilen von metrics_official.csv (Kanal-Cache-Nachbau,
      c94226f6) sollen auf vier Stellen getroffen werden; Abweichungen werden
      nur GEMELDET (metrics_official.csv wird nie angefasst).

Ablage: final_results/stage2/metrics_official_traces.csv (21 Zeilen, sechs
Nachkommastellen) + metrics_official_traces_manifest.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi3dor_official_metrics import official_metrics  # noqa: E402

_OR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_OR)

ARMS = [
    "clip_only", "dino_only_full", "ulip_only_full", "clip_dino_ulip_full",
    "oscar_maxview", "oscar_softmax", "clip_pruned_dino_ulip",
]

RUNS = [
    # (run_id, gallery, views, results_dir)
    ("oscarplus_v2_tau037_dinomean_partialforce", "partial", 42,
     "results_mi3dor_oscarplus_v2_tau037_dinomean_partialforce/partial"),
    ("oscarplus_v2_tau037_dinomean_ulipfix", "fullmesh", 42,
     "results_mi3dor_oscarplus_v2_tau037_dinomean_ulipfix/fullmesh"),
    ("oscar_legacy_v8", "fullmesh", 8,
     "results_mi3dor_oscar_legacy_v8/fullmesh"),
]

GALLERY_LEN = 3848
N_QUERIES = 10500
C_MIN, C_MAX = 31, 250
T_MAX_EXPECTED = 250
METRIC_COLS = ["NN", "FT", "ST", "F", "DCG", "ANMRR", "AUC"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_integrity(records) -> dict:
    """Strukturpruefung VOR dem Rechnen; wirft bei jedem Verstoss."""
    assert len(records) == N_QUERIES, f"{len(records)} != {N_QUERIES} Records"
    c_vals = []
    for i, rec in enumerate(records):
        et = rec.get("eval_trace")
        assert et is not None, f"Record {i}: eval_trace fehlt"
        c = int(et["num_rel_true"])
        assert C_MIN <= c <= C_MAX, f"Record {i}: C={c} nicht in {C_MIN}..{C_MAX}"
        c_vals.append(c)
        arms = et["arms"]
        missing = [a for a in ARMS if a not in arms]
        assert not missing, f"Record {i}: Arme fehlen: {missing}"
        for a in ARMS:
            t = arms[a]
            assert int(t["len"]) == GALLERY_LEN, \
                f"Record {i}/{a}: len={t['len']} != {GALLERY_LEN}"
            pos = np.asarray(t["rel_positions"], dtype=np.int64)
            assert len(pos) == c, \
                f"Record {i}/{a}: {len(pos)} Positionen != num_rel_true {c}"
            assert pos[0] >= 1 and pos[-1] <= GALLERY_LEN and \
                np.all(np.diff(pos) > 0), \
                f"Record {i}/{a}: Positionen nicht streng steigend in 1..{GALLERY_LEN}"
    t_max = max(c_vals)
    assert t_max == T_MAX_EXPECTED, f"T_max={t_max} != {T_MAX_EXPECTED}"
    return {"records": len(records), "C_min": min(c_vals), "C_max": t_max,
            "T_max": t_max, "status": "OK"}


def arm_metrics(records, arm: str, t_max: int):
    """Binaere Relevanzvektoren des Arms rekonstruieren und official_metrics rechnen."""
    rres, cs, nn_count = [], [], 0
    for rec in records:
        et = rec["eval_trace"]
        t = et["arms"][arm]
        v = np.zeros(int(t["len"]), dtype=np.int64)
        v[np.asarray(t["rel_positions"], dtype=np.int64) - 1] = 1
        nn_count += int(v[0])
        rres.append(v)
        cs.append(int(et["num_rel_true"]))
    m = official_metrics(rres, cs, t_max)
    return m, nn_count


def load_reference_csv(path: str):
    ref = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ref[(row["run"], row["gallery"], row["arm"])] = {
                k: float(row[k]) for k in METRIC_COLS}
    return ref


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--script-commit", default="",
                    help="Git-Commit dieses Skripts (fuer das Manifest)")
    ap.add_argument("--port-commit", default="c94226f6",
                    help="Git-Commit des Metrik-Ports (fuer das Manifest)")
    args = ap.parse_args()

    out_dir = os.path.join(_ROOT, "final_results", "stage2")
    ref_csv = os.path.join(out_dir, "metrics_official.csv")
    for run_id, gallery, views, rel_dir in RUNS:
        d = os.path.join(_OR, rel_dir)
        for fn in ("results_topk_15.json", "metrics_summary_topk_15.json"):
            p = os.path.join(d, fn)
            assert os.path.isfile(p), f"Eingabe fehlt: {p}"
    assert os.path.isfile(ref_csv), f"Referenz fehlt: {ref_csv}"
    ref = load_reference_csv(ref_csv)

    manifest = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "evaluator_quelle": ("github.com/tianbao-li/MI3DOR "
                             "Retrieval/cross_performance.m, Commit "
                             "4325c24c91553283ec98d948513063f659771767"),
        "port": {"datei": "object_retrieval/mi3dor_official_metrics.py",
                 "commit": args.port_commit, "sha256": sha256_of(
                     os.path.join(_OR, "mi3dor_official_metrics.py"))},
        "skript": {"datei": "object_retrieval/mi3dor_official_from_traces.py",
                   "commit": args.script_commit or "(nicht uebergeben)",
                   "sha256": sha256_of(os.path.abspath(__file__))},
        "traces_seit": "eval_common Commit ea84ffb8 (2026-08-07): eval_trace "
                       "mit num_rel_true + je Arm len/rel_positions",
        "eingaben": {}, "w1": {}, "w2": {}, "w3": {},
    }

    rows = []
    w1_fail, w2_fail, w3_diff = [], [], []
    for run_id, gallery, views, rel_dir in RUNS:
        res_p = os.path.join(_OR, rel_dir, "results_topk_15.json")
        sum_p = os.path.join(_OR, rel_dir, "metrics_summary_topk_15.json")
        st = os.stat(res_p)
        print(f"[{run_id}] lade {res_p} ({st.st_size} B) ...", flush=True)
        records = json.load(open(res_p))
        integ = check_integrity(records)
        print(f"[{run_id}] Integritaet: {integ}", flush=True)
        manifest["eingaben"][run_id] = {
            "pfad": os.path.relpath(res_p, _ROOT), "groesse": st.st_size,
            "datum": datetime.datetime.fromtimestamp(st.st_mtime)
            .isoformat(timespec="seconds"),
            "sha256": sha256_of(res_p), "gallery": gallery, "views": views,
            "integritaet": integ,
        }
        variants = json.load(open(sum_p))["variants"]
        manifest["w1"][run_id] = {}
        for arm in ARMS:
            m, nn_count = arm_metrics(records, arm, integ["T_max"])
            nn_prod = variants[arm]["NN_accuracy"] / 100.0
            ft_prod = variants[arm]["FT_mean"]
            d_nn, d_ft = abs(m["NN"] - nn_prod), abs(m["FT"] - ft_prod)
            ok = d_nn <= 1e-9 and d_ft <= 1e-9
            manifest["w1"][run_id][arm] = {
                "NN_prod": nn_prod, "FT_prod": ft_prod,
                "NN_traces": m["NN"], "FT_traces": m["FT"],
                "status": "exakt (1e-9)" if ok else
                f"ABWEICHUNG dNN={d_nn:.3e} dFT={d_ft:.3e}"}
            if not ok:
                w1_fail.append((run_id, arm, d_nn, d_ft))
            print(f"[{run_id}] {arm:22s} NN {m['NN']:.6f} FT {m['FT']:.6f} "
                  f"{'OK' if ok else 'W1-FEHLER'}", flush=True)
            rows.append([run_id, gallery, views, arm, N_QUERIES, nn_count] +
                        [f"{m[k]:.6f}" for k in METRIC_COLS])
            key = (run_id, gallery, arm)
            if key in ref:
                diffs = {k: round(m[k], 4) - ref[key][k] for k in METRIC_COLS}
                hit = all(abs(v) < 1e-9 for v in diffs.values())
                tgt = manifest["w2"] if gallery == "partial" else manifest["w3"]
                tgt[f"{run_id}/{arm}"] = {
                    "status": "getroffen (4 Stellen)" if hit else
                    {k: round(v, 6) for k, v in diffs.items() if abs(v) >= 1e-9}}
                if not hit:
                    (w2_fail if gallery == "partial" else w3_diff).append(
                        (run_id, arm, diffs))
        del records

    if w1_fail or w2_fail:
        print("\nABBRUCH — Waechter verletzt, keine Ablage:")
        for f in w1_fail:
            print("  W1:", f)
        for f in w2_fail:
            print("  W2:", f)
        sys.exit(1)

    out_csv = os.path.join(out_dir, "metrics_official_traces.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "gallery", "views", "arm", "N", "NN_count"] +
                   METRIC_COLS)
        w.writerows(rows)
    man_p = os.path.join(out_dir, "metrics_official_traces_manifest.json")
    json.dump(manifest, open(man_p, "w"), indent=1)
    print(f"\n{len(rows)} Zeilen -> {out_csv}\nManifest -> {man_p}")
    if w3_diff:
        print("\nW3-ABWEICHUNGEN (metrics_official.csv NICHT angefasst; "
              "Entscheidung beim Nutzer):")
        for run_id, arm, diffs in w3_diff:
            print(f"  {run_id}/{arm}: " + ", ".join(
                f"{k}{v:+.6f}" for k, v in diffs.items() if abs(v) >= 1e-9))
    else:
        print("W3: beide full-mesh-Zeilen auf vier Stellen getroffen.")


if __name__ == "__main__":
    main()
