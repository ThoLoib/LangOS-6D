#!/usr/bin/env python3
"""
run_provenance.py — Arm/Lauf → Datum → gespeicherte Konfiguration.

    python3 tools/run_provenance.py             # alle Stages
    python3 tools/run_provenance.py --stage 1   # nur eine

Liest, was die Treiber tatsaechlich mitgeschrieben haben. Die **Umgebungs-
variablen** stehen NICHT in den Ergebnisdateien — sie sind der Teil von
`docs/RUN_PROVENANCE.md`, der von Hand gepflegt wird. Diese Trennung ist der
Grund, warum ein still auf Full-Mesh zurueckgefallener Lauf am 2026-09-06 erst
auffiel, als zwei Arme bitgleiche Zahlen lieferten.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OR = os.path.join(_ROOT, "object_retrieval")
S1 = os.path.join(OR, "results_shrec18_v2_stage1_42v_k5")
S3 = os.path.join(OR, "results_bop_stage3_v2")
S4 = os.path.join(_ROOT, "results_stage4")


# Skript- und Umgebungszuordnung. Steht HIER und nicht in einer Markdown-Datei,
# damit Konfiguration und Ergebnis aus EINER Quelle in EINE Tabelle laufen.
# Zuordnung ueber das Datum des Laufs (Ende-Zeitpunkt der jeweiligen Kette).
S1_RUNS = [
    ("2026-08-26 13:00", "2026-08-27 07:00", "run_stage1_full.sh",
     "FORCE_PARTIAL=✓ · DINO=mean · GEO_BACKEND=dgedi · DGEDI=shrec"),
    ("2026-08-26 14:50", "2026-08-26 15:10", "run_a7.sh",
     "FORCE_PARTIAL=✓ (colored) · DINO=mean"),
    ("2026-09-03 00:00", "2026-09-03 23:59", "run_stage1_fullmesh_color.sh",
     "FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean"),
    ("2026-09-04 03:00", "2026-09-04 05:00", "run_stage1_cross_fullmesh.sh",
     "FORCE_PARTIAL=— (Full-Mesh gewollt) · DINO=mean"),
    ("2026-09-04 14:00", "2026-09-04 15:00", "run_stage1_geo_on_best.sh",
     "GEO_BACKEND=dgedi · DGEDI=shrec · --geom-k 50 · DINO=mean"),
    ("2026-09-06 18:00", "2026-09-06 23:59", "run_stage1_cross_fullmesh.sh",
     "FORCE_PARTIAL=✓ (colored) · DINO=mean"),
]


def run_of(ts, table):
    for a, b, script, env in table:
        if a <= ts <= b:
            return script, env
    return "—", "—"


def when(p):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))


def stage1():
    print("\n=== STAGE 1 — SHREC'18 " + "=" * 54)
    print(f"  {'Arm':<36}{'Datum':<18}{'Shape-Pass':<24}{'Gewichte':<18}Geo")
    rows = []
    for d in sorted(os.listdir(S1)):
        f = os.path.join(S1, d, "metrics_summary.json")
        if not os.path.isfile(f):
            continue
        m = json.load(open(f))
        c = m.get("config") or {}
        ch = (c.get("channels") or {}).get("shape")
        rows.append((when(f), d, (ch[0] if ch else "—"),
                     str(c.get("weights")), c.get("geometry") or "—"))
    for t, d, sp, w, g in sorted(rows):
        print(f"  {d:<36}{t:<18}{sp:<24}{w:<18}{g}")
    # Verdaechtig: identische Zahlen bei verschiedenen Shape-Paessen
    # Bitgleiche Werte sind bei ALIASEN normal (A2_view_only_V42 ist E1_view_only).
    # Verdaechtig sind sie nur, wenn die Arme VERSCHIEDENE Shape-Paesse fahren —
    # dann hat einer der beiden nicht das gerechnet, was sein Name sagt.
    vals = {}
    for d in sorted(os.listdir(S1)):
        f = os.path.join(S1, d, "metrics_summary.json")
        if not os.path.isfile(f):
            continue
        m = json.load(open(f))
        ch = ((m.get("config") or {}).get("channels") or {}).get("shape")
        vals.setdefault(round(m["metrics"]["nDCG"], 10), []).append(
            (d, ch[0] if ch else None))
    # Dokumentierte Aliase: derselbe Pass unter zwei Namen. `ulip_pc_rgb_v42`
    # IST `ulip_pc_rgb` mit allen 42 Views — gleiche Zahlen sind dort korrekt.
    ALIAS = {"ulip_pc_rgb": "ulip_pc_rgb_v42", "ulip_pc_rgb_v42": "ulip_pc_rgb"}
    def same(a, b): return a == b or ALIAS.get(a) == b
    flagged = []
    for k, v in vals.items():
        passes = [p for _, p in v]
        if len(v) > 1 and not all(same(passes[0], p) for p in passes[1:]):
            flagged.append((k, v))
    if flagged:
        print("\n  ⚠ GLEICHE ZAHL BEI VERSCHIEDENEM SHAPE-PASS — einer der Arme hat")
        print("    nicht gerechnet, was sein Name sagt (meist stiller Full-Mesh-Fallback,")
        print("    siehe docs/RUN_PROVENANCE.md §0):")
        for k, v in flagged:
            print(f"      {k:.6f}")
            for d, sp in v:
                print(f"         {d:<40}{sp}")
    else:
        print("\n  ✓ keine verdaechtigen Doppelwerte (gleiche Zahl nur bei gleichem Pass)")


def stage3():
    print("\n=== STAGE 3 — BOP " + "=" * 60)
    print(f"  {'Lauf':<28}{'Datum':<18}{'Modus':<8}{'arm_ranks?':<12}Gallery")
    for d in sorted(os.listdir(S3)):
        p = os.path.join(S3, d)
        if not os.path.isdir(p) or d == "gt":
            continue
        f = next((x for x in sorted(glob.glob(os.path.join(p, "combined_stage3*.json")))), None)
        if not f:
            continue
        j = json.load(open(f))
        mode = os.path.basename(f).replace("combined_stage", "").replace(".json", "")
        has = "—"
        rf = next(iter(glob.glob(os.path.join(p, "*_stage3a", "records.json"))), None)
        if rf:
            r = json.load(open(rf))
            has = "ja" if r and "arm_ranks" in r[0] else "nein"
        print(f"  {d:<28}{when(f):<18}{mode:<8}{has:<12}{j.get('gallery_size','—')}")


def stage4():
    print("\n=== STAGE 4 — Latenz " + "=" * 57)
    for f in sorted(glob.glob(os.path.join(S4, "*.json"))):
        j = json.load(open(f))
        prov = j.get("provenance") or {}
        extra = []
        for k in ("views", "view_counts", "stages", "n_warmup", "gallery_size",
                  "geometry", "geo_k", "pose", "proxy_only", "dataset"):
            if k in j:
                extra.append(f"{k}={j[k]}")
        print(f"  {os.path.basename(f):<32}{when(f):<18}{prov.get('gpu','?')}")
        print(f"      {' · '.join(extra)}")


def markdown():
    """Eine Tabelle je Stage: Konfiguration UND Ergebnis nebeneinander."""
    out = ["# Konfiguration → Ergebnis (generiert)\n",
           "> Erzeugt von `tools/run_provenance.py --markdown`. Nicht von Hand aendern.",
           "> Die Skript- und Variablenspalten stammen aus `S1_RUNS` im Werkzeug,",
           "> alles andere aus den Ergebnisdateien.\n"]

    out.append("\n## Stage 1 — SHREC'18\n")
    out.append("| Arm | Shape-Pass | Gewichte | Geo | Skript | Variablen | nDCG | NN_sub |")
    out.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for d in sorted(os.listdir(S1)):
        f = os.path.join(S1, d, "metrics_summary.json")
        if not os.path.isfile(f):
            continue
        m = json.load(open(f)); c = m.get("config") or {}
        ch = (c.get("channels") or {}).get("shape")
        sc, env = run_of(when(f), S1_RUNS)
        rows.append((when(f), d, ch[0] if ch else "—",
                     "/".join(str(x) for x in (c.get("weights") or [])),
                     c.get("geometry") or "—", sc, env,
                     m["metrics"]["nDCG"], (m.get("metrics_depth") or {}).get("NN_sub")))
    for t, d, sp, w, g, sc, env, nd, nn in sorted(rows, key=lambda r: r[1]):
        out.append(f"| `{d}` | `{sp}` | {w} | {g} | `{sc}` | {env} | "
                   f"{nd:.4f} | {nn:.4f} |" if nn is not None else
                   f"| `{d}` | `{sp}` | {w} | {g} | `{sc}` | {env} | {nd:.4f} | — |")

    out.append("\n## Stage 3 — BOP\n")
    out.append("| Lauf | Modus | Gallery | arm_ranks | R@1 / D_sym | MRR |")
    out.append("|---|---|---|---|---|---|")
    for d in sorted(os.listdir(S3)):
        pth = os.path.join(S3, d)
        if not os.path.isdir(pth) or d == "gt":
            continue
        f = next(iter(sorted(glob.glob(os.path.join(pth, "combined_stage3*.json")))), None)
        if not f:
            continue
        j = json.load(open(f)); o = j.get("overall") or j
        mode = os.path.basename(f)[len("combined_stage"):-len(".json")]
        rf = next(iter(glob.glob(os.path.join(pth, "*_stage3a", "records.json"))), None)
        ar = "—"
        if rf:
            r = json.load(open(rf)); ar = "ja" if r and "arm_ranks" in r[0] else "nein"
        if mode == "3a":
            main_v, second = f"{o.get('recall@1',0):.4f}", f"{o.get('mrr',0):.4f}"
        else:
            sy = (o.get("dsym") or j.get("dsym") or {})
            main_v, second = f"{sy.get('d_sym_median',0):.2f} mm", "—"
        out.append(f"| `{d}` | {mode} | {j.get('gallery_size','—')} | {ar} | {main_v} | {second} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=[1, 3, 4])
    ap.add_argument("--markdown", metavar="DATEI",
                    help="Konfiguration+Ergebnis als Markdown-Tabelle schreiben.")
    a = ap.parse_args()
    if a.markdown:
        t = markdown()
        p = a.markdown if os.path.isabs(a.markdown) else os.path.join(_ROOT, a.markdown)
        open(p, "w").write(t)
        print(f"geschrieben: {p} ({len(t.splitlines())} Zeilen)")
        return
    if a.stage in (None, 1):
        stage1()
    if a.stage in (None, 3):
        stage3()
    if a.stage in (None, 4):
        stage4()
    print("\n  Umgebungsvariablen und Skriptzuordnung: docs/RUN_PROVENANCE.md "
          "(von Hand gepflegt — die Treiber schreiben sie nicht mit).")


if __name__ == "__main__":
    main()
