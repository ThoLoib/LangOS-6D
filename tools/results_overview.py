#!/usr/bin/env python3
"""
results_overview.py — erzeugt docs/RESULTS_OVERVIEW.md aus den ECHTEN Ergebnisordnern.

    python3 tools/results_overview.py                 # nach stdout
    python3 tools/results_overview.py -o docs/RESULTS_OVERVIEW.md

Warum generiert statt getippt: eine handgepflegte Ergebnistabelle driftet. Am
2026-09-03 fiel auf, dass Stage 1 die Zelle cross x full-mesh gar nicht hat —
ausgerechnet die, die auf BOP gewinnt — und dass ein Stage-1-Lauf unbemerkt im
falschen Verzeichnis gelandet war. Beides waere in einer generierten Uebersicht
sofort sichtbar gewesen.

Das Skript liest nur, es rechnet nichts nach und aendert nichts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OR = os.path.join(_ROOT, "object_retrieval")

# Stage-1-Ergebnisse koennen an zwei Orten liegen: --results-root ist relativ,
# und je nach Arbeitsverzeichnis des Containers landet ein Lauf unter /app oder
# unter /app/object_retrieval. Beide Orte werden gelesen und die Herkunft in der
# Tabelle vermerkt — sonst faellt ein verirrter Lauf nicht auf.
S1_ROOTS = [
    ("object_retrieval/", os.path.join(OR, "results_shrec18_v2_stage1_42v_k5")),
    ("repo-root/", os.path.join(_ROOT, "results_shrec18_v2_stage1_42v_k5")),
]
S3_DIR = os.path.join(OR, "results_bop_stage3_v2")


def _load(path) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


# --------------------------------------------------------------------------
def stage1_rows():
    """(arm, nDCG, NN_sub, Herkunft) fuer jeden Stage-1-Arm."""
    rows, seen = [], {}
    for tag, root in S1_ROOTS:
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            f = os.path.join(root, d, "metrics_summary.json")
            if not os.path.isfile(f):
                continue
            m = _load(f) or {}
            rows.append((d, (m.get("metrics") or {}).get("nDCG"),
                         (m.get("metrics_depth") or {}).get("NN_sub"), tag))
            seen.setdefault(d, []).append(tag)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    return sorted(rows), dupes


def stage2_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(OR, "results_mi3dor*"))):
        for f in sorted(glob.glob(os.path.join(d, "*", "metrics_summary_topk_*.json"))):
            m = _load(f) or {}
            mode = os.path.basename(os.path.dirname(f))
            for arm, v in (m.get("variants") or {}).items():
                rows.append((os.path.basename(d), mode, arm,
                             v.get("NN_accuracy"), v.get("FT_mean"),
                             v.get("mAP_mean") or v.get("mAP")))
    return rows


def stage3_rows():
    """3a/3b/3c je Arm, inklusive isoliertem Shape-Kanal wo vorhanden."""
    import statistics
    rows = []
    if not os.path.isdir(S3_DIR):
        return rows
    for d in sorted(os.listdir(S3_DIR)):
        p = os.path.join(S3_DIR, d)
        if not os.path.isdir(p) or d == "gt":
            continue
        entry = {"arm": d, "mode": None}
        a = _load(os.path.join(p, "combined_stage3a.json"))
        if a:
            o = a.get("overall") or a
            entry.update(mode="3a", r1=o.get("recall@1"), mrr=o.get("mrr"),
                         per={k: (v or {}).get("recall@1")
                              for k, v in (a.get("per_dataset") or {}).items()})
            # isolierter Shape-Kanal aus arm_ranks (nur in Laeufen ab 2026-09-01)
            ranks = []
            for rf in glob.glob(os.path.join(p, "*_stage3a", "records.json")):
                for r in (_load(rf) or []):
                    v = (r.get("arm_ranks") or {}).get("ulip_only_full")
                    if v is not None:
                        ranks.append(v)
            if ranks:
                entry["iso"] = statistics.fmean(1.0 if r == 1 else 0.0 for r in ranks)
        for mode in ("b", "c"):
            b = _load(os.path.join(p, f"combined_stage3{mode}.json"))
            if b:
                s = (b.get("overall") or b).get("dsym") or b.get("dsym") or {}
                de = (b.get("overall") or b).get("delta") or b.get("delta") or {}
                entry.update(mode=f"3{mode}", dsym=s.get("d_sym_median"),
                             delta=de.get("delta_median"), cov=s.get("coverage"))
        if entry.get("mode"):
            rows.append(entry)
    return rows


def stage4_rows():
    out = {}
    for name in ("query_latency_ycbv", "onboarding", "onboarding_render"):
        d = _load(os.path.join(_ROOT, "results_stage4", f"{name}.json"))
        if d:
            out[name] = d
    return out


# --------------------------------------------------------------------------
def fmt(v, n=4):
    return f"{v:.{n}f}" if isinstance(v, (int, float)) else "—"


def render() -> str:
    L = []
    add = L.append
    add("# Ergebnisübersicht — alle Stages\n")
    add("> **Generiert** von `tools/results_overview.py`. Nicht von Hand ändern —\n"
        "> nach jedem Lauf neu erzeugen. Handgepflegte Tabellen driften.\n")
    import datetime
    add(f"Stand: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")

    # ---- Stage 1
    rows, dupes = stage1_rows()
    add(f"\n## Stage 1 — SHREC'18 ({len(rows)} Arme)\n")
    if dupes:
        add("> ⚠ **Arme an zwei Orten:** " + ", ".join(sorted(dupes)) +
            ". `--results-root` ist relativ; je nach Arbeitsverzeichnis landet ein\n"
            "> Lauf unter `object_retrieval/` oder im Repo-Wurzelverzeichnis.\n")
    add("| Arm | nDCG | NN_sub | Ort |")
    add("|---|---|---|---|")
    for arm, nd, nn, tag in rows:
        add(f"| `{arm}` | {fmt(nd)} | {fmt(nn)} | {tag} |")

    # ---- Stage 2
    s2 = stage2_rows()
    add(f"\n## Stage 2 — MI3DOR ({len(s2)} Zeilen)\n")
    add("| Ordner | Modus | Arm | NN | FT | mAP |")
    add("|---|---|---|---|---|---|")
    for d, mode, arm, nn, ft, ap in sorted(s2):
        add(f"| `{d}` | {mode} | `{arm}` | {fmt(nn, 2)} | {fmt(ft, 3)} | {fmt(ap, 3)} |")

    # ---- Stage 3
    s3 = stage3_rows()
    add(f"\n## Stage 3 — BOP ({len(s3)} Arme)\n")
    add("### 3a — Retrieval\n")
    add("| Arm | R@1 | MRR | Shape allein | YCB-V | T-LESS | LM-O |")
    add("|---|---|---|---|---|---|---|")
    for e in s3:
        if e["mode"] != "3a":
            continue
        per = e.get("per") or {}
        add(f"| `{e['arm']}` | {fmt(e.get('r1'))} | {fmt(e.get('mrr'))} | "
            f"{fmt(e.get('iso'))} | {fmt(per.get('ycbv'), 3)} | "
            f"{fmt(per.get('tless'), 3)} | {fmt(per.get('lmo'), 3)} |")
    add("\n### 3b / 3c — Pose\n")
    add("| Arm | Modus | D_sym Median (mm) | Δ Median (mm) | Deckung |")
    add("|---|---|---|---|---|")
    for e in s3:
        if e["mode"] not in ("3b", "3c"):
            continue
        add(f"| `{e['arm']}` | {e['mode']} | {fmt(e.get('dsym'), 2)} | "
            f"{fmt(e.get('delta'), 2)} | {fmt(e.get('cov'), 3)} |")

    # ---- Stage 4
    s4 = stage4_rows()
    add("\n## Stage 4 — Latenz\n")
    if not s4:
        add("_keine Messungen gefunden_")
    for name, d in s4.items():
        add(f"\n**`{name}`** — {d.get('provenance', {}).get('gpu', '?')}\n")
        key = "by_views"
        for v, blk in (d.get(key) or {}).items():
            steps = blk.get("per_step") or {}
            tot = (blk.get("per_query_total_s") or blk.get("per_object_total_s") or {})
            parts = ", ".join(f"{k} {s['median']*1000:.0f} ms"
                              for k, s in steps.items() if s.get("n"))
            add(f"- {v} Views — gesamt {fmt(tot.get('median'), 3)} s · {parts}")

    # ---- Lücken
    add("\n## Bekannte Lücken\n")
    s1_arms = {a for a, _, _, _ in rows}
    if "E7_ulip2_cross_fullmesh_shape_only" not in s1_arms:
        add("- **Stage 1 hat keine Zelle cross × full-mesh.** Vorhanden sind "
            "pc×partial (`E1_shape_only`), pc×full-mesh (`E2b_fullmesh_shape_only`) "
            "und cross×partial (`E7_ulip2_cross_shape_only`). Auf BOP ist genau die "
            "fehlende Kombination der beste Arm (R@1 0.5151).")
    have3 = {e["arm"] for e in s3}
    for want, why in [("3b_cross_fullmesh", "Pose für den neuen besten Retrieval-Arm"),
                      ("3c_cross_fullmesh", "Zerlegung dazu")]:
        if want not in have3:
            add(f"- **`{want}` fehlt** — {why}.")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", help="Zieldatei (Default: stdout)")
    a = ap.parse_args()
    txt = render()
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(_ROOT, a.out)
        with open(p, "w") as fh:
            fh.write(txt)
        print(f"geschrieben: {p}  ({len(txt.splitlines())} Zeilen)")
    else:
        print(txt)


if __name__ == "__main__":
    main()
