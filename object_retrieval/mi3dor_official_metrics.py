#!/usr/bin/env python3
"""Exakte Python-Portierung des offiziellen MI3DOR-Evaluators.

Quelle: github.com/tianbao-li/MI3DOR, Retrieval/cross_performance.m,
Commit 4325c24c91553283ec98d948513063f659771767 (2023-10-24),
abgerufen 2026-09-25. KEINE "Verbesserungen" — jede Formel folgt dem
MATLAB-Original zeilengetreu (inkl. der Eigenheiten):

  * NN   = Anteil Rang-1-Treffer.
  * FT   = Treffer in Top C / C (C = Galerie-Klassengroesse der Query),
           makro-gemittelt.
  * ST   = Treffer in Top 2C-1 / C (NICHT 2C), makro-gemittelt.
  * F    = "20-Measure", MIKRO: s = Treffersumme in Top 20 ueber alle Queries;
           p = s/(N*20); r = s/(Summe aller Relevanten); F = 2/(1/p + 1/r).
  * DCG  = je Query ueber Raenge 1..C; Rang 1 UND Rang 2 wiegen je 1
           (1/log2(k) ab k>=2 — der Off-by-one-Diskont des Originals);
           Ideal = alle C relevant; makro-gemittelt.
  * ANMRR: S = min(4C, 2*T_max); Schleife ueber die RANG-POSITIONEN k=1..C
           (nicht ueber die relevanten Items!): r_k = k bei Treffer, sonst
           S+1; NMRR = (mean(r) - C/2 - 0.5) / (S - C/2 + 0.5);
           makro-gemittelt. Weicht von MPEG-7 UND von Pullis Scorer ab.
  * AUC  = Mikro-PR-Kurve ueber ALLE Tiefen (p_i = s_i/(N*i),
           r_i = s_i/Gesamt-Relevante), Trapezregel — wie AUC_0 im Original.

Eingabe: ``rresult`` — je Query der binaere Relevanzvektor UEBER DIE VOLLE
GALERIE in Rangreihenfolge (bestes Modell zuerst; das Original sortiert eine
Distanzmatrix aufsteigend — wer Aehnlichkeiten hat, sortiert absteigend und
uebergibt die Treffervektoren) plus ``C`` je Query und ``T_max``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np


def official_metrics(rresult: Sequence[np.ndarray], C: Sequence[int],
                     T_max: int) -> Dict[str, float]:
    """Alle sieben Groessen aus cross_performance.m."""
    N = len(rresult)
    assert N == len(C) and N > 0
    total_rel = float(sum(int(r.sum()) for r in rresult))

    nn = ft = st = dcg = anmrr = 0.0
    s_top20 = 0.0
    for r, c in zip(rresult, C):
        r = np.asarray(r, dtype=np.int64)
        c = int(c)
        nn += float(r[0])
        ft += float(r[:c].sum()) / c
        st += float(r[:2 * c - 1].sum()) / c
        s_top20 += float(r[:20].sum())
        # DCG: Rang 1 Gewicht 1; k=2..C Gewicht 1/log2(k)
        num = float(r[0])
        den = 1.0
        for k in range(2, c + 1):
            w = 1.0 / math.log2(k)
            num += float(r[k - 1]) * w
            den += w
        dcg += num / den
        # ANMRR: Positions-Schleife k=1..C
        S = min(4 * c, 2 * int(T_max))
        rsum = 0.0
        for k in range(1, c + 1):
            rsum += k if r[k - 1] == 1 else S + 1
        anmrr += ((rsum / c) - c / 2.0 - 0.5) / (S - c / 2.0 + 0.5)

    # AUC: Mikro-PR ueber alle Tiefen (vektorisiert: Summe der Treffer je Rang)
    G = max(len(r) for r in rresult)
    hits_by_rank = np.zeros(G, dtype=np.float64)
    for r in rresult:
        hits_by_rank[:len(r)] += r
    s_i = np.cumsum(hits_by_rank)
    depths = np.arange(1, G + 1, dtype=np.float64)
    p_i = s_i / (N * depths)
    r_i = s_i / total_rel
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 vs. <2
    auc = float(_trapz(p_i, r_i))

    return {"NN": nn / N, "FT": ft / N, "ST": st / N,
            "F": 2.0 / (1.0 / (s_top20 / (N * 20.0))
                        + 1.0 / (s_top20 / total_rel)),
            "DCG": dcg / N, "ANMRR": anmrr / N, "AUC": auc}


def _selftest() -> None:
    """Handnachgerechnetes Beispiel: 2 Queries, Galerie 6, T_max = 3.

    Q1: C=2, Treffer auf Rang 1 und 3 -> r = [1,0,1,0,0,0]
    Q2: C=3, Treffer auf Rang 2, 4, 5 -> r = [0,1,0,1,1,0]
    Nachrechnung von Hand (MATLAB-Semantik):
      NN = (1+0)/2 = 0.5
      FT = (1/2 + 1/3)/2 = 0.416667
      ST = (Top3: 2/2 ; Top5: 3/3)/2 = 1.0
      DCG1: num = 1 + 0/log2(2) = 1; den = 1 + 1/log2(2) = 2 -> 0.5
      DCG2: num = 0 + 1/log2(2) + 0 = 1; den = 1 + 1 + 1/log2(3) = 2.63093
            -> 0.380093
      DCG = 0.440047
      ANMRR1: S=min(8,6)=6; r=[1, 7]; mean=4; (4-1-0.5)/(6-1+0.5)=0.454545
      ANMRR2: S=min(12,6)=6; r=[7,2,7]; mean=16/3; (16/3-1.5-0.5)/(6-1.5+0.5)=0.666667
      ANMRR = 0.560606
      F: s_top20 = 2+3 = 5; total_rel = 5; p=5/40=0.125; r=1; F=2/(8+1)=0.222222
    """
    r1 = np.array([1, 0, 1, 0, 0, 0])
    r2 = np.array([0, 1, 0, 1, 1, 0])
    m = official_metrics([r1, r2], [2, 3], T_max=3)
    soll = {"NN": 0.5, "FT": 0.4166667, "ST": 1.0, "DCG": 0.4400469,
            "ANMRR": 0.5606061, "F": 0.2222222}
    for k, v in soll.items():
        assert abs(m[k] - v) < 1e-6, (k, m[k], v)
    print("[selftest] alle Handwerte getroffen:",
          {k: round(v, 6) for k, v in m.items()})


if __name__ == "__main__":
    _selftest()
