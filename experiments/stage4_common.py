#!/usr/bin/env python3
# =============================================================================
# stage4_common.py — Messwerkzeug fuer die beiden Stage-4-Latenzexperimente.
#
# Enthaelt nur, was BEIDE Skripte brauchen: eine GPU-korrekte Stoppuhr, die
# Statistik ueber viele Messungen und die Ausgabe. Keine Pipeline-Logik.
#
# Warum eine eigene Stoppuhr: CUDA-Kernel laufen asynchron. Ein blosses
# time.perf_counter() um einen Encoder-Aufruf misst die Zeit, bis der Kernel
# EINGEREIHT ist, nicht bis er fertig ist — bei kurzen Schritten kann das um
# Groessenordnungen danebenliegen. Jede Messung synchronisiert deshalb vorher
# und nachher, sofern CUDA ueberhaupt aktiv ist.
# =============================================================================
from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager
from typing import Dict, List, Optional


def _sync() -> None:
    """CUDA-Kernel abwarten, damit die gemessene Zeit die echte Rechenzeit ist."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class Timings:
    """Sammelt benannte Dauern. Ein Name kann mehrfach gemessen werden."""

    def __init__(self) -> None:
        self.runs: Dict[str, List[float]] = {}

    @contextmanager
    def measure(self, name: str):
        _sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            self.runs.setdefault(name, []).append(time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self.runs.setdefault(name, []).append(seconds)

    def total(self) -> float:
        return sum(sum(v) for v in self.runs.values())

    def as_dict(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self.runs.items()}


def summarize(samples: List[float]) -> Dict[str, float]:
    """Median + Streuung. Median und IQR statt Mittelwert und Stdabw., weil
    Latenzverteilungen rechtsschief sind: ein einzelner Ausreisser (Swap,
    Cache-Miss, Thermal-Throttling) verschiebt den Mittelwert, den Median
    nicht. p95 steht daneben, weil fuer ein interaktives System der schlechte
    Fall die relevante Groesse ist, nicht der typische."""
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    n = len(s)

    def q(p: float) -> float:
        if n == 1:
            return s[0]
        i = p * (n - 1)
        lo = int(i)
        hi = min(lo + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (i - lo)

    return {
        "n": n,
        "median": statistics.median(s),
        "mean": statistics.fmean(s),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "q25": q(0.25),
        "q75": q(0.75),
        "iqr": q(0.75) - q(0.25),
        "p95": q(0.95),
        "sum": sum(s),
    }


def aggregate(per_item: List[Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Pro Schritt ueber alle Objekte/Queries zusammenfassen.

    Mehrfachmessungen desselben Schritts innerhalb eines Items werden zuerst
    aufsummiert (42 View-Encodes = EIN Onboarding-Schritt), damit die Statistik
    ueber Items laeuft und nicht ueber Einzelaufrufe — sonst wuerde ein Objekt
    mit vielen Views die Verteilung dominieren.
    """
    names: List[str] = []
    for item in per_item:
        for k in item:
            if k not in names:
                names.append(k)
    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        out[name] = summarize([sum(item[name]) for item in per_item if item.get(name)])
    return out


def print_table(title: str, stats: Dict[str, Dict[str, float]],
                total_key: Optional[str] = None) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'Schritt':<26}{'n':>5}{'Median':>11}{'IQR':>11}"
          f"{'p95':>11}{'Anteil':>9}")
    grand = sum(v.get("median", 0.0) for k, v in stats.items() if k != total_key)
    for name, v in stats.items():
        if not v.get("n"):
            continue
        share = ("" if name == total_key or grand <= 0
                 else f"{100 * v['median'] / grand:7.1f}%")
        print(f"  {name:<26}{v['n']:>5}{_fmt(v['median']):>11}"
              f"{_fmt(v['iqr']):>11}{_fmt(v['p95']):>11}{share:>9}")
    if grand > 0:
        print(f"  {'-' * 71}")
        print(f"  {'SUMME (Mediane)':<26}{'':>5}{_fmt(grand):>11}")


def _fmt(sec: float) -> str:
    if sec >= 60:
        return f"{sec / 60:.2f} min"
    if sec >= 1:
        return f"{sec:.2f} s"
    return f"{sec * 1000:.1f} ms"


def write_results(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[stage4] geschrieben: {path}")


def host_provenance() -> dict:
    """Hardware und Commit mitschreiben — eine Latenzzahl ohne die Maschine,
    auf der sie entstand, ist in der Arbeit nicht zitierfaehig."""
    info: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        import subprocess
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        ).stdout.strip()
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:
        pass
    try:
        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    return info
