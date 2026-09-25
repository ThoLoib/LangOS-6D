#!/usr/bin/env python3
"""MI3DOR-Gewichts-Sweep v2 — byte-identisch zur Produktionsfusion.

Ersetzt mi3dor_weight_sweep.py (eigene Refusion, BASE traf die Produktion
nicht: 0.6851 vs. 0.6918). v2 repliziert die SCORE-ASSEMBLIERUNG von
pipeline/step6_fusion._weighted_sum exakt — Einfuegereihenfolge (CLIP-, dann
DINO-, dann Shape-Kandidaten), clip = max(clip_res.score, dino.clip_score),
ulip-NaN bleibt 0, Min-Max ueber die assemblierten Werte (Nullen inklusive),
stabile Sortierung — und BEWEIST das doppelt:

  1. Fuer jede ~35. Query wird zusaetzlich das echte
     ``fusion_mod.fuse(...)`` (ScoreFusion, Produktionsgewichte) gerechnet;
     die Top-100 muessen der vektorisierten BASE-Rangfolge exakt entsprechen.
  2. Der BASE-Punkt (0.3, 0.4, 0.3) muss den Produktionslauf treffen
     (Partial-Fusion 07.09.: FT 0.6918, NN 88.4381; Toleranz 2e-4) —
     sonst Abbruch ohne Heatmap.

Raster wie der Stage-1-Sweep: 66 Simplex-Punkte, Schritt 0.1. Die
Kanal-Score-Matrizen werden PERSISTIERT (channel_scores.npz), damit spaetere
Re-Sweeps ohne GPU-Scoring auskommen; manifest.json haelt Commit, Konfig und
Selfcheck-Ergebnisse fest.

    cd object_retrieval && python3 mi3dor_weight_sweep_v2.py
    # Smoke: MI3DOR_MAX_QUERIES_PER_CAT=15 (Selfcheck 2 wird dann uebersprungen)
"""
import csv
import datetime as _dt
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retrieval_mi3dor_eval_oscarplus import (  # noqa: E402
    cfg, to_category_label, _get_categories,
    _collect_filtered_cad_mesh_items, _make_query_factory, _collect_query_paths,
)
from eval_common import (  # noqa: E402
    build_pipeline, run_query, make_accum, update_accum, finalize_accum,
    load_ulip_query_cache, pre_encode_ulip_queries,
)

BASE_W = (0.3, 0.4, 0.3)
PROD_FT, PROD_NN = 0.6918, 88.4381      # Partial-Fusion, Rerun 2026-09-07
TOL = 2e-4
STEP = 0.1                              # 66 Punkte — wie der Stage-1-Sweep
OUT_DIR = os.environ.get("MI3DOR_SWEEP_DIR", "results_mi3dor_wsweep_v2")
CHECK_EVERY = 35                        # jede 35. Query: echtes fusion_mod.fuse


def _simplex(step):
    n = int(round(1.0 / step))
    return [(round(i * step, 4), round(j * step, 4), round((n - i - j) * step, 4))
            for i in range(n + 1) for j in range(n + 1 - i)]


def assemble(out):
    """Score-Assemblierung EXAKT wie ScoreFusion._weighted_sum (step6:172-197):
    Einfuegereihenfolge clip -> dino -> shape; clip=max(., dino.clip_score);
    ulip-NaN wird uebersprungen (bleibt 0). Rueckgabe: (ids, clip, dino, ulip)."""
    order, idx = [], {}

    def ent(oid):
        if oid not in idx:
            idx[oid] = len(order)
            order.append(oid)
            for a in (c_v, d_v, u_v):
                a.append(0.0)
        return idx[oid]

    c_v, d_v, u_v = [], [], []
    if out["clip_res"]:
        for c in out["clip_res"].candidates:
            i = ent(c.object_id)
            c_v[i] = max(c_v[i], float(c.score))
    if out["dino_res_full"]:
        for c in out["dino_res_full"].candidates:
            i = ent(c.object_id)
            d_v[i] = max(d_v[i], float(c.dino_score))
            c_v[i] = max(c_v[i], float(getattr(c, "clip_score", 0.0) or 0.0))
    if out["shape_res_full"]:
        for c in out["shape_res_full"].candidates:
            i = ent(c.object_id)
            s = c.shape_score
            if not (isinstance(s, float) and np.isnan(s)):
                u_v[i] = max(u_v[i], float(s))
    # float64 wie die Produktions-Python-Floats — float32 kippte bei
    # Beinahe-Gleichstaenden die Reihenfolge (2/300 Proben am 24.09.)
    return (order, np.asarray(c_v, np.float64), np.asarray(d_v, np.float64),
            np.asarray(u_v, np.float64))


def _norm(v):
    """Min-Max wie step6._minmax auf NaN-freien Vektoren (Nullen inklusive)."""
    lo, hi = float(v.min()), float(v.max())
    rng = hi - lo
    if rng <= 0:
        return np.zeros_like(v)
    return (v - lo) / rng


def rank_ids(ids, c, d, u, w):
    fused = w[0] * _norm(c) + w[1] * _norm(d) + w[2] * _norm(u)
    order = np.argsort(-fused, kind="stable")     # stabil == Python-sort
    return [(ids[i], float(fused[i])) for i in order]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    smoke = bool(int(os.environ.get("MI3DOR_MAX_QUERIES_PER_CAT", "0") or "0"))
    categories = _get_categories()
    cad_mesh_items = _collect_filtered_cad_mesh_items(categories)
    cfg.ulip2_use_partial_views = True          # Partial-Galerie (berichtete Config)
    cfg.result_folder = OUT_DIR
    assert tuple(round(x, 4) for x in
                 (cfg.weight_clip, cfg.weight_dino, cfg.weight_ulip)) == BASE_W, \
        "Produktionsgewichte weichen von BASE ab — Selfcheck 1 waere sinnlos"
    print(f"[sweep2] Pipeline (partial) ueber {len(categories)} Kategorien ...")
    components = build_pipeline(cfg, cad_mesh_items=cad_mesh_items)
    pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m = components

    ulip_cache = load_ulip_query_cache(cfg.ulip_query_cache_path)
    if ulip_cache is None:
        print("[sweep2] ULIP-Query-Cache fehlt — enkodiere einmalig ...")
        ulip_cache = pre_encode_ulip_queries(_collect_query_paths(categories), shape_m)
        try:
            import torch
            torch.save(ulip_cache, cfg.ulip_query_cache_path)
            print(f"[sweep2] ULIP-Query-Cache gesichert -> {cfg.ulip_query_cache_path}")
        except Exception as exc:                       # noqa: BLE001
            print(f"[sweep2][warn] Cache nicht gesichert: {exc}")

    gallery_ids = set(getattr(dino_rer, "_ref_embeddings", {}) or {})
    if shape_m is not None and getattr(shape_m, "_cad_embeddings", None):
        gallery_ids |= set(shape_m._cad_embeddings)
    glc = {}
    for oid in gallery_ids:
        glc[to_category_label(oid)] = glc.get(to_category_label(oid), 0) + 1
    ref_objects = len(getattr(dino_rer, "_ref_embeddings", {}) or {})
    cad_objects = len(shape_m._cad_embeddings) if (shape_m and shape_m._cad_embeddings) else 0
    dino_k = max(cfg.dino_top_k, ref_objects) if ref_objects else cfg.dino_top_k
    ulip_k = max(cfg.ulip2_top_k, cad_objects) if cad_objects else cfg.ulip2_top_k
    clip_rows = len(getattr(clip_retr, "_desc_labels", []) or [])
    clip_k = max(cfg.clip_top_k, clip_rows, ref_objects, 1_000_000 if clip_rows == 0 else 0)

    # ---- Scoring: je Query EINMAL run_query; Produktions-Assemblierung ----
    print("[sweep2] Scoring (run_query) + Assemblierung wie _weighted_sum ...")
    cache = []                   # (ids, c, d, u, gt_label, |C|)
    checks = []                  # (query_nr, echte-top100-ids)
    n = fail_ident = 0
    for roi, gt_label, img_path, category, fname in _make_query_factory(categories)(cfg.topk[0]):
        try:
            emb = ulip_cache.get(img_path) if ulip_cache else None
            out = run_query(pipeline_cfg, clip_retr, dino_rer, fusion_mod, shape_m,
                            roi, cfg, ulip_query_emb=emb,
                            dino_full_top_k=dino_k, ulip_full_top_k=ulip_k,
                            clip_full_top_k=clip_k)
        except Exception as exc:                       # noqa: BLE001
            print(f"[sweep2][warn] Query fehlgeschlagen ({img_path}): {exc}")
            continue
        ids, c, d, u = assemble(out)
        cache.append((ids, c, d, u, gt_label, glc.get(gt_label, 0)))
        # SELFCHECK 1: vektorisierte BASE-Rangfolge == echtes fusion_mod.fuse
        if n % CHECK_EVERY == 0:
            echte = [cd.object_id for cd in out["fused_full"].candidates[:100]]
            meine = [oid for oid, _ in rank_ids(ids, c, d, u, BASE_W)[:100]]
            checks.append(echte == meine)
            if echte != meine:
                fail_ident += 1
                print(f"[sweep2] IDENTITAETSFEHLER bei Query {n} ({fname})")
                with open(os.path.join(OUT_DIR, "identity_failures.jsonl"), "a") as fh:
                    fh.write(json.dumps({"n": n, "fname": fname,
                                         "echte": echte, "meine": meine}) + "\n")
        n += 1
        if n % 1000 == 0:
            print(f"[sweep2]   {n} Queries gescort", flush=True)
    print(f"[sweep2] {len(cache)} Queries | Identitaetsproben: "
          f"{sum(checks)}/{len(checks)} exakt")

    # ---- Persistenz VOR dem Abbruch-Check: Scoring nie wieder verlieren ---
    id_list = sorted({oid for ids, *_ in cache for oid in ids})
    id_idx = {oid: i for i, oid in enumerate(id_list)}
    offs, flat_idx, flat_c, flat_d, flat_u, labels, csize = [0], [], [], [], [], [], []
    for ids, c, d, u, gt, nr in cache:
        flat_idx.append(np.fromiter((id_idx[o] for o in ids), np.int32, len(ids)))
        flat_c.append(c.astype(np.float32)); flat_d.append(d.astype(np.float32)); flat_u.append(u.astype(np.float32))
        offs.append(offs[-1] + len(ids))
        labels.append(gt); csize.append(nr)
    np.savez_compressed(
        os.path.join(OUT_DIR, "channel_scores.npz"),
        gallery_ids=np.array(id_list), offsets=np.array(offs, np.int64),
        obj_idx=np.concatenate(flat_idx), clip=np.concatenate(flat_c),
        dino=np.concatenate(flat_d), ulip=np.concatenate(flat_u),
        gt_label=np.array(labels), cat_size=np.array(csize, np.int32))
    print(f"[sweep2] Kanal-Matrizen persistiert -> {OUT_DIR}/channel_scores.npz")
    if fail_ident:
        sys.exit(f"[sweep2] ABBRUCH: {fail_ident} Identitaetsproben verfehlt — "
                 f"Assemblierung NICHT produktionsgleich (Details: "
                 f"{OUT_DIR}/identity_failures.jsonl; Matrizen sind gesichert).")

    # ---- Sweep (66 Punkte) + SELFCHECK 2 ----------------------------------
    def eval_point(w):
        acc = make_accum()
        for ids, c, d, u, gt, nr in cache:
            update_accum(acc, rank_ids(ids, c, d, u, w), gt,
                         to_category_label, cfg.TOP_F, nr)
        m = finalize_accum(acc)
        return float(m["FT_mean"]), float(m["NN_accuracy"])

    bft, bnn = eval_point(BASE_W)
    print(f"[sweep2] SELFCHECK 2 — BASE {BASE_W}: FT={bft:.4f} NN={bnn:.4f} "
          f"(Produktion: FT={PROD_FT} NN={PROD_NN})")
    if smoke:
        print("[sweep2] Smoke-Modus (Query-Cap) — Produktionsvergleich uebersprungen.")
    elif abs(bft - PROD_FT) > TOL or abs(bnn - PROD_NN) > TOL * 100:
        sys.exit(f"[sweep2] ABBRUCH: BASE verfehlt die Produktion "
                 f"(ΔFT={bft-PROD_FT:+.4f}, ΔNN={bnn-PROD_NN:+.4f}).")

    rows, best = [], (-1.0, None)
    for w in _simplex(STEP):
        ft, nn = eval_point(w)
        rows.append((*w, round(ft, 4), round(nn, 4)))
        if ft > best[0]:
            best = (ft, w)
    out_csv = os.path.join(OUT_DIR, "weight_sweep_66.csv")
    with open(out_csv, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["w_clip", "w_dino", "w_ulip", "FT", "NN"])
        wtr.writerows(rows)

    manifest = dict(
        ts=_dt.datetime.now().isoformat(timespec="seconds"),
        git=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                           text=True, cwd=os.path.dirname(os.path.abspath(__file__))
                           ).stdout.strip(),
        grid=f"{len(rows)} Punkte, Schritt {STEP} (wie Stage-1-Sweep)",
        config=dict(partial=True, dino_pooling=os.environ.get("MI3DOR_DINO_POOLING", "mean"),
                    num_views=int(os.environ.get("MI3DOR_NUM_VIEWS", "42")),
                    weights_prod=BASE_W, ulip_ckpt=cfg.ulip2_checkpoint
                    if hasattr(cfg, "ulip2_checkpoint") else ""),
        selfcheck=dict(identitaetsproben=f"{sum(checks)}/{len(checks)}",
                       base_ft=round(bft, 4), base_nn=round(bnn, 4),
                       produktion_ft=PROD_FT, produktion_nn=PROD_NN,
                       queries=len(cache)),
        fusion="pipeline/step6_fusion._weighted_sum (Assemblierung repliziert, "
               "per Stichproben gegen fusion_mod.fuse bewiesen)")
    json.dump(manifest, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=1)
    print(f"[sweep2] {len(rows)} Punkte -> {out_csv}")
    print(f"[sweep2] BASE FT={bft:.4f} | Optimum FT={best[0]:.4f} bei w={best[1]} "
          f"(Δ={best[0]-bft:+.4f})")


if __name__ == "__main__":
    main()
