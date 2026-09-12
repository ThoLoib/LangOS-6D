#!/usr/bin/env python3
"""Solo-Trial — Szenario 2 (OBJEKTAUSWAHL_SOLO.md): EIN Objekt allein auf dem Tisch.

Wiederverwendet die Stage-5-Bausteine unveraendert; der einzige Unterschied zum
Studien-Trial (experiment_proxy_grasp.run_trial): die PyBullet-Welt enthaelt NUR
das Zielobjekt ("BOP-Frame minus Clutter" — Kamera, Konventionen und Ausgangspose
kommen weiterhin aus dem eingefrorenen Frame des Plans). FoundationPose sieht
deshalb das SIM-Rendering (RGB-D + Seg-Maske) statt des echten Sensorbilds, denn
im echten Bild steht der Clutter. D_sym wird gegen die gesettelte Solo-Pose
gerechnet, nicht gegen die BOP-Annotation.

====================================================================
OBJEKT UND PROXY TAUSCHEN — der freie Modus, ohne Plan, ohne Code-Aenderung:

    # gt-Serie: eigenes CAD, 10 Laeufe (je +36 Grad gedreht), Erfolgsrate am Ende
    docker compose run --rm oscar python3 -m grasping.stage_5 --object tless:12

    # Proxy-Serie: auf dem Proxy geplant, das echte Objekt gegriffen
    docker compose run --rm oscar python3 -m grasping.stage_5 \
        --object tless:12 --proxy itodd/obj_000013

  Bedingung automatisch: --proxy leer oder gleich dem Objekt selbst -> gt-Lauf
  (eigenes CAD); --proxy gesetzt -> Proxy-Lauf. Am Ende steht die Erfolgsrate
  (X/N). Seriengroesse: --runs (Default 10), Drehung je Lauf: --yaw-step (36).

  --object <ds>:<id>   das Zielobjekt: ycbv | tless | lmo + BOP-obj_id.
                       Szene/Frame (Kamera + Ausgangspose) sucht das Skript
                       selbst (sichtbarster Frame); Overrides: --scene, --im.
  --proxy <id>         das CAD, auf dem geplant wird: Pool-CAD (gso/... ,
                       housecat6d/... , itodd/...; Liste: --list-proxies
                       [filter]) ODER ein BOP-Objekt (z.B. tless/obj_000006).

  Weitere Parameter, die man je nach Objekt anpassen koennte:
  --conditions gt_pose,gt,proxy   welche Arme laufen (gt_pose = wahre Pose,
                                  gt = FoundationPose + eigenes CAD,
                                  proxy = FoundationPose + --proxy)
  --n-tries 5                     Greifversuche je Trial
  --mass 0.411                    Masse des Ziels in kg (Default: bekannte
                                  YCB-Masse aus sim_scene.OBJECT_MASS_KG,
                                  sonst 0.2 kg)
  --csv <datei>                   eigene Ergebnisdatei (Default solo_custom/)
  Greifer: 5-80 mm Oeffnung (PROTOCOL in experiment_proxy_grasp).

Serienbetrieb (die vorbereitete Liste): --all bzw. --case/--inst, gespeist aus
--plan (Default _s5_out/solo_full/plan.json, gebaut von stage_5_full.py:
alle greifbaren BOP-Objekte, je mit 3b-Proxy und 3c-Substitut). Vierte
Bedingung im Serienbetrieb: proxy3c (FoundationPose + 3c-Substitut; kann ein
BOP-Geschwisterobjekt sein, das wie ein Ziel-CAD aufgeloest wird).
====================================================================

    docker compose run --rm oscar python3 -m grasping.stage_5 --case ycbv14
    docker compose run --rm oscar python3 -m grasping.stage_5 --case tless30 --inst 2 --verbose
    docker compose run --rm oscar python3 -m grasping.stage_5 --all        # ganze Serie, Resume
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_ROOT, os.path.join(_ROOT, "object_retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from grasping.experiment_proxy_grasp import (Ctx, PROTOCOL, _append_csv,      # noqa: E402
                                             _cad_under_test, _fp_pose,
                                             _grasp_step)
from grasping.sim_scene import TabletopSim                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--object", default="",
                    help="freier Modus: Zielobjekt als <ds>:<id>, z.B. tless:12")
    ap.add_argument("--proxy", default="",
                    help="freier Modus: Proxy-CAD als <quelle>/<name>, "
                         "z.B. itodd/obj_000013 (Liste: --list-proxies)")
    ap.add_argument("--scene", default="", help="freier Modus: Szene erzwingen (z.B. 000012)")
    ap.add_argument("--im", type=int, default=-1, help="freier Modus: Frame erzwingen")
    ap.add_argument("--list-proxies", nargs="?", const="", default=None, metavar="FILTER",
                    help="alle Proxy-Kandidaten (optional gefiltert) ausgeben und beenden")
    ap.add_argument("--runs", type=int, default=10,
                    help="freier Modus: Anzahl Laeufe der Serie (Default 10; "
                         "je Lauf um --yaw-step gedreht)")
    ap.add_argument("--mass", type=float, default=0.0,
                    help="Masse des Zielobjekts in kg (ueberschreibt die "
                         "Standardmasse; 0 = Standard)")
    ap.add_argument("--canonical", action="store_true",
                    help="Objekt GREIFBAR hinstellen statt der BOP-Liegepose: "
                         "stabilste Standpose, deren horizontale Ausdehnung in "
                         "den Greifer (78 mm) passt; dazu je Instanz eine "
                         "deterministische Drehung um die Hochachse "
                         "(Instanz-Index x --yaw-step). Ergebnis-CSV: solo_full/")
    ap.add_argument("--yaw-step", type=float, default=36.0,
                    help="Grad je Instanz-Index im --canonical-Modus (Default 36)")
    ap.add_argument("--case", default="",
                    help="Fall wie in plan.json (z.B. ycbv14, tless30); mit --all: "
                         "nur diesen Fall laufen lassen")
    ap.add_argument("--inst", type=int, default=0, help="Instanz-Index im Plan (0..9)")
    ap.add_argument("--all", action="store_true",
                    help="alle Faelle x Instanzen des Plans (mit Resume aus der CSV)")
    ap.add_argument("--plan", default=os.path.join(_ROOT, "_s5_out", "solo_full", "plan.json"),
                    help="Plan-Datei (Default: der Vollplan aus stage_5_full.py)")
    ap.add_argument("--conditions", default="gt_pose,gt,proxy",
                    help="Arme: gt_pose (wahre Pose), gt (FP + eigenes CAD), proxy "
                         "(FP + 3b-Proxy des Plans), proxy3c (FP + 3c-Substitut des Plans)")
    ap.add_argument("--csv", default="",
                    help="Default: _s5_out/solo_full/trials.csv bei --all --canonical, "
                         "sonst solo_smoke")
    ap.add_argument("--n-tries", type=int, default=PROTOCOL["executor"]["n_tries"])
    ap.add_argument("--no-exec", dest="exec", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    # Felder, die Ctx/_grasp_step aus der Studie erwarten:
    ap.add_argument("--world", default="auto")
    ap.add_argument("--pose-source", default="fp", choices=["fp"])   # solo: immer frisch
    args = ap.parse_args()

    if args.list_proxies is not None:
        from grasping.proxy_grasp_cases import proxy_pool
        for pid in proxy_pool():
            if args.list_proxies.lower() in pid.lower():
                print(pid)
        return

    if not args.csv:
        args.csv = os.path.join(
            _ROOT, "_s5_out",
            "solo_custom" if args.object else
            "solo_full" if (args.canonical and args.all) else
            "solo_run" if args.all else "solo_smoke",
            "trials.csv")
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if args.object:
        base = _custom_tr(args)
        self_id = f"{base['dataset']}/obj_{base['obj_id']:06d}"
        if args.conditions == "gt_pose,gt,proxy":     # Default -> automatisch:
            # Proxy leer oder das Objekt selbst -> gt-Lauf (eigenes CAD);
            # Proxy gesetzt -> Proxy-Lauf (auf dem Proxy geplant).
            conds = ["gt"] if args.proxy in ("", self_id) else ["proxy"]
            print(f"[stage5] Bedingung: {conds[0]} "
                  f"({'auf dem Proxy geplant' if conds == ['proxy'] else 'eigenes CAD'})")
        if args.runs > 1 and not args.canonical:
            args.canonical = True                     # Serie steht immer kanonisch
            print(f"[stage5] Serie: {args.runs} Laeufe, kanonische Standpose, "
                  f"je Lauf +{args.yaw_step:g}° Drehung")
        todo = [dict(base, inst=i) for i in range(max(1, args.runs))]
    elif args.all:
        todo = json.load(open(args.plan))["plan"]
        if args.case:                                 # --all --case X = nur dieser Fall
            todo = [t for t in todo if t["case"] == args.case]
        if args.proxy:                                # Proxy-Override fuer Sonderlaeufe
            todo = [dict(t, proxy=args.proxy) for t in todo]
            print(f"[stage5] Proxy-Override: {args.proxy}")
    else:
        if not args.case:
            sys.exit("Einzeltrial braucht --case (z.B. --case ycbv14), "
                     "oder --all fuer die Serie, oder --object fuer den freien Modus")
        plan = json.load(open(args.plan))["plan"]
        trs = [t for t in plan if t["case"] == args.case]
        if not trs:
            sys.exit(f"Fall {args.case} nicht im Plan ({sorted({t['case'] for t in plan})})")
        todo = [trs[args.inst]]

    done = set()
    if args.object:
        pass          # freier Modus: kein Resume — jede Eingabe = frische Serie
    elif os.path.exists(args.csv):
        import csv as _csv
        for d in _csv.DictReader(open(args.csv)):
            done.add((d["case"], str(d["scene"]), str(d["im"]), str(d["gt_idx"]),
                      d["condition"]))
        if done:
            print(f"[stage5] Resume: {len(done)} Trials bereits in {args.csv}")

    ctx = Ctx(args)
    os.environ.setdefault("GRASP_QUIET", "1")
    if args.mass > 0:
        from grasping import sim_scene
        for t in todo:
            sim_scene.OBJECT_MASS_KG[(t["dataset"], t["obj_id"])] = args.mass
        print(f"[stage5] Zielmasse per CLI: {args.mass} kg")
    n_total = sum(1 for t in todo for c in conds
                  if (t["case"], str(t["scene"]), str(t["im"]), str(t["gt_idx"]), c)
                  not in done)
    n_run = 0
    import csv as _csv2
    n_pre = (sum(1 for _ in _csv2.DictReader(open(args.csv)))
             if args.object and os.path.exists(args.csv) else 0)
    for tr in todo:
        pend = [c for c in conds
                if (tr["case"], str(tr["scene"]), str(tr["im"]), str(tr["gt_idx"]), c)
                not in done]
        if not pend:
            continue
        print(f"[stage5] {tr['case']} ({tr['name']}) inst s{tr['scene']}/im{tr['im']} — "
              f"Proxy {tr['proxy']}")
        fr, objs, cam, winfo = ctx.scene(tr["dataset"], tr["scene"], tr["im"])
        tgt = objs[tr["gt_idx"]]
        assert tgt.obj_id == tr["obj_id"]
        if args.canonical:
            idx = tr["inst"] if "inst" in tr else \
                [t for t in todo if t["case"] == tr["case"]].index(tr)
            yaw = idx * args.yaw_step
            tgt, graspable = canonical_object(ctx, tgt, yaw)
            tr = dict(tr, yaw_deg=yaw, _graspable=int(graspable), _table_z=0.0)
            if not graspable:
                print(f"[stage5]   HINWEIS {tr['case']}: keine greifbare Standpose "
                      f"(<=78 mm) — stabilste Pose verwendet, als Ausnahme markiert")
        solo = [tgt]                                # <- der ganze Szenariowechsel
        n_run += run_conditions(ctx, args, tr, fr, objs, cam, winfo, tgt, solo, pend,
                                n_run, n_total)
    print(f"[stage5] fertig: {n_run} neue Trials, CSV: {args.csv}")
    if args.object:
        new = list(_csv2.DictReader(open(args.csv)))[n_pre:]
        wins = sum(1 for r in new if r["succ"] == "1")
        what = args.proxy if conds == ["proxy"] else "eigenes CAD"
        print(f"\n[stage5] ============================================")
        print(f"[stage5] ERGEBNIS {todo[0]['name']} | {what}: "
              f"Erfolgsrate {wins}/{len(new)}")
        print(f"[stage5] ============================================")


_CANON_CACHE: dict = {}


def canonical_pose(mesh_m, max_grip: float = 0.078):
    """Deterministische Aufstellpose: unter den stabilen Liegeposen des Meshes
    (trimesh, sigma=0 -> kein Zufall) die STABILSTE, deren horizontale
    Ausdehnung <= max_grip ist (Objekt ist dann von der Seite greifbar).
    Gibt (T_stable 4x4, greifbar: bool) zurueck; ohne greifbare Standpose die
    stabilste ueberhaupt + False (Anwendbarkeits-Ausnahme, wird geloggt)."""
    import numpy as np
    import trimesh
    key = id(mesh_m)
    if key in _CANON_CACHE:
        return _CANON_CACHE[key]
    Ts, probs = trimesh.poses.compute_stable_poses(mesh_m, sigma=0.0, n_samples=1)
    best, best_any = None, None
    for T, p in zip(Ts, probs):
        m2 = mesh_m.copy()
        m2.apply_transform(T)
        horiz = float(min(m2.extents[0], m2.extents[1]))
        if best_any is None or p > best_any[1]:
            best_any = (T, p)
        if horiz <= max_grip and (best is None or p > best[1]):
            best = (T, p)
    out = ((best[0], True) if best is not None else (best_any[0], False))
    _CANON_CACHE[key] = out
    return out


def canonical_object(ctx, tgt, yaw_deg: float):
    """SceneObject-Kopie: greifbar aufgestellt an der Tischposition des Frames,
    um yaw_deg um die Hochachse gedreht, Unterkante auf z=0."""
    import dataclasses
    import numpy as np
    mesh_m = ctx.mesh_m(tgt.mesh_path, True)
    T_st, graspable = canonical_pose(mesh_m)
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    T = Rz @ T_st
    m2 = mesh_m.copy()
    m2.apply_transform(T)
    # Zentrum der Grundflaeche an die Tischposition des Frames, Unterkante auf z=0
    cx, cy = m2.bounds.mean(axis=0)[:2]
    T[0, 3] += tgt.T_world[0, 3] - cx
    T[1, 3] += tgt.T_world[1, 3] - cy
    T[2, 3] += -float(m2.bounds[0, 2])
    return dataclasses.replace(tgt, T_world=T), graspable


def _custom_tr(args) -> dict:
    """Freier Modus: aus --object (+ optional --scene/--im) einen Trial-Eintrag
    bauen. Frame-Wahl: steht das Objekt im Vollplan, wird der Frame seiner
    Experiment-Instanz 1 verwendet (gleiche Kamera-Geometrie wie die Tabellen —
    der bloss "sichtbarste" Frame liefert z. T. deutlich schlechtere FP-Fits);
    sonst der sichtbarste Frame. Eigener Frame jederzeit per --scene/--im."""
    import glob as _glob
    from grasping.build_grasp_instances import _test_root
    from grasping.sim_scene import load_bop_frame, object_name
    try:
        ds, oid = args.object.replace("/", ":").split(":")
        oid = int(oid)
    except ValueError:
        sys.exit(f"--object '{args.object}' nicht lesbar — Format <ds>:<id>, z.B. tless:12")
    plan_gt_idx = None
    if not args.scene and args.im < 0:
        plan_p = os.path.join(_ROOT, "_s5_out", "solo_full", "plan.json")
        if os.path.isfile(plan_p):
            for t in json.load(open(plan_p))["plan"]:
                if t["dataset"] == ds and t["obj_id"] == oid:
                    args.scene = f"{int(t['scene']):06d}"
                    args.im = int(t["im"])
                    plan_gt_idx = t["gt_idx"]
                    print(f"[stage5] Frame aus dem Experiment-Plan: Szene "
                          f"{args.scene}/im {args.im} (wie Tabellen-Instanz 1)")
                    break
    scene = args.scene
    if not scene:
        for sdir in sorted(_glob.glob(os.path.join(_test_root(ds), "*"))):
            gt = os.path.join(sdir, "scene_gt.json")
            if os.path.isfile(gt):
                first = next(iter(json.load(open(gt)).values()))
                if any(e["obj_id"] == oid for e in first):
                    scene = os.path.basename(sdir)
                    break
        if not scene:
            sys.exit(f"kein Testsplit-Frame mit {ds} obj {oid} gefunden.")
    if args.im >= 0:
        im = args.im
    else:
        # sichtbarster Frame des Objekts, robust je Frame bestimmt
        sdir = os.path.join(_test_root(ds), scene)
        gt_all = json.load(open(os.path.join(sdir, "scene_gt.json")))
        info_all = json.load(open(os.path.join(sdir, "scene_gt_info.json")))
        best = None
        for k, entries in gt_all.items():
            for gi, e in enumerate(entries):
                if e["obj_id"] == oid and k in info_all and gi < len(info_all[k]):
                    v = info_all[k][gi].get("visib_fract", 0.0)
                    if best is None or v > best[0]:
                        best = (v, int(k))
        if best is None:
            sys.exit(f"{ds} obj {oid}: kein annotierter Frame in Szene {scene}.")
        im = best[1]
    fr = load_bop_frame(ds, scene, im)
    gt_idx = plan_gt_idx if plan_gt_idx is not None else fr.instances_of(oid)[0]
    import json as _json
    mi = _json.load(open(os.path.join(_ROOT, "eval", "datasets", ds,
                                      "models_eval", "models_info.json")))
    print(f"[stage5] freier Modus: {ds} obj {oid} ({object_name(ds, oid)}) — "
          f"Szene {scene}/im {im} (sichtbarster Frame), Proxy {args.proxy or '—'}")
    return dict(case=f"custom_{ds}{oid}", rank=0, tier=0, dataset=ds, obj_id=oid,
                name=object_name(ds, oid), proxy=args.proxy, pool="", n_top1="",
                scene=int(scene), im=int(im), gt_idx=gt_idx,
                visib=fr.visib(gt_idx), diameter=mi.get(str(oid), {}).get("diameter"))


def resolve_cad(ctx, tr, cond):
    """CAD unter Test aufloesen: (cad_id, pfad, units_m).

    proxy   = 3b-Proxy (tr["proxy"]) — Pool-CAD oder BOP-Geschwister.
    proxy3c = 3c-Substitut (tr["proxy3c"]) — kann BOP-Geschwister sein.
    BOP-Geschwister werden EXAKT wie das eigene CAD aufgeloest (gleiche Datei,
    gleiche Einheiten) — bop_mesh_path liefert je Datensatz unterschiedliche
    Einheiten, target_cad kapselt das korrekt. Fehlt die ID: KeyError."""
    if cond in ("proxy", "proxy3c"):
        pid = tr["proxy"] if cond == "proxy" else (tr.get("proxy3c") or "")
        if not pid:
            raise KeyError(f"{cond}: keine CAD-Id im Plan")
        src = pid.split("/", 1)[0]
        if src in ("ycbv", "tless", "lmo"):
            cad_path, units_m = ctx.target_cad(src, int(pid.split("obj_")[1]))
        else:
            from grasping.proxy_grasp_cases import proxy_mesh
            cad_path, units_m = proxy_mesh(pid)
        return pid, cad_path, units_m
    return _cad_under_test(ctx, tr, cond)


def run_conditions(ctx, args, tr, fr, objs, cam, winfo, tgt, solo, conds,
                   n_done, n_total) -> int:
    import numpy as np                               # noqa: F811
    ran = 0
    for cond in conds:
        t0 = time.time()
        row = dict(case=tr["case"], rank=tr["rank"], tier=tr["tier"], dataset=tr["dataset"],
                   obj_id=tr["obj_id"], name=tr["name"], scene=tr["scene"], im=tr["im"],
                   gt_idx=tr["gt_idx"], visib="", condition=cond, cad="", cad_units_m="",
                   pose_source="", fp_conf="", fp_fail="", dsym_mm="", dsym_norm="", f05="",
                   place_mm="", settle_mm="", world="solo", plane_inlier="",
                   plane_angle_deg="", bottom_gap_mm="",
                   n_cand=0, n_reach=0, n_blocked=0, n_att=0, n_succ=0, first_succ=0,
                   succ=0, lift_cm=0.0, att_seq="", fail_reason="", runtime_s=0.0,
                   ts=_dt.datetime.now().isoformat(timespec="seconds"))
        if "yaw_deg" in tr:                     # --canonical: eigene Spalten
            row["yaw_deg"] = tr["yaw_deg"]
            row["graspable_pose"] = tr["_graspable"]
        try:
            cad_id, cad_path, units_m = resolve_cad(ctx, tr, cond)
        except KeyError:
            row.update(cad="", fail_reason="cad_missing")
            _append_csv(args.csv, row)
            continue
        row.update(cad=cad_id, cad_units_m=int(units_m))
        sim = TabletopSim().connect()
        try:
            # Ohne Roboter bauen: die Kameraseiten-Platzierung stellt den Panda
            # in die Sichtlinie der BOP-Kamera — im Solo-Rendering wuerde er das
            # Ziel komplett verdecken (in der Studie sah FP das ECHTE Bild, da
            # war das egal). Der Roboter kommt nach dem Rendern dazu.
            tz = tr.get("_table_z")
            sim.build(solo, cam, target_gt_idx=tr["gt_idx"], with_robot=False,
                      table_z=tz if tz is not None else
                      min(0.0, winfo["bottom_z"].get(tr["gt_idx"], 0.0) - 0.001))
            sim.settle(PROTOCOL["physics"]["settle_steps"])
            row["settle_mm"] = round(sim.target_displacement_mm(tgt.T_world), 1)
            sim.freeze_initial()
            T_true = sim.target_pose()

            extra = 1.0
            if cond == "gt_pose":
                T_m2w, row["pose_source"] = T_true, "sim_true"
            else:
                rd = sim.render_rgbd()
                mask = sim.object_mask(tgt.obj_id, rd["seg"]).astype(np.uint8)
                if mask.sum() < 200:
                    row.update(fail_reason="fp_error", fp_fail="Solo-Maske leer")
                    continue
                try:
                    R, t, conf = _fp_pose(cad_path, rd["rgb"], rd["depth"], mask,
                                          cam.K, units_m, extra)
                    row["pose_source"] = "fp_solo_render"
                    row["fp_conf"] = round(float(conf), 3) if conf is not None else ""
                except Exception as exc:                              # noqa: BLE001
                    row.update(fp_fail=str(exc).splitlines()[-1][:80],
                               fail_reason="fp_error")
                    continue
                T_m2c = np.eye(4)
                T_m2c[:3, :3], T_m2c[:3, 3] = R, np.asarray(t, float) / 1000.0
                T_m2w = cam.T_world @ T_m2c
                # D_sym gegen die GESETTELTE Solo-Pose (Kameraframe, mm)
                from stage3_metrics import d_sym
                T_true_c = np.linalg.inv(cam.T_world) @ T_true
                tpath, tunits = ctx.target_cad(tr["dataset"], tr["obj_id"])
                dsr = d_sym(ctx.pts_mm(tpath, tunits), T_true_c[:3, :3],
                            T_true_c[:3, 3] * 1000.0,
                            ctx.pts_mm(cad_path, units_m, extra), R, np.asarray(t, float),
                            tr.get("diameter") or 0.0)
                row.update(dsym_mm=round(dsr["d_sym"], 2),
                           dsym_norm=round(dsr["d_sym_norm"], 4)
                           if dsr["d_sym_norm"] is not None else "",
                           f05=round(dsr["fscore"]["0.05"]["f"], 4))
            sim._add_panda(solo)          # jetzt erst der Roboter (nach dem Rendern)
            sim.freeze_initial()          # Reset-Zustand inkl. Roboter einfrieren
            mesh_m = ctx.mesh_m(cad_path, units_m, extra)
            c_cad = T_m2w[:3, :3] @ mesh_m.centroid + T_m2w[:3, 3]
            c_tgt = T_true[:3, :3] @ ctx.mesh_m(tgt.mesh_path, True).centroid + T_true[:3, 3]
            row["place_mm"] = round(float(np.linalg.norm(c_cad - c_tgt) * 1000), 1)
            _grasp_step(ctx, sim, cad_path, units_m, extra, T_m2w, row)
        finally:
            sim.disconnect()
            row["runtime_s"] = round(time.time() - t0, 1)
            _append_csv(args.csv, row)
            ran += 1
            print(f"[stage5] [{n_done + ran:>3}/{n_total}] {cond:8s} "
                  f"cad={row['cad'][:36]:38s} "
                  f"D_sym={row['dsym_mm'] or '—':>6} place={row['place_mm'] or '—':>6} "
                  f"cand={row['n_cand']:>2} reach={row['n_reach']:>2} "
                  f"blocked={row['n_blocked']:>2} att={row['n_att']} "
                  f"-> {'ERFOLG' if row['succ'] else row['fail_reason'] or 'kein Erfolg'} "
                  f"({row['runtime_s']} s)")
    return ran


if __name__ == "__main__":
    main()
