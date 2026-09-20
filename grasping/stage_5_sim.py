#!/usr/bin/env python3
"""Stage-5-Sim — die komplette OSCAR+-Pipeline live im PyBullet-Fenster.

Eine "echte" Situation: Prompt rein, zusehen, wie der Roboter in 10 Laeufen
(Objekt je Lauf um 36 Grad gedreht) versucht zu greifen. Jeder Lauf durchlaeuft
die ECHTE Pipeline auf dem Sim-Bild — keine GT-Abkuerzungen:

  1  GroundingDINO + SAM       Segmentierung per Prompt (immer echt)
  2  Punktwolke                maskierte Rueckprojektion (fuer Schritt 7)
  3-6  CLIP / DINOv2 / ULIP-2  Retrieval + Fusion gegen die VOLLE Datenbank
                               (G_proxy + alle BOP-Ziel-CADs — das exakte
                               Modell DARF gefunden werden)
  7  Geometrischer Check       dGeDi-Re-Ranking der Shortlist (abschaltbar)
  8  FoundationPose            Pose des abgerufenen CADs
  →  Greifen                   antipodale Griffe auf dem abgerufenen CAD,
                               bis zu --n-tries Versuche je Lauf

Anzeige (GUI-Modus, Default): PyBullet-Fenster, Beobachter-Kamera frei mit der
Maus drehbar/zoombar. Das abgerufene CAD schwebt als gruener Geist in der
geschaetzten Pose ueber dem echten Objekt; Griffkandidaten als gelbe Linien,
der aktive Versuch blau, Erfolg gruen / Fehlschlag rot. Am Ende zaehlt genau
eine Metrik: die Erfolgsrate X/N.

    # Solo-Objekt auf dem Tisch (starre Default-Kamera):
    docker compose run --rm oscar python3 -m grasping.stage_5_sim \
        --prompt "the red mug" --object ycbv:14

    # BOP-Szene mit Clutter (Kamera der Szene, Ziel waehlt der Prompt):
    docker compose run --rm oscar python3 -m grasping.stage_5_sim \
        --prompt "the mustard bottle" --scene ycbv:48:1

  --skip geometry[,shape,dino,clip]  Schritte auslassen (geometry = Schritt 7;
                                     shape/dino/clip = Fusionskanal auf 0)
  --runs 10 --yaw-step 36 --n-tries 5   die Stage-5-Seriensemantik
  --no-scale-fit    abgerufenes CAD NICHT auf die beobachtete Groesse skalieren
  --headless        ohne GUI (DIRECT) — fuer Tests; nur Konsolen-Ausgabe

Dienste: FoundationPose (docker compose up -d foundationpose) und — ausser bei
--skip geometry — dGeDi mit der BOP-Galerie (docker compose up -d dgedi).
GUI braucht die X-Durchreichung (in docker-compose.yml hinterlegt, DISPLAY=:1).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "object_retrieval"))

import numpy as np                                                  # noqa: E402

BASE_WEIGHTS = (0.3, 0.4, 0.3)          # (Text, Bild, Form) — Stage-1 BASE
GEO_TOP_K = 5                           # Schritt-7-Shortlist wie Stage 3


# ---------------------------------------------------------------------------
# Aufbau-Helfer
# ---------------------------------------------------------------------------

def fixed_camera(w: int = 640, h: int = 480,
                 eye=(0.55, 0.0, 0.45), look=(0.0, 0.0, 0.05)):
    """Starre Default-Kamera (OpenCV-Konvention: x rechts, y runter, z vor)."""
    from grasping.sim_scene import SceneCamera
    eye = np.asarray(eye, float)
    fwd = np.asarray(look, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, eye
    K = np.array([[600.0, 0, w / 2], [0, 600.0, h / 2], [0, 0, 1]])
    return SceneCamera(K=K, T_world=T, width=w, height=h)


def solo_object(ds: str, oid: int, yaw_deg: float):
    """Zielobjekt allein, kanonisch greifbar aufgestellt, um yaw_deg gedreht."""
    import trimesh
    from grasping.sim_scene import SceneObject, bop_mesh_path
    from grasping.stage_5 import canonical_pose
    mp = bop_mesh_path(ds, oid)                       # Meter-.obj (Sim-Format)
    mesh = trimesh.load(mp, force="mesh")
    T_st, graspable = canonical_pose(mesh)
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    T = Rz @ T_st
    m2 = mesh.copy()
    m2.apply_transform(T)
    cx, cy = m2.bounds.mean(axis=0)[:2]
    T[0, 3] -= cx
    T[1, 3] -= cy
    T[2, 3] -= float(m2.bounds[0, 2])                 # Unterkante auf z=0
    if not graspable:
        print(f"[sim] HINWEIS {ds}:{oid}: keine Standpose <=78 mm — "
              f"stabilste Pose verwendet")
    return SceneObject(obj_id=oid, T_world=T, mesh_path=mp, dataset=ds)


def rotate_about_z(T: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Objektpose um die eigene Hochachse drehen (Szenen-Modus)."""
    a = np.radians(yaw_deg)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    out = T.copy()
    out[:3, :3] = Rz @ T[:3, :3]
    return out


def load_perception(weights):
    """Volle Datenbank (Proxys + ALLE Ziel-CADs) + Pipeline-Komponenten."""
    from stage3_gallery import assemble_gallery, TARGET_DATASETS
    from grasping.perceive import Perception
    gal = assemble_gallery(target_datasets=TARGET_DATASETS, weights=weights)
    print(f"[sim] Datenbank: {len(gal.gallery_ids)} CADs "
          f"(G_proxy + Ziel-CADs)  Gewichte {weights}")
    return Perception(gallery=gal, components=gal.components(),
                      cfg=gal.eval_cfg)


def parse_skip(arg: str):
    skips = {t.strip() for t in arg.split(",") if t.strip()}
    bad = skips - {"geometry", "clip", "dino", "shape"}
    if bad:
        raise SystemExit(f"--skip: unbekannt {sorted(bad)} "
                         f"(erlaubt: geometry, clip, dino, shape)")
    w = np.array(BASE_WEIGHTS)
    for tok, i in (("clip", 0), ("dino", 1), ("shape", 2)):
        if tok in skips:
            w[i] = 0.0
    if w.sum() <= 0:
        raise SystemExit("--skip: mindestens ein Retrieval-Kanal muss bleiben")
    return skips, tuple(float(x) for x in np.round(w / w.sum(), 4))


# ---------------------------------------------------------------------------
# Overlays (GUI): Geist + Griff-Linien
# ---------------------------------------------------------------------------

class Overlay:
    def __init__(self, p):
        self.p = p
        self.ghost = None
        self.lines = []

    def show_ghost(self, cad_path, units_m: bool, size_scale: float, T_w):
        import trimesh
        from grasping.sim_scene import _as_metre_obj
        q = trimesh.transformations.quaternion_from_matrix(T_w)
        vis = self.p.createVisualShape(
            self.p.GEOM_MESH,
            fileName=_as_metre_obj(cad_path, 1.0 if units_m else 0.001),
            meshScale=[size_scale] * 3,
            rgbaColor=(0.15, 0.95, 0.35, 0.5))
        self.ghost = self.p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis,
            basePosition=T_w[:3, 3].tolist(),
            baseOrientation=[q[1], q[2], q[3], q[0]])

    def draw_grasps(self, grasps, color=(1.0, 0.85, 0.1), width=2.0):
        from grasping.grasp_execute import _unit
        for g in grasps:
            half = _unit(g.axis) * g.width / 2
            a = _unit(g.approach)
            self.lines.append(self.p.addUserDebugLine(
                (g.center - half).tolist(), (g.center + half).tolist(),
                color, lineWidth=width))
            self.lines.append(self.p.addUserDebugLine(
                (g.center - a * 0.06).tolist(), (g.center - a * 0.01).tolist(),
                color, lineWidth=max(1.0, width - 1)))

    def mark(self, g, color, width=4.0):
        self.draw_grasps([g], color=color, width=width)

    def clear(self):
        for lid in self.lines:
            try:
                self.p.removeUserDebugItem(lid)
            except Exception:                              # noqa: BLE001
                pass
        self.lines = []
        if self.ghost is not None:
            try:
                self.p.removeBody(self.ghost)
            except Exception:                              # noqa: BLE001
                pass
            self.ghost = None


# ---------------------------------------------------------------------------
# Ein Lauf
# ---------------------------------------------------------------------------

def run_once(sim, args, per, loc_mod, run_idx: int, state: dict) -> bool:
    """Aufbauen -> Pipeline -> Greifen. True bei Lauf-Erfolg."""
    import trimesh
    from PIL import Image
    from grasping.sim_scene import TabletopSim, load_bop_scene  # noqa: F401
    from grasping.antipodal_grasp_sampler import (GripperConfig,
                                                  sample_antipodal_grasps,
                                                  transform_grasps)
    from grasping.grasp_execute import (PandaGrasper, feasible_grasps,
                                        reachable_order)
    from query_cloud import backproject_masked
    p = sim._p
    yaw = run_idx * args.yaw_step
    label = f"Lauf {run_idx + 1}/{args.runs} ({yaw:g} Grad)"

    # ---- Welt aufbauen (ohne Roboter; Wahrnehmung braucht freie Sicht) ----
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    sim.body = {}
    sim.robot = None
    if args.object:
        ds, oid = args.object.replace("/", ":").split(":")
        objs = [solo_object(ds, int(oid), yaw)]
        cam = fixed_camera()
        target_gt_idx = 0
    else:
        ds, scene_id, frame = (args.scene.split(":") + ["1"])[:3]
        objs, cam = load_bop_scene(ds, scene_id, int(frame))
        tgt_oid = state.get("target_oid")
        if tgt_oid is not None and yaw:
            for o in objs:
                if o.obj_id == tgt_oid:
                    o.T_world = rotate_about_z(o.T_world, yaw)
        target_gt_idx = None                    # kennt erst Schritt 1
    sim.build(objs, cam, target_gt_idx=target_gt_idx,
              target_id=state.get("target_oid"), with_robot=False,
              table_z=0.0 if args.object else None)
    sim.settle(120)
    sim.freeze_initial()
    obs = sim.render_rgbd()

    # ---- Schritt 1: Segmentierung per Prompt (immer echt) -----------------
    loc = loc_mod.localize(Image.fromarray(obs["rgb"]), args.prompt)
    if loc is None or loc.mask is None or not loc.mask.any():
        print(f"[sim] {label}: Schritt 1 fand nichts fuer {args.prompt!r} — "
              f"Lauf zaehlt als Fehlschlag")
        return False
    mask = np.asarray(loc.mask).astype(bool)
    print(f"[sim] {label}: Schritt 1 ok ({int(mask.sum())} px, "
          f"conf {loc.confidence:.2f})")

    # Ziel-Body identifizieren (fuer Reset-Buchhaltung im Szenen-Modus)
    seg = obs["seg"]
    ids, counts = np.unique(seg[mask], return_counts=True)
    ids = [i for i, _ in sorted(zip(ids, counts), key=lambda t: -t[1])
           if i in sim.body.values()]
    if ids and not args.object:
        tgt = next(o for o, b in sim.body.items() if b == ids[0])
        rebuild = state.get("target_oid") != tgt
        state["target_oid"] = tgt
        if rebuild:
            # Erst jetzt ist klar, WELCHES Objekt dynamisch sein muss —
            # Welt identisch neu aufbauen, Ziel als dynamischen Koerper.
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            sim.body = {}
            sim.robot = None
            sim.build(objs, cam, target_id=tgt, with_robot=False)
            sim.settle(120)
            sim.freeze_initial()

    # ---- Schritte 3-6: Retrieval + Fusion ---------------------------------
    top_id, cad_path, ranking = per.retrieve_proxy(obs["rgb"], mask)

    # ---- Schritt 7: geometrischer Check (optional) ------------------------
    if "geometry" not in args.skips:
        from dgedi_bridge import dgedi_rerank
        from pipeline.step7_geometry_reranking import geo_rerank
        q_cloud, _ = backproject_masked(obs["depth"], mask, cam.K)
        geo = dgedi_rerank(q_cloud, [oid_ for oid_, _ in ranking[:GEO_TOP_K]],
                           ransac_keypoints=6000, ransac_max_iter=10000,
                           use_icp=True)
        n_ok = sum(1 for v in (geo or {}).values() if v.get("ok"))
        if n_ok:
            ranking = geo_rerank(ranking, geo, GEO_TOP_K)
            if ranking[0][0] != top_id:
                print(f"[sim] {label}: Schritt 7 dreht Rang 1: "
                      f"{top_id} -> {ranking[0][0]}")
            top_id = ranking[0][0]
            cad_path, per._proxy_units_m = per.gallery.id_to_pose_mesh[top_id]
        else:
            print(f"[sim] {label}: Schritt 7 ohne Registrierung "
                  f"(dGeDi erreichbar? Galerie passend?) — Rangfolge bleibt")
    print(f"[sim] {label}: abgerufen = {top_id}")

    # ---- Schritt 8: FoundationPose + Griffe -------------------------------
    size_scale = (per.observed_scale(obs["depth"], mask, cam.K, cad_path)
                  if args.scale_fit else 1.0)
    if not 0.5 <= size_scale <= 2.0:
        # Ausreisser stammen praktisch immer aus einer schlechten Maske
        # (Schritt-1-Fehler blaeht die beobachtete Diagonale auf) — klemmen,
        # damit das Greifen nicht an einem absurd skalierten CAD plant.
        print(f"[sim] {label}: size_scale {size_scale:.2f} ausserhalb "
              f"[0.5, 2.0] — geklemmt (Maske pruefen / --no-scale-fit)")
        size_scale = float(np.clip(size_scale, 0.5, 2.0))
    pose, conf = per.estimate_pose(obs["rgb"], obs["depth"], mask, cad_path,
                                   cam.K, size_scale=size_scale)
    T_o2w = cam.T_world @ pose
    print(f"[sim] {label}: FoundationPose conf {conf:.1f}  "
          f"size_scale {size_scale:.2f}")

    sim._add_panda(objs)
    sim.freeze_initial()
    grasper = PandaGrasper(sim)
    grasper.reset()
    base_xy = p.getBasePositionAndOrientation(sim.robot)[0][:2]

    mesh = trimesh.load(cad_path, force="mesh")
    scale = (1.0 if per._proxy_units_m else 0.001) * size_scale
    if abs(scale - 1.0) > 1e-9:
        mesh.apply_scale(scale)
    gs = sample_antipodal_grasps(mesh, GripperConfig(), n_samples=800, top_k=40)
    feas = feasible_grasps(grasper, reachable_order(
        transform_grasps(gs, T_o2w), base_xy))
    print(f"[sim] {label}: {len(gs)} Kandidaten, {len(feas)} erreichbar")

    ov = Overlay(p)
    ov.show_ghost(cad_path, per._proxy_units_m, size_scale, T_o2w)
    ov.draw_grasps(feas[:20])

    ok, n_att = False, 0
    for i, g in enumerate(feas):
        if n_att >= args.n_tries:
            break
        sim.reset_objects()
        grasper.reset()
        sim.settle(30)
        ov.mark(g, color=(0.2, 0.55, 1.0))
        r = grasper.execute(g)
        if r.get("blocked"):
            ov.mark(g, color=(0.5, 0.5, 0.5), width=2.5)
            continue
        n_att += 1
        ov.mark(g, color=((0.15, 0.9, 0.3) if r["success"] else (0.95, 0.2, 0.2)))
        print(f"[sim] {label}: Versuch {n_att} -> "
              f"{'ERFOLG' if r['success'] else 'Fehlschlag'}")
        if r["success"]:
            ok = True
            break
    if args.gui:
        time.sleep(1.5)                                 # Endzustand kurz zeigen
    ov.clear()
    return ok


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", required=True,
                    help='Zielbeschreibung, z.B. "the red mug"')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--object", help="Solo-Modus: <ds>:<id>, z.B. ycbv:14")
    g.add_argument("--scene", help="Szenen-Modus: <ds>:<scene>[:frame], "
                                   "z.B. ycbv:48:1")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--yaw-step", type=float, default=36.0)
    ap.add_argument("--n-tries", type=int, default=5)
    ap.add_argument("--skip", default="",
                    help="Komma-Liste: geometry, clip, dino, shape")
    ap.add_argument("--no-scale-fit", dest="scale_fit", action="store_false",
                    help="abgerufenes CAD in Originalgroesse verwenden")
    ap.add_argument("--headless", action="store_true",
                    help="ohne GUI-Fenster (nur Konsole)")
    args = ap.parse_args()
    args.gui = not args.headless
    args.skips, weights = parse_skip(args.skip)

    os.environ.setdefault("GRASP_QUIET", "1")
    # eval_common/stage3_gallery arbeiten mit relativen ../object_database-
    # Pfaden — cwd wie bei den Eval-Treibern auf object_retrieval setzen
    # (alle Sim-Pfade hier sind absolut, das ist gefahrlos).
    os.chdir(os.path.join(_ROOT, "object_retrieval"))
    from pipeline.config import PipelineConfig
    from pipeline.step1_localization import ObjectLocalizer
    from grasping.sim_scene import TabletopSim

    print(f"[sim] Prompt: {args.prompt!r}  |  "
          f"{'Solo ' + args.object if args.object else 'Szene ' + args.scene}"
          f"  |  {args.runs} Laeufe x max. {args.n_tries} Griffe"
          + (f"  |  skip: {sorted(args.skips)}" if args.skips else ""))
    loc_mod = ObjectLocalizer(PipelineConfig())
    per = load_perception(weights)

    sim = TabletopSim(gui=args.gui).connect()
    p = sim._p
    if args.gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=55,
                                     cameraPitch=-28,
                                     cameraTargetPosition=[0.12, 0.0, 0.12])
        # Echtzeit-Tempo: jeder Physikschritt schlaeft 1/240 s
        real_step = p.stepSimulation

        def _paced(*a, **kw):
            out = real_step(*a, **kw)
            time.sleep(1.0 / 240.0)
            return out

        p.stepSimulation = _paced

    wins = 0
    state: dict = {}
    try:
        for run_idx in range(max(1, args.runs)):
            try:
                wins += int(run_once(sim, args, per, loc_mod, run_idx, state))
            except SystemExit:
                raise
            except Exception as exc:                       # noqa: BLE001
                print(f"[sim] Lauf {run_idx + 1}: FEHLER {exc} — "
                      f"zaehlt als Fehlschlag")
    finally:
        sim.disconnect()

    print("\n[sim] ==========================================")
    print(f"[sim] ERGEBNIS  {args.prompt!r}: Erfolgsrate {wins}/{args.runs}")
    print("[sim] ==========================================")


if __name__ == "__main__":
    main()
