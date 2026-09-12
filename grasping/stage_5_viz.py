#!/usr/bin/env python3
"""Visualisierung einer Stage-5-Serie — ausfuehrbar per Terminal.

Beobachter-Kamera (Roboter + Tisch + Objekt im Bild), dazu:
  grasp.mp4     DAS VIDEO: alle Laeufe der Serie (Default 10, Objekt je Lauf
                um 36 Grad weitergedreht) hintereinander. Je Lauf: kurze
                Pose-Einblendung (das CAD unter Test als gruene Silhouette in
                der FoundationPose-Pose, die geplanten Griffe gelb), dann die
                Greifausfuehrung in 3x Zeitlupe. Geist und Griffe bleiben
                waehrend der ANFAHRT eingeblendet (aktiver Versuch blau,
                gescheitert rot, Erfolg gruen) und verschwinden im
                Zugriffsmoment. Abschluss: Erfolgsraten-Karte (X/N).
  overlay.png   hochaufgeloestes Standbild von Lauf 1 (Pose + Griffe samt
                Ausfuehrungs-Markierung) — als Abbildung fuer die Arbeit.

Bedienung wie der freie Modus von stage_5:

    # Proxy-Serie: auf dem Proxy geplant, das echte Objekt gegriffen
    docker compose run --rm oscar python3 -m grasping.stage_5_viz \
        --object ycbv:14 --proxy housecat6d/cup-red_heart

    # gt-Serie (Proxy leer oder gleich dem Objekt selbst): eigenes CAD
    docker compose run --rm oscar python3 -m grasping.stage_5_viz --object ycbv:14

  --runs 10       Anzahl Laeufe; --yaw-step 36 Grad je Lauf; --yaw Startwinkel
  --out DIR       Zielordner (Default _s5_out/stage_5_viz/<objekt>_<bedingung>/)
  --scene/--im/--mass wie im freien Modus (Frame kommt sonst aus dem
                  Experiment-Plan, damit das Video die Tabellen reproduziert).
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "object_retrieval"))

import numpy as np                                                  # noqa: E402

from grasping.experiment_proxy_grasp import Ctx, PROTOCOL, _fp_pose  # noqa: E402
from grasping.sim_scene import TabletopSim, _as_metre_obj            # noqa: E402
from grasping.stage_5 import _custom_tr, canonical_object, resolve_cad  # noqa: E402

W, H = 1600, 1200          # Standbild
GW, GH = 1280, 960         # Video-Frames (durch 16 teilbar fuer den Encoder)
FARBE_KANDIDAT = ((255, 210, 40), 2)     # geplant (gelb)
FARBE_AKTIV = ((80, 160, 255), 4)        # gerade ausgefuehrt (blau)
FARBE_FEHL = ((235, 60, 60), 4)          # ausgefuehrt, gescheitert (rot)
FARBE_ERFOLG = ((60, 220, 90), 5)        # ausgefuehrt, Erfolg (gruen)


def observer(p, target_xyz, robot_xy, dist=0.88, pitch=-22.0, w=W, h=H):
    """Beobachter-Kamera: seitliche Dreiviertel-Ansicht, Roboterarm UND Objekt."""
    t, r = np.asarray(target_xyz, float), np.asarray(robot_xy, float)
    v = t[:2] - r
    yaw = np.degrees(np.arctan2(v[1], v[0])) + 78.0      # schraeg hinter dem Arm
    look = [0.5 * (t[0] + r[0]), 0.5 * (t[1] + r[1]), 0.26]
    view = p.computeViewMatrixFromYawPitchRoll(look, dist, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(55, w / h, 0.05, 4.0)
    return view, proj


def snap(p, view, proj, w=W, h=H):
    out = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.reshape(out[2], (h, w, 4))[:, :, :3].astype(np.uint8)
    seg = np.reshape(out[4], (h, w))
    return rgb, seg


def to_px(view, proj, pts, w, h):
    """Weltpunkte (N,3) -> Pixel (N,2)."""
    V = np.array(view).reshape(4, 4, order="F")
    P = np.array(proj).reshape(4, 4, order="F")
    q = np.c_[np.atleast_2d(pts), np.ones(len(np.atleast_2d(pts)))] @ (P @ V).T
    q = q[:, :3] / np.clip(q[:, 3:4], 1e-9, None)
    return np.c_[(q[:, 0] + 1) * 0.5 * w, (1 - q[:, 1]) * 0.5 * h]


def draw_grasps(img, view, proj, feas, marks):
    """Griffe als Fingerlinie + Anfahrpfeil zeichnen (in Bildaufloesung!).
    marks: index -> (Farbe, Breite); alles andere gelber Kandidat."""
    from PIL import Image, ImageDraw
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, g in enumerate(feas):
        col, wd = marks.get(i, FARBE_KANDIDAT)
        half = g.axis / max(np.linalg.norm(g.axis), 1e-9) * g.width / 2
        f1, f2 = to_px(view, proj, np.stack([g.center - half, g.center + half]), w, h)
        a0, a1 = to_px(view, proj, np.stack(
            [g.center - g.approach * 0.07, g.center - g.approach * 0.015]), w, h)
        d.line([tuple(f1), tuple(f2)], fill=col, width=wd)
        d.line([tuple(a0), tuple(a1)], fill=col, width=max(1, wd - 1))
        d.ellipse([a1[0] - wd, a1[1] - wd, a1[0] + wd, a1[1] + wd], fill=col)
    return np.asarray(im)


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:                                  # noqa: BLE001
        return ImageFont.load_default()


def stamp(img, text):
    """Beschriftungszeile oben ins Bild schreiben (skaliert mit Aufloesung)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    strip = max(18, img.shape[0] // 30)
    d.rectangle([0, 0, img.shape[1], strip], fill=(0, 0, 0))
    d.text((8, strip // 6), text, fill=(255, 255, 255), font=_font(strip - strip // 3))
    return np.asarray(im)


def karte(text_lines, w=GW, h=GH):
    """Abschlusskarte (schwarz, zentrierter Text)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (10, 10, 10))
    d = ImageDraw.Draw(im)
    f = _font(h // 16)
    y = h // 2 - (h // 12) * len(text_lines) // 2 - h // 24
    for ln in text_lines:
        tw = d.textlength(ln, font=f)
        d.text(((w - tw) / 2, y), ln, fill=(240, 240, 240), font=f)
        y += h // 12
    return np.asarray(im)


def ghost_tint(img, seg, gb):
    """Geist-Pixel gruen einfaerben (verdeckungs-korrekt via Seg-Maske)."""
    m = seg == gb
    if m.any():
        sh = img[m].astype(float).mean(axis=1, keepdims=True) / 255.0
        img[m] = (np.array([60, 235, 110]) * (0.35 + 0.65 * sh)).astype(np.uint8)
    return img


def ghost_overlay(p, view, proj, w, h, gb, feas, marks):
    """Aktuelle Szene + Geist (muss sichtbar stehen) + Griffe."""
    img, seg = snap(p, view, proj, w, h)
    return draw_grasps(ghost_tint(img, seg, gb), view, proj, feas, marks)


def run_once(ctx, args, tr, cam, tgt_base, cad_path, units_m, yaw, label,
             frames, want_hires):
    """Ein Lauf der Serie: aufstellen, FP-Pose, Griffe, Ausfuehrung — mit
    Frame-Capture an jedem 4. Simulationsschritt (240 Hz / 4 bei 20 fps = 3x Zeitlupe).
    Rueckgabe (erfolg: bool, hires_overlay | None)."""
    from grasping.antipodal_grasp_sampler import transform_grasps
    from grasping.grasp_execute import (PandaGrasper, feasible_grasps,
                                        reachable_order)
    tgt, _ = canonical_object(ctx, tgt_base, yaw)
    sim = TabletopSim().connect()
    hires = None
    try:
        p = sim._p
        sim.build([tgt], cam, target_gt_idx=tr["gt_idx"], with_robot=False,
                  table_z=0.0)
        sim.settle(PROTOCOL["physics"]["settle_steps"])
        sim.freeze_initial()
        T_true = sim.target_pose()

        rd = sim.render_rgbd()
        mask = sim.object_mask(tgt.obj_id, rd["seg"]).astype(np.uint8)
        if mask.sum() < 200:
            print(f"[viz] {label}: Solo-Maske leer — Lauf uebersprungen")
            return False, None
        try:
            R, t, _conf = _fp_pose(cad_path, rd["rgb"], rd["depth"], mask,
                                   cam.K, units_m, 1.0)
        except Exception as exc:                              # noqa: BLE001
            print(f"[viz] {label}: FP-Fehler ({str(exc).splitlines()[-1][:60]})")
            return False, None
        T_m2c = np.eye(4)
        T_m2c[:3, :3], T_m2c[:3, 3] = R, np.asarray(t, float) / 1000.0
        T_m2w = cam.T_world @ T_m2c

        sim._add_panda([tgt])
        sim.freeze_initial()
        gs = ctx.grasp_candidates(cad_path, units_m, 1.0)
        grasper = PandaGrasper(sim)
        grasper.reset()
        base_xy = p.getBasePositionAndOrientation(sim.robot)[0][:2]
        feas = feasible_grasps(grasper, reachable_order(
            transform_grasps(gs, T_m2w), base_xy),
            tol_mm=PROTOCOL["executor"]["reach_tol_mm"])[:20]

        view, proj = observer(p, T_true[:3, 3], base_xy)
        gview, gproj = observer(p, T_true[:3, 3], base_xy, w=GW, h=GH)
        quat = __import__("trimesh").transformations.quaternion_from_matrix(T_m2w)
        ghost_obj = _as_metre_obj(cad_path, 1.0 if units_m else 0.001)
        # Geist als dauerhaften (rein visuellen) Body anlegen: sichtbar waehrend
        # Intro UND Anfahrt, versteckt ab dem Zugriff. Leicht aufgeblasen: liegt
        # das CAD unter Test VOLLSTAENDIG im Zielobjekt (kleineres Substitut),
        # waere die verdeckungs-korrekte Silhouette sonst unsichtbar.
        gvis = p.createVisualShape(p.GEOM_MESH, fileName=ghost_obj,
                                   meshScale=[1.05] * 3,
                                   rgbaColor=(0.15, 0.95, 0.35, 1.0))
        gpos = T_m2w[:3, 3].tolist()
        gorn = [quat[1], quat[2], quat[3], quat[0]]
        gb = p.createMultiBody(baseMass=0, baseVisualShapeIndex=gvis,
                               basePosition=gpos, baseOrientation=gorn)

        def ghost_show(on):
            p.resetBasePositionAndOrientation(gb, gpos if on else [0, 0, -5], gorn)

        marks, cur = {}, [None]
        hide, setg_n = [False], [0]
        real_setg = grasper.set_gripper

        def _setg(width, force=40):
            # 1. Aufruf je Versuch = Oeffnen (Geist+Pfeile sichtbar); ab dem 2.
            # beginnt das Greifen -> beide ausblenden, freie Sicht auf den Griff
            if setg_n[0] > 0 and not hide[0]:
                hide[0] = True
                ghost_show(False)
            setg_n[0] += 1
            return real_setg(width, force)

        grasper.set_gripper = _setg
        intro = ghost_overlay(p, gview, gproj, GW, GH, gb, feas, marks)
        frames.extend([stamp(intro, f"{label} | Pose + geplante Griffe")] * 40)

        real_step = p.stepSimulation
        nstep = [0]

        def dyn():
            d = dict(marks)
            if cur[0] is not None and cur[0] not in d:
                d[cur[0]] = FARBE_AKTIV
            return d

        def _step(*a, **kw):
            out = real_step(*a, **kw)
            nstep[0] += 1
            if nstep[0] % 4 == 0:
                img, seg = snap(p, gview, gproj, GW, GH)
                if not hide[0]:
                    img = draw_grasps(ghost_tint(img, seg, gb), gview, gproj,
                                      feas, dyn())
                frames.append(stamp(img, label))
            return out

        succ_i, n_att = None, 0
        for i, g in enumerate(feas):
            if n_att >= args.n_tries:
                break
            sim.reset_objects()                       # ohne Aufnahme (Teleport)
            grasper.reset()
            sim.settle(30)
            hide[0], setg_n[0] = False, 0             # Geist+Pfeile wieder an
            ghost_show(True)
            p.stepSimulation = _step
            cur[0] = i
            r = grasper.execute(g)
            cur[0] = None
            p.stepSimulation = real_step
            if r.get("blocked"):
                continue
            n_att += 1
            marks[i] = FARBE_FEHL
            if r["success"]:
                marks[i] = FARBE_ERFOLG
                succ_i = i
                break
        if frames:
            frames.extend([frames[-1]] * 20)          # 1 s Endzustand halten
        print(f"[viz] {label}: "
              f"{'ERFOLG (Versuch ' + str(n_att) + ')' if succ_i is not None else 'kein Erfolg (' + str(n_att) + ' Versuche)'}")

        if want_hires:
            sim.reset_objects()
            grasper.reset()
            ghost_show(True)
            hires = ghost_overlay(p, view, proj, W, H, gb, feas, marks)
        return succ_i is not None, hires
    finally:
        sim.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--object", required=True, help="<ds>:<id>, z.B. ycbv:14")
    ap.add_argument("--proxy", default="", help="CAD, auf dem geplant wird "
                    "(leer/eigenes Objekt = gt-Lauf); Pool- oder BOP-Id")
    ap.add_argument("--runs", type=int, default=10, help="Laeufe der Serie")
    ap.add_argument("--yaw", type=float, default=0.0, help="Startwinkel")
    ap.add_argument("--yaw-step", type=float, default=36.0, help="Grad je Lauf")
    ap.add_argument("--scene", default="")
    ap.add_argument("--im", type=int, default=-1)
    ap.add_argument("--mass", type=float, default=0.0)
    ap.add_argument("--n-tries", type=int, default=PROTOCOL["executor"]["n_tries"])
    ap.add_argument("--out", default="")
    # Felder, die Ctx erwartet:
    ap.add_argument("--world", default="auto")
    ap.add_argument("--pose-source", default="fp", choices=["fp"])
    args = ap.parse_args()
    args.exec, args.verbose = True, False

    ctx = Ctx(args)
    os.environ.setdefault("GRASP_QUIET", "1")
    tr = _custom_tr(args)
    self_id = f"{tr['dataset']}/obj_{tr['obj_id']:06d}"
    cond = "gt" if args.proxy in ("", self_id) else "proxy"
    out = args.out or os.path.join(_ROOT, "_s5_out", "stage_5_viz",
                                   f"{tr['dataset']}{tr['obj_id']}_{cond}")
    os.makedirs(out, exist_ok=True)
    if args.mass > 0:
        from grasping import sim_scene
        sim_scene.OBJECT_MASS_KG[(tr["dataset"], tr["obj_id"])] = args.mass

    cad_id, cad_path, units_m = resolve_cad(ctx, tr, cond)
    print(f"[viz] {tr['name']} | Bedingung {cond} | CAD unter Test: {cad_id} | "
          f"{args.runs} Laeufe")

    fr, objs, cam, winfo = ctx.scene(tr["dataset"], tr["scene"], tr["im"])
    tgt_base = objs[tr["gt_idx"]]

    frames, wins = [], 0
    png = os.path.join(out, "overlay.png")
    for ridx in range(max(1, args.runs)):
        yaw = args.yaw + ridx * args.yaw_step
        label = f"{tr['name']} auf {cad_id.split('/')[-1][:24]} | Lauf {ridx + 1}/{args.runs} ({yaw:g} Grad)"
        ok, hires = run_once(ctx, args, tr, cam, tgt_base, cad_path, units_m,
                             yaw, label, frames, want_hires=(ridx == 0))
        wins += int(ok)
        if hires is not None:
            from PIL import Image
            Image.fromarray(stamp(hires, label)).save(png)

    frames.extend([karte([f"{tr['name']}  |  geplant auf: {cad_id}",
                          f"Erfolgsrate: {wins}/{args.runs}"])] * 60)
    import imageio
    try:
        vid = os.path.join(out, "grasp.mp4")
        imageio.mimsave(vid, frames, fps=20, macro_block_size=16)
    except Exception:                                  # ohne ffmpeg: GIF
        vid = os.path.join(out, "grasp.gif")
        imageio.mimsave(vid, frames, fps=20, loop=0)
    print(f"[viz] ERGEBNIS {tr['name']} | "
          f"{cad_id if cond == 'proxy' else 'eigenes CAD'}: "
          f"Erfolgsrate {wins}/{args.runs}")
    print(f"[viz] geschrieben: {vid} ({len(frames)} Frames) + {png}")


if __name__ == "__main__":
    main()
