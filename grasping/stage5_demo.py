#!/usr/bin/env python3
"""Stage-5 · top-level demo — perceive-then-grasp on a YCB-V tabletop.

Ties the pieces together into the full open-set loop:

    RGB-D scene (YCB-V)  ─5.1─▶  segment target  ─5.2─▶  retrieve PROXY (G_proxy,
    exact CAD excluded)  ─5.2─▶  FoundationPose(proxy) → pose  ─E1─▶  antipodal
    grasps on the proxy  ─5.3─▶  Panda executes → success + GIF.

Everything is a thin wrapper over the existing modules; run it yourself:

    python -m grasping.stage5_demo --scene 000048 --prompt "the mustard bottle" --gif demo.gif
    python -m grasping.stage5_demo --scene 000048 --target 6 --no-pose        # skip FoundationPose
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)


def resolve_target(prompt: str, objs, names) -> int:
    """Map a free-text prompt to a scene object id by its YCB name."""
    p = prompt.lower()
    best, best_hits = None, 0
    for o in objs:
        nm = names.get(o.obj_id, "")
        hits = sum(1 for w in nm.split() if w in p)      # word overlap
        if nm and nm in p:
            hits += 3
        if hits > best_hits:
            best, best_hits = o.obj_id, hits
    if best is None:
        raise SystemExit(f"prompt {prompt!r} did not match any object in the scene: "
                         + ", ".join(names.get(o.obj_id, '?') for o in objs))
    return best


def main():
    ap = argparse.ArgumentParser(description="Stage-5 perceive-then-grasp demo")
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--frame", type=int, default=1)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help='free text, e.g. "the mustard bottle"')
    g.add_argument("--target", type=int, help="YCB obj_id directly")
    ap.add_argument("--uni3d", action="store_true", help="Uni3D shape arm")
    ap.add_argument("--no-pose", action="store_true",
                    help="skip FoundationPose; grasp using the GT object pose (mechanics test)")
    ap.add_argument("--n-tries", type=int, default=5)
    ap.add_argument("--gif", default=None)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    import trimesh
    from grasping.sim_scene import TabletopSim, load_ycbv_scene, YCBV_NAMES
    from grasping.antipodal_grasp_sampler import (sample_antipodal_grasps,
                                                  GripperConfig, transform_grasps)
    from grasping.grasp_execute import PandaGrasper, reachable_order

    # ---- 5.1 scene ----------------------------------------------------------
    objs, cam = load_ycbv_scene(args.scene, args.frame)
    target = args.target or resolve_target(args.prompt, objs, YCBV_NAMES)
    print(f"[demo] scene {args.scene} · target = {target}:{YCBV_NAMES.get(target,'?')}"
          + (f'  (prompt: "{args.prompt}")' if args.prompt else ""))
    sim = TabletopSim(gui=args.gui).connect()
    sim.build(objs, cam, target_id=target, with_robot=True)
    sim.settle(120)
    obs = sim.render_rgbd()
    mask = obs["seg"] == sim.body[target]
    print(f"[demo] segmented target: {int(mask.sum())} px")

    tgt_obj = next(o for o in objs if o.obj_id == target)
    if args.no_pose:
        # mechanics path: proxy = target's own mesh, pose = GT (isolates grasping)
        cad_path, T_obj2world = tgt_obj.mesh_path, tgt_obj.T_world
        print("[demo] --no-pose: using GT pose + own mesh (grasp mechanics only)")
    else:
        # ---- 5.2 perceive: proxy retrieval + FoundationPose -----------------
        from grasping.perceive import Perception
        per = Perception.load(use_uni3d=args.uni3d)
        proxy_id, cad_path, _ = per.retrieve_proxy(obs["rgb"], mask)
        print(f"[demo] retrieved PROXY {proxy_id}\n       cad = {cad_path}")
        pose, conf = per.estimate_pose(obs["rgb"], obs["depth"], mask, cad_path, cam.K)
        T_obj2world = pose            # proxy pose in the CAMERA frame == world here
        print(f"[demo] FoundationPose conf {conf:.3f}")

    # ---- E1 grasps on the (proxy) mesh, placed by the estimated pose --------
    mesh = trimesh.load(cad_path, force="mesh")
    grasps = sample_antipodal_grasps(mesh, GripperConfig(), n_samples=800, top_k=40)
    grasps_w = transform_grasps(grasps, T_obj2world)
    base = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
    grasps_w = reachable_order(grasps_w, base)
    print(f"[demo] {len(grasps_w)} grasp candidates on the proxy")

    # ---- 5.3 execute --------------------------------------------------------
    frames = []
    cap = (lambda: frames.append(sim.render_rgbd()["rgb"])) if args.gif else None
    grasper = PandaGrasper(sim)
    result = {"success": False}
    for i, gr in enumerate(grasps_w[:args.n_tries]):
        result = grasper.execute(gr, cap=cap)
        print(f"[demo]   try {i}: q={gr.quality:.3f} -> {result}")
        if result["success"]:
            break

    print(f"\n[demo] ===== {'GRASP SUCCESS' if result['success'] else 'grasp failed'} "
          f"({result.get('lift_cm','?')} cm lift) =====")
    if args.gif and frames:
        try:
            import imageio
            imageio.mimsave(args.gif, frames, fps=20)
            print(f"[demo] wrote {len(frames)} frames -> {args.gif}")
        except Exception as e:
            print(f"[demo] gif save failed ({e})")
    sim.disconnect()


if __name__ == "__main__":
    main()
