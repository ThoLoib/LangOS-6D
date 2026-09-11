#!/usr/bin/env python3
"""Ablation: does sizing the retrieved proxy to the observed object ("scale-fit")
improve the faithful pipeline  OSCAR+ retrieval -> top-1 proxy -> FoundationPose
-> grasp?  Proxies are arbitrary real-world CADs (a 16 cm GSO plant pot may be
retrieved for a 9 cm mug), so FP and the grasps inherit that size mismatch.

Loads the gallery ONCE, then for each YCB-V target compares scale-fit OFF vs ON:
  * FoundationPose PLACEMENT error (proxy centroid vs the real object)
  * # REACHABLE parallel-jaw grasps
  * grasp SUCCESS (shake-verified lift)

    cd /app/object_retrieval && PYTHONPATH=/app python3 -m grasping.experiment_scale_ablation \
        --targets 14,6,19,20 --exec
"""
import argparse
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "object_retrieval"))


def grasp_pipeline(per, objs, cam, obs, mask, tgt, scale_fit, execute, n_tries):
    """Run top-1 proxy -> FP -> grasp for one target at one scale setting."""
    import trimesh
    from grasping.sim_scene import TabletopSim
    from grasping.grasp_execute import (PandaGrasper, reachable_order,
                                        feasible_grasps)
    from grasping.antipodal_grasp_sampler import (sample_antipodal_grasps,
                                                  GripperConfig, transform_grasps)

    proxy_id, cad_path, _ = per.retrieve_proxy(obs["rgb"], mask)
    size_scale = (per.observed_scale(obs["depth"], mask, cam.K, cad_path)
                  if scale_fit else 1.0)
    gscale = per._units() * size_scale
    pose, conf = per.estimate_pose(obs["rgb"], obs["depth"], mask, cad_path, cam.K,
                                   size_scale=size_scale)
    T = cam.T_world @ pose

    proxy = trimesh.load(cad_path, force="mesh")
    real = trimesh.load(tgt.mesh_path, force="mesh")
    proxy_c = (T @ np.append(proxy.centroid * gscale, 1.0))[:3]
    real_c = (tgt.T_world @ np.append(real.centroid, 1.0))[:3]
    placement = float(np.linalg.norm(proxy_c - real_c) * 1000)

    sim = TabletopSim().connect()
    sim.build(objs, cam, target_id=tgt.obj_id, with_robot=True)
    sim.settle(60)
    gmesh = proxy.copy()
    if abs(gscale - 1.0) > 1e-6:
        gmesh.apply_scale(gscale)
    grasps = sample_antipodal_grasps(gmesh, GripperConfig(), n_samples=800, top_k=40)
    base = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
    gw = reachable_order(transform_grasps(grasps, T), base)
    grasper = PandaGrasper(sim); grasper.home()
    feas = feasible_grasps(grasper, gw)

    success, lift = False, 0.0
    if execute and feas:
        attempts = 0
        for g in feas:
            if attempts >= n_tries:
                break
            sim.reset_objects(); grasper.home(); sim.settle(30)
            r = grasper.execute(g)
            if r.get("blocked"):
                continue
            attempts += 1
            if r["success"]:
                success, lift = True, r["lift_cm"]
                break
    sim.disconnect()
    return dict(proxy=proxy_id.split("/")[-1][:22], conf=conf, scale=size_scale,
                placement=placement, reachable=len(feas), success=success, lift=lift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--targets", default="14,6,19,20")
    ap.add_argument("--proxy", default="gso")
    ap.add_argument("--exec", action="store_true", help="also execute the grasps")
    ap.add_argument("--n-tries", type=int, default=8)
    args = ap.parse_args()
    targets = [int(t) for t in args.targets.split(",") if t.strip()]

    from grasping.sim_scene import TabletopSim, load_ycbv_scene, YCBV_NAMES
    from grasping.perceive import Perception

    objs, cam = load_ycbv_scene(args.scene, args.frame)
    # render the scene once (no robot); per-target masks come from the seg image
    sim = TabletopSim().connect()
    sim.build(objs, cam, target_id=None, with_robot=False)
    obs = sim.render_rgbd()
    body = dict(sim.body)
    sim.disconnect()

    per = Perception.load(proxy_ds=[args.proxy])          # gallery loaded ONCE

    print("\n================ SCALE-FIT ABLATION (top-1 proxy) ================")
    hdr = f"{'target':<16}{'scale_fit':<10}{'proxy':<24}{'FPconf':>7}{'scale':>7}{'place_mm':>9}{'reach':>7}{'success':>9}"
    print(hdr); print("-" * len(hdr))
    for t in targets:
        tgt = next((o for o in objs if o.obj_id == t), None)
        if tgt is None:
            continue
        mask = obs["seg"] == body[t]
        for sf in (False, True):
            r = None
            for _ in range(2):                             # FP can be flaky on small meshes
                try:
                    r = grasp_pipeline(per, objs, cam, obs, mask, tgt, sf,
                                       args.exec, args.n_tries)
                    break
                except Exception as exc:
                    err = str(exc).splitlines()[-1][:38]
            if r is None:
                print(f"{str(t)+':'+YCBV_NAMES.get(t,'?'):<16}{str(sf):<10}"
                      f"{'FP/grasp error: '+err:<47}")
                continue
            succ = (f"{'YES' if r['success'] else 'no':>4} {r['lift']:>4.1f}cm"
                    if args.exec else "  n/a")
            print(f"{str(t)+':'+YCBV_NAMES.get(t,'?'):<16}{str(sf):<10}"
                  f"{r['proxy']:<24}{r['conf']:>7.0f}{r['scale']:>7.2f}"
                  f"{r['placement']:>9.0f}{r['reachable']:>7}{succ:>9}")
    print("=" * len(hdr))


if __name__ == "__main__":
    main()
