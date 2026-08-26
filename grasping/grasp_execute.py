#!/usr/bin/env python3
"""Stage-5 · 5.3 — Grasp execution with a Franka Panda (PyBullet).

Take grasp candidates (from `antipodal_grasp_sampler`, in the object frame),
transform them by the object's pose into the world, and drive the Panda to
execute the best **reachable** one: pre-grasp → approach → close → lift →
success check. CPU-only (no GPU) so it runs alongside anything else.

Frame convention: a `Grasp` has x = closing axis, z = approach. The Panda's
grasp-target link is aligned so its **+z = approach** (into the object) and the
fingers close along the grasp's closing axis.

CLI (standalone test — grasps the target's OWN mesh at GT pose, to validate the
mechanics without the perception stage):
    python -m grasping.grasp_execute --scene 000048 --target 6 --gif /tmp/grasp.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)

from grasping.antipodal_grasp_sampler import Grasp, _unit          # noqa: E402


# Panda joint layout (pybullet franka_panda/panda.urdf)
PANDA_ARM = list(range(7))            # 7 revolute arm joints
PANDA_FINGERS = [9, 10]               # prismatic finger joints (0=closed, 0.04=open)
PANDA_EE = 11                         # panda_grasptarget link (TCP)
FINGER_OPEN = 0.04


class PandaGrasper:
    def __init__(self, sim):
        self.sim = sim
        self.p = sim._p
        self.robot = sim.robot
        assert self.robot is not None, "scene built without a robot"

    # ---- low-level control -------------------------------------------------
    def _drive_arm(self, q: List[float]):
        for j, qi in zip(PANDA_ARM, q):
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL,
                                         qi, force=200)

    def set_gripper(self, width: float, force: float = 40):
        w = float(np.clip(width / 2, 0.0, FINGER_OPEN))
        for j in PANDA_FINGERS:
            self.p.setJointMotorControl2(self.robot, j, self.p.POSITION_CONTROL,
                                         w, force=force)

    def move_to(self, pos, orn_quat, steps: int = 240, cap=None):
        """IK to an EE pose, drive there, step the sim (rendering frames if cap)."""
        q = self.p.calculateInverseKinematics(self.robot, PANDA_EE, pos, orn_quat,
                                              maxNumIterations=100,
                                              residualThreshold=1e-4)
        self._drive_arm(q[:7])
        for i in range(steps):
            self.p.stepSimulation()
            if cap is not None and i % 12 == 0:
                cap()

    # ---- one grasp attempt -------------------------------------------------
    def execute(self, grasp_world: Grasp, cap=None) -> dict:
        """Pre-grasp → approach → close → lift; return success + diagnostics."""
        p = self.p
        z = _unit(grasp_world.approach)               # approach (into object)
        orn = _hand_quat(grasp_world)
        center = grasp_world.center
        pregrasp = center - z * 0.12                  # 12 cm back along approach

        self.set_gripper(FINGER_OPEN * 2)             # open
        self.move_to(pregrasp, orn, 180, cap)
        self.move_to(center - z * 0.01, orn, 160, cap)   # approach to contacts
        self.set_gripper(grasp_world.width * 0.6, force=60)  # close (slightly < width)
        for _ in range(80):
            p.stepSimulation()
            if cap:
                cap()
        # lift
        target_body = None
        z0 = None
        for oid, b in self.sim.body.items():
            # the dynamic (grasped) object is the one with mass>0
            if p.getDynamicsInfo(b, -1)[0] > 0:
                target_body = b
                z0 = p.getBasePositionAndOrientation(b)[0][2]
        self.move_to(center - z * 0.01 + np.array([0, 0, 0.15]), orn, 240, cap)  # lift 15 cm
        lifted = 0.0
        if target_body is not None:
            z1 = p.getBasePositionAndOrientation(target_body)[0][2]
            lifted = z1 - z0
        success = lifted > 0.05                       # object rose ≥ 5 cm
        return {"success": bool(success), "lift_cm": round(100 * lifted, 1)}


# ---------------------------------------------------------------------------
def _hand_quat(g: Grasp):
    """Panda grasp-target orientation. The Franka fingers translate along the
    hand's **local y-axis**, so hand-y := the grasp closing axis, and
    hand-z := approach (into the object). Returns a pybullet quat [x,y,z,w]."""
    from scipy.spatial.transform import Rotation
    z = _unit(g.approach)             # hand +z = approach
    y = _unit(g.axis)                 # hand +y = closing axis (finger travel)
    x = _unit(np.cross(y, z))
    y = np.cross(z, x)                # re-orthogonalise
    R = np.column_stack([x, y, z])
    return Rotation.from_matrix(R).as_quat()


def grasps_to_world(grasps: List[Grasp], T_obj2world: np.ndarray) -> List[Grasp]:
    """Convenience re-export (transform object-frame grasps to world)."""
    from grasping.antipodal_grasp_sampler import transform_grasps
    return transform_grasps(grasps, T_obj2world)


def reachable_order(grasps: List[Grasp], base_xy) -> List[Grasp]:
    """Heuristic: prefer top-down-ish grasps near the robot (better IK odds)."""
    def key(g):
        downness = -_unit(g.approach)[2]          # approach pointing downward = good
        near = -np.linalg.norm(g.center[:2] - np.asarray(base_xy))
        return 0.7 * g.quality + 0.2 * downness + 0.1 * near
    return sorted(grasps, key=key, reverse=True)


# ---------------------------------------------------------------------------
# CLI — standalone mechanics test on the target's own mesh + GT pose.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 grasp execution (5.3)")
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--n-tries", type=int, default=5)
    ap.add_argument("--gif", default=None, help="save a GIF of the attempt")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    import trimesh
    from grasping.sim_scene import TabletopSim, load_ycbv_scene, YCBV_NAMES
    from grasping.antipodal_grasp_sampler import (sample_antipodal_grasps,
                                                  GripperConfig, transform_grasps)

    objs, cam = load_ycbv_scene(args.scene, args.frame)
    tgt = next(o for o in objs if o.obj_id == args.target)
    sim = TabletopSim(gui=args.gui).connect()
    sim.build(objs, cam, target_id=args.target, with_robot=True)
    sim.settle(120)

    # sample grasps on the target's OWN mesh (object frame) — stand-in for the
    # proxy in this mechanics test — then place them by the object's world pose.
    mesh = trimesh.load(tgt.mesh_path, force="mesh")
    grasps = sample_antipodal_grasps(mesh, GripperConfig(), n_samples=800, top_k=40)
    grasps_w = transform_grasps(grasps, tgt.T_world)
    base = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
    grasps_w = reachable_order(grasps_w, base)
    print(f"[grasp-exec] target {args.target}:{YCBV_NAMES.get(args.target,'?')} — "
          f"{len(grasps_w)} candidates, trying top {args.n_tries}")

    frames = []
    cap = (lambda: frames.append(sim.render_rgbd()["rgb"])) if args.gif else None
    grasper = PandaGrasper(sim)
    result = {"success": False}
    for i, g in enumerate(grasps_w[:args.n_tries]):
        print(f"  try {i}: q={g.quality:.3f} width={g.width*1000:.0f}mm")
        result = grasper.execute(g, cap=cap)
        print(f"    -> {result}")
        if result["success"]:
            break
        sim.build  # (a full reset per try is omitted for brevity)

    print(f"[grasp-exec] RESULT: {result}")
    if args.gif and frames:
        try:
            import imageio
            imageio.mimsave(args.gif, frames, fps=20)
            print(f"[grasp-exec] wrote {len(frames)} frames -> {args.gif}")
        except Exception as e:
            print(f"[grasp-exec] gif save failed ({e})")
    sim.disconnect()


if __name__ == "__main__":
    main()
