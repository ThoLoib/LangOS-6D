#!/usr/bin/env python3
"""Stage-5 · 5.1 — Tabletop sim scene from a YCB-V capture (PyBullet).

Reconstructs a real **YCB-V** scene inside PyBullet: the same objects at their
ground-truth poses on a table, a Franka-Panda robot, and an RGB-D camera placed
at the YCB-V camera pose so the rendered view matches the real capture. Produces
the **RGB + depth + segmentation** that the perception stage (5.2) consumes.

Coordinate handling (explicit on purpose):
  * BOP poses are in **mm**, camera convention **OpenCV** (x-right, y-down, z-fwd).
  * `scene_gt.json`   gives  T_model→cam  (cam_R_m2c, cam_t_m2c).
  * `scene_camera.json` gives K, T_world→cam (cam_R_w2c, cam_t_w2c), depth_scale.
  * We rebuild in the BOP **world** frame (z-up): T_model→world = T_cam→world · T_model→cam,
    all lengths converted to metres, and place a ground plane at the objects' base.

CLI (run standalone):
    python -m grasping.sim_scene --scene 000048 --frame 1 --out /tmp/scene --headless
    python -m grasping.sim_scene --scene 000048 --gui            # interactive
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)

# Data locations (relative to the OSCAR repo root).
YCBV_TEST = os.path.join(_ROOT, "eval/datasets/ycbv/test")
YCBV_MESHES = os.path.join(_ROOT, "object_database/ycbv")     # obj_000001/textured_simple.obj


# ---------------------------------------------------------------------------
# Scene description parsed from the YCB-V ground truth
# ---------------------------------------------------------------------------
@dataclass
class SceneObject:
    obj_id: int
    T_world: np.ndarray                 # 4×4, model→world (metres)
    mesh_path: str


@dataclass
class SceneCamera:
    K: np.ndarray                       # 3×3 intrinsics (px)
    T_world: np.ndarray                 # 4×4 camera→world (metres)
    width: int
    height: int


def _pose(R, t_mm) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t_mm, float) / 1000.0        # mm → m
    return T


def load_ycbv_scene(scene_id: str, frame: int = 1
                    ) -> Tuple[List[SceneObject], SceneCamera]:
    """Parse one YCB-V frame's GT into world-frame objects + camera."""
    sdir = os.path.join(YCBV_TEST, scene_id)
    gt = json.load(open(os.path.join(sdir, "scene_gt.json")))
    cam = json.load(open(os.path.join(sdir, "scene_camera.json")))
    fk = str(frame)
    if fk not in gt:
        fk = sorted(gt, key=int)[0]
    c = cam[fk]
    T_w2c = _pose(c["cam_R_w2c"], c["cam_t_w2c"])
    T_c2w = np.linalg.inv(T_w2c)
    K = np.asarray(c["cam_K"], float).reshape(3, 3)

    objs: List[SceneObject] = []
    for o in gt[fk]:
        T_m2c = _pose(o["cam_R_m2c"], o["cam_t_m2c"])
        T_m2w = T_c2w @ T_m2c
        mesh = os.path.join(YCBV_MESHES, f"obj_{o['obj_id']:06d}", "textured_simple.obj")
        objs.append(SceneObject(o["obj_id"], T_m2w, mesh))

    # image size from the rgb (fallback to BOP default 640×480)
    W, H = 640, 480
    return objs, SceneCamera(K, T_c2w, W, H)


# ---------------------------------------------------------------------------
# PyBullet world
# ---------------------------------------------------------------------------
@dataclass
class TabletopSim:
    gui: bool = False
    _p: object = field(default=None, repr=False)
    body: Dict[int, int] = field(default_factory=dict)     # obj_id → bullet body id
    table_z: float = 0.0
    robot: Optional[int] = None
    cam: Optional[SceneCamera] = None

    def connect(self):
        import pybullet as p
        import pybullet_data
        self._p = p
        p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        return self

    # ---- world construction ------------------------------------------------
    def build(self, objects: List[SceneObject], camera: SceneCamera,
              target_id: Optional[int] = None, with_robot: bool = True):
        """Static clutter + a dynamic target + table + (optional) Panda."""
        p = self._p
        self.cam = camera
        # table plane at the objects' base (min z of the object origins minus a margin)
        base_z = min(o.T_world[2, 3] for o in objects)
        self.table_z = base_z - 0.005
        p.loadURDF("plane.urdf", [0, 0, self.table_z])

        for o in objects:
            dynamic = (o.obj_id == target_id)
            self.body[o.obj_id] = self._spawn_mesh(o, dynamic=dynamic)

        if with_robot:
            self._add_panda(objects)

    def _spawn_mesh(self, o: SceneObject, dynamic: bool) -> int:
        """Load a YCB mesh at its world pose. Static clutter uses concave
        collision; the dynamic target uses a convex hull (stable to grasp)."""
        p = self._p
        pos = o.T_world[:3, 3]
        quat = _mat2quat(o.T_world[:3, :3])
        # object_database YCB meshes are already in METRES (native ~0.1 m), so
        # meshScale = 1.0. (Only the BOP *poses* are mm and get ÷1000 in _pose.)
        S = [1.0, 1.0, 1.0]
        vis = p.createVisualShape(p.GEOM_MESH, fileName=o.mesh_path, meshScale=S)
        if dynamic:
            col = p.createCollisionShape(p.GEOM_MESH, fileName=o.mesh_path,
                                         meshScale=S)               # convex hull
            mass = 0.2
        else:
            col = p.createCollisionShape(p.GEOM_MESH, fileName=o.mesh_path,
                                         meshScale=S,
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
            mass = 0.0                                              # static clutter
        return p.createMultiBody(mass, col, vis, pos, quat)

    def _add_panda(self, objects: List[SceneObject]):
        """Place a Panda so the objects are within reach, base on the table."""
        p = self._p
        cx = np.mean([o.T_world[0, 3] for o in objects])
        cy = np.mean([o.T_world[1, 3] for o in objects])
        base = [cx - 0.55, cy, self.table_z]                       # behind the objects
        self.robot = p.loadURDF("franka_panda/panda.urdf", base, useFixedBase=True)

    def settle(self, steps: int = 240):
        for _ in range(steps):
            self._p.stepSimulation()

    # ---- sensing -----------------------------------------------------------
    def render_rgbd(self) -> Dict[str, np.ndarray]:
        """RGB (H,W,3 uint8), depth (H,W float m), seg (H,W int body id)."""
        p, cam = self._p, self.cam
        eye = cam.T_world[:3, 3]
        R = cam.T_world[:3, :3]
        target = eye + R @ np.array([0, 0, 1.0])          # OpenCV: look along +z
        up = R @ np.array([0, -1.0, 0])                   # OpenCV y is down
        view = p.computeViewMatrix(eye, target, up)
        fx, fy = cam.K[0, 0], cam.K[1, 1]
        near, far = 0.01, 3.0
        fov_y = 2 * np.degrees(np.arctan(cam.height / (2 * fy)))
        proj = p.computeProjectionMatrixFOV(fov_y, cam.width / cam.height, near, far)
        w, h, rgba, depth_buf, seg = p.getCameraImage(
            cam.width, cam.height, view, proj,
            renderer=p.ER_TINY_RENDERER)                  # CPU renderer (headless-safe)
        rgb = np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)
        depth_buf = np.reshape(depth_buf, (h, w))
        depth = far * near / (far - (far - near) * depth_buf)   # buffer → metric depth
        seg = np.reshape(seg, (h, w)).astype(np.int32)
        return {"rgb": rgb, "depth": depth, "seg": seg}

    def object_mask(self, obj_id: int, seg: np.ndarray) -> np.ndarray:
        """Boolean mask of one object in a seg image (by its bullet body id)."""
        return seg == self.body.get(obj_id, -999)

    def disconnect(self):
        if self._p is not None:
            self._p.disconnect()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mat2quat(R: np.ndarray):
    """3×3 rotation → PyBullet quaternion [x,y,z,w]."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()             # [x,y,z,w]


# YCB-V obj_id → readable name (for the CLI / prompts).
YCBV_NAMES = {
    1: "master chef can", 2: "cracker box", 3: "sugar box", 4: "tomato soup can",
    5: "mustard bottle", 6: "tuna fish can", 7: "pudding box", 8: "gelatin box",
    9: "potted meat can", 10: "banana", 11: "pitcher base", 12: "bleach cleanser",
    13: "bowl", 14: "mug", 15: "power drill", 16: "wood block", 17: "scissors",
    18: "large marker", 19: "large clamp", 20: "extra large clamp", 21: "foam brick",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 tabletop sim scene (5.1)")
    ap.add_argument("--scene", default="000048", help="YCB-V test scene id")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--target", type=int, default=None,
                    help="obj_id of the dynamic (graspable) target; default = none dynamic")
    ap.add_argument("--no-robot", action="store_true")
    ap.add_argument("--settle", type=int, default=0, help="physics settle steps")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default=None, help="save <out>_rgb.png/_depth.png/_seg.png")
    args = ap.parse_args()

    objs, cam = load_ycbv_scene(args.scene, args.frame)
    print(f"[scene] YCB-V {args.scene} frame {args.frame}: "
          f"{len(objs)} objects -> " +
          ", ".join(f"{o.obj_id}:{YCBV_NAMES.get(o.obj_id, '?')}" for o in objs))

    sim = TabletopSim(gui=args.gui).connect()
    sim.build(objs, cam, target_id=args.target, with_robot=not args.no_robot)
    if args.settle:
        sim.settle(args.settle)
    out = sim.render_rgbd()
    print(f"[scene] rendered RGB {out['rgb'].shape}, depth range "
          f"[{out['depth'][out['depth']<2.9].min():.2f},{out['depth'][out['depth']<2.9].max():.2f}] m, "
          f"{len(np.unique(out['seg']))} seg ids")

    if args.out:
        from PIL import Image
        Image.fromarray(out["rgb"]).save(f"{args.out}_rgb.png")
        d = out["depth"].copy(); d[d > 2.9] = 0
        Image.fromarray((255 * d / (d.max() + 1e-9)).astype(np.uint8)).save(f"{args.out}_depth.png")
        segv = (out["seg"] - out["seg"].min())
        Image.fromarray((255 * segv / (segv.max() + 1e-9)).astype(np.uint8)).save(f"{args.out}_seg.png")
        print(f"[scene] wrote {args.out}_{{rgb,depth,seg}}.png")
    if args.gui:
        print("[scene] GUI up — Ctrl-C to exit"); import time
        while True:
            sim._p.stepSimulation(); time.sleep(1 / 240)
    sim.disconnect()


if __name__ == "__main__":
    main()
