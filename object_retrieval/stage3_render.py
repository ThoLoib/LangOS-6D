"""stage3_render.py — headless depth renderer for BOP VSD.

bop_toolkit's ``vsd()`` needs a renderer exposing
``render_object(obj_id, R, t, fx, fy, cx, cy) -> {"depth": HxW mm}`` in the BOP
camera convention (camera at origin, +z forward, +y down; X_cam = R·X_model + t).

The vendored bop_toolkit renderers need a working vispy/glumpy GL *app* backend,
which will not initialise headless in the oscar container. pyrender + EGL does
(with NVIDIA_DRIVER_CAPABILITIES=all injecting libEGL_nvidia and the glvnd
loader). So we wrap pyrender to the same tiny interface. Everything is in mm —
models_eval is mm, t is mm, depth_test is mm — so no unit conversion here.

pyrender/OpenGL uses camera -z forward, +y up. We convert by posing the camera
with a 180° flip about x (diag(1,-1,-1,1)); pyrender's IntrinsicsCamera already
takes (fx,fy,cx,cy) in pixels. The __main__ self-test validates the convention
against a real GT frame (silhouette IoU vs mask, depth vs GT z).
"""

import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# vendored pyrender/PyOpenGL live under third_party/pylibs (see stage3_metrics)
_PYLIBS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "third_party", "pylibs")
_ap = os.path.abspath(_PYLIBS)
if os.path.isdir(_ap) and _ap not in sys.path:
    sys.path.insert(0, _ap)

import numpy as np
import trimesh


class PyrenderDepthRenderer:
    """Minimal BOP-convention depth renderer backed by pyrender+EGL (mm)."""

    # BOP camera (+z fwd, +y down) -> OpenGL camera (-z fwd, +y up)
    _CAM_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])

    def __init__(self, width, height, znear=1.0, zfar=1.0e5):
        import pyrender
        self._pyrender = pyrender
        self.width, self.height = int(width), int(height)
        self.znear, self.zfar = float(znear), float(zfar)
        self._r = pyrender.OffscreenRenderer(self.width, self.height)
        self._meshes = {}          # obj_id -> pyrender.Mesh

    def add_object(self, obj_id, model_path, units_m=False):
        m = trimesh.load(model_path, force="mesh")
        if units_m:                # proxies in metres -> mm (VSD works in mm)
            m.apply_scale(1000.0)
        self._meshes[obj_id] = self._pyrender.Mesh.from_trimesh(m, smooth=False)

    def render_object(self, obj_id, R, t, fx, fy, cx, cy):
        pr = self._pyrender
        scene = pr.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
        T = np.eye(4)
        T[:3, :3] = np.asarray(R, float)
        T[:3, 3] = np.asarray(t, float).reshape(3)
        scene.add(self._meshes[obj_id], pose=T)
        cam = pr.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                  znear=self.znear, zfar=self.zfar)
        scene.add(cam, pose=self._CAM_FLIP)
        depth = self._r.render(scene, flags=pr.RenderFlags.DEPTH_ONLY)
        return {"depth": depth}      # HxW float, mm, 0 where no surface


if __name__ == "__main__":
    # Convention self-test on the first YCB-V test frame: render the GT-posed
    # object and confirm it lands where the mask says and at the GT depth.
    import json
    from PIL import Image

    DS = "/app/eval/datasets/ycbv"
    sd = f"{DS}/test/000048"
    cam = json.load(open(f"{sd}/scene_camera.json"))["1"]
    gt = json.load(open(f"{sd}/scene_gt.json"))["1"][0]
    K = np.array(cam["cam_K"], float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    R = np.array(gt["cam_R_m2c"], float).reshape(3, 3)
    t = np.array(gt["cam_t_m2c"], float)
    obj = gt["obj_id"]
    mask = np.array(Image.open(f"{sd}/mask_visib/000001_000000.png")) > 0
    H, W = mask.shape

    r = PyrenderDepthRenderer(W, H)
    r.add_object(obj, f"{DS}/models_eval/obj_{obj:06d}.ply")
    depth = r.render_object(obj, R, t, fx, fy, cx, cy)["depth"]
    sil = depth > 0
    inter = np.logical_and(sil, mask).sum()
    union = np.logical_or(sil, mask).sum()
    iou = inter / max(union, 1)
    print(f"obj={obj} t_z(GT)={t[2]:.1f}mm  rendered z median={np.median(depth[sil]):.1f}mm")
    print(f"silhouette IoU vs GT mask = {iou:.3f}  "
          f"(rendered {int(sil.sum())} px, mask {int(mask.sum())} px)")
    print("CONVENTION OK" if iou > 0.5 else "CONVENTION WRONG — check cx/cy/flip")
