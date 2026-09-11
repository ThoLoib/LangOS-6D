#!/usr/bin/env python3
"""Stage-5 · Proxy-grasp study — is a retrieved proxy CAD good enough to grasp with?

Protocol (frozen before the first run): docs/STAGE5_PROTOCOL.md. In short:

For each object of the frozen Top-20 set (`proxy_grasp_cases.py`) and each of
its frozen instances (`proxy_grasp_instances.json` = the Stage-3b instances in
which the fixed proxy was retrieved at rank 1) the BOP frame is rebuilt in
PyBullet [R14] — annotated objects at their GT poses, the target dynamic, a
Franka Panda — and the SAME grasp planner and executor run per condition:

    gt        target's own CAD posed by FoundationPose [R4]  — the reference
    proxy     the retrieved proxy CAD posed by FoundationPose — the pipeline
    gt_pose   target's own CAD at the TRUE (settled) sim pose — planner ceiling (control)
    random    a hash-drawn random gallery CAD, FoundationPose — chance baseline (control)
    proxy_scaled  proxy sized to the observed depth cloud    — size ablation (optional)

FoundationPose sees the REAL sensor RGB-D and the BOP GT visible mask, exactly as
Stage 3b did (`eval_bop_pose.estimate_pose`); pose quality is scored with the
Stage-3 D_sym (`stage3_metrics.d_sym`). A trial succeeds when one of at most
`--n-tries` executed grasp candidates lifts the target, holds it for 1 s and
survives a shake test [R3] (`grasp_execute`). Every attempt starts from a full
reset. Reporting: success rates per condition, dataset and object, paired as
Δ plus win split — no intervals (docs/AGREEMENTS.md 2026-09-03).

Run from the repo root of the HOST; the script wraps itself into the oscar
container (`docker compose run`, as the host user), starts FoundationPose when a
run needs it and restarts it when it stops answering. `python3
repro_experiment.py --stage 5 ...` is the same entry. Reference list: grasping/README.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "object_retrieval"))   # eval_bop_pose, stage3_metrics

from grasping.proxy_grasp_cases import (CASES, EXCLUDED, ALL_CASES, select_cases,   # noqa: E402
                                        proxy_mesh, proxy_pool, random_proxy)

IN_CONTAINER = os.path.exists("/.dockerenv")
CONDITIONS = ("gt", "proxy", "gt_pose", "random", "proxy_scaled")
DEFAULT_CONDITIONS = "gt,proxy"          # the measurement; the rest are opt-in controls
INSTANCES_JSON = os.path.join(_THIS, "proxy_grasp_instances.json")
OUT_DIR = os.path.join(_ROOT, "_s5_out", "proxy_grasp")
GRASP_CACHE = os.path.join(_ROOT, "_grasp_cache")
EXAMPLES = """examples:
  python3 grasping/experiment_proxy_grasp.py --check       data, FoundationPose, table-plane self-check
  python3 grasping/experiment_proxy_grasp.py --plan        print + freeze the trial plan (plan.json)
  python3 grasping/experiment_proxy_grasp.py               run gt vs proxy, rank 1-10 (resumes from the CSV)
  python3 grasping/experiment_proxy_grasp.py --report      REPORT.md from the CSV
  python3 grasping/experiment_proxy_grasp.py --conditions gt_pose,gt,proxy      + planner ceiling
  python3 grasping/experiment_proxy_grasp.py --ranks 1-20 --per-object 8         larger sample
  python3 grasping/experiment_proxy_grasp.py --cases tless22 --per-object 1 --verbose --csv _s5_out/one/trials.csv
  python3 grasping/experiment_proxy_grasp.py --pose-source stage3                archived Stage-3 poses, no GPU
"""

# Every value that shapes an outcome, frozen here and written to the manifest.
PROTOCOL = dict(
    sampler=dict(n_samples=800, top_k=40, friction_mu=0.5, n_approach=4, seed=0,
                 gripper_min_width_m=0.005, gripper_max_width_m=0.08,
                 method="antipodal contact pairs inside the friction cone [R1, R2]"),
    executor=dict(n_tries=5, pregrasp_m=0.12, lift_m=0.15, rise_min_m=0.05,
                  hold_steps=240, shake_amp_m=0.05, in_hand_m=0.15,
                  close_force_n=(20, 120), reach_tol_mm=30, blocked_mm=30,
                  success="rose >= 5 cm AND held 1 s AND survived ±5 cm shakes [R3]; "
                          "trial = any of <= n_tries attempts, full reset before each"),
    physics=dict(engine="PyBullet [R14]", timestep_s=1 / 240, gravity=-9.81, target_mass_kg=0.2,
                 target_friction=1.6, finger_friction=1.5, settle_steps=60,
                 collision="V-HACD [R13] for the target / concave static clutter"),
    robot=dict(arm="Franka Panda (pybullet_data franka_panda/panda.urdf)", pedestal_m=0.35,
               placement="on the camera's side: 0.55 m from the object centroid along the "
                         "camera viewing direction projected onto the table"),
    world=dict(frame="auto: BOP extrinsics where present (YCB-V, T-LESS), else the table plane "
                     "fitted to the real depth (LM-O)",
               table_plane="RANSAC [R12], objects masked out, ±0.5 m around the object depths, "
                           "thr 6 mm, candidate planes must carry every object 0–40 cm above them",
               table_z="0 = lowest object vertex (bop) / the fitted plane; lowered to the "
                       "target's own lowest vertex if that is below"),
    pose=dict(method="FoundationPose [R4] via eval_bop_pose.estimate_pose", fp_refine_iter=5,
              input="real RGB-D + BOP GT mask_visib (Stage 3b)",
              d_sym="stage3_metrics.d_sym, 10 000 surface samples, seed 0"),
    sampling=dict(min_visib=0.5, per_object=6,
                  rule="round-robin over scenes, evenly spaced frames within a scene"),
)


# ===========================================================================
# 1. plan: which instances run (deterministic, no outcome, no RNG)
# ===========================================================================
def load_instances(path: str = INSTANCES_JSON) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"missing {path} — run grasping/build_grasp_instances.py first")
    return json.load(open(path))


def plan_trials(cases: List[dict], db: dict, per_object: int, min_visib: float) -> List[dict]:
    """Per case: keep instances with visib >= min_visib, give `per_object` picks
    round-robin to the scenes (sorted by id), and within a scene take frames at
    evenly spaced positions of its frame-sorted list."""
    plan = []
    for c in cases:
        entry = db["cases"].get(c["key"])
        if not entry:
            print(f"[plan] {c['key']}: no instance list")
            continue
        pool = [i for i in entry["instances"] if i.get("visib") is None or i["visib"] >= min_visib]
        by_scene: Dict[int, list] = defaultdict(list)
        for i in pool:
            by_scene[i["scene"]].append(i)
        scenes = sorted(by_scene)
        for s in scenes:
            by_scene[s].sort(key=lambda i: (i["im"], i["gt_idx"]))
        want = min(per_object, len(pool))
        quota: Counter = Counter()
        k = 0
        while sum(quota.values()) < want and scenes:
            s = scenes[k % len(scenes)]
            if quota[s] < len(by_scene[s]):
                quota[s] += 1
            k += 1
        for s in scenes:
            lst, q = by_scene[s], quota[s]
            idx = sorted({min(int((j + 0.5) / q * len(lst)), len(lst) - 1) for j in range(q)})
            idx += [j for j in range(len(lst)) if j not in idx][:q - len(idx)]   # tiny lists
            for i in sorted(idx):
                plan.append(dict(case=c["key"], rank=c["rank"], tier=c["tier"],
                                 dataset=c["dataset"], obj_id=c["obj_id"], name=c["name"],
                                 proxy=c["proxy"], pool=len(pool), n_top1=entry["n_top1"],
                                 **lst[i]))
    return plan


def print_plan(cases, plan, conds, args):
    print(f"== plan: {len(cases)} objects, {len(plan)} instances, conditions {conds} "
          f"(per-object {args.per_object}, min visib {args.min_visib}) ==")
    cur = None
    for tr in plan:
        if tr["case"] != cur:
            cur = tr["case"]
            print(f"  #{tr['rank'] or '–'} {tr['case']:<8} {tr['name']:<15} proxy {tr['proxy']}"
                  f"   pool {tr['pool']}/{tr['n_top1']}")
        print(f"      s{tr['scene']:02d} im {tr['im']:04d} gt{tr['gt_idx']} visib {tr.get('visib')}")


# ===========================================================================
# 2. per-run caches: scenes, meshes, Stage-3 surface samples, grasp candidates
# ===========================================================================
class Ctx:
    def __init__(self, args):
        self.args = args
        self.scenes: dict = {}
        self.meshes: dict = {}
        self.pts: dict = {}
        self.grasps: dict = {}
        self.pool = proxy_pool()
        self.eval_fallback: set = set()

    def scene(self, ds: str, scene: int, im: int):
        """(frame, objects, camera, world-check info) for one BOP frame."""
        key = (ds, scene, im)
        if key not in self.scenes:
            from grasping.sim_scene import load_bop_frame, scene_from_frame
            fr = load_bop_frame(ds, scene, im)
            self.scenes[key] = (fr,) + scene_from_frame(fr, world=self.args.world)
        return self.scenes[key]

    def mesh_m(self, path: str, units_m: bool, extra: float = 1.0):
        """trimesh in METRES (native units × extra)."""
        key = (path, units_m, round(extra, 6))
        if key not in self.meshes:
            import trimesh
            m = trimesh.load(path, force="mesh")
            s = (1.0 if units_m else 0.001) * extra
            if abs(s - 1.0) > 1e-9:
                m.apply_scale(s)
            self.meshes[key] = m
        return self.meshes[key]

    def pts_mm(self, path: str, units_m: bool, extra: float = 1.0):
        """Stage-3 surface sample (mm) of a CAD — `stage3_metrics.sample_surface_mm`."""
        key = (path, units_m, round(extra, 6))
        if key not in self.pts:
            from stage3_metrics import sample_surface_mm
            p = sample_surface_mm(path, units_m=units_m)
            self.pts[key] = p * extra if abs(extra - 1.0) > 1e-9 else p
        return self.pts[key]

    def target_cad(self, ds: str, obj_id: int) -> Tuple[str, bool]:
        """The CAD Stage 3 posed the target with (BOP models_eval, mm) — or the
        sim mesh when that split is absent on this machine (reported)."""
        from grasping.sim_scene import eval_mesh_path
        path, units_m = eval_mesh_path(ds, obj_id)
        if units_m:
            self.eval_fallback.add(ds)
        return path, units_m

    def grasp_candidates(self, path: str, units_m: bool, extra: float = 1.0):
        """Object-frame grasps of a CAD (independent of the pose), cached on
        disk per CAD + sampler protocol."""
        P = PROTOCOL["sampler"]
        key = hashlib.md5(f"{path}|{units_m}|{extra:.6f}|{P}".encode()).hexdigest()[:16]
        if key in self.grasps:
            return self.grasps[key]
        import numpy as np
        from grasping.antipodal_grasp_sampler import Grasp, GripperConfig, sample_antipodal_grasps
        os.makedirs(GRASP_CACHE, exist_ok=True)
        f = os.path.join(GRASP_CACHE, key + ".json")
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            gs = [Grasp(center=np.array(g["center"]), axis=np.array(g["axis"]),
                        approach=np.array(g["approach"]), width=g["width"], quality=g["quality"],
                        contacts=(np.array(g["contacts"][0]), np.array(g["contacts"][1])))
                  for g in json.load(open(f))["grasps"]]
        else:
            t0 = time.time()
            gs = sample_antipodal_grasps(
                self.mesh_m(path, units_m, extra),
                GripperConfig(max_width=P["gripper_max_width_m"], min_width=P["gripper_min_width_m"]),
                n_samples=P["n_samples"], friction_mu=P["friction_mu"],
                n_approach=P["n_approach"], top_k=P["top_k"], seed=P["seed"])
            tmp = f"{f}.{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"cad": path, "units_m": units_m, "extra": extra, "protocol": P,
                           "seconds": round(time.time() - t0, 1),
                           "grasps": [g.to_dict() for g in gs]}, fh)
            os.replace(tmp, f)
        self.grasps[key] = gs
        return gs


# ===========================================================================
# 3. one trial = (instance, condition): CAD -> pose -> grasps -> execution
# ===========================================================================
def _cad_under_test(ctx: Ctx, tr: dict, cond: str) -> Tuple[str, Optional[str], bool]:
    """(cad id, mesh path or None, units_m) for a condition."""
    if cond in ("gt", "gt_pose"):
        path, units_m = ctx.target_cad(tr["dataset"], tr["obj_id"])
        return "gt", path, units_m
    cid = random_proxy(tr["dataset"], tr["obj_id"], ctx.pool) if cond == "random" else tr["proxy"]
    path, units_m = proxy_mesh(cid)
    return cid, path, units_m


def _fp_pose(cad_path, rgb, depth_m, mask, K, units_m: bool, extra: float):
    """FoundationPose [R4] in the Stage-3 configuration -> (R 3x3, t mm, conf).
    `extra` != 1 only for the proxy_scaled ablation (Stage 3 never scales)."""
    from eval_bop_pose import estimate_pose, FP_URL
    if abs(extra - 1.0) < 1e-9:
        return estimate_pose(cad_path, rgb, depth_m, mask, K, mesh_units_m=units_m,
                             refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    from pipeline.foundationpose_bridge import call_foundationpose
    pose, conf = call_foundationpose(FP_URL, rgb=rgb, depth=depth_m, mask=mask, K=K,
                                     cad_path=cad_path, scale=(1.0 if units_m else 1e-3) * extra,
                                     refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    return pose[:3, :3], pose[:3, 3] * 1000.0, float(conf)


def _pose_step(ctx: Ctx, tr: dict, cond: str, fr, cad_path: str, units_m: bool, row: dict):
    """Camera-frame pose of the CAD under test + its Stage-3 score. Returns
    (R, t_mm, extra) or None when the pose service failed (row says why)."""
    import numpy as np
    from stage3_metrics import d_sym
    rgb, depth_m = fr.rgb(), fr.depth_m()
    mask = fr.mask_visib(tr["gt_idx"]).astype(np.uint8)
    extra = 1.0
    if cond == "proxy_scaled":                    # size the proxy to the observed depth cloud
        from grasping.perceive import observed_diag
        d_obs = observed_diag(depth_m, mask.astype(bool), fr.K)
        d_cad = float(np.linalg.norm(ctx.mesh_m(cad_path, units_m).extents))
        if d_obs and d_cad > 1e-6:
            extra = d_obs / d_cad
    stored = None
    if ctx.args.pose_source == "stage3" and cond in ("gt", "proxy"):
        stored = tr.get("fp_gt" if cond == "gt" else "fp_proxy")
    if stored is not None:                        # archived Stage-3 pose of this very instance
        R = np.asarray(stored["R"], float).reshape(3, 3)
        t = np.asarray(stored["t"], float).reshape(3)
        conf, row["pose_source"] = stored.get("conf"), "stage3"
    else:
        try:
            R, t, conf = _fp_pose(cad_path, rgb, depth_m, mask, fr.K, units_m, extra)
            row["pose_source"] = "fp"
        except Exception as exc:
            row.update(fp_fail=str(exc).splitlines()[-1][:80], fail_reason="fp_error")
            return None
    row["fp_conf"] = round(float(conf), 3) if conf is not None else ""
    # Stage-3 score: D_sym between the GT-posed target and the posed CAD (camera frame, mm)
    R_gt = np.asarray(fr.gt[tr["gt_idx"]]["cam_R_m2c"], float).reshape(3, 3)
    t_gt = np.asarray(fr.gt[tr["gt_idx"]]["cam_t_m2c"], float).reshape(3)
    tpath, tunits = ctx.target_cad(tr["dataset"], tr["obj_id"])
    dsr = d_sym(ctx.pts_mm(tpath, tunits), R_gt, t_gt,
                ctx.pts_mm(cad_path, units_m, extra), R, t, tr.get("diameter") or 0.0)
    row.update(dsym_mm=round(dsr["d_sym"], 2),
               dsym_norm=round(dsr["d_sym_norm"], 4) if dsr["d_sym_norm"] is not None else "",
               f05=round(dsr["fscore"]["0.05"]["f"], 4))
    return R, t, extra


def _grasp_step(ctx: Ctx, sim, cad_path: str, units_m: bool, extra: float, T_m2w, row: dict):
    """Plan on the posed CAD, execute on the real target: up to n_tries attempts,
    full reset before each, stop at the first success."""
    import numpy as np
    from grasping.grasp_execute import PandaGrasper, reachable_order, feasible_grasps
    from grasping.antipodal_grasp_sampler import transform_grasps, _unit
    gs = ctx.grasp_candidates(cad_path, units_m, extra)
    grasper = PandaGrasper(sim)
    grasper.reset()
    base_xy = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
    feas = feasible_grasps(grasper, reachable_order(transform_grasps(gs, T_m2w), base_xy),
                           tol_mm=PROTOCOL["executor"]["reach_tol_mm"])
    row.update(n_cand=len(gs), n_reach=len(feas))
    seq, lifts = [], []
    for g in (feas if ctx.args.exec else []):
        if row["n_att"] >= ctx.args.n_tries:
            break
        sim.reset_objects()
        grasper.reset()
        sim.settle(30)
        r = grasper.execute(g)
        if ctx.args.verbose:
            print(f"        attempt {len(seq) + 1:>2}: q={g.quality:.2f} w={g.width * 1000:.0f}mm "
                  f"approach_z={_unit(g.approach)[2]:+.2f} -> {r}", flush=True)
        if r.get("blocked"):                        # approach corridor occupied: not an attempt
            row["n_blocked"] += 1
            seq.append("B")
            continue
        row["n_att"] += 1
        if r["success"]:
            row["n_succ"] += 1
            row["first_succ"] = int(row["n_att"] == 1)
            lifts.append(r["lift_cm"])
            seq.append("S")
            break
        seq.append("F")
    row.update(att_seq=",".join(seq), succ=int(row["n_succ"] > 0),
               lift_cm=round(float(np.mean(lifts)), 1) if lifts else 0.0)
    if not row["succ"]:
        row["fail_reason"] = ("no_candidates" if not gs else "unreachable" if not feas
                              else "all_blocked" if row["n_att"] == 0 else "grasp_failed")


def run_trial(ctx: Ctx, tr: dict, cond: str) -> dict:
    import numpy as np
    from grasping.sim_scene import TabletopSim
    t0 = time.time()
    ds, gt_idx = tr["dataset"], tr["gt_idx"]
    fr, objs, cam, winfo = ctx.scene(ds, tr["scene"], tr["im"])
    tgt = objs[gt_idx]
    assert tgt.obj_id == tr["obj_id"], f"gt_idx {gt_idx} is obj {tgt.obj_id}, not {tr['obj_id']}"
    row = dict(case=tr["case"], rank=tr["rank"], tier=tr["tier"], dataset=ds, obj_id=tr["obj_id"],
               name=tr["name"], scene=tr["scene"], im=tr["im"], gt_idx=gt_idx, visib=tr.get("visib"),
               condition=cond, cad="", cad_units_m="", pose_source="", fp_conf="", fp_fail="",
               dsym_mm="", dsym_norm="", f05="", place_mm="", settle_mm="",
               world=winfo.get("world"), plane_inlier=winfo.get("plane_inlier", ""),
               plane_angle_deg=winfo.get("plane_angle_deg", ""),
               bottom_gap_mm=winfo.get("bottom_gap_mm", ""),
               n_cand=0, n_reach=0, n_blocked=0, n_att=0, n_succ=0, first_succ=0, succ=0,
               lift_cm=0.0, att_seq="", fail_reason="", runtime_s=0.0,
               ts=_dt.datetime.now().isoformat(timespec="seconds"))

    cad_id, cad_path, units_m = _cad_under_test(ctx, tr, cond)
    row.update(cad=cad_id, cad_units_m=int(units_m))
    if not cad_path:
        row.update(fail_reason="cad_missing", runtime_s=round(time.time() - t0, 1))
        return row

    # -- world: annotated objects at their GT poses, only the target dynamic --
    os.environ.setdefault("GRASP_QUIET", "1")
    sim = TabletopSim().connect()
    # the table is z = 0 in every world; lower it only if THIS target's annotated
    # pose intersects it (a penetrating dynamic body gets kicked out otherwise)
    sim.build(objs, cam, target_gt_idx=gt_idx, with_robot=True,
              table_z=min(0.0, winfo["bottom_z"].get(gt_idx, 0.0) - 0.001))
    sim.settle(PROTOCOL["physics"]["settle_steps"])
    row["settle_mm"] = round(sim.target_displacement_mm(tgt.T_world), 1)
    sim.freeze_initial()
    T_true = sim.target_pose()                          # settled model→world pose
    try:
        # -- pose of the CAD under test ----------------------------------------
        if cond == "gt_pose":
            extra, T_m2w, row["pose_source"] = 1.0, T_true, "sim_true"
        else:
            res = _pose_step(ctx, tr, cond, fr, cad_path, units_m, row)
            if res is None:
                return row
            R, t, extra = res
            T_m2c = np.eye(4)
            T_m2c[:3, :3], T_m2c[:3, 3] = R, t / 1000.0
            T_m2w = cam.T_world @ T_m2c
        mesh_m = ctx.mesh_m(cad_path, units_m, extra)
        c_cad = T_m2w[:3, :3] @ mesh_m.centroid + T_m2w[:3, 3]
        c_tgt = T_true[:3, :3] @ ctx.mesh_m(tgt.mesh_path, True).centroid + T_true[:3, 3]
        row["place_mm"] = round(float(np.linalg.norm(c_cad - c_tgt) * 1000), 1)
        # -- grasps --------------------------------------------------------------
        _grasp_step(ctx, sim, cad_path, units_m, extra, T_m2w, row)
    finally:
        sim.disconnect()
        row["runtime_s"] = round(time.time() - t0, 1)
    return row


# ===========================================================================
# 4. results: CSV per trial (append + fsync, resume), manifest
# ===========================================================================
def _key(r: dict) -> tuple:
    return (r["case"], int(r["scene"]), int(r["im"]), int(r["gt_idx"]), r["condition"])


def _append_csv(path: str, row: dict):
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if fresh:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())          # a hard crash otherwise leaves a torn line


def _load_csv(path: str) -> List[dict]:
    """Valid rows, the LAST one per (instance, condition): a re-run appends."""
    if not path or not os.path.exists(path):
        return []
    known = {c["key"] for c in ALL_CASES}
    last: dict = {}
    with open(path) as fh:
        for d in csv.DictReader(fh):
            try:
                assert d.get("case", "").strip() in known and d.get("condition") in CONDITIONS
                last[_key(d)] = d
                int(d["n_att"]); int(d["succ"])
            except (AssertionError, TypeError, ValueError, KeyError):
                print(f"[resume] skipping malformed row: {str(d)[:70]}")
    return list(last.values())


def _git_state() -> dict:
    st = {"rev": "unknown", "dirty": "unknown"}
    try:
        head = open(os.path.join(_ROOT, ".git", "HEAD")).read().strip()
        ref = os.path.join(_ROOT, ".git", head[5:]) if head.startswith("ref: ") else None
        st["rev"] = (open(ref).read().strip() if ref and os.path.isfile(ref) else head)[:12]
        if ref:
            st["branch"] = head[5:].split("/")[-1]
        # the repo is bind-mounted into the container under another owner, hence safe.directory
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", _ROOT, "status", "--porcelain",
                              "--", "grasping", "object_retrieval", "pipeline"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            st["dirty"] = bool(out.stdout.strip())
    except Exception:
        pass
    return st


def _write_manifest(args, plan, conds) -> dict:
    import platform
    versions = {}
    for mod in ("numpy", "scipy", "trimesh", "pybullet"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", None) or str(m.getAPIVersion())
        except Exception as exc:
            versions[mod] = f"unavailable ({exc.__class__.__name__})"
    man = dict(ts=_dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
               python=platform.python_version(), versions=versions, git=_git_state(),
               args=vars(args), conditions=conds, protocol=PROTOCOL,
               instances_sha=hashlib.sha256(open(INSTANCES_JSON, "rb").read()).hexdigest()[:16],
               n_planned=len(plan), cases=sorted({p["case"] for p in plan}),
               threads=os.environ.get("OMP_NUM_THREADS"))
    path = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hist = []
    if os.path.isfile(path):
        try:
            hist = json.load(open(path)).get("runs", [])
        except ValueError:
            pass
    with open(path + ".tmp", "w") as fh:
        json.dump({"runs": hist + [man]}, fh, indent=1)
    os.replace(path + ".tmp", path)
    return man


# ===========================================================================
# 5. --check: data, FoundationPose, world self-check
# ===========================================================================
def check(args, plan) -> int:
    from grasping.sim_scene import BOP_DATASETS, bop_mesh_path, eval_mesh_path
    missing, seen = 0, set()
    print("== data ==")
    for tr in plan:
        ds = tr["dataset"]
        sdir = os.path.join(BOP_DATASETS[ds]["test"], f"{tr['scene']:06d}")
        for rel in (f"rgb/{tr['im']:06d}.png", f"depth/{tr['im']:06d}.png",
                    f"mask_visib/{tr['im']:06d}_{tr['gt_idx']:06d}.png",
                    "scene_gt.json", "scene_camera.json"):
            p = os.path.join(sdir, rel)
            if not os.path.isfile(p) and not (rel.endswith(".png") and os.path.isfile(p[:-4] + ".jpg")):
                print(f"  MISSING {p}")
                missing += 1
        for cid in (tr["proxy"], random_proxy(ds, tr["obj_id"], proxy_pool())):
            if cid not in seen:
                seen.add(cid)
                if not proxy_mesh(cid)[0]:
                    print(f"  MISSING proxy CAD for {cid}")
                    missing += 1
        if (ds, tr["obj_id"]) not in seen:
            seen.add((ds, tr["obj_id"]))
            try:
                bop_mesh_path(ds, tr["obj_id"])
                ep, em = eval_mesh_path(ds, tr["obj_id"])
                if em:
                    print(f"  note: {ds} obj {tr['obj_id']}: models_eval absent -> "
                          f"pose/score with {os.path.relpath(ep, _ROOT)}")
            except Exception as exc:
                print(f"  MISSING target CAD {ds} obj {tr['obj_id']}: {exc}")
                missing += 1
    print(f"  {len(plan)} planned instances, {missing} missing file(s)")

    print("== FoundationPose ==")
    url = os.environ.get("FP_URL", "http://foundationpose:5050")
    h, _, port = url.split("//")[-1].split("/")[0].partition(":")
    try:
        socket.create_connection((h, int(port or 80)), timeout=3).close()
        print(f"  {url}: reachable")
    except OSError as exc:
        print(f"  {url}: NOT reachable ({exc}) — start it: docker compose up -d foundationpose")

    print("== world self-check (table plane from the real depth vs BOP up-vector) ==")
    from grasping.sim_scene import load_bop_frame, scene_from_frame
    done = set()
    for tr in plan:
        if tr["dataset"] in done:
            continue
        done.add(tr["dataset"])
        _, _, info = scene_from_frame(load_bop_frame(tr["dataset"], tr["scene"], tr["im"]),
                                      world=args.world)
        print(f"  {tr['dataset']} {tr['scene']:06d}/{tr['im']:06d}: world={info['world']} "
              f"inliers={info.get('plane_inlier')} angle_vs_BOP_up={info.get('plane_angle_deg', 'n/a')}° "
              f"bottom_gap={info['bottom_gap_mm']} mm")
    return 0 if not missing else 1


# ===========================================================================
# 6. --report: tables from the CSV (Δ + win split, no intervals)
# ===========================================================================
def _pct(a, b):
    return f"{100.0 * a / b:.0f}%" if b else "–"


def _med(vals):
    v = [float(x) for x in vals if x not in ("", None)]
    return f"{statistics.median(v):.1f}" if v else "–"


def _headline_table(rows, conds, title):
    out = [f"**{title}**", "",
           "| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | "
           "reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rc = [r for r in rows if r["condition"] == c]
        if not rc:
            continue
        n, att = len(rc), sum(int(r["n_att"]) for r in rc)
        s5, s1, sa = (sum(int(r[k]) for r in rc) for k in ("succ", "first_succ", "n_succ"))
        out.append(f"| {c} | {len({r['case'] for r in rc})} | {n} | {s5}/{n} ({_pct(s5, n)}) | "
                   f"{s1}/{n} ({_pct(s1, n)}) | {sa}/{att} ({_pct(sa, att)}) | "
                   f"{_med(r['n_cand'] for r in rc)} | {_med(r['n_reach'] for r in rc)} | "
                   f"{sum(int(r['n_blocked']) for r in rc)} | "
                   f"{sum(r['fail_reason'] == 'no_candidates' for r in rc)} | "
                   f"{sum(r['fail_reason'] == 'fp_error' for r in rc)} | "
                   f"{_med(r['dsym_mm'] for r in rc)} | {_med(r['place_mm'] for r in rc)} | "
                   f"{_med(r['runtime_s'] for r in rc)} |")
    return out


def _paired_table(rows, pairs, title):
    by: dict = defaultdict(dict)
    for r in rows:
        by[_key(r)[:4]][r["condition"]] = int(r["succ"])
    out = [f"**{title}** (same instance under both conditions; Δ in percentage points)", "",
           "| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |",
           "|---|---|---|---|---|---|"]
    for a, b in pairs:
        ks = [k for k, d in by.items() if a in d and b in d]
        if not ks:
            continue
        sa, sb = sum(by[k][a] for k in ks), sum(by[k][b] for k in ks)
        ao = sum(1 for k in ks if by[k][a] and not by[k][b])
        bo = sum(1 for k in ks if by[k][b] and not by[k][a])
        both = sum(1 for k in ks if by[k][a] and by[k][b])
        out.append(f"| {a} → {b} | {len(ks)} | {_pct(sa, len(ks))} | {_pct(sb, len(ks))} | "
                   f"{100.0 * (sb - sa) / len(ks):+.0f} pp | {ao} : {bo} : {both} : {len(ks) - ao - bo - both} |")
    return out


def report(rows: List[dict], csv_path: str, conds: List[str]) -> str:
    rows = [r for r in rows if r["condition"] in conds]
    counted = [r for r in rows if str(r["tier"]) in ("1", "2")]
    exhibits = [r for r in rows if str(r["tier"]) == "x"]
    extra = [r for r in rows if str(r["tier"]) == "-"]
    runs = []
    man = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "manifest.json")
    if os.path.isfile(man):
        try:
            runs = json.load(open(man)).get("runs", [])
        except ValueError:
            pass
    L = ["# Stage 5 — Proxy-grasp study (generated report)", "",
         f"CSV: `{os.path.relpath(csv_path, _ROOT)}` · {len(rows)} trials · "
         f"{len({r['case'] for r in rows})} objects · {len({_key(r)[:4] for r in rows})} instances · "
         f"pose source: {sorted({r['pose_source'] for r in rows if r['pose_source']})} · "
         f"world: {sorted({r['world'] for r in rows})}",
         f"Runs: {len(runs)}" + (f", last {runs[-1]['ts']} on {runs[-1]['host']} "
                                 f"(git {runs[-1]['git'].get('rev')}, dirty={runs[-1]['git'].get('dirty')})"
                                 if runs else ""),
         "", "success@k = trial succeeded within ≤ n_tries executed attempts (headline); "
             "success@1 = the first executed candidate succeeded; attempts S/N = per-attempt rate. "
             "No confidence intervals by agreement (2026-09-03): Δ plus the per-instance win split.", "",
         "## 1. Success rate", ""]
    for ds in ("ycbv", "tless", "lmo"):                     # per dataset first, never only pooled
        rd = [r for r in counted if r["dataset"] == ds]
        if rd:
            L += _headline_table(rd, conds, f"{ds.upper()} ({len({r['case'] for r in rd})} objects)") + [""]
    L += _headline_table(counted, conds, "ALL datasets (pooled — composition differs per dataset)") + [""]
    pairs = [(a, b) for a, b in (("gt", "proxy"), ("gt_pose", "gt"), ("gt_pose", "proxy"),
                                 ("proxy", "random"), ("proxy", "proxy_scaled"))
             if a in conds and b in conds]
    L += ["## 2. Paired comparisons", ""] + _paired_table(counted, pairs, "All counted objects") + [""]
    for ds in ("ycbv", "tless", "lmo"):
        rd = [r for r in counted if r["dataset"] == ds]
        if rd:
            L += _paired_table(rd, pairs, ds.upper()) + [""]
    L += ["## 3. Per object", "",
          "| rank | object | proxy (fixed) | inst | " + " | ".join(f"{c} succ@k | {c} D_sym" for c in conds) + " |",
          "|" + "---|" * (4 + 2 * len(conds))]
    for c in [c for c in CASES if c["tier"] in (1, 2)] + [c for c in CASES if c["tier"] == "x"] + EXCLUDED:
        rc = [r for r in rows if r["case"] == c["key"]]
        if not rc:
            continue
        cells = []
        for cond in conds:
            rr = [r for r in rc if r["condition"] == cond]
            cells += [f"{sum(int(r['succ']) for r in rr)}/{len(rr)}" if rr else "–",
                      _med(r["dsym_mm"] for r in rr) if rr else "–"]
        tag = {"x": " (exhibit)", "-": " (excluded)"}.get(str(c["tier"]), "")
        L.append(f"| {c['rank'] or '–'} | {c['dataset']} {c['obj_id']} {c['name']}{tag} | "
                 f"{c['proxy'].split('/', 1)[1][:34]} | {len({_key(r)[:4] for r in rc})} | " + " | ".join(cells) + " |")
    rp = sorted({(r["case"], r["cad"]) for r in rows if r["condition"] == "random"})
    if rp:
        L += ["", "Random proxies (one per object, hash-drawn from the 1257-CAD gallery): " +
              "; ".join(f"{k} → {v}" for k, v in rp)]
    L += ["", "## 4. Failure taxonomy and validity", "",
          "| condition | success | grasp_failed | all_blocked | unreachable | no_candidates | fp_error | cad_missing |",
          "|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        rc = [r for r in rows if r["condition"] == cond]
        if rc:
            cnt = Counter(r["fail_reason"] or "success" for r in rc)
            L.append(f"| {cond} | " + " | ".join(str(cnt.get(k, 0)) for k in (
                "success", "grasp_failed", "all_blocked", "unreachable", "no_candidates",
                "fp_error", "cad_missing")) + " |")
    L.append("")
    for k, unit in (("settle_mm", "mm"), ("plane_angle_deg", "°"), ("bottom_gap_mm", "mm"),
                    ("plane_inlier", ""), ("visib", "")):
        vals = [float(r[k]) for r in rows if r.get(k) not in ("", None)]
        if vals:
            L.append(f"- {k}: median {statistics.median(vals):.2f}{unit}, "
                     f"min {min(vals):.2f}, max {max(vals):.2f} (n={len(vals)})")
    appl = sorted({r["case"] for r in rows if r["condition"] == "gt_pose" and int(r["n_cand"]) == 0})
    L.append(f"- applicability (0 antipodal candidates on the target's own CAD within the gripper): "
             f"{appl if appl else 'none'}")
    if exhibits:
        L += ["", "## 5. Mechanism exhibits (not in the rate)", ""]
        L += _headline_table(exhibits, conds, "Exhibits: " + ", ".join(sorted({r['case'] for r in exhibits})))
    if extra:
        L += ["", "## 5b. Explicitly run excluded objects (not in the rate)", ""]
        L += _headline_table(extra, conds, "Excluded: " + ", ".join(sorted({r['case'] for r in extra})))
    L += ["", "## 6. Predeclared exclusions", ""]
    L += [f"- {c['dataset']} {c['obj_id']} {c['name']}: {c['note']}" for c in EXCLUDED]
    L += ["", "## 7. Protocol", "", "```", json.dumps(PROTOCOL, indent=1, default=str), "```", "",
          "Caveats: simulation only; clutter is static and consists of the annotated objects "
          "(LM-O's unannotated clutter is absent); masks are BOP GT (`mask_visib`), as in Stage 3; "
          "the `gt_pose` ceiling uses the settled sim pose, so it also bounds the sampler+executor."]
    return "\n".join(L)


def _write_report(csv_path: str, md: str) -> str:
    """REPORT.md next to the CSV; /tmp when the folder is not writable (root-owned)."""
    out = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "REPORT.md")
    try:
        with open(out, "w") as fh:
            fh.write(md + "\n")
        return out
    except PermissionError:
        alt = "/tmp/proxy_grasp_REPORT.md"
        with open(alt, "w") as fh:
            fh.write(md + "\n")
        return f"{alt} ({out} is not writable for this user)"


# ===========================================================================
# 7. host side: wrap into the oscar container (pattern of repro_experiment.py)
# ===========================================================================
OUT_DIRS = ("_s5_out", "_grasp_cache", "_vhacd_cache", "_bop_obj_cache")
THREAD_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}


def _docker() -> list:
    """A WORKING `docker compose` CLI. On WSL the `docker` shim can be present
    but dead after a Docker Desktop restart (I/O error), so each candidate is
    tried, not just looked up."""
    import shutil
    for c in ("docker", "docker.exe"):
        if shutil.which(c):
            try:
                if subprocess.run([c, "compose", "version"], capture_output=True, timeout=30).returncode == 0:
                    return [c, "compose"]
            except (OSError, subprocess.TimeoutExpired):
                pass
    sys.exit("no working docker CLI — start Docker Desktop (WSL: check the WSL integration) / install docker")


def _fp_healthy(dc) -> bool:
    out = subprocess.run(dc + ["ps", "--format", "{{.Name}} {{.Status}}"], capture_output=True, text=True).stdout
    return any("foundationpose" in l and "healthy" in l for l in out.splitlines())


def _wait_fp(dc, timeout_s: int = 600):
    subprocess.run(dc + ["up", "-d", "foundationpose"], capture_output=True)
    print("[host] waiting for foundationpose ...", flush=True)
    for _ in range(timeout_s // 5):
        if _fp_healthy(dc):
            return
        time.sleep(5)
    sys.exit("foundationpose did not become healthy — check `docker compose logs foundationpose`")


def _own_outputs(dc):
    """Output/cache folders must belong to the host user; an earlier root run
    may have left them root-owned (chowned once via a root container)."""
    uid, gid = os.getuid(), os.getgid()
    foreign = any(os.stat(os.path.join(r, x)).st_uid != uid
                  for d in OUT_DIRS if os.path.isdir(os.path.join(_ROOT, d))
                  for r, ds, fs in os.walk(os.path.join(_ROOT, d)) for x in [""] + ds + fs)
    if foreign:
        print(f"[host] taking ownership of {' '.join(OUT_DIRS)} (written by an earlier root run)")
        subprocess.run(dc + ["run", "--rm", "oscar", "bash", "-lc",
                             f"cd /app && mkdir -p {' '.join(OUT_DIRS)} && chown -R {uid}:{gid} {' '.join(OUT_DIRS)}"],
                       capture_output=True)
    for d in OUT_DIRS:
        os.makedirs(os.path.join(_ROOT, d), exist_ok=True)


def run_in_container(argv: List[str], needs_fp: bool) -> int:
    """Re-run this script inside the oscar container as the host user. Exit
    code 3 (FoundationPose stopped answering) restarts the service and resumes
    — the experiment continues from its CSV, failed trials are re-run."""
    dc = _docker()
    _own_outputs(dc)
    if needs_fp:
        _wait_fp(dc)
    # --csv is a HOST path (relative = repo root); inside the container it must be
    # absolute under /app, otherwise it lands relative to /app/object_retrieval
    argv = list(argv)
    for i, a in enumerate(argv):
        if a == "--csv" and i + 1 < len(argv):
            argv[i + 1] = os.path.abspath(argv[i + 1])
        elif a.startswith("--csv="):
            argv[i] = "--csv=" + os.path.abspath(a[6:])
    argv = [a.replace(_ROOT, "/app") if _ROOT in a else a for a in argv]
    cmd = dc + ["run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    for k, v in THREAD_ENV.items():
        cmd += ["-e", f"{k}={os.environ.get(k, v)}"]
    cmd += ["oscar", "bash", "-lc", "cd /app/object_retrieval && PYTHONPATH=/app exec python3 -u -m "
            "grasping.experiment_proxy_grasp " + " ".join(argv)]
    caps = ", ".join(f"{k}={os.environ.get(k, v)}" for k, v in THREAD_ENV.items())
    print(f"[host] running in the oscar container as uid {os.getuid()} ({caps})", flush=True)
    for attempt in range(1, 5):
        rc = subprocess.call(cmd)
        if rc != 3:
            return rc
        print(f"[host] restarting foundationpose (attempt {attempt}) ...", flush=True)
        subprocess.run(dc + ["restart", "foundationpose"], capture_output=True)
        _wait_fp(dc)
    return 3


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Stage-5 proxy-grasp study (docs/STAGE5_PROTOCOL.md)",
                                 epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("what runs")
    g.add_argument("--ranks", default="1-10", help="rank range of the Top-20 set (default 1-10)")
    g.add_argument("--cases", default=None, help="explicit case keys instead of --ranks, e.g. ycbv14,tless22")
    g.add_argument("--exhibits", action="store_true", help="also run the mechanism exhibits (ycbv5, tless16)")
    g.add_argument("--conditions", default=DEFAULT_CONDITIONS,
                   help=f"comma list from {CONDITIONS} (default {DEFAULT_CONDITIONS})")
    g.add_argument("--per-object", type=int, default=PROTOCOL["sampling"]["per_object"],
                   help="instances per object, round-robin over scenes (default 6)")
    g.add_argument("--min-visib", type=float, default=PROTOCOL["sampling"]["min_visib"],
                   help="skip instances below this BOP visib_fract (default 0.5)")
    g = ap.add_argument_group("how it runs")
    g.add_argument("--pose-source", default="fp", choices=("fp", "stage3"),
                   help="fp = call FoundationPose now (default); stage3 = archived Stage-3 poses for gt/proxy, no GPU")
    g.add_argument("--world", default="auto", choices=("auto", "plane", "bop"),
                   help="sim world: auto = BOP extrinsics where present, else the depth-fitted table plane")
    g.add_argument("--n-tries", type=int, default=PROTOCOL["executor"]["n_tries"],
                   help="executed grasp attempts per trial, the k of success@k (default 5)")
    g.add_argument("--no-exec", dest="exec", action="store_false", help="pose + candidate metrics only")
    g.add_argument("--verbose", action="store_true", help="print every executed attempt (rose/hold/held)")
    g.add_argument("--no-docker", action="store_true", help="do not wrap into the oscar container")
    g = ap.add_argument_group("modes and output")
    g.add_argument("--check", action="store_true", help="verify data, FoundationPose and the world fit, then exit")
    g.add_argument("--plan", action="store_true", help="print + freeze the trial plan (plan.json), then exit")
    g.add_argument("--report", action="store_true", help="write REPORT.md from the CSV, then exit")
    g.add_argument("--csv", default=os.path.join(OUT_DIR, "trials.csv"),
                   help="one row per finished trial; re-running resumes from it")
    g.add_argument("--fresh", action="store_true", help="delete the CSV and start over")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if any(c not in CONDITIONS for c in conds):
        sys.exit(f"unknown condition in {conds}; choose from {CONDITIONS}")
    # everything but --plan/--report needs the sim stack -> run in the container
    if not IN_CONTAINER and not args.no_docker and not (args.plan or args.report):
        needs_fp = not args.check and any(
            (c in ("gt", "proxy", "proxy_scaled") and args.pose_source == "fp") or c == "random"
            for c in conds)
        sys.exit(run_in_container(sys.argv[1:], needs_fp))

    cases = select_cases(args.ranks, args.cases, args.exhibits)
    plan = plan_trials(cases, load_instances(), args.per_object, args.min_visib)
    PROTOCOL["executor"]["n_tries"] = args.n_tries

    if args.report:
        rows = _load_csv(args.csv)
        if not rows:
            sys.exit(f"no trials in {args.csv}")
        md = report(rows, args.csv, conds)
        print(md)
        print(f"\n[report] -> {_write_report(args.csv, md)}")
        return
    if args.check:
        sys.exit(check(args, plan))
    print_plan(cases, plan, conds, args)
    if args.plan:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "plan.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(dict(ts=_dt.datetime.now().isoformat(timespec="seconds"), args=vars(args),
                           conditions=conds, plan=plan), fh, indent=1)
        print(f"[plan] {len(plan)} instances x {len(conds)} conditions = {len(plan) * len(conds)} trials -> {out}")
        return

    # ---- run -----------------------------------------------------------------
    if args.fresh and os.path.exists(args.csv):
        os.remove(args.csv)
    # rows that failed on the pose SERVICE are re-run: a poisoned CUDA context is
    # not an outcome of the instance
    done = {_key(r) for r in _load_csv(args.csv) if r.get("fail_reason") != "fp_error"}
    if done:
        print(f"[resume] {len(done)} finished trial(s) in {args.csv}")
    man = _write_manifest(args, plan, conds)
    print(f"[run] git {man['git'].get('rev')} dirty={man['git'].get('dirty')} · "
          f"pose-source {args.pose_source} · world {args.world} · n-tries {args.n_tries} · csv {args.csv}")
    ctx = Ctx(args)
    todo = [(tr, c) for tr in plan for c in conds
            if (tr["case"], tr["scene"], tr["im"], tr["gt_idx"], c) not in done]
    print(f"[run] {len(todo)} trials to run ({len(plan) * len(conds) - len(todo)} cached)\n")
    hdr = (f"{'#':>7} {'instance':<28}{'condition':<13}{'CAD':<30}{'result':<16}"
           f"{'cand/reach/blk':>15}{'D_sym':>7}{'place':>7}{'settle':>7}{'s':>6}")
    print(hdr + "\n" + "-" * len(hdr))
    fp_dead, times = 0, []
    for i, (tr, cond) in enumerate(todo, 1):
        label = f"{tr['case']:<8}s{tr['scene']:02d} im{tr['im']:04d} g{tr['gt_idx']}"
        row = None
        for attempt in range(2):                     # one retry on a transient error
            try:
                row = run_trial(ctx, tr, cond)
                break
            except Exception as exc:
                traceback.print_exc(limit=2)
                print(f"[run] {label} {cond}: {exc.__class__.__name__}: {str(exc)[:80]}"
                      f"{' — retrying' if attempt == 0 else ''}")
        if row is None:
            continue
        _append_csv(args.csv, row)
        times.append(row["runtime_s"])
        fp_dead = fp_dead + 1 if row["fail_reason"] == "fp_error" else 0
        if fp_dead >= 3:
            print("[run] FoundationPose failed 3x in a row — exit code 3: the host side restarts "
                  "the service and resumes (these trials are re-run)")
            sys.exit(3)
        res = ("S@1 ✔" if row["first_succ"] else "S@k ✔" if row["succ"] else "✘ " + (row["fail_reason"] or "?"))
        print(f"{i:>3}/{len(todo):<3} {label:<28}{cond:<13}{row['cad'].split('/')[-1][:29]:<30}"
              f"{res:<16}{row['n_cand']:>5}/{row['n_reach']:<4}/{row['n_blocked']:<4}"
              f"{str(row['dsym_mm']):>7}{str(row['place_mm']):>7}{str(row['settle_mm']):>7}"
              f"{row['runtime_s']:>6.0f}", flush=True)
        if i % 10 == 0:
            print(f"        … {i}/{len(todo)} done, mean {statistics.mean(times):.0f} s/trial, "
                  f"ETA {statistics.mean(times) * (len(todo) - i) / 60:.0f} min", flush=True)
    print("=" * len(hdr))
    if ctx.eval_fallback:
        print(f"[note] models_eval absent for {sorted(ctx.eval_fallback)} — targets were posed/"
              f"scored with the sim mesh instead (see --check)")
    rows = _load_csv(args.csv)
    md = report(rows, args.csv, conds)
    out = _write_report(args.csv, md)
    print("\n" + "\n".join(md.split("\n## 3.")[0].split("\n")[6:]))      # headline + paired tables
    print(f"\n[out] {len(rows)} trials in {args.csv}\n[out] report -> {out}")


if __name__ == "__main__":
    main()
