#!/usr/bin/env python3
"""Stage-5 · Proxy-grasp study — is a retrieved proxy CAD good enough to grasp with?

The predeclared protocol is in docs/STAGE5_PROTOCOL.md; in one paragraph:

For each object of the frozen Top-20 set (`proxy_grasp_cases.py`) and each of
its frozen instances (`proxy_grasp_instances.json` = the Stage-3 3b instances in
which the fixed proxy was retrieved at rank 1), the same BOP frame is rebuilt in
PyBullet — the annotated objects at their GT poses on the table plane fitted to
the real depth image, the target dynamic, a Franka Panda — and the SAME grasp
planner and executor run under four conditions:

    gt_pose   target's own CAD at the TRUE (settled) sim pose  — mechanics ceiling
    gt        target's own CAD posed by FoundationPose          — pose loss only
    proxy     the retrieved proxy CAD posed by FoundationPose   — the pipeline
    random    a hash-drawn random gallery CAD posed by FP       — chance baseline
   (proxy_scaled: the proxy sized to the observed depth cloud — optional ablation)

FoundationPose sees the REAL sensor RGB-D and the GT visible mask, exactly as in
Stage 3 (`eval_bop_pose.estimate_pose`); pose quality is scored with Stage 3's
D_sym (`stage3_metrics.d_sym`). A trial succeeds when one of at most `--n-tries`
executed grasp candidates lifts the target, holds it and survives a shake test
(`grasp_execute`). Every attempt starts from the identical reset state.

Runs from the repo root of the HOST and wraps itself into the oscar container
(`docker compose run`, as the host user; FoundationPose is started when needed
and restarted if it stops answering). Also reachable as
`python3 repro_experiment.py --stage 5 ...`.

    python3 grasping/experiment_proxy_grasp.py --check      # data + FoundationPose + plane self-check
    python3 grasping/experiment_proxy_grasp.py --plan       # print / freeze the trial plan
    python3 grasping/experiment_proxy_grasp.py              # run gt vs proxy (resumes from the CSV)
    python3 grasping/experiment_proxy_grasp.py --report     # tables from the CSV
    python3 grasping/experiment_proxy_grasp.py --ranks 1-20 --per-object 8   # larger sample
    python3 grasping/experiment_proxy_grasp.py --conditions gt_pose,gt,proxy,random
    python3 grasping/experiment_proxy_grasp.py --pose-source stage3          # archived Stage-3 poses, no GPU
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
CONDITIONS = ("gt_pose", "gt", "proxy", "random", "proxy_scaled")
DEFAULT_CONDITIONS = "gt,proxy"          # the headline pair; gt_pose/random are opt-in controls
INSTANCES_JSON = os.path.join(_THIS, "proxy_grasp_instances.json")
OUT_DIR = os.path.join(_ROOT, "_s5_out", "proxy_grasp")
GRASP_CACHE = os.path.join(_ROOT, "_grasp_cache")

# Every value that shapes an outcome, frozen here and written to the manifest.
PROTOCOL = dict(
    sampler=dict(n_samples=800, top_k=40, friction_mu=0.5, n_approach=4, seed=0,
                 gripper_min_width_m=0.005, gripper_max_width_m=0.08),
    executor=dict(n_tries=5, pregrasp_m=0.12, lift_m=0.15, rise_min_m=0.05,
                  hold_steps=240, shake_amp_m=0.05, in_hand_m=0.15,
                  close_force_n=(20, 120), reach_tol_mm=30, blocked_mm=30),
    physics=dict(timestep_s=1 / 240, gravity=-9.81, target_mass_kg=0.2,
                 target_friction=1.6, finger_friction=1.5, settle_steps=60,
                 collision="V-HACD (target) / concave static clutter"),
    robot=dict(arm="Franka Panda (pybullet franka_panda/panda.urdf)", pedestal_m=0.35,
               placement="on the camera's side: 0.55 m from the object centroid along the "
                         "camera viewing direction projected onto the table"),
    world=dict(frame="auto: BOP extrinsics where present (YCB-V, T-LESS), else the table plane "
                     "fitted to the real depth (LM-O)",
               table_plane="RANSAC, objects masked out, ±0.5 m around the object depths, thr 6 mm, "
                           "candidate planes must carry every object 0–40 cm above them",
               table_z="0 = lowest object vertex (bop) / the fitted plane; lowered to the "
                       "target's own lowest vertex if that is below"),
    pose=dict(fp_refine_iter=5, fp_input="bop = real RGB-D + GT mask_visib (Stage 3)",
              d_sym="stage3_metrics.d_sym, 10 000 surface samples, seed 0"),
    sampling=dict(min_visib=0.5, per_object=6,
                  rule="round-robin over scenes, evenly spaced frames within a scene"),
    success="rose >= 5 cm AND held 1 s AND survived ±5 cm shakes; trial = any of <= n_tries attempts",
)


# ---------------------------------------------------------------------------
# frozen instance lists + trial plan
# ---------------------------------------------------------------------------
def load_instances(path: str = INSTANCES_JSON) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"missing {path} — run grasping/build_grasp_instances.py first")
    return json.load(open(path))


def plan_trials(cases: List[dict], db: dict, per_object: int, min_visib: float
                ) -> List[dict]:
    """Deterministic instance choice per case: keep instances with
    visib >= min_visib, distribute `per_object` picks round-robin over the
    scenes (sorted by id), and within a scene take frames at evenly spaced
    positions of its (frame-sorted) instance list. No outcome, no RNG."""
    plan = []
    for c in cases:
        entry = db["cases"].get(c["key"])
        if not entry:
            print(f"[plan] {c['key']}: no instance list")
            continue
        pool = [i for i in entry["instances"]
                if i.get("visib") is None or i["visib"] >= min_visib]
        by_scene: Dict[int, list] = defaultdict(list)
        for i in pool:
            by_scene[i["scene"]].append(i)
        scenes = sorted(by_scene)
        for s in scenes:
            by_scene[s].sort(key=lambda i: (i["im"], i["gt_idx"]))
        want = min(per_object, len(pool))
        quota = Counter()
        k = 0
        while sum(quota.values()) < want and scenes:
            s = scenes[k % len(scenes)]
            if quota[s] < len(by_scene[s]):
                quota[s] += 1
            k += 1
            if k > 10 * want + len(scenes):
                break
        chosen = []
        for s in scenes:
            lst, q = by_scene[s], quota[s]
            if q == 0:
                continue
            idx = sorted({int((j + 0.5) / q * len(lst)) for j in range(q)})
            idx = [min(i, len(lst) - 1) for i in idx]
            # evenly spaced positions may collide on tiny lists — fill up in order
            for j in range(len(lst)):
                if len(idx) >= q:
                    break
                if j not in idx:
                    idx.append(j)
            chosen += [lst[i] for i in sorted(idx)[:q]]
        for i in chosen:
            plan.append(dict(case=c["key"], rank=c["rank"], tier=c["tier"],
                             dataset=c["dataset"], obj_id=c["obj_id"], name=c["name"],
                             proxy=c["proxy"], pool=len(pool), n_top1=entry["n_top1"], **i))
    return plan


# ---------------------------------------------------------------------------
# per-run context: caches for scenes, meshes, surface samples, grasps
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args = args
        self.scenes: dict = {}
        self.meshes: dict = {}
        self.pts: dict = {}
        self.grasps: dict = {}
        self.pool = proxy_pool()
        self.eval_fallback = set()

    # -- scene (world from the real depth image) ------------------------------
    def scene(self, ds: str, scene: int, im: int):
        key = (ds, scene, im)
        if key not in self.scenes:
            from grasping.sim_scene import load_bop_frame, scene_from_frame
            fr = load_bop_frame(ds, scene, im)
            objs, cam, info = scene_from_frame(fr, world=self.args.world)
            self.scenes[key] = (fr, objs, cam, info)
        return self.scenes[key]

    # -- meshes ---------------------------------------------------------------
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
        """The CAD Stage 3 posed the target with (models_eval, mm) — or the sim
        mesh when that split is absent on this machine (reported)."""
        from grasping.sim_scene import eval_mesh_path
        path, units_m = eval_mesh_path(ds, obj_id)
        if units_m:
            self.eval_fallback.add(ds)
        return path, units_m

    # -- grasp candidates (object frame, cached on disk per CAD) --------------
    def grasp_candidates(self, path: str, units_m: bool, extra: float = 1.0):
        P = PROTOCOL["sampler"]
        key = hashlib.md5(f"{path}|{units_m}|{extra:.6f}|{P}".encode()).hexdigest()[:16]
        if key in self.grasps:
            return self.grasps[key]
        from grasping.antipodal_grasp_sampler import Grasp, GripperConfig, sample_antipodal_grasps
        import numpy as np
        os.makedirs(GRASP_CACHE, exist_ok=True)
        f = os.path.join(GRASP_CACHE, key + ".json")
        if os.path.isfile(f) and os.path.getsize(f) > 0:
            raw = json.load(open(f))
            gs = [Grasp(center=np.array(g["center"]), axis=np.array(g["axis"]),
                        approach=np.array(g["approach"]), width=g["width"],
                        quality=g["quality"],
                        contacts=(np.array(g["contacts"][0]), np.array(g["contacts"][1])))
                  for g in raw["grasps"]]
        else:
            t0 = time.time()
            gs = sample_antipodal_grasps(
                self.mesh_m(path, units_m, extra),
                GripperConfig(max_width=P["gripper_max_width_m"], min_width=P["gripper_min_width_m"]),
                n_samples=P["n_samples"], friction_mu=P["friction_mu"],
                n_approach=P["n_approach"], top_k=P["top_k"], seed=P["seed"])
            tmp = f + f".{os.getpid()}.tmp"
            with open(tmp, "w") as fh:
                json.dump({"cad": path, "units_m": units_m, "extra": extra, "protocol": P,
                           "seconds": round(time.time() - t0, 1),
                           "grasps": [g.to_dict() for g in gs]}, fh)
            os.replace(tmp, f)
        self.grasps[key] = gs
        return gs


# ---------------------------------------------------------------------------
# one trial = (instance, condition)
# ---------------------------------------------------------------------------
def _fp_pose(cad_path, rgb, depth_m, mask, K, units_m: bool, extra: float):
    """FoundationPose in the Stage-3 configuration. Returns (R 3x3, t mm, conf).
    `extra` != 1 only for the proxy_scaled ablation (Stage 3 never scales)."""
    from eval_bop_pose import estimate_pose, FP_URL
    if abs(extra - 1.0) < 1e-9:
        return estimate_pose(cad_path, rgb, depth_m, mask, K, mesh_units_m=units_m,
                             refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    from pipeline.foundationpose_bridge import call_foundationpose
    import numpy as np
    scale = (1.0 if units_m else 1e-3) * extra
    pose, conf = call_foundationpose(FP_URL, rgb=rgb, depth=depth_m, mask=mask, K=K,
                                     cad_path=cad_path, scale=scale,
                                     refine_iter=PROTOCOL["pose"]["fp_refine_iter"])
    return pose[:3, :3], pose[:3, 3] * 1000.0, float(conf)


def run_trial(ctx: Ctx, tr: dict, cond: str) -> dict:
    import numpy as np
    from stage3_metrics import d_sym
    from grasping.sim_scene import TabletopSim
    from grasping.grasp_execute import PandaGrasper, reachable_order, feasible_grasps
    from grasping.antipodal_grasp_sampler import transform_grasps
    args = ctx.args
    t0 = time.time()
    ds, obj_id, gt_idx = tr["dataset"], tr["obj_id"], tr["gt_idx"]
    fr, objs, cam, winfo = ctx.scene(ds, tr["scene"], tr["im"])
    tgt = objs[gt_idx]
    assert tgt.obj_id == obj_id, f"gt_idx {gt_idx} is obj {tgt.obj_id}, not {obj_id}"
    row = dict(case=tr["case"], rank=tr["rank"], tier=tr["tier"], dataset=ds, obj_id=obj_id,
               name=tr["name"], scene=tr["scene"], im=tr["im"], gt_idx=gt_idx,
               visib=tr.get("visib"), condition=cond, cad="", cad_units_m="",
               pose_source="", fp_input=args.fp_input, fp_conf="", fp_fail="",
               dsym_mm="", dsym_norm="", f05="", place_mm="", settle_mm="",
               world=winfo.get("world"), plane_inlier=winfo.get("plane_inlier", ""),
               plane_angle_deg=winfo.get("plane_angle_deg", ""),
               bottom_gap_mm=winfo.get("bottom_gap_mm", ""),
               n_cand=0, n_reach=0, n_blocked=0, n_att=0, n_succ=0, first_succ=0, succ=0,
               lift_cm=0.0, att_seq="", fail_reason="", runtime_s=0.0,
               ts=_dt.datetime.now().isoformat(timespec="seconds"))

    # ---- the CAD under test ---------------------------------------------------
    extra = 1.0
    if cond in ("gt_pose", "gt"):
        cad_path, units_m = ctx.target_cad(ds, obj_id)
        cad_id = "gt"
    elif cond in ("proxy", "proxy_scaled"):
        cad_id = tr["proxy"]
        cad_path, units_m = proxy_mesh(cad_id)
    elif cond == "random":
        cad_id = random_proxy(ds, obj_id, ctx.pool)
        cad_path, units_m = proxy_mesh(cad_id)
    else:
        raise ValueError(cond)
    if not cad_path:
        row.update(fail_reason="cad_missing", cad=cad_id, runtime_s=round(time.time() - t0, 1))
        return row
    row.update(cad=cad_id, cad_units_m=int(units_m))

    # ---- sim world: annotated objects at GT poses, target dynamic ------------
    os.environ.setdefault("GRASP_QUIET", "1")
    sim = TabletopSim().connect()
    # the table is z = 0 in every world; only lower it if THIS target's annotated
    # pose would intersect it (a penetrating dynamic body gets kicked out otherwise)
    table_z = min(0.0, winfo["bottom_z"].get(gt_idx, 0.0) - 0.001)
    sim.build(objs, cam, target_gt_idx=gt_idx, with_robot=True, table_z=table_z)
    sim.settle(PROTOCOL["physics"]["settle_steps"])
    row["settle_mm"] = round(sim.target_displacement_mm(tgt.T_world), 1)
    sim.freeze_initial()
    T_true = sim.target_pose()                       # settled model→world pose
    try:
        # ---- observation for FoundationPose ------------------------------------
        if cond != "gt_pose":
            if args.fp_input == "sim":
                obs = sim.render_rgbd()
                rgb, depth_m = obs["rgb"], obs["depth"].astype(np.float32)
                mask = (obs["seg"] == sim.target_body()).astype(np.uint8)
            else:
                rgb, depth_m = fr.rgb(), fr.depth_m()
                mask = fr.mask_visib(gt_idx).astype(np.uint8)
            if cond == "proxy_scaled":
                from grasping.perceive import observed_diag
                d_obs = observed_diag(depth_m, mask.astype(bool), fr.K)
                d_cad = float(np.linalg.norm(ctx.mesh_m(cad_path, units_m).extents))
                if d_obs and d_cad > 1e-6:
                    extra = d_obs / d_cad
        mesh_m = ctx.mesh_m(cad_path, units_m, extra)

        # ---- pose (camera frame, mm) ----------------------------------------------
        R_gt = np.asarray(fr.gt[gt_idx]["cam_R_m2c"], float).reshape(3, 3)
        t_gt = np.asarray(fr.gt[gt_idx]["cam_t_m2c"], float).reshape(3)
        if cond == "gt_pose":
            row["pose_source"] = "sim_true"
            T_m2w = T_true
        else:
            stored = None
            if args.pose_source == "stage3":
                stored = tr.get("fp_gt") if cond == "gt" else (
                    tr.get("fp_proxy") if cond == "proxy" else None)
            if stored is not None:
                R = np.asarray(stored["R"], float).reshape(3, 3)
                t = np.asarray(stored["t"], float).reshape(3)
                conf = stored.get("conf")
                row["pose_source"] = "stage3"
            else:
                try:
                    R, t, conf = _fp_pose(cad_path, rgb, depth_m, mask, fr.K, units_m, extra)
                    row["pose_source"] = "fp"
                except Exception as exc:
                    row.update(fp_fail=str(exc).splitlines()[-1][:80], fail_reason="fp_error",
                               runtime_s=round(time.time() - t0, 1))
                    return row
            row["fp_conf"] = round(float(conf), 3) if conf is not None else ""
            # Stage-3 pose score: D_sym between the GT-posed target and the posed CAD
            tpath, tunits = ctx.target_cad(ds, obj_id)
            dsr = d_sym(ctx.pts_mm(tpath, tunits), R_gt, t_gt,
                        ctx.pts_mm(cad_path, units_m, extra), R, t, tr.get("diameter") or 0.0)
            row.update(dsym_mm=round(dsr["d_sym"], 2),
                       dsym_norm=(round(dsr["d_sym_norm"], 4) if dsr["d_sym_norm"] is not None else ""),
                       f05=round(dsr["fscore"]["0.05"]["f"], 4))
            T_m2c = np.eye(4)
            T_m2c[:3, :3], T_m2c[:3, 3] = R, t / 1000.0
            T_m2w = cam.T_world @ T_m2c
        # placement: centroid of the posed CAD vs the settled target's centroid
        tgt_mesh = ctx.mesh_m(tgt.mesh_path, True)
        c_cad = T_m2w[:3, :3] @ mesh_m.centroid + T_m2w[:3, 3]
        c_tgt = T_true[:3, :3] @ tgt_mesh.centroid + T_true[:3, 3]
        row["place_mm"] = round(float(np.linalg.norm(c_cad - c_tgt) * 1000), 1)

        # ---- grasp planning on the posed CAD, execution on the real target ------
        gs = ctx.grasp_candidates(cad_path, units_m, extra)
        row["n_cand"] = len(gs)
        grasper = PandaGrasper(sim)
        grasper.reset()
        base = sim._p.getBasePositionAndOrientation(sim.robot)[0][:2]
        gw = reachable_order(transform_grasps(gs, T_m2w), base)
        feas = feasible_grasps(grasper, gw, tol_mm=PROTOCOL["executor"]["reach_tol_mm"])
        row["n_reach"] = len(feas)
        seq, lifts = [], []
        if args.exec:
            for g in feas:
                if row["n_att"] >= args.n_tries:
                    break
                sim.reset_objects()
                grasper.reset()
                sim.settle(30)
                r = grasper.execute(g)
                if args.verbose:
                    from grasping.antipodal_grasp_sampler import _unit
                    print(f"        attempt {len(seq) + 1:>2}: q={g.quality:.2f} w={g.width * 1000:.0f}mm "
                          f"approach_z={_unit(g.approach)[2]:+.2f} -> {r}", flush=True)
                if r.get("blocked"):
                    row["n_blocked"] += 1
                    seq.append("B")
                    continue
                row["n_att"] += 1
                if r["success"]:
                    row["n_succ"] += 1
                    lifts.append(r["lift_cm"])
                    seq.append("S")
                    if row["n_att"] == 1:
                        row["first_succ"] = 1
                    break                          # success within k: stop at the first
                seq.append("F")
        row["att_seq"] = ",".join(seq)
        row["succ"] = int(row["n_succ"] > 0)
        row["lift_cm"] = round(float(np.mean(lifts)), 1) if lifts else 0.0
        if not row["succ"]:
            row["fail_reason"] = ("no_candidates" if not gs else "unreachable" if not feas
                                  else "all_blocked" if row["n_att"] == 0 else "grasp_failed")
    finally:
        sim.disconnect()
    row["runtime_s"] = round(time.time() - t0, 1)
    return row


# ---------------------------------------------------------------------------
# CSV (append per trial, resume) + manifest
# ---------------------------------------------------------------------------
KEY = ("case", "scene", "im", "gt_idx", "condition")


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
    if not path or not os.path.exists(path):
        return []
    known = {c["key"] for c in ALL_CASES}
    out = []
    with open(path) as fh:
        for d in csv.DictReader(fh):
            if (d.get("case") or "").strip() not in known or d.get("condition") not in CONDITIONS:
                print(f"[resume] skipping malformed row: {str(d)[:70]}")
                continue
            try:
                int(d["n_att"]); int(d["succ"]); int(d["scene"]); int(d["im"])
            except (TypeError, ValueError, KeyError):
                print(f"[resume] skipping incomplete row: {str(d)[:70]}")
                continue
            out.append(d)
    # a re-run of a trial appends a new row (e.g. after a FoundationPose restart):
    # the LAST row per (instance, condition) is the valid one
    last = {}
    for d in out:
        last[_key(d)] = d
    return list(last.values())


def _key(r: dict) -> tuple:
    return (r["case"], int(r["scene"]), int(r["im"]), int(r["gt_idx"]), r["condition"])


def _git_state() -> dict:
    import subprocess
    st = {"rev": "unknown", "dirty": "unknown"}
    try:
        head = open(os.path.join(_ROOT, ".git", "HEAD")).read().strip()
        if head.startswith("ref: "):
            ref = os.path.join(_ROOT, ".git", head[5:])
            st["rev"] = open(ref).read().strip()[:12] if os.path.isfile(ref) else head
            st["branch"] = head[5:].split("/")[-1]
        else:
            st["rev"] = head[:12]
    except OSError:
        pass
    try:
        # the repo is bind-mounted into the container under another owner, hence safe.directory
        out = subprocess.run(["git", "-c", "safe.directory=*", "-C", _ROOT, "status", "--porcelain",
                              "--", "grasping", "object_retrieval", "pipeline"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            st["dirty"] = bool(out.stdout.strip())
    except Exception:
        pass
    return st


def _write_manifest(args, plan, conds):
    import platform
    versions = {}
    for mod in ("numpy", "scipy", "trimesh", "pybullet"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", None) or str(getattr(m, "getAPIVersion", lambda: "?")())
        except Exception as exc:
            versions[mod] = f"unavailable ({exc.__class__.__name__})"
    man = dict(ts=_dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
               python=platform.python_version(), versions=versions, git=_git_state(),
               args=vars(args), conditions=conds, protocol=PROTOCOL,
               instances_json=INSTANCES_JSON,
               instances_sha=hashlib.sha256(open(INSTANCES_JSON, "rb").read()).hexdigest()[:16],
               n_planned=len(plan), cases=sorted({p["case"] for p in plan}),
               fp_url=os.environ.get("FP_URL", "http://foundationpose:5050"),
               threads=os.environ.get("OMP_NUM_THREADS"))
    path = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "manifest.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hist = []
    if os.path.isfile(path):
        try:
            hist = json.load(open(path)).get("runs", [])
        except ValueError:
            hist = []
    hist.append(man)
    with open(path + ".tmp", "w") as fh:
        json.dump({"runs": hist}, fh, indent=1)
    os.replace(path + ".tmp", path)
    return man


# ---------------------------------------------------------------------------
# --check: data, FoundationPose, world self-check
# ---------------------------------------------------------------------------
def check(args, plan) -> int:
    from grasping.sim_scene import BOP_DATASETS, bop_mesh_path, eval_mesh_path
    missing = 0
    print("== data ==")
    seen = set()
    for tr in plan:
        ds = tr["dataset"]
        sdir = os.path.join(BOP_DATASETS[ds]["test"], f"{tr['scene']:06d}")
        for rel in (f"rgb/{tr['im']:06d}.png", f"depth/{tr['im']:06d}.png",
                    f"mask_visib/{tr['im']:06d}_{tr['gt_idx']:06d}.png", "scene_gt.json",
                    "scene_camera.json"):
            p = os.path.join(sdir, rel)
            if not os.path.isfile(p) and not (rel.endswith(".png") and os.path.isfile(p[:-4] + ".jpg")):
                print(f"  MISSING {p}")
                missing += 1
        for cid in (tr["proxy"], random_proxy(ds, tr["obj_id"], proxy_pool())):
            if cid not in seen:
                seen.add(cid)
                p, _ = proxy_mesh(cid)
                if not p:
                    print(f"  MISSING proxy CAD for {cid}")
                    missing += 1
        k = (ds, tr["obj_id"])
        if k not in seen:
            seen.add(k)
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
    host = url.split("//")[-1].split("/")[0]
    h, _, port = host.partition(":")
    try:
        socket.create_connection((h, int(port or 80)), timeout=3).close()
        print(f"  {url}: reachable")
    except OSError as exc:
        print(f"  {url}: NOT reachable ({exc}) — start it: docker compose up -d foundationpose")

    print("== world self-check (table plane from the real depth) ==")
    try:
        import numpy as np  # noqa: F401
        from grasping.sim_scene import load_bop_frame, scene_from_frame
        done = set()
        for tr in plan:
            if tr["dataset"] in done:
                continue
            done.add(tr["dataset"])
            fr = load_bop_frame(tr["dataset"], tr["scene"], tr["im"])
            _, _, info = scene_from_frame(fr, world=args.world)
            print(f"  {tr['dataset']} {tr['scene']:06d}/{tr['im']:06d}: world={info['world']} "
                  f"inliers={info.get('plane_inlier')} angle_vs_BOP_up={info.get('plane_angle_deg', 'n/a')}° "
                  f"bottom_gap={info['bottom_gap_mm']} mm")
    except ImportError as exc:
        print(f"  skipped (needs the sim stack: {exc}) — run inside the oscar container")
    return 0 if not missing else 1


# ---------------------------------------------------------------------------
# --report: tables from the CSV (Δ + win split, no intervals — AGREEMENTS 2026-09-03)
# ---------------------------------------------------------------------------
def _pct(a, b):
    return f"{100.0 * a / b:.0f}%" if b else "–"


def _med(vals):
    v = [float(x) for x in vals if x not in ("", None)]
    return f"{statistics.median(v):.1f}" if v else "–"


def _cond_stats(rows):
    n = len(rows)
    s5 = sum(int(r["succ"]) for r in rows)
    s1 = sum(int(r["first_succ"]) for r in rows)
    att = sum(int(r["n_att"]) for r in rows)
    sa = sum(int(r["n_succ"]) for r in rows)
    return dict(n=n, objects=len({r["case"] for r in rows}), s5=s5, s1=s1, att=att, sa=sa,
                cand=_med(r["n_cand"] for r in rows), reach=_med(r["n_reach"] for r in rows),
                blocked=sum(int(r["n_blocked"]) for r in rows),
                nocand=sum(1 for r in rows if r["fail_reason"] == "no_candidates"),
                fpfail=sum(1 for r in rows if r["fail_reason"] == "fp_error"),
                dsym=_med(r["dsym_mm"] for r in rows), place=_med(r["place_mm"] for r in rows),
                rt=_med(r["runtime_s"] for r in rows))


def _headline_table(rows, conds, title):
    out = [f"**{title}**", "",
           "| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | "
           "reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        rc = [r for r in rows if r["condition"] == c]
        if not rc:
            continue
        s = _cond_stats(rc)
        out.append(f"| {c} | {s['objects']} | {s['n']} | {s['s5']}/{s['n']} ({_pct(s['s5'], s['n'])}) | "
                   f"{s['s1']}/{s['n']} ({_pct(s['s1'], s['n'])}) | {s['sa']}/{s['att']} ({_pct(s['sa'], s['att'])}) | "
                   f"{s['cand']} | {s['reach']} | {s['blocked']} | {s['nocand']} | {s['fpfail']} | "
                   f"{s['dsym']} | {s['place']} | {s['rt']} |")
    return out


def _paired_table(rows, pairs, title):
    by = defaultdict(dict)
    for r in rows:
        by[_key(r)[:4]][r["condition"]] = int(r["succ"])
    out = [f"**{title}** (same instance under both conditions; Δ in percentage points)", "",
           "| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |",
           "|---|---|---|---|---|---|"]
    for a, b in pairs:
        ks = [k for k, d in by.items() if a in d and b in d]
        if not ks:
            continue
        sa = sum(by[k][a] for k in ks); sb = sum(by[k][b] for k in ks)
        ao = sum(1 for k in ks if by[k][a] and not by[k][b])
        bo = sum(1 for k in ks if by[k][b] and not by[k][a])
        both = sum(1 for k in ks if by[k][a] and by[k][b])
        nei = len(ks) - ao - bo - both
        out.append(f"| {a} → {b} | {len(ks)} | {_pct(sa, len(ks))} | {_pct(sb, len(ks))} | "
                   f"{100.0 * (sb - sa) / len(ks):+.0f} pp | {ao} : {bo} : {both} : {nei} |")
    return out


def report(rows: List[dict], csv_path: str, conds: List[str]) -> str:
    rows = [r for r in rows if r["condition"] in conds]
    counted = [r for r in rows if str(r["tier"]) in ("1", "2")]
    exhibits = [r for r in rows if str(r["tier"]) == "x"]
    extra = [r for r in rows if str(r["tier"]) == "-"]
    man_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "manifest.json")
    runs = []
    if os.path.isfile(man_path):
        try:
            runs = json.load(open(man_path)).get("runs", [])
        except ValueError:
            pass
    L = ["# Stage 5 — Proxy-grasp study (generated report)", "",
         f"CSV: `{os.path.relpath(csv_path, _ROOT)}` · {len(rows)} trials · "
         f"{len({r['case'] for r in rows})} objects · "
         f"{len({_key(r)[:4] for r in rows})} instances · "
         f"pose source: {sorted({r['pose_source'] for r in rows if r['pose_source']})} · "
         f"FP input: {sorted({r['fp_input'] for r in rows})} · "
         f"world: {sorted({r['world'] for r in rows})}",
         f"Runs: {len(runs)}" + (f", last {runs[-1]['ts']} on {runs[-1]['host']} "
                                 f"(git {runs[-1]['git'].get('rev')}, dirty={runs[-1]['git'].get('dirty')})"
                                 if runs else ""),
         "", "success@k = trial succeeded within ≤ n_tries executed attempts (headline); "
             "success@1 = the first executed candidate succeeded; attempts S/N = per-attempt rate. "
             "No confidence intervals by agreement (2026-09-03): Δ plus the per-instance win split.", ""]
    # -- headline: per dataset first (never only the aggregate), then all
    L.append("## 1. Success rate — rank 1–10/20 set")
    L.append("")
    for ds in ("ycbv", "tless", "lmo"):
        rd = [r for r in counted if r["dataset"] == ds]
        if rd:
            L += _headline_table(rd, conds, f"{ds.upper()} ({len({r['case'] for r in rd})} objects)") + [""]
    L += _headline_table(counted, conds, "ALL datasets (pooled — composition differs per dataset)") + [""]
    # -- paired
    pairs = [("gt_pose", "gt"), ("gt", "proxy"), ("proxy", "random"), ("gt_pose", "proxy")]
    if "proxy_scaled" in conds:
        pairs.append(("proxy", "proxy_scaled"))
    L.append("## 2. Paired comparisons")
    L.append("")
    L += _paired_table(counted, pairs, "All counted objects") + [""]
    for ds in ("ycbv", "tless", "lmo"):
        rd = [r for r in counted if r["dataset"] == ds]
        if rd:
            L += _paired_table(rd, pairs, ds.upper()) + [""]
    # -- per object
    L.append("## 3. Per object")
    L.append("")
    hdr = "| rank | object | proxy (fixed) | inst | " + " | ".join(
        f"{c} succ@k | {c} D_sym" for c in conds) + " |"
    L += [hdr, "|" + "---|" * (4 + 2 * len(conds))]
    for c in [c for c in CASES if c["tier"] in (1, 2)] + [c for c in CASES if c["tier"] == "x"] + EXCLUDED:
        rc = [r for r in rows if r["case"] == c["key"]]
        if not rc:
            continue
        cells = []
        for cond in conds:
            rr = [r for r in rc if r["condition"] == cond]
            cells.append(f"{sum(int(r['succ']) for r in rr)}/{len(rr)}" if rr else "–")
            cells.append(_med(r["dsym_mm"] for r in rr) if rr else "–")
        tag = {"x": " (exhibit)", "-": " (excluded)"}.get(str(c["tier"]), "")
        L.append(f"| {c['rank'] or '–'} | {c['dataset']} {c['obj_id']} {c['name']}{tag} | "
                 f"{c['proxy'].split('/', 1)[1][:34]} | {len({_key(r)[:4] for r in rc})} | "
                 + " | ".join(cells) + " |")
    L.append("")
    # -- random proxies drawn
    rp = sorted({(r["case"], r["cad"]) for r in rows if r["condition"] == "random"})
    if rp:
        L += ["Random proxies (one per object, hash-drawn from the 1257-CAD gallery): " +
              "; ".join(f"{k} → {v}" for k, v in rp), ""]
    # -- failures + validity
    L.append("## 4. Failure taxonomy and validity")
    L.append("")
    L += ["| condition | success | grasp_failed | all_blocked | unreachable | no_candidates | fp_error | cad_missing |",
          "|---|---|---|---|---|---|---|---|"]
    for cond in conds:
        rc = [r for r in rows if r["condition"] == cond]
        if not rc:
            continue
        cnt = Counter(r["fail_reason"] or "success" for r in rc)
        L.append(f"| {cond} | " + " | ".join(str(cnt.get(k, 0)) for k in (
            "success", "grasp_failed", "all_blocked", "unreachable", "no_candidates",
            "fp_error", "cad_missing")) + " |")
    L.append("")
    v = lambda k: [float(r[k]) for r in rows if r.get(k) not in ("", None)]  # noqa: E731
    for k, unit in (("settle_mm", "mm"), ("plane_angle_deg", "°"), ("bottom_gap_mm", "mm"),
                    ("plane_inlier", ""), ("visib", "")):
        vals = v(k)
        if vals:
            L.append(f"- {k}: median {statistics.median(vals):.2f}{unit}, "
                     f"min {min(vals):.2f}, max {max(vals):.2f} (n={len(vals)})")
    appl = sorted({(r["case"]) for r in rows if r["condition"] == "gt_pose" and int(r["n_cand"]) == 0})
    L.append(f"- applicability (0 antipodal candidates on the target's own CAD within the "
             f"gripper): {appl if appl else 'none'}")
    L.append("")
    # -- exhibits / excluded
    if exhibits:
        L += ["## 5. Mechanism exhibits (not in the rate)", ""]
        L += _headline_table(exhibits, conds, "Exhibits: " + ", ".join(sorted({r['case'] for r in exhibits}))) + [""]
    if extra:
        L += ["## 5b. Explicitly run excluded objects (not in the rate)", ""]
        L += _headline_table(extra, conds, "Excluded: " + ", ".join(sorted({r['case'] for r in extra}))) + [""]
    L += ["## 6. Predeclared exclusions", ""]
    for c in EXCLUDED:
        L.append(f"- {c['dataset']} {c['obj_id']} {c['name']}: {c['note']}")
    L += ["", "## 7. Protocol", "", "```", json.dumps(PROTOCOL, indent=1, default=str), "```", "",
          "Caveats: simulation only; clutter is static and consists of the annotated objects "
          "(LM-O's unannotated clutter is absent); masks are BOP GT (`mask_visib`), as in Stage 3; "
          "the `gt_pose` ceiling uses the settled sim pose, so it also bounds the sampler+executor."]
    return "\n".join(L)


def _write_report(csv_path: str, md: str) -> str:
    """REPORT.md next to the CSV; falls back to /tmp when the output folder was
    created by the container as root and the host user cannot write there."""
    out = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "REPORT.md")
    try:
        with open(out, "w") as fh:
            fh.write(md + "\n")
        return out
    except PermissionError:
        alt = os.path.join("/tmp", "proxy_grasp_REPORT.md")
        with open(alt, "w") as fh:
            fh.write(md + "\n")
        return f"{alt} ({out} is not writable for this user — the container wrote it as root; " \
               f"use ./grasping/run_stage5.sh proxy-grasp --report)"


# ---------------------------------------------------------------------------
# host side: wrap into the oscar container (same pattern as repro_experiment.py)
# ---------------------------------------------------------------------------
OUT_DIRS = ("_s5_out", "_grasp_cache", "_vhacd_cache", "_bop_obj_cache")
THREAD_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}


def _docker() -> list:
    import shutil
    for c in ("docker", "docker.exe"):
        if shutil.which(c):
            return [c, "compose"]
    sys.exit("docker CLI not found — start Docker Desktop / install docker")


def _fp_healthy(dc) -> bool:
    import subprocess
    out = subprocess.run(dc + ["ps", "--format", "{{.Name}} {{.Status}}"],
                         capture_output=True, text=True).stdout
    return any("foundationpose" in l and "healthy" in l for l in out.splitlines())


def _wait_fp(dc, timeout_s: int = 600):
    import subprocess
    subprocess.run(dc + ["up", "-d", "foundationpose"], capture_output=True)
    print("[host] waiting for foundationpose ...", flush=True)
    for _ in range(timeout_s // 5):
        if _fp_healthy(dc):
            return
        time.sleep(5)
    sys.exit("foundationpose did not become healthy — check `docker compose logs foundationpose`")


def _own_outputs(dc):
    """Output/cache folders must belong to the host user; an earlier root run
    (or the container's defaults) may have left them root-owned."""
    import subprocess
    uid, gid = os.getuid(), os.getgid()
    need = False
    for d in OUT_DIRS:
        p = os.path.join(_ROOT, d)
        if os.path.isdir(p):
            for root, dirs, files in os.walk(p):
                if any(os.stat(os.path.join(root, x)).st_uid != uid for x in dirs + files) \
                        or os.stat(root).st_uid != uid:
                    need = True
                    break
        if need:
            break
    if need:
        print(f"[host] taking ownership of {' '.join(OUT_DIRS)} (written by an earlier root run)")
        subprocess.run(dc + ["run", "--rm", "oscar", "bash", "-lc",
                             f"cd /app && mkdir -p {' '.join(OUT_DIRS)} && chown -R {uid}:{gid} {' '.join(OUT_DIRS)}"],
                       capture_output=True)
    for d in OUT_DIRS:
        os.makedirs(os.path.join(_ROOT, d), exist_ok=True)


def run_in_container(argv: List[str], needs_fp: bool) -> int:
    """Re-run this script inside the oscar container as the host user. Exit
    code 3 (FoundationPose stopped answering) restarts the service and resumes
    — the experiment continues from its CSV."""
    import subprocess
    dc = _docker()
    _own_outputs(dc)
    if needs_fp:
        _wait_fp(dc)
    argv = [a.replace(_ROOT, "/app") if a.startswith(_ROOT) else a for a in argv]
    cmd = dc + ["run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    for k, v in THREAD_ENV.items():
        cmd += ["-e", f"{k}={os.environ.get(k, v)}"]
    cmd += ["oscar", "bash", "-lc",
            "cd /app/object_retrieval && PYTHONPATH=/app exec python3 -u -m "
            "grasping.experiment_proxy_grasp " + " ".join(argv)]
    caps = ", ".join(f"{k}={os.environ.get(k, v)}" for k, v in THREAD_ENV.items())
    print(f"[host] running in the oscar container as uid {os.getuid()} ({caps})", flush=True)
    for attempt in range(4):
        rc = subprocess.call(cmd)
        if rc != 3:
            return rc
        print(f"[host] restarting foundationpose (attempt {attempt + 1}) ...", flush=True)
        subprocess.run(dc + ["restart", "foundationpose"], capture_output=True)
        _wait_fp(dc)
    return 3


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stage-5 proxy-grasp study",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--ranks", default="1-10", help="rank range of the Top-20 set to run")
    ap.add_argument("--cases", default=None, help="explicit case keys instead of --ranks, e.g. ycbv14,tless22")
    ap.add_argument("--exhibits", action="store_true", help="also run the mechanism exhibits (ycbv5, tless16)")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--per-object", type=int, default=PROTOCOL["sampling"]["per_object"],
                    help="instances per object (round-robin over scenes)")
    ap.add_argument("--min-visib", type=float, default=PROTOCOL["sampling"]["min_visib"],
                    help="instances below this BOP visib_fract are not used")
    ap.add_argument("--pose-source", default="fp", choices=("fp", "stage3"),
                    help="fp = call FoundationPose now; stage3 = reuse the archived Stage-3 poses "
                         "for gt/proxy (random still needs FP)")
    ap.add_argument("--fp-input", default="bop", choices=("bop", "sim"),
                    help="bop = real RGB-D + GT mask (Stage 3); sim = the PyBullet render")
    ap.add_argument("--world", default="auto", choices=("auto", "plane", "bop"),
                    help="sim world: auto = BOP extrinsics where present (YCB-V, T-LESS), else the "
                         "table plane fitted to the real depth (LM-O); plane = fit everywhere")
    ap.add_argument("--n-tries", type=int, default=PROTOCOL["executor"]["n_tries"],
                    help="executed grasp attempts per trial (k of success@k)")
    ap.add_argument("--no-exec", dest="exec", action="store_false",
                    help="pose + candidate metrics only, no grasp execution")
    ap.add_argument("--csv", default=os.path.join(OUT_DIR, "trials.csv"),
                    help="one row per finished trial; re-running resumes from it")
    ap.add_argument("--fresh", action="store_true", help="delete the CSV and start over")
    ap.add_argument("--check", action="store_true", help="verify data, FoundationPose and the world fit")
    ap.add_argument("--plan", action="store_true", help="print the frozen trial plan and exit")
    ap.add_argument("--report", action="store_true", help="write REPORT.md from the CSV and exit")
    ap.add_argument("--seed", type=int, default=0, help="numpy seed (sampler seed is in PROTOCOL)")
    ap.add_argument("--verbose", action="store_true", help="print every executed attempt (rose/hold/held)")
    ap.add_argument("--no-docker", action="store_true",
                    help="do not wrap into the oscar container (you are already in one)")
    args = ap.parse_args()

    # Everything but --plan/--report needs the sim stack -> run in the container.
    if not IN_CONTAINER and not args.no_docker and not (args.plan or args.report):
        conds_h = [c.strip() for c in args.conditions.split(",") if c.strip()]
        needs_fp = (not args.check) and any(
            c in ("gt", "proxy", "proxy_scaled") and args.pose_source == "fp" or c == "random"
            for c in conds_h)
        sys.exit(run_in_container(sys.argv[1:], needs_fp))

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        sys.exit(f"unknown condition(s) {bad}; choose from {CONDITIONS}")
    cases = select_cases(args.ranks, args.cases, args.exhibits)
    db = load_instances()
    plan = plan_trials(cases, db, args.per_object, args.min_visib)
    if args.n_tries != PROTOCOL["executor"]["n_tries"]:
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

    # ---- the plan ------------------------------------------------------------
    print(f"== plan: {len(cases)} objects, {len(plan)} instances, conditions {conds} "
          f"(per-object {args.per_object}, min visib {args.min_visib}) ==")
    cur = None
    for tr in plan:
        if tr["case"] != cur:
            cur = tr["case"]
            print(f"  #{tr['rank'] or '–'} {tr['case']:<8} {tr['name']:<15} proxy {tr['proxy']}"
                  f"   pool {tr['pool']}/{tr['n_top1']}")
        print(f"      s{tr['scene']:02d} im {tr['im']:04d} gt{tr['gt_idx']} visib {tr.get('visib')}")
    if args.plan:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "plan.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(dict(ts=_dt.datetime.now().isoformat(timespec="seconds"), args=vars(args),
                           conditions=conds, plan=plan), fh, indent=1)
        print(f"[plan] {len(plan)} instances x {len(conds)} conditions = "
              f"{len(plan) * len(conds)} trials -> {out}")
        return

    # ---- run -------------------------------------------------------------------
    if args.fresh and os.path.exists(args.csv):
        os.remove(args.csv)
    # rows that failed on the pose SERVICE (not on the trial) are re-run: a poisoned
    # CUDA context is not an outcome of the instance
    done = {_key(r): r for r in _load_csv(args.csv) if r.get("fail_reason") != "fp_error"}
    if done:
        print(f"[resume] {len(done)} finished trial(s) in {args.csv}")
    man = _write_manifest(args, plan, conds)
    print(f"[run] git {man['git'].get('rev')} dirty={man['git'].get('dirty')} · "
          f"pose-source {args.pose_source} · fp-input {args.fp_input} · world {args.world} · "
          f"n-tries {args.n_tries} · csv {args.csv}")
    import numpy as np
    np.random.seed(args.seed)
    ctx = Ctx(args)
    todo = [(tr, c) for tr in plan for c in conds
            if (tr["case"], tr["scene"], tr["im"], tr["gt_idx"], c) not in done]
    print(f"[run] {len(todo)} trials to run ({len(plan) * len(conds) - len(todo)} cached)\n")
    hdr = (f"{'#':>7} {'instance':<28}{'condition':<13}{'CAD':<30}{'result':<16}"
           f"{'cand/reach/blk':>15}{'D_sym':>7}{'place':>7}{'settle':>7}{'s':>6}")
    print(hdr)
    print("-" * len(hdr))
    fp_dead = 0
    times = []
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
        if row["fail_reason"] == "fp_error":
            fp_dead += 1
            if fp_dead >= 3:
                print("[run] FoundationPose failed 3x in a row — exiting with code 3 so the "
                      "runner can restart the service and resume")
                sys.exit(3)
        else:
            fp_dead = 0
        res = ("S@1 ✔" if row["first_succ"] else "S@k ✔" if row["succ"] else
               ("✘ " + (row["fail_reason"] or "?")))
        print(f"{i:>3}/{len(todo):<3} {label:<28}{cond:<13}{row['cad'].split('/')[-1][:29]:<30}"
              f"{res:<16}{row['n_cand']:>5}/{row['n_reach']:<4}/{row['n_blocked']:<4}"
              f"{str(row['dsym_mm']):>7}{str(row['place_mm']):>7}{str(row['settle_mm']):>7}"
              f"{row['runtime_s']:>6.0f}", flush=True)
        if i % 10 == 0 and times:
            eta = statistics.mean(times) * (len(todo) - i) / 60
            print(f"        … {i}/{len(todo)} done, mean {statistics.mean(times):.0f} s/trial, "
                  f"ETA {eta:.0f} min", flush=True)
    print("=" * len(hdr))
    if ctx.eval_fallback:
        print(f"[note] models_eval absent for {sorted(ctx.eval_fallback)} — targets were posed/"
              f"scored with the sim mesh instead (see --check)")
    rows = _load_csv(args.csv)
    md = report(rows, args.csv, conds)
    out = _write_report(args.csv, md)
    # console: the headline + paired tables only
    print("\n" + "\n".join(l for l in md.split("\n## 3.")[0].split("\n")[6:]))
    print(f"\n[out] {len(rows)} trials in {args.csv}\n[out] report -> {out}")


if __name__ == "__main__":
    main()
