# Stage 5 — Proxy-grasp study (and the perceive-then-grasp demo)

Downstream test of the OSCAR+ thesis: **is the retrieved proxy CAD good enough to act
on?** A BOP tabletop frame is rebuilt in PyBullet, the target's pose is estimated with
FoundationPose from the *real* RGB-D (exactly the Stage-3 path), antipodal grasps are
planned on the CAD under test and executed with a Franka Panda on the real target
geometry. The predeclared protocol — object set, instance lists, conditions, success
definition, predictions — is `docs/STAGE5_PROTOCOL.md`.

```
BOP frame ─▶ sim: annotated objects at GT poses, table at z=0 (BOP frame or depth-fitted plane), target dynamic
          ─▶ FoundationPose(CAD under test | real RGB-D + GT mask)  ─▶ pose, D_sym (Stage 3)
          ─▶ antipodal grasps on the posed CAD ─▶ Panda: reset → pre-grasp → approach → close
          ─▶ lift → hold 1 s → shake  ─▶ success@k, success@1, per-attempt rate
```

## Run it (from the repo root on the host; the script wraps itself into the `oscar` container)

```bash
python3 grasping/experiment_proxy_grasp.py --check      # data + FoundationPose + table-plane self-check
python3 grasping/experiment_proxy_grasp.py --plan       # print + freeze the trial plan (plan.json)
python3 grasping/experiment_proxy_grasp.py              # run rank 1–10, gt vs proxy (resumes from the CSV)
python3 grasping/experiment_proxy_grasp.py --report     # REPORT.md from the CSV
python3 repro_experiment.py --stage 5 [same flags]      # the eval_final entry point, identical
```

Useful flags (all pass straight through): `--ranks 1-20`, `--cases ycbv14,tless22`,
`--exhibits` (mustard, tless 16 — shown, not counted), `--per-object 8`, `--min-visib 0.5`,
`--conditions gt_pose,gt,proxy,random,proxy_scaled`, `--pose-source stage3` (reuse the
archived Stage-3 poses for `gt`/`proxy`; no GPU needed except for `random`), `--fp-input sim`
(pose from the PyBullet render instead of the real image), `--n-tries 5`, `--no-exec`,
`--fresh`, `--no-docker` (already inside the container). The host side starts FoundationPose
when a run needs it, runs the container as your user, caps threads (this laptop bluescreens
under sustained all-core load) and restarts FoundationPose when it stops answering (exit
code 3 → restart → resume).

Outputs in `_s5_out/proxy_grasp/`: `trials.csv` (one row per trial, appended + fsynced as it
finishes — a crash loses nothing, re-running resumes), `manifest.json` (args, protocol, git
revision, library versions, per run), `plan.json`, `REPORT.md`.

## Conditions (paired per instance; default `gt,proxy`, the rest via `--conditions`)

| condition | CAD for planning | pose | measures |
|---|---|---|---|
| `gt_pose` | target's own CAD | true settled sim pose | ceiling of sampler + executor in this scene |
| `gt` | target's own CAD | FoundationPose | pose-estimation loss alone |
| `proxy` | the fixed Stage-3 proxy, native size | FoundationPose | the implemented pipeline |
| `random` | hash-drawn random CAD from G_proxy (1257) | FoundationPose | chance baseline |
| `proxy_scaled` | proxy sized to the observed depth cloud | FoundationPose | optional size ablation |

Success of an attempt: rose ≥ 5 cm **and** held 1 s **and** survived ±5 cm shakes. Trial
success (`success@k`): any of ≤ 5 executed attempts; each attempt starts from a full reset.
Reporting follows the 2026-09-03 agreement: Δ plus the per-instance win split, per dataset
and pooled, no intervals.

## Files

| file | role |
|---|---|
| `proxy_grasp_cases.py` | the frozen Top-20 object set, fixed proxies, tiers, exclusions with reasons; proxy CAD resolution; the 1257-CAD pool + hash draw for `random` |
| `build_grasp_instances.py` | freezes the instance lists from the Stage-3 3b records (+ visibility, + Stage-3 poses) into `proxy_grasp_instances.json` |
| `proxy_grasp_instances.json` | 1440 instances / 25 cases: (scene, im, gt_idx) where the fixed proxy was the 3b top-1, with `visib`, `diameter`, `fp_proxy` and `fp_gt` poses |
| `experiment_proxy_grasp.py` | the study: plan → trials → CSV → report (`--check/--plan/--report`); wraps itself into the container, starts/restarts FoundationPose |
| `sim_scene.py` | BOP frame → PyBullet world (BOP extrinsics or the depth-fitted table plane, object-constrained), V-HACD target, Panda on the camera side, RGB-D render; `load_bop_frame`, `scene_from_frame` |
| `antipodal_grasp_sampler.py` | analytic parallel-jaw antipodal sampler (trimesh only) |
| `grasp_execute.py` | Panda IK executor: reset → pre-grasp → approach (blocked check) → two-stage close → lift → hold → shake |
| `perceive.py`, `stage5_demo.py`, `experiment_scale_ablation.py` | the interactive demo: live OSCAR+ retrieval of a proxy (gallery load, GPU) + FoundationPose + grasp; scale-fit ablation |

## Demo (interactive, YCB-V, live retrieval)

```bash
docker compose up -d foundationpose
docker compose run --rm oscar bash -lc "cd /app/object_retrieval && PYTHONPATH=/app \
    python3 -m grasping.stage5_demo --scene 000048 --prompt 'the mug' --proxy gso --gif /tmp/demo.gif"
python -m grasping.sim_scene --dataset tless --scene 000008 --frame 194 --out /tmp/s   # render only
python -m grasping.grasp_execute --scene 000048 --target 14                            # mechanics only
```

The demo bypasses GroundingDINO/SAM (prompt → YCB name, mask from the seg buffer) and must
not be presented as a language-grounding evaluation. Only GSO is a complete local proxy
gallery on this laptop (`--proxy gso`).

## Caveats
- Simulation only; clutter is static and consists of the annotated objects (LM-O's
  unannotated clutter is missing); masks are BOP GT, as in Stage 3.
- YCB-V has no `models_eval` on this laptop: targets are posed/scored with the textured YCB
  mesh (`--check` reports it).
- The rendering path (`--fp-input sim`, GIFs) is CPU: the "EGL" plugin in this image is
  llvmpipe, off by default (`GRASP_EGL=1`).
