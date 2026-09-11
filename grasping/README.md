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
archived Stage-3 poses for `gt`/`proxy`; no GPU needed except for `random`), `--n-tries 5`, `--no-exec`,
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
- Rendering (the demo's GIFs) is CPU: the "EGL" plugin in this image is
  llvmpipe, off by default (`GRASP_EGL=1`).

## References (cited in the code as [Rn])

The grasp planner, the success test and the simulation setup are simple,
transparent implementations of published ideas; the concrete parameter choices
(800 samples, μ = 0.5, ±5 cm shakes, 0.35 m pedestal, …) are this project's own.

- [R1] V.-D. Nguyen. *Constructing force-closure grasps.* The International Journal of
  Robotics Research 7(3), 1988. — antipodal contacts inside their friction cones give a
  force-closure parallel-jaw grasp (the sampler's acceptance test).
- [R2] I.-M. Chen and J. W. Burdick. *Finding antipodal point grasps on irregularly shaped
  objects.* IEEE Transactions on Robotics and Automation 9(4), 1993. — searching antipodal
  point pairs on arbitrary surfaces (the sampler's search idea).
- [R3] C. Eppner, A. Mousavian, D. Fox. *ACRONYM: A large-scale grasp dataset based on
  simulation.* ICRA 2021. — simulated grasp success judged by a shaking motion (the shake test).
- [R4] B. Wen, W. Yang, J. Kautz, S. Birchfield. *FoundationPose: Unified 6D pose estimation
  and tracking of novel objects.* CVPR 2024. — the pose estimator (HTTP service, `pipeline/foundationpose_bridge.py`).
- [R5] T. Hodaň et al. *BOP: Benchmark for 6D object pose estimation.* ECCV 2018; and
  T. Hodaň et al. *BOP Challenge 2020 on 6D object localization.* ECCV Workshops 2020. — the
  BOP dataset format (`scene_gt.json`, `scene_camera.json`, `mask_visib`, `visib_fract`,
  `models_eval`) used throughout.
- [R6] Y. Xiang, T. Schmidt, V. Narayanan, D. Fox. *PoseCNN: A convolutional neural network for
  6D object pose estimation in cluttered scenes.* RSS 2018. — YCB-Video (YCB-V).
- [R7] T. Hodaň, P. Haluza, Š. Obdržálek, J. Matas, M. Lourakis, X. Zabulis. *T-LESS: An RGB-D
  dataset for 6D pose estimation of texture-less objects.* WACV 2017.
- [R8] E. Brachmann, A. Krull, F. Michel, S. Gumhold, J. Shotton, C. Rother. *Learning 6D object
  pose estimation using 3D object coordinates.* ECCV 2014. — LM-O (Occluded LINEMOD).
- [R9] B. Drost, M. Ulrich, P. Bergmann, P. Härtinger, C. Steger. *Introducing MVTec ITODD — A
  dataset for 3D object recognition in industry.* ICCV Workshops 2017.
- [R10] L. Downs et al. *Google Scanned Objects: A high-quality dataset of 3D scanned household
  items.* ICRA 2022.
- [R11] H. Jung et al. *HouseCat6D: A large-scale multi-modal category level 6D object perception
  dataset with household objects in realistic scenarios.* CVPR 2024.
- [R12] M. A. Fischler and R. C. Bolles. *Random sample consensus: A paradigm for model fitting
  with applications to image analysis and automated cartography.* Communications of the ACM
  24(6), 1981. — the table-plane fit.
- [R13] K. Mamou and F. Ghorbel. *A simple and efficient approach for 3D mesh approximate convex
  decomposition.* ICIP 2009. — V-HACD, the collision shape of the dynamic target.
- [R14] E. Coumans and Y. Bai. *PyBullet, a Python module for physics simulation for games,
  robotics and machine learning.* 2016–2021, http://pybullet.org — the simulator and the
  Franka Panda model (`pybullet_data`).
- [R15] C. Gümeli, A. Dai, M. Nießner. *ROCA: Robust CAD model retrieval and alignment from a
  single image.* CVPR 2022. — the category-substitute (proxy CAD) framing the thesis builds on.
- [R16] B. Calli, A. Singh, A. Walsman, S. Srinivasa, P. Abbeel, A. M. Dollar. *The YCB object and
  model set: Towards common benchmarks for manipulation research.* ICAR 2015. — the textured
  YCB meshes (`object_database/ycbv/*/textured_simple.obj`).

Not from a publication and therefore uncited: the grasp scoring heuristic, the reachability
ordering, the two-stage close, the hold duration, the robot placement rule and the instance
sampling rule — all documented in `docs/STAGE5_PROTOCOL.md` and `PROTOCOL` in the script.
