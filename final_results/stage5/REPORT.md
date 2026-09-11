# Stage 5 — Proxy-grasp study (generated report)

CSV: `_s5_out/proxy_grasp/trials.csv` · 180 trials · 10 objects · 60 instances · pose source: ['fp', 'sim_true'] · world: ['bop', 'plane']
Runs: 3, last 2026-09-11T12:52:09 on 076dd9f3fc1b (git e937a3c8ab04, dirty=True)

success@k = trial succeeded within ≤ n_tries executed attempts (headline); success@1 = the first executed candidate succeeded; attempts S/N = per-attempt rate. No confidence intervals by agreement (2026-09-03): Δ plus the per-instance win split.

## 1. Success rate

**YCBV (3 objects)**

| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt_pose | 3 | 18 | 2/18 (11%) | 0/18 (0%) | 2/55 (4%) | 40.0 | 11.5 | 39 | 0 | 0 | – | 8.5 | 22.9 |
| gt | 3 | 18 | 5/18 (28%) | 1/18 (6%) | 5/53 (9%) | 40.0 | 13.0 | 22 | 0 | 0 | 2.0 | 11.4 | 25.4 |
| proxy | 3 | 18 | 8/18 (44%) | 6/18 (33%) | 8/51 (16%) | 40.0 | 11.0 | 14 | 0 | 0 | 4.2 | 11.1 | 8.1 |

**TLESS (6 objects)**

| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt_pose | 6 | 36 | 17/36 (47%) | 10/36 (28%) | 17/54 (31%) | 40.0 | 9.5 | 35 | 0 | 0 | – | 0.0 | 2.5 |
| gt | 6 | 36 | 20/36 (56%) | 11/36 (31%) | 20/61 (33%) | 40.0 | 10.5 | 34 | 0 | 0 | 1.5 | 3.0 | 7.5 |
| proxy | 6 | 36 | 8/36 (22%) | 6/36 (17%) | 8/69 (12%) | 16.0 | 3.0 | 20 | 0 | 0 | 7.2 | 15.8 | 3.2 |

**LMO (1 objects)**

| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt_pose | 1 | 6 | 5/6 (83%) | 5/6 (83%) | 5/10 (50%) | 40.0 | 12.5 | 3 | 0 | 0 | – | 0.0 | 0.8 |
| gt | 1 | 6 | 6/6 (100%) | 4/6 (67%) | 6/9 (67%) | 40.0 | 13.0 | 3 | 0 | 0 | 3.8 | 7.9 | 1.9 |
| proxy | 1 | 6 | 3/6 (50%) | 1/6 (17%) | 3/22 (14%) | 40.0 | 15.5 | 13 | 0 | 0 | 14.8 | 32.5 | 20.9 |

**ALL datasets (pooled — composition differs per dataset)**

| condition | objects | trials | success@k | success@1 | attempts S/N | cand (med) | reach (med) | blocked | no cand | FP fail | D_sym med [mm] | place med [mm] | s/trial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gt_pose | 10 | 60 | 24/60 (40%) | 15/60 (25%) | 24/119 (20%) | 40.0 | 11.0 | 77 | 0 | 0 | – | 0.0 | 5.2 |
| gt | 10 | 60 | 31/60 (52%) | 16/60 (27%) | 31/123 (25%) | 40.0 | 11.5 | 59 | 0 | 0 | 1.7 | 6.2 | 11.9 |
| proxy | 10 | 60 | 19/60 (32%) | 13/60 (22%) | 19/142 (13%) | 40.0 | 6.0 | 47 | 0 | 0 | 6.2 | 14.2 | 6.8 |

## 2. Paired comparisons

**All counted objects** (same instance under both conditions; Δ in percentage points)

| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |
|---|---|---|---|---|---|
| gt → proxy | 60 | 52% | 32% | -20 pp | 18 : 6 : 13 : 23 |
| gt_pose → gt | 60 | 40% | 52% | +12 pp | 0 : 7 : 24 : 29 |
| gt_pose → proxy | 60 | 40% | 32% | -8 pp | 13 : 8 : 11 : 28 |

**YCBV** (same instance under both conditions; Δ in percentage points)

| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |
|---|---|---|---|---|---|
| gt → proxy | 18 | 28% | 44% | +17 pp | 2 : 5 : 3 : 8 |
| gt_pose → gt | 18 | 11% | 28% | +17 pp | 0 : 3 : 2 : 13 |
| gt_pose → proxy | 18 | 11% | 44% | +33 pp | 0 : 6 : 2 : 10 |

**TLESS** (same instance under both conditions; Δ in percentage points)

| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |
|---|---|---|---|---|---|
| gt → proxy | 36 | 56% | 22% | -33 pp | 13 : 1 : 7 : 15 |
| gt_pose → gt | 36 | 47% | 56% | +8 pp | 0 : 3 : 17 : 16 |
| gt_pose → proxy | 36 | 47% | 22% | -25 pp | 11 : 2 : 6 : 17 |

**LMO** (same instance under both conditions; Δ in percentage points)

| A → B | n paired | A | B | Δ (B−A) | A only : B only : both : neither |
|---|---|---|---|---|---|
| gt → proxy | 6 | 100% | 50% | -50 pp | 3 : 0 : 3 : 0 |
| gt_pose → gt | 6 | 83% | 100% | +17 pp | 0 : 1 : 5 : 0 |
| gt_pose → proxy | 6 | 83% | 50% | -33 pp | 2 : 0 : 3 : 1 |

## 3. Per object

| rank | object | proxy (fixed) | inst | gt_pose succ@k | gt_pose D_sym | gt succ@k | gt D_sym | proxy succ@k | proxy D_sym |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ycbv 14 mug | cup-red_heart | 6 | 2/6 | – | 4/6 | 2.1 | 5/6 | 4.2 |
| 2 | tless 30 obj_30 | BIA_Porcelain_Ramekin_With_Glazed_ | 6 | 5/6 | – | 6/6 | 1.2 | 0/6 | 4.8 |
| 3 | tless 19 obj_19 | obj_000018 | 6 | 2/6 | – | 2/6 | 1.7 | 1/6 | 6.2 |
| 4 | ycbv 2 cracker box | Nestle_Carnation_Cinnamon_Coffeeca | 6 | 0/6 | – | 1/6 | 2.0 | 3/6 | 5.0 |
| 5 | ycbv 17 scissors | Diamond_Visions_Scissors_Red | 6 | 0/6 | – | 0/6 | 2.0 | 0/6 | 3.4 |
| 6 | tless 22 obj_22 | obj_000026 | 6 | 4/6 | – | 4/6 | 1.5 | 2/6 | 8.3 |
| 7 | lmo 9 duck | CHICKEN_RACER | 6 | 5/6 | – | 6/6 | 3.8 | 3/6 | 14.8 |
| 8 | tless 10 obj_10 | obj_000018 | 6 | 6/6 | – | 6/6 | 1.3 | 3/6 | 9.3 |
| 9 | tless 28 obj_28 | obj_000018 | 6 | 0/6 | – | 0/6 | 1.5 | 0/6 | 6.8 |
| 10 | tless 18 obj_18 | obj_000013 | 6 | 0/6 | – | 2/6 | 1.4 | 2/6 | 6.8 |

## 4. Failure taxonomy and validity

| condition | success | grasp_failed | all_blocked | unreachable | no_candidates | fp_error | cad_missing |
|---|---|---|---|---|---|---|---|
| gt_pose | 24 | 21 | 11 | 4 | 0 | 0 | 0 |
| gt | 31 | 16 | 9 | 4 | 0 | 0 | 0 |
| proxy | 19 | 24 | 7 | 10 | 0 | 0 | 0 |

- settle_mm: median 2.20mm, min 0.20, max 22.40 (n=180)
- plane_angle_deg: median 0.65°, min 0.24, max 73.92 (n=162)
- bottom_gap_mm: median 0.00mm, min -39.30, max 0.00 (n=180)
- plane_inlier: median 0.88, min 0.15, max 1.00 (n=180)
- visib: median 0.94, min 0.50, max 1.00 (n=180)
- applicability (0 antipodal candidates on the target's own CAD within the gripper): none

## 6. Predeclared exclusions

- ycbv 1 master chef can: Anwendbarkeit: 102 mm Durchmesser > 80 mm Greiferöffnung (vorab, aus der Geometrie)
- tless 20 obj_20: FoundationPose scheitert im Sim schon mit dem eigenen CAD (Pilot 2026-09-06)
- lmo 11 glue: Bbox-Abweichung 0.05 täuscht — Pose-Median 64 mm

## 7. Protocol

```
{
 "sampler": {
  "n_samples": 800,
  "top_k": 40,
  "friction_mu": 0.5,
  "n_approach": 4,
  "seed": 0,
  "gripper_min_width_m": 0.005,
  "gripper_max_width_m": 0.08,
  "method": "antipodal contact pairs inside the friction cone [R1, R2]"
 },
 "executor": {
  "n_tries": 5,
  "pregrasp_m": 0.12,
  "lift_m": 0.15,
  "rise_min_m": 0.05,
  "hold_steps": 240,
  "shake_amp_m": 0.05,
  "in_hand_m": 0.15,
  "close_force_n": [
   20,
   120
  ],
  "reach_tol_mm": 30,
  "blocked_mm": 30,
  "success": "rose >= 5 cm AND held 1 s AND survived \u00b15 cm shakes [R3]; trial = any of <= n_tries attempts, full reset before each"
 },
 "physics": {
  "engine": "PyBullet [R14]",
  "timestep_s": 0.004166666666666667,
  "gravity": -9.81,
  "target_mass_kg": "sim_scene.OBJECT_MASS_KG (YCB object list) else 0.2",
  "target_friction": 1.0,
  "finger_friction": 1.0,
  "friction_combination": "product (PyBullet; measured 1.0x1.0 -> 1.01, 1.6x1.5 -> 2.42)",
  "settle_steps": 60,
  "collision": "V-HACD [R13] for the target / concave static clutter"
 },
 "robot": {
  "arm": "Franka Panda (pybullet_data franka_panda/panda.urdf)",
  "pedestal_m": 0.35,
  "placement": "on the camera's side: 0.55 m from the object centroid along the camera viewing direction projected onto the table"
 },
 "world": {
  "frame": "auto: BOP extrinsics where present (YCB-V, T-LESS), else the table plane fitted to the real depth (LM-O)",
  "table_plane": "RANSAC [R12], objects masked out, \u00b10.5 m around the object depths, thr 6 mm, candidate planes must carry every object 0\u201340 cm above them",
  "table_z": "0 = lowest object vertex (bop) / the fitted plane; lowered to the target's own lowest vertex if that is below"
 },
 "pose": {
  "method": "FoundationPose [R4] via eval_bop_pose.estimate_pose",
  "fp_refine_iter": 5,
  "input": "real RGB-D + BOP GT mask_visib (Stage 3b)",
  "d_sym": "stage3_metrics.d_sym, 10 000 surface samples, seed 0"
 },
 "sampling": {
  "min_visib": 0.5,
  "per_object": 6,
  "rule": "round-robin over scenes, evenly spaced frames within a scene"
 }
}
```

Caveats: simulation only; clutter is static and consists of the annotated objects (LM-O's unannotated clutter is absent); masks are BOP GT (`mask_visib`), as in Stage 3; the `gt_pose` ceiling uses the settled sim pose, so it also bounds the sampler+executor.
