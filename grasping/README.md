# Stage 5 — Perceive-then-Grasp demo

The downstream demonstration of the OSCAR+ thesis: a robot with an RGB-D camera
looks at a cluttered YCB-V tabletop, is asked (by a text prompt) to grasp one
object, and runs the **full pipeline** to do it — segment → retrieve a **proxy**
CAD (the target's own model is *not* in the gallery) → estimate its pose with
FoundationPose → sample antipodal grasps on the proxy → execute with a Franka
Panda in a PyBullet physics simulation.

It is built as thin, CLI-runnable modules over the existing pipeline — nothing
is re-implemented.

```
RGB-D scene (YCB-V) ─5.1─▶ segment target ─5.2─▶ retrieve PROXY (G_proxy, exact
CAD excluded) ─5.2─▶ FoundationPose(proxy) → pose ─E1─▶ antipodal grasps ─5.3─▶
Panda: pre-grasp → approach → close → lift → success + GIF
```

## Modules (each importable and a standalone CLI)

| file | stage | what it does | reuses |
|---|---|---|---|
| `antipodal_grasp_sampler.py` | E1 | sample + score parallel-jaw antipodal grasps on a mesh; matplotlib render | trimesh |
| `sim_scene.py` | 5.1 | rebuild a YCB-V scene in PyBullet (objects at GT poses) + Panda + RGB-D camera | YCB-V GT, YCB meshes |
| `perceive.py` | 5.2 | segment → retrieve top-1 proxy from **G_proxy** → FoundationPose | `stage3_gallery.assemble_gallery`, `eval_common.run_query`/`fusion_ranking`/`crop_by_bbox`, `pipeline.foundationpose_bridge.call_foundationpose` |
| `grasp_execute.py` | 5.3 | Panda IK grasp: pre-grasp → approach → close → lift → success | `antipodal_grasp_sampler`, `sim_scene` |
| `stage5_demo.py` | top | the full loop with a `--prompt` CLI | all of the above |

## Run it

Dependencies (`pybullet`, `rtree`) are baked into the `oscar` image (Dockerfile);
if you're on an older image, `pip install pybullet rtree` first. Everything runs
inside the `oscar` container:

```bash
docker compose run --rm oscar bash -lc "cd /app && <command>"
```

```bash
# 5.1 — just render the reconstructed scene (RGB + depth + seg)
python -m grasping.sim_scene   --scene 000048 --no-robot --out /tmp/scene48

# E1 — grasp candidates + visualization on any mesh
python -m grasping.antipodal_grasp_sampler object_database/ycbv/obj_000005/textured_simple.obj \
       --render /tmp/grasps.png --out /tmp/grasps.json

# 5.2 — segment → retrieve proxy → pose (needs the GPU: encoders + FoundationPose)
python -m grasping.perceive    --scene 000048 --target 6 --topk 10

# 5.3 — grasp mechanics test (grasps the target's own mesh at GT pose)
python -m grasping.grasp_execute --scene 000048 --target 19 --gif /tmp/grasp.gif

# top — the whole thing, prompt-driven
python -m grasping.stage5_demo --scene 000048 --prompt "the mustard bottle" --gif /tmp/demo.gif
python -m grasping.stage5_demo --scene 000048 --target 19 --no-pose            # mechanics only (no GPU)
```

`--no-pose` skips retrieval + FoundationPose and grasps using the GT object pose
+ its own mesh — the CPU-only path to test the robot mechanics in isolation.
Without it, the demo runs the real open-set perception (GPU).

## Key parameters
- `--scene` YCB-V test scene id (e.g. `000048`, objects `[1,6,14,19,20]`).
- `--prompt` free text → matched to a scene object by its YCB name; or `--target <obj_id>`.
- `--uni3d` use the Uni3D shape arm instead of ULIP-2.
- `--gif` record the attempt. `--gui` opens the interactive PyBullet window.
- Gripper model: `GripperConfig` in `antipodal_grasp_sampler.py` (parallel-jaw,
  8 cm max width by default — objects wider than that are only graspable at thin
  features; pick a graspable target or widen the gripper).

## Notes / status
- **5.1** verified: scene reconstructs and renders with textures.
- **5.3** grasp execution is the fiddly part (Panda IK + finger-frame + physics);
  the finger axis is the hand's local **y**. Some YCB objects exceed the 8 cm
  gripper (cans, boxes) — the clamps, mug, and smaller items grasp cleanly.
- **5.2** requires the GPU (retrieval encoders + FoundationPose service running:
  `docker compose up -d foundationpose`); the exact target CAD is excluded by
  construction (query = a YCB object, gallery = GSO ∪ HouseCat6D ∪ ITODD).
