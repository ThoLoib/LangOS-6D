#!/usr/bin/env bash
# =============================================================================
# Stage-3 new-config run: Uni3D (partial pc-query) + dGeDi geometry re-rank.
# Launch ONLY after the current 3b run has finished (it holds the GPU).
#
# Steps (each logged; GPU-serial to avoid contention):
#   0. dGeDi GPU smoke + real latency (fail fast)
#   1. Uni3D partial gallery caches (oscar container)         [encode, GPU]
#   2. dGeDi gallery descriptor cache (dgedi container)       [encode, GPU]
#   3. bring up dgedi + foundationpose services               [:5061 / :5050]
#   4. Stage-3 3a with --uni3d --dgedi --pose  -> Recall/MRR + BOP-AR
#   5. Stage-3 3b with --uni3d --dgedi --pose  -> proxy pose + D_sym
#
# 3a+3b both include POSE (FoundationPose): BOP-AR (3a) and D_sym (3b) under the
# new Uni3D+dGeDi retrieval. This is a long run (~1.5-2 days) — pose per instance
# dominates. Same OUT dir; the driver names outputs per mode (stage3a/stage3b).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=object_retrieval/results_bop_stage3_uni3d_dgedi
CACHE=object_retrieval/.dgedi_gallery
LOG=_stage3_newcfg.log
exec > >(tee -a "$LOG") 2>&1
echo "===== $(date -Is) START new-config Stage-3 ====="

echo "[0/4] dGeDi GPU smoke + real per-pair latency (fail fast before precompute)"
docker compose run --rm -T dgedi python3 - <<'PYEOF'
import time, sys, numpy as np
sys.path.insert(0, "/dgedi"); sys.path.insert(0, "/oscar/dgedi_service")
import server
server._STATE["model"] = server.load_model("/dgedi/config_dgedi.yaml", "multi_scale", "cuda")
server._STATE["device"] = "cuda"
# real dGeDi features on two synthetic clouds -> one RANSAC+chamfer pair, timed
pq, fq = server.compute_feats(np.random.randn(6000,3).astype(np.float32))
pt, ft = server.compute_feats(np.random.randn(6000,3).astype(np.float32))
print("feat dim =", fq.shape[1], "| query pts =", len(pq.points))
kq, kfq = server._keypoints(pq, fq, server.RANSAC_KEYPOINTS)
kt, kft = server._keypoints(pt, ft, server.RANSAC_KEYPOINTS)
server.ransac_only(kq, kfq, kt, kft, 0.03)  # warm
N=20; t0=time.time()
for _ in range(N):
    r = server.ransac_only(kq, kfq, kt, kft, 0.03)
    T=np.asarray(r.transformation); qa=np.asarray(pq.points)@T[:3,:3].T+T[:3,3]
    server.trimmed_chamfer(qa, np.asarray(pt.points), 0.1)
per=(time.time()-t0)/N
print(f"REAL per-pair: {per*1000:.1f} ms | K=20 {per*20:.2f}s/query | full-3a-K20 ~{per*20*12284/3600:.1f}h")
print("dGeDi GPU smoke OK")
PYEOF

echo "[1/4] Uni3D partial gallery precompute (all datasets)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && python3 precompute_uni3d_partial.py --datasets all"

echo "[2/4] dGeDi gallery descriptor precompute"
docker compose run --rm dgedi python3 /oscar/dgedi_service/precompute_gallery.py \
  --manifest /oscar/$CACHE/manifest.json --out /oscar/$CACHE

echo "[3/5] dgedi + foundationpose services up"
docker compose up -d dgedi foundationpose
# wait for dgedi health
for i in $(seq 1 60); do
  if docker compose exec -T dgedi python3 -c \
      "import urllib.request;urllib.request.urlopen('http://localhost:5061/health',timeout=5)" 2>/dev/null; then
    echo "  dgedi healthy"; break; fi
  sleep 5
done
# wait for foundationpose health (pose needs it; else pose degrades to misses)
for i in $(seq 1 60); do
  if curl -fs http://localhost:5050/health >/dev/null 2>&1; then
    echo "  foundationpose healthy"; break; fi
  sleep 5
done

# dGeDi geometry re-rank = E2_both (RANSAC fitness + trimmed Chamfer, Borda,
# no ICP). Stage-1's best used geometry_k=50; with 6x the queries here that is
# ~expensive, so start at K=20 (Stage-1 K=20 was 0.6292 vs 0.6428 at K=50) and
# bump after the smoke test measures real per-pair RANSAC throughput.
echo "[4/5] Stage-3 3a WITH POSE (Uni3D + partial + dGeDi E2_both -> Recall/MRR + BOP-AR)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && python3 -u eval_bop_pose.py \
     --datasets all --mode 3a --pose --uni3d --dgedi --dgedi-top-k 20 \
     --output $OUT"

echo "[5/5] Stage-3 3b WITH POSE (Uni3D + partial + dGeDi -> proxy pose + D_sym)"
docker compose run --rm oscar bash -lc \
  "cd /app/object_retrieval && python3 -u eval_bop_pose.py \
     --datasets all --mode 3b --pose --uni3d --dgedi --dgedi-top-k 20 \
     --output $OUT"

echo "===== $(date -Is) DONE -> $OUT ====="
