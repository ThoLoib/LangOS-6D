"""
gedi_server.py — HTTP server for GeDi descriptor computation.

Runs inside the GeDi Docker container. OSCAR calls this via HTTP
to compute GeDi descriptors without dependency conflicts.

Endpoints:
    GET  /health              — Health check
    POST /compute_descriptors — Compute GeDi descriptors on a point cloud

Protocol (same pattern as FoundationPose bridge):
    Request:  JSON with base64-encoded point cloud data
    Response: JSON with base64-encoded descriptors + keypoint indices
"""

import argparse
import base64
import io
import logging
import sys
import time

import numpy as np

sys.path.insert(0, "/gedi")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GeDi] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load GeDi model
# ---------------------------------------------------------------------------

_gedi_instance = None
_gedi_config = {
    "dim": 32,
    "samples_per_batch": 500,
    "samples_per_patch_lrf": 4000,
    "samples_per_patch_out": 512,
    "r_lrf": 0.5,
    "fchkpt_gedi_net": "/gedi/data/chkpts/3dmatch/chkpt.tar",
}


def get_gedi():
    global _gedi_instance
    if _gedi_instance is None:
        logger.info("Loading GeDi model...")
        t0 = time.time()
        from gedi import GeDi
        _gedi_instance = GeDi(config=_gedi_config)
        logger.info("GeDi loaded in %.1fs", time.time() - t0)
    return _gedi_instance


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "gedi"})


@app.route("/compute_descriptors", methods=["POST"])
def compute_descriptors():
    """Compute GeDi descriptors on a point cloud.

    Request JSON:
        points: base64-encoded float32 array (N, 3)
        num_keypoints: int (optional, default 5000)
        r_lrf: float (optional, override LRF radius)

    Response JSON:
        keypoint_indices: list of int — indices into the input points
        descriptors: base64-encoded float32 array (K, dim)
        dim: int — descriptor dimension
        num_keypoints: int — actual number of keypoints computed
    """
    import torch

    try:
        data = request.get_json()

        # Decode point cloud
        pts_bytes = base64.b64decode(data["points"])
        pts_np = np.frombuffer(pts_bytes, dtype=np.float32).reshape(-1, 3)

        num_kp = data.get("num_keypoints", 5000)
        num_kp = min(num_kp, len(pts_np))

        if len(pts_np) < 100:
            return jsonify({
                "error": f"Point cloud too small ({len(pts_np)} points)",
                "keypoint_indices": [],
                "descriptors": "",
                "dim": _gedi_config["dim"],
                "num_keypoints": 0,
            }), 400

        gedi = get_gedi()

        # Sample keypoints
        kp_indices = np.random.choice(len(pts_np), num_kp, replace=False)
        kp_pts = torch.tensor(pts_np[kp_indices]).float()
        pcd_tensor = torch.tensor(pts_np).float()

        # Compute descriptors
        t0 = time.time()
        descriptors = gedi.compute(pts=kp_pts, pcd=pcd_tensor)
        elapsed = time.time() - t0

        logger.info(
            "Computed %d descriptors (dim=%d) on %d-point cloud in %.2fs",
            len(descriptors), descriptors.shape[1], len(pts_np), elapsed,
        )

        # Encode response
        desc_bytes = descriptors.astype(np.float32).tobytes()

        return jsonify({
            "keypoint_indices": kp_indices.tolist(),
            "descriptors": base64.b64encode(desc_bytes).decode("ascii"),
            "dim": int(descriptors.shape[1]),
            "num_keypoints": len(kp_indices),
            "compute_time_s": elapsed,
        })

    except Exception as exc:
        logger.exception("compute_descriptors failed")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Pre-load model
    logger.info("Pre-loading GeDi model...")
    get_gedi()

    logger.info("GeDi server starting on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=False)
