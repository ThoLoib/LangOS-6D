import json
import numpy as np
import os
import point_cloud_utils as pcu
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm  # <-- add tqdm
import trimesh

# ---------------------------
# Configuration
# ---------------------------
EVAL_FILE = Path("../object_retrieval/results_threshold_eval_ycbv_gso/results_thr_0.35.json")
MODEL_DIR = Path("../object_database/gso/models_orig")
CACHE_FILE = Path("asfddg.json")

# ---------------------------
# Utilities
# ---------------------------

@lru_cache(maxsize=None)
def load_point_cloud(label: str, base_dir: Path = MODEL_DIR, num_points: int = 1024) -> np.ndarray:
    """
    Load and cache a point cloud for an object label.
    Supports either `points.xyz` or `.glb` mesh files.
    """

    xyz_file = base_dir / label / "points.xyz"
    glb_file = base_dir / label / "meshes" /  "model.obj"  # adjust if your .glb filename differs
    print(glb_file)

    # For housecat6d, files are directly under MODEL_DIR
    #xyz_file = base_dir / f"{label}.xyz"
    #glb_file = base_dir / f"{label}.glb"

    if xyz_file.exists():
        # Load from existing point cloud
        return np.loadtxt(xyz_file)

    elif glb_file.exists():
        # Load mesh and sample points
        mesh = trimesh.load(glb_file)
        if not isinstance(mesh, trimesh.Trimesh) and isinstance(mesh, trimesh.Scene):
            # Convert scene to single mesh
            mesh = trimesh.util.concatenate(mesh.dump(concatenate=True))
        # Sample points uniformly on mesh surface
        points = mesh.sample(num_points)
        return points

    else:
        raise FileNotFoundError(f"No point cloud or GLB file found for label: {label}")

def sinkhorn_distance(pc1: np.ndarray, pc2: np.ndarray, eps: float = 1e-3) -> float:
    """
    Compute the Sinkhorn distance between two point clouds.
    """
    if pc1.size == 0 or pc2.size == 0:
        return np.inf

    # Uniform weights (normalized)
    w1 = np.ones(pc1.shape[0]) / pc1.shape[0]
    w2 = np.ones(pc2.shape[0]) / pc2.shape[0]

    # Pairwise L2 distances
    M = pcu.pairwise_distances(pc1, pc2)

    # Optimal transport plan
    P = pcu.sinkhorn(w1, w2, M, eps=eps)

    return float((M * P).sum())

def load_cache() -> Dict[tuple, float]:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        return {tuple(k.split("|||")): v for k, v in data.items()}
    return {}

def save_cache(cache: Dict[tuple, float]):
    with open(CACHE_FILE, "w") as f:
        json.dump({"|||".join(k): v for k, v in cache.items()}, f, indent=2)

def evaluate(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    cache: Dict[tuple, float] = load_cache()

    # Extract all unique relevant pairs
    relevant_pairs = {(item["gt"], item["pred"]) for item in dataset if item["pred"] is not None}

    # Precompute distances for only relevant pairs
    for gt, pred in tqdm(relevant_pairs, desc="Computing Sinkhorn distances", unit="pair"):
        if (gt, pred) in cache:
            continue
        if gt == pred:
            cache[(gt, pred)] = 0.0
        else:
            try:
                pc_gt = load_point_cloud(gt)
                pc_pred = load_point_cloud(pred)
                dist = sinkhorn_distance(pc_gt, pc_pred)
            except Exception as e:
                dist = np.inf
                print(f"[Warning] Could not process {gt} vs {pred}: {e}")
            cache[(gt, pred)] = dist

    # Evaluate dataset using precomputed distances
    for item in tqdm(dataset, desc="Evaluating dataset", unit="item"):
        pair = (item["gt"], item["pred"])
        dist = cache.get(pair, np.inf)
        results.append({
            "scene": item.get("scene"),
            "img_id": item.get("img_id"),
            "inst_id": item.get("inst_id"),
            "gt": item["gt"],
            "pred": item["pred"],
            "sinkhorn_dist": dist
        })

    save_cache(cache)
    return results


def main():
    with open(EVAL_FILE, "r") as f:
        dataset = json.load(f)


    results = evaluate(dataset)

    # Save evaluation results
    results_file = Path("sinkhorn_results_ycbv_gso.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {results_file.resolve()}")

    # Compute average Sinkhorn distance
    valid_distances = [r["sinkhorn_dist"] for r in results if np.isfinite(r["sinkhorn_dist"])]
    avg_distance = np.mean(valid_distances) if valid_distances else np.inf
    print(f"Average Sinkhorn distance over dataset: {avg_distance:.6f}")

if __name__ == "__main__":
    main()
