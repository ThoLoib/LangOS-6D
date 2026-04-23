"""
precompute_ulip_query_embeddings.py
====================================

Standalone pre-encoder for ULIP-2 cross-modal query embeddings.

Run this BEFORE retrieval_mi3dor_eval_oscarplus.py when the GPU does not have
enough VRAM to hold ViT-bigG-14 alongside CLIP + DINOv2 + PointBERT.

Strategy
--------
Load ONLY OpenCLIP ViT-bigG-14 in float16 (~5 GB).  With nothing else in
VRAM, it fits on a 6 GB GPU.  Batch-encode all query images, save embeddings
to a .pt cache file.  The main eval script detects the cache and skips the
per-query forward pass entirely.

How to run
----------
    cd OSCAR/object_retrieval
    python precompute_ulip_query_embeddings.py

Edit the CONFIG block to match your dataset / paths.
The output file (ulip_query_cache_path) must match the same variable in
retrieval_mi3dor_eval_oscarplus.py.
"""

import os
import sys
import glob as _glob

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# ============================================================================
# CONFIG  — keep in sync with retrieval_mi3dor_eval_oscarplus.py
# ============================================================================
bop_root            = "../eval/datasets/mi3dor/image/test"
ulip_query_cache_path = "ulip_query_cache_mi3dor.pt"   # output file

# OpenCLIP model (must match what ULIP-2 uses)
openclip_model      = "ViT-bigG-14"
openclip_pretrained = "laion2b_s39b_b160k"

# Encoding settings
batch_size = 8      # conservative default for 6 GB VRAM; raise if no OOM
dtype      = torch.float16   # float16 halves VRAM vs float32
device     = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Collect query image paths
# ============================================================================

def collect_query_paths(bop_root: str):
    if not os.path.isdir(bop_root):
        raise FileNotFoundError(f"bop_root not found: {bop_root}")
    paths = []
    for category in sorted(os.listdir(bop_root)):
        cat_dir = os.path.join(bop_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for fname in sorted(os.listdir(cat_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(cat_dir, fname))
    return paths


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"[precompute] device = {device}")
    print(f"[precompute] dtype  = {dtype}")
    print(f"[precompute] batch  = {batch_size}")

    # --- Load model ---
    try:
        import open_clip
    except ImportError:
        sys.exit("open_clip not installed. Run: pip install open-clip-torch")

    print(f"[precompute] Loading {openclip_model} ({openclip_pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        openclip_model, pretrained=openclip_pretrained
    )
    model = model.visual          # we only need the image tower
    model = model.to(dtype).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[precompute] Model loaded: {n_params / 1e6:.0f}M params")

    # --- Collect paths ---
    img_paths = collect_query_paths(bop_root)
    print(f"[precompute] {len(img_paths)} query images found under {bop_root}")

    # --- Preprocess (CPU, one-time) ---
    print("[precompute] Preprocessing images...")
    tensors, valid_paths = [], []
    for p in tqdm(img_paths, desc="preprocess", unit="img"):
        try:
            tensors.append(preprocess(Image.open(p).convert("RGB")))
            valid_paths.append(p)
        except Exception as exc:
            tqdm.write(f"[precompute] skip {p}: {exc}")
    print(f"[precompute] {len(tensors)} images preprocessed.")

    # --- Encode in batches ---
    cache = {}
    for i in tqdm(range(0, len(tensors), batch_size),
                  desc="encode", unit="batch"):
        batch_tensors = tensors[i:i + batch_size]
        batch = torch.stack(batch_tensors).to(device)
        if dtype == torch.float16:
            batch = batch.half()

        try:
            with torch.no_grad():
                emb = model(batch)                      # (B, embed_dim)
                emb = F.normalize(emb.float(), p=2, dim=-1)  # store as float32
        except torch.cuda.OutOfMemoryError:
            print(f"\n[precompute] OOM at batch_size={batch_size}. "
                  "Reduce batch_size in CONFIG and restart.")
            raise

        batch_paths = valid_paths[i:i + batch_size]
        for j, p in enumerate(batch_paths):
            cache[p] = emb[j:j + 1].cpu()              # (1, embed_dim), float32

    # --- Save ---
    print(f"[precompute] Saving {len(cache)} embeddings → {ulip_query_cache_path}")
    torch.save(cache, ulip_query_cache_path)
    size_mb = os.path.getsize(ulip_query_cache_path) / 1024 / 1024
    print(f"[precompute] Done. Cache size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
