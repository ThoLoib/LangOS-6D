# Thesis Methodology vs. Implementation — Point-by-Point Comparison

This document maps every technical specification from the thesis methodology
(Chapter 3) and implementation (Chapter 4) chapters to the corresponding
code in the OSCAR+ pipeline.

---

## Step A: Scene Grounding

### A1: Object Localisation

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Detection model | GroundingDINO (open-set detector) | `AutoModelForZeroShotObjectDetection` loaded from HuggingFace | `step1_localization.py:121` | Yes |
| Segmentation model | SAM (Segment Anything) | SAM2.1 (`facebook/sam2.1-hiera-large`) via `Sam2Model` | `step1_localization.py:129-141` | Yes |
| Detection confidence threshold | 0.3 | `detection_confidence: float = 0.3` | `config.py:35` | Yes |
| Prompt format | Free-text, dot-appended for GroundingDINO | `if not text.endswith("."): text += "."` | `step1_localization.py:153-154` | Yes |
| Best detection selection | Highest confidence score | `best_idx = det["scores"].argmax()` | `step1_localization.py:285` | Yes |

### A2: Mask Post-Processing

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Largest connected component | Retain only largest CC; discard spurious fragments | `cv2.connectedComponentsWithStats` → keep largest by area | `step1_localization.py:226-238` | Yes |
| Mask dilation kernel | 5x5 pixels | `mask_dilation_kernel: int = 5` | `config.py:39` | Yes |
| Mask dilation iterations | 1 iteration | `mask_dilation_iterations: int = 1` | `config.py:40` | Yes |
| Dilation purpose | Compensate depth-shadow at object boundaries (structured-light: Shen et al., 2013; ToF: Chugunov et al., 2021) | Docstring references both papers; `cv2.dilate` with configurable kernel | `step1_localization.py:199-254` | Yes |
| Order | CC first, then dilation | CC at line 226, dilation at line 242 | `step1_localization.py` | Yes |

### A3: Partial Point Cloud Reconstruction

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Back-projection model | Pinhole camera: X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy | Vectorised implementation matching exact formula | `step2_pointcloud.py:293-296` | Yes |
| Depth filtering | Discard zero/NaN; reject pixels deviating substantially from masked-region median depth | `_gate_depth()`: median-relative gating, +-tolerance | `step2_pointcloud.py:95-123` | Yes |
| Depth gate tolerance | +/-30% around median | `depth_gate_tolerance: float = 0.3` | `config.py:57` | Yes |
| Voxel downsampling size | 2 mm | `voxel_size: float = 0.002` | `config.py:53` | Yes |
| Statistical outlier removal | Applied | `pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.0)` | `step2_pointcloud.py:209-218` | Yes |
| SOR nb_neighbors | 10 | `sor_nb_neighbors: int = 10` | `config.py:60` | Yes |
| SOR std_ratio | 1.0 | `sor_std_ratio: float = 1.0` | `config.py:61` | Yes |
| Output triple | (I_ROI, M_ROI, P_ROI) — cropped image, mask, partial PC | `LocalizationResult` returns `roi_image`, `mask`; `PointCloudResult` returns `point_cloud` | `step1_localization.py:58-75`, `step2_pointcloud.py:39-58` | Yes |

### A4: ROI Extraction

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Background colour for masked ROI | Neutral grey | `background_color: Tuple = (205, 205, 205)` | `step1_localization.py:363` | Yes |
| Crop strategy | Tight crop from mask bounding box, not detection box | Uses `np.argwhere(mask)` min/max for crop bounds | `step1_localization.py:383-391` | Yes |

---

## Step B1: Multi-Signal CAD Retrieval

### B1.1: Semantic Channel S_text (CLIP)

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Model | CLIP (frozen) | `clip.load(config.clip_model_name)` | `step3_clip_retrieval.py:119-122` | Yes |
| Model variant | ViT-B/32 | `clip_model_name: str = "ViT-B/32"` | `config.py:72` | Yes |
| Query | ROI image embedding vs. pre-encoded text description embeddings | `encode_image()` → cosine similarity with `_desc_embeddings` | `step3_clip_retrieval.py:258-261` | Yes |
| Similarity metric | Cosine similarity | `query_emb @ self._desc_embeddings.T` (both L2-normalised) | `step3_clip_retrieval.py:261` | Yes |
| Per-object deduplication | Best score per object ID | `if obj_id not in seen_objects` — keeps highest-scoring description | `step3_clip_retrieval.py:305-311` | Yes |
| Full-database scoring (default) | All candidates scored, no hard pre-filter | Default path scores all descriptions; top-k used for output | `step3_clip_retrieval.py:294` | Yes |
| OSCAR cascade variant | Retained as ablation O2 | `clip_top_k` can limit candidates for downstream steps | `config.py:73` | Yes |

### B1.2: Appearance Channel S_view (DINOv2 / SigLIP)

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Default encoder | DINOv2 (frozen) | `appearance_encoder: str = "dinov2"`; model loaded via `AutoModel.from_pretrained` | `config.py:85`, `step4_dino_reranking.py` | Yes |
| Alternative encoder (ablation E4) | SigLIP | `appearance_encoder = "siglip"` → loads `SiglipVisionModel` | `config.py:85`, `step4_dino_reranking.py` | Yes |
| DINOv2 model ID | facebook/dinov2-base | `dino_model_name: str = "facebook/dinov2-base"` | `config.py:86` | Yes |
| SigLIP model ID | google/siglip-base-patch16-224 | `siglip_model_name: str = "google/siglip-base-patch16-224"` | `config.py:87` | Yes |
| Feature pooling | CLS token (CNOS convention) | `dino_pooling: str = "cls"` → extracts `outputs.last_hidden_state[:, 0]` | `config.py:88`, `step4_dino_reranking.py` | Yes |
| Number of rendered views V | 8 (default, ablated via O4) | Reference images stored per object in `reference_images_dir` | `config.py:99` | Yes |
| Per-object aggregation | Top-k_v=5 views, softmax-weighted | `dino_view_aggregation: str = "topk_softmax"`, `dino_view_topk: int = 5` | `config.py:94-95` | Yes |
| Softmax temperature | 0.5 | `dino_view_temperature: float = 0.5` | `config.py:96` | Yes |
| Aggregation reference | CNOS (top-5 views); OPEN (query-conditioned attention, approximated training-free via softmax) | Header comment references CNOS k_v=5, OPEN Eq. 2-3 | `step4_dino_reranking.py` header | Yes |

### B1.3: Shape Channel S_shape (ULIP-2 / Uni3D)

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Default encoder | ULIP-2 with Point-BERT backbone (frozen) | `shape_encoder: str = "ulip2"`, `ulip2_backbone: str = "pointbert_colored"` | `config.py:108,114` | Yes |
| Alternative encoder (ablation E7) | Uni3D | `shape_encoder = "uni3d"` → loads `Uni3DEncoder` | `config.py:108`, `step5_shape_matching.py` | Yes |
| Uni3D model ID | BAAI/Uni3D | `uni3d_model_name: str = "BAAI/Uni3D"` | `config.py:109` | Yes |
| Input point count | 10,000 points | `ulip2_num_points: int = 10000`, `uni3d_num_points: int = 10000` | `config.py:110,116` | Yes |
| Point upsampling for partial PCs | Sampling with replacement + Gaussian jitter | `normalize_pointcloud()` in step5 uses replacement sampling when partial PC has fewer than N points | `step5_shape_matching.py` | Yes |
| Input modality (ablation O5) | XYZ+RGB (default) vs XYZ-only | `ulip2_use_colors: bool = True` | `config.py:117` | Yes |
| Partial reference views | Partial PCs per rendered view, not full mesh | `ulip2_use_partial_views: bool = False` (available but not default) | `config.py:128` | Partial |
| Uni3D mode restriction | PC-only (no cross-modal image encoder) | `if self._encoder_type == "uni3d" and mode != "pc": mode = "pc"` | `step5_shape_matching.py` | Yes |
| ULIP-2 query modes | pc, cross (OpenCLIP image branch), both | `ulip2_mode: str = "cross"` with options "pc", "cross", "both" | `config.py:126` | Yes |

### B1.4: Score Fusion

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Default fusion method | Weighted sum with min-max normalisation | `fusion_method: str = "weighted_sum"` | `config.py:148` | Yes |
| Normalisation | Min-max per channel to [0,1] (Jain et al., 2005) | `_minmax()` helper in `_weighted_sum()` | `step6_fusion.py:203-213` | Yes |
| Fusion formula | S(o) = alpha * S_text + beta * S_view + gamma * S_shape | `fused = w_clip * norm_clip[i] + w_dino * norm_dino[i] + w_ulip * norm_ulip[i]` | `step6_fusion.py:222-225` | Yes |
| Fusion weights (alpha, beta, gamma) | (0.3, 0.4, 0.3) | `weight_clip=0.3, weight_dino=0.4, weight_ulip=0.3` | `config.py:149-151` | Yes |
| Alternative fusion (ablation E6) | Majority voting (Borda count, SAMURAI-inspired) | `_majority_voting()` method; `fusion_method = "majority_voting"` | `step6_fusion.py:383-488` | Yes |
| Additional fusion methods | RRF (Cormack et al., 2009), intersection | `_reciprocal_rank_fusion()`, `_intersection()` | `step6_fusion.py:322-377, 256-316` | Yes |

---

## Sub-step B2: Geometry Re-ranking

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Operates on | Top-k=5 fused candidates | `geometry_reranking_top_k: int = 5` | `config.py:161` | Yes |
| Signal 1: GeDi correspondence score | GeDi descriptors (Poiesi & Boscaini, 2022) + RANSAC → inlier count | `_gedi_ransac()` computes RANSAC, returns `float(inlier_count)` | `step_b2_geometry_reranking.py:241-289` | Yes |
| Signal 2: Trimmed Chamfer | One-sided NN distances, top 10% trimmed | `trimmed_chamfer_distance()` with `trim_ratio=0.1` | `utils.py:91-126`, `config.py:175` | Yes |
| GeDi descriptor model | Pre-trained on 3DMatch, 32-dim descriptors | `gedi_dim: int = 32` | `config.py:167` | Yes |
| GeDi service architecture | Separate Docker container via HTTP | `gedi_url: str = "http://gedi:5060"`; HTTP POST in `gedi_descriptors.py` | `config.py:164`, `gedi_descriptors.py:134-146` | Yes |
| RANSAC minimal set | 3 points | `ransac_n=3` | `step_b2_geometry_reranking.py:275` | Yes |
| RANSAC inlier distance | 1.5 x voxel size | `max_correspondence_distance=voxel_size * 1.5` | `step_b2_geometry_reranking.py:271` | Yes |
| RANSAC max iterations | 100,000 | `RANSACConvergenceCriteria(100000, 0.999)` | `step_b2_geometry_reranking.py:281-282` | Yes |
| RANSAC confidence | 0.999 | `RANSACConvergenceCriteria(100000, 0.999)` | `step_b2_geometry_reranking.py:282` | Yes |
| Active signals | One at a time by default | `geometry_reranking_signal: str = "gedi"` (options: "gedi", "chamfer", "both") | `config.py:160` | Yes |
| RANSAC transform forwarded to Step C | Yes, reused as ICP init | `best_transformation` in `GeometryReRankingResult`; consumed by `step7` via `init_transform` | `step_b2_geometry_reranking.py:88-94`, `step7_scale_estimation.py:120` | Yes |
| FPFH fallback | When GeDi unavailable | `if signal == "gedi": signal = "chamfer"` on GeDi failure; Step 7 has explicit FPFH fallback | `step_b2_geometry_reranking.py:165-166`, `step7_scale_estimation.py:350-388` | Yes |

---

## Step C: CAD-to-Pose Estimation

### C1: Coarse Alignment

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Descriptor reuse from B2 | B2 RANSAC transform as ICP initialisation | `init_transform` parameter; when provided, skips RANSAC | `step7_scale_estimation.py:282-285` | Yes |
| Primary descriptors | GeDi | `_ransac_with_descriptors()` tries GeDi first | `step7_scale_estimation.py:324-348` | Yes |
| Fallback descriptors | FPFH (Rusu et al., 2009) | Explicit FPFH fallback in `_ransac_with_descriptors()` | `step7_scale_estimation.py:350-388` | Yes |
| ICP variant | Point-to-Plane | `TransformationEstimationPointToPlane()` | `step7_scale_estimation.py:304` | Yes |
| ICP max iterations | 50 | `icp_max_iterations: int = 50` | `config.py:204` | Yes |
| ICP correspondence distance | 3 x voxel size | `max_correspondence_distance=voxel_size * 3` | `step7_scale_estimation.py:302` | Yes |

### C2: Scale Estimation

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Primary method | Derived from ICP-aligned point clouds; per-axis extent ratios; 2 best axes | `ratios = obs_aligned_size / safe_cad`, use top 2 axes by ratio | `step7_scale_estimation.py:146-151` | Yes |
| Confidence formula | min(ICP_fitness, 1 - ratio_spread) | `confidence = float(min(fitness, max(0.0, 1.0 - ratio_spread)))` | `step7_scale_estimation.py:155` | Yes |
| Fallback | Sorted bounding-box extent ratio when ICP confidence < threshold | `estimate_fast()` uses sorted dims; triggered when `confidence < _min_conf` | `step7_scale_estimation.py:169-178` | Yes |
| Fallback threshold | 0.15 | `scale_icp_min_confidence: float = 0.15` | `config.py:196` | Yes |
| Scale gate (legacy, ablation) | Not default; available for ablation E2 | `scale_gate_enabled: bool = False` | `config.py:182` | Yes |

### C3: 6D Pose Estimation

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Primary method | FoundationPose (Wen et al., 2024), frozen, decoupled service | `pose_method = "foundationpose"` → HTTP call via `foundationpose_bridge.py` | `config.py:203`, `step8_pose_estimation.py:129-179` | Yes |
| FoundationPose service URL | Separate Docker container | `foundationpose_url: str = "http://foundationpose:5050"` | `config.py:206` | Yes |
| Fallback | ICP (FPFH RANSAC + Point-to-Plane ICP) | `_estimate_icp()` triggered on FoundationPose failure | `step8_pose_estimation.py:218-342` | Yes |
| ICP correspondence distance | 3 x voxel size (thesis Sec. 3.4) | `max_correspondence_distance=voxel_size * 3` (comment: "thesis: 3x voxel_size") | `step8_pose_estimation.py:318` | Yes |
| ICP max iterations | 50 | `self.config.icp_max_iterations` = 50 | `config.py:204` | Yes |
| Initial pose from Step 7 | Coarse alignment used as ICP init | `initial_pose` parameter; skips RANSAC when provided | `step8_pose_estimation.py:293-295` | Yes |

---

## Global Settings

| Thesis Specification | Value | Implementation | File | Match |
|---|---|---|---|---|
| Random seed | 42 | `seed: int = 42` | `config.py:25` | Yes |
| All neural components frozen | No training/fine-tuning | All models loaded with `.eval()`, `torch.no_grad()` context | all step files | Yes |
| Modularity / ablation friendliness | Each signal switchable | Config flags: `appearance_encoder`, `shape_encoder`, `fusion_method`, `geometry_reranking_signal`, `pose_method` | `config.py` | Yes |
| Dependency isolation | FoundationPose + GeDi in separate containers | `docker-compose.yml` defines `oscar`, `foundationpose`, `gedi` services | `docker-compose.yml`, `Dockerfile.gedi` | Yes |

---

## Ablation Configuration Summary

| Ablation | Thesis Description | Config Parameter(s) | How to Activate |
|---|---|---|---|
| E2 | Geometry re-ranking (B2) vs. legacy scale gate | `geometry_reranking_enabled`, `scale_gate_enabled` | `--no-geometry-reranking` + `--scale-gate` |
| E4 | DINOv2 vs. SigLIP for S_view | `appearance_encoder` | `--appearance-encoder siglip` |
| E6 | Weighted sum vs. majority voting | `fusion_method` | `--fusion-method majority_voting` |
| E7 | ULIP-2 vs. Uni3D for S_shape | `shape_encoder` | `--shape-encoder uni3d` |
| O1 | Remove shape channel from fusion | `weight_ulip` | Set `--weight-ulip 0.0` |
| O2 | Full-database vs. OSCAR cascade | `clip_top_k` | Limit `--clip-top-k` to e.g. 10 |
| O4 | Number of rendered views V | Number of images in `reference_images_dir` | Re-render with different V |
| O5 | XYZ-only vs. XYZ+RGB for ULIP-2 | `ulip2_use_colors` | `--ulip-no-colors` |

---

## Deviations and Open Items

| Item | Thesis Says | Implementation Status | Notes |
|---|---|---|---|
| FreeZe full registration (DINOv2 + GeDi concatenated descriptors) | Planned extension: `f(p) = [f_vis/norm ; f_geo/norm]` | **Not implemented** | Thesis explicitly marks as "not yet implemented" |
| Partial reference views default | Partial PCs per view for shape matching | `ulip2_use_partial_views = False` (full mesh is default) | Config supports it but default differs from thesis recommendation |
| Thesis lists `ulip2_mode` default as unspecified | Cross-modal as strong option | `ulip2_mode = "cross"` (default) | Thesis discusses PC mode as the "pure shape" channel; cross uses image branch |
| FoundationPose vs. ICP accuracy ablation | "Reported as runtime configuration, not as accuracy ablation" | No FoundationPose-vs-ICP accuracy comparison | Thesis acknowledges this is out of scope |
| `icp_threshold` config field | Unused legacy field | Still present in config at value 0.02 | `step7` and `step8` both use `voxel_size * 3` directly; legacy field is harmless |
