# LangOS-6D

Language and shape based 3D object retrieval for 6D pose estimation. This repo uses OSCAR as the baseline and extends it with shape-aware retrieval from partial point clouds.

## Summary
Open-vocabulary CAD model retrieval from a single RGB image and language prompt works well semantically (OSCAR), but often misses geometric consistency. This thesis investigates whether shape embeddings from segmented RGB-D point clouds improve retrieval and downstream 6D pose estimation.

## Research Questions (from the expose)
- Do shape embeddings from partial point clouds improve open-vocabulary CAD retrieval beyond image + language?
- How do different shape encoders (e.g., PointNet-style vs. ULIP-2) compare for partial observations?
- How should image, language, and shape similarity be fused, and how do failure modes change?
- What is the impact on downstream 6D pose estimation when using the retrieved CAD model?

## Approach (planned)
- Reproduce the OSCAR pipeline (language-guided detection + text/image retrieval) as the baseline.
- Reconstruct a partial point cloud from the segmented RGB-D observation.
- Compute shape embeddings (focus on ULIP-2; compare with simpler baselines).
- Re-rank CAD candidates using shape similarity and fusion strategies.
- Evaluate retrieval quality and 6D pose estimation accuracy.

## Repo Structure
- docs/ project notes, paper summaries, and figures
- experiments/ experiment configs and run logs
- esults/ tables, plots, and metrics outputs
- scripts/ helper scripts for data prep and evaluation
- data/ datasets and symlinks (not committed)
- 
otes/ thesis writing notes and planning
- src/ core code for retrieval and pose evaluation
- ssets/ images for README and reports

## Branching Strategy
- oscar keeps the original baseline (mirror of pullover00/OSCAR)
- main is the thesis workspace scaffold
- exp/* branches hold ablations (e.g., exp/pointbert, exp/ulip)

