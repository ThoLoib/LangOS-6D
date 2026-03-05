# =============================================================================
# OSCAR+ Pipeline – Shape-Aware Object Retrieval & Pose Estimation
# =============================================================================
#
# Modulare Pipeline für:
#   1. Objektlokalisierung (GroundingDINO + SAM)
#   2. Punktwolkenerzeugung (Open3D)
#   3. Semantische Kandidatensuche (CLIP)
#   4. Bildbasiertes Re-Ranking (DINOv2)
#   5. Shape Matching (ULIP-2)
#   6. Score-Fusion
#   7. Skalenbestimmung
#   8. Pose Estimation (FoundationPose / ICP)
#
# Basiert auf dem OSCAR-Framework (https://github.com/pullover00/OSCAR)
# Erweitert um shape-aware retrieval via ULIP-2.
# =============================================================================

__version__ = "0.1.0"
