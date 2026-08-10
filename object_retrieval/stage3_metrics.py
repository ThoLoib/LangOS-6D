"""
stage3_metrics.py
=================
Instance-level retrieval metrics for Stage-3 (3a).

Unlike the category-level MI3DOR/SHREC metrics (recall@|C| over a whole relevant
class), Stage-3a has **exactly one** relevant gallery item per query — the exact
target CAD `d/obj_0000NN`. So the retrieval metrics are the standard
single-relevant ones from the concept doc:

    Recall@1, Recall@5, Recall@10   — is the exact target in the top-k?
    MRR                              — 1 / rank of the exact target (0 if absent)

Pose metrics (BOP-AR) and the 3b D_sym surface metric live in later phases
(they depend on bop_toolkit / FoundationPose and are added in Phase B/C).
"""

from typing import List, Optional, Sequence, Tuple


def rank_of_target(ranking: Sequence[Tuple[str, float]],
                   target_id: str) -> Optional[int]:
    """1-indexed rank of ``target_id`` in a descending (id, score) ranking.

    Returns None if the target is not present (e.g. it was pruned, or — in 3b —
    deliberately removed from the gallery)."""
    for i, (oid, _score) in enumerate(ranking):
        if oid == target_id:
            return i + 1
    return None


def summarize_retrieval(ranks: List[Optional[int]],
                        ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Aggregate per-query target ranks into Recall@k + MRR.

    ``ranks`` holds one entry per evaluated query: the 1-indexed rank of that
    query's exact target, or None if it never appeared in the ranking. Missing
    targets count as a miss for every Recall@k and contribute 0 to MRR."""
    n = len(ranks)
    out = {"n_queries": n}
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        out["mrr"] = 0.0
        out["n_target_found"] = 0
        return out
    found = [r for r in ranks if r is not None]
    for k in ks:
        hits = sum(1 for r in found if r <= k)
        out[f"recall@{k}"] = hits / n
    out["mrr"] = sum(1.0 / r for r in found) / n
    out["n_target_found"] = len(found)
    return out
