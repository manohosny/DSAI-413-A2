"""Retrieval metrics for the RAG QA mode.

Each generated QA pair records the ``source_study_id`` it was written
from. A retrieval is therefore "correct" when that source report appears
in the retrieved top-k. From the rank of the source report we compute:

  * Recall@k - fraction of questions whose source report is in the top-k.
  * MRR      - mean reciprocal rank of the source report.

Latency is timed too as a practical performance measure.
"""

from __future__ import annotations

import time
from typing import Any

from cxr.config import CONFIG
from cxr.models.retrievers import BaseRetriever


def _source_rank(hits: list, source_id: str) -> int | None:
    """1-based rank of the source report among hits, or None if absent."""
    for hit in hits:
        if str(hit.study_id) == str(source_id):
            return hit.rank
    return None


def evaluate_retriever(
    retriever: BaseRetriever,
    qa_items: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Score a retriever on a list of QA items.

    Each item needs ``question`` and ``source_study_id`` keys.
    """
    k_values = k_values or list(CONFIG.evaluation.retrieval_k_values)
    max_k = max(k_values)

    ranks: list[int | None] = []
    latencies: list[float] = []
    for item in qa_items:
        start = time.time()
        hits = retriever.retrieve(item["question"], top_k=max_k)
        latencies.append(time.time() - start)
        ranks.append(_source_rank(hits, item["source_study_id"]))

    n = len(ranks)
    metrics: dict[str, Any] = {"retriever": retriever.name, "num_queries": n}
    for k in k_values:
        hit_at_k = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"recall@{k}"] = round(hit_at_k / n, 4) if n else 0.0
    metrics["mrr"] = (
        round(sum((1.0 / r) for r in ranks if r is not None) / n, 4) if n else 0.0
    )
    metrics["avg_latency_s"] = round(sum(latencies) / n, 4) if n else 0.0
    return metrics
