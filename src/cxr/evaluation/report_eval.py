"""Report-generation metrics.

Scores generated radiology reports against reference reports with three
complementary metrics:

  * BLEU      - n-gram precision (surface overlap).
  * ROUGE-L   - longest-common-subsequence recall (structure/coverage).
  * BERTScore - contextual-embedding similarity (semantic agreement).

Surface metrics (BLEU/ROUGE) and the semantic metric (BERTScore) often
disagree for clinical text - reporting all three is what makes the
zero-shot vs RAG comparison meaningful.
"""

from __future__ import annotations


def _bleu(predictions: list[str], references: list[str]) -> float:
    """Corpus BLEU (0-100) via sacrebleu."""
    from sacrebleu import corpus_bleu

    return round(corpus_bleu(predictions, [references]).score, 2)


def _rouge_l(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F1 (0-1) via rouge_score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _bertscore(predictions: list[str], references: list[str]) -> float:
    """Mean BERTScore F1 (0-1). Heavy - downloads a transformer on first use."""
    from bert_score import score

    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return round(float(f1.mean()), 4)


_METRIC_FNS = {"bleu": _bleu, "rougeL": _rouge_l, "bertscore": _bertscore}


def evaluate_reports(
    predictions: list[str],
    references: list[str],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute the requested metrics for ``predictions`` vs ``references``."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length.")
    if not predictions:
        raise ValueError("Nothing to evaluate - empty prediction list.")

    from cxr.config import CONFIG

    metrics = metrics or list(CONFIG.evaluation.report_metrics)
    results: dict[str, float] = {}
    for name in metrics:
        if name not in _METRIC_FNS:
            raise ValueError(f"Unknown metric {name!r}. Options: {list(_METRIC_FNS)}")
        results[name] = _METRIC_FNS[name](predictions, references)
    return results
