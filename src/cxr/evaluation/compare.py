"""Model-comparison harness.

Produces the two comparisons the assignment requires and writes them as
markdown tables to ``reports/comparison_results.md``:

  1. Retriever comparison - BiomedCLIP vs ColPali on the QA eval set
     (Recall@k, MRR, latency).
  2. Report-strategy comparison - MedGemma zero-shot vs MedGemma + RAG
     few-shot (BLEU, ROUGE-L, BERTScore).

Each comparison is wrapped so a missing model/index degrades gracefully
(e.g. the ColPali index only exists after running the Colab notebook).

Run:  python -m cxr.evaluation.compare
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from cxr.config import CONFIG, PROJECT_ROOT
from cxr.data.loader import load_records
from cxr.data.qa_builder import load_qa_dataset
from cxr.evaluation.report_eval import evaluate_reports
from cxr.evaluation.retrieval_eval import evaluate_retriever
from cxr.models.retrievers import build_retriever


# ── markdown helpers ──────────────────────────────────────────────────
def _markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a list of flat dicts as a GitHub-flavoured markdown table."""
    if not rows:
        return "_No results._\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


# ── comparison 1: retrievers ──────────────────────────────────────────
def compare_retrievers(qa_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every available retriever on the QA sample."""
    rows: list[dict[str, Any]] = []
    for name in ("biomedclip", "colpali"):
        try:
            retr = build_retriever(name)
            retr.load_index()
            rows.append(evaluate_retriever(retr, qa_sample))
            print(f"  [{name}] scored on {len(qa_sample)} queries.")
        except Exception as exc:  # noqa: BLE001 - skip retriever, keep the rest
            print(f"  [{name}] skipped: {exc}")
    return rows


# ── comparison 2: report strategies ───────────────────────────────────
def compare_report_strategies(n_reports: int) -> list[dict[str, Any]]:
    """Score MedGemma zero-shot vs RAG few-shot report generation."""
    from cxr.modes.report_generation import ReportGenerator

    records = load_records(limit=None)
    random.Random(CONFIG.seed).shuffle(records)
    records = records[:n_reports]
    references = [r.report_text for r in records]

    generator = ReportGenerator()
    rows: list[dict[str, Any]] = []

    # Zero-shot.
    preds = [generator.generate(r.load_image(), "zero_shot").report for r in records]
    rows.append({"strategy": "zero_shot", **evaluate_reports(preds, references)})
    print(f"  [zero_shot] scored on {len(records)} reports.")

    # RAG few-shot (needs a retriever index).
    try:
        retr = build_retriever()
        retr.load_index()
        preds = [
            generator.generate(r.load_image(), "rag_fewshot", retriever=retr).report
            for r in records
        ]
        rows.append({"strategy": "rag_fewshot", **evaluate_reports(preds, references)})
        print(f"  [rag_fewshot] scored on {len(records)} reports.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [rag_fewshot] skipped: {exc}")
    return rows


# ── entry point ───────────────────────────────────────────────────────
def main() -> None:
    sample_size = CONFIG.evaluation.sample_size
    rng = random.Random(CONFIG.seed)

    print("Loading QA dataset ...")
    qa = load_qa_dataset()
    qa_sample = rng.sample(qa, min(sample_size, len(qa)))

    print("\n[1/2] Comparing retrievers ...")
    retriever_rows = compare_retrievers(qa_sample)

    print("\n[2/2] Comparing report-generation strategies ...")
    # Report generation is slow on an M1; use a smaller sample.
    try:
        report_rows = compare_report_strategies(n_reports=min(10, sample_size))
    except Exception as exc:  # noqa: BLE001
        print(f"  report comparison skipped: {exc}")
        report_rows = []

    out = PROJECT_ROOT / "reports" / "comparison_results.md"
    out.write_text(
        "# Model Comparison Results\n\n"
        f"_QA eval sample: {len(qa_sample)} questions._\n\n"
        "## 1. Retriever comparison (QA mode)\n\n"
        + _markdown_table(retriever_rows)
        + "\n## 2. Report-generation strategy comparison\n\n"
        + _markdown_table(report_rows),
        encoding="utf-8",
    )
    print(f"\nWrote comparison tables -> {out}")


if __name__ == "__main__":
    sys.exit(main())
