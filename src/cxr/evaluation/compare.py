"""Model-comparison harness.

Writes the assignment's comparison/evaluation tables as markdown to
``reports/comparison_results.md``:

  1. Retriever evaluation - BiomedCLIP Recall@k / MRR / latency on the
     QA eval set.
  2. Report-strategy comparison - MedGemma zero-shot vs MedGemma + RAG
     few-shot (BLEU, ROUGE-L). This is the core performance comparison:
     MedGemma alone vs MedGemma augmented by the BiomedCLIP retriever.

Each section degrades gracefully if a model or index is missing.

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


# ── section 1: retriever evaluation ───────────────────────────────────
def evaluate_retrievers(qa_sample: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score the BiomedCLIP retriever on the QA sample."""
    rows: list[dict[str, Any]] = []
    try:
        retr = build_retriever("biomedclip")
        retr.load_index()
        rows.append(evaluate_retriever(retr, qa_sample))
        print(f"  [biomedclip] scored on {len(qa_sample)} queries.")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"  [biomedclip] skipped: {exc}")
    return rows


# ── section 2: report strategies ──────────────────────────────────────
def compare_report_strategies(n_reports: int) -> list[dict[str, Any]]:
    """Score MedGemma zero-shot vs MedGemma + RAG few-shot report generation."""
    from cxr.modes.report_generation import ReportGenerator

    # Only records whose X-ray image is available locally can be scored.
    records = [r for r in load_records(limit=None) if r.image_path.exists()]
    if not records:
        raise FileNotFoundError(
            "No records have a local image file - download X-rays into data/raw/."
        )
    random.Random(CONFIG.seed).shuffle(records)
    records = records[:n_reports]
    references = [r.report_text for r in records]

    generator = ReportGenerator()
    rows: list[dict[str, Any]] = []

    # MedGemma alone.
    preds = [generator.generate(r.load_image(), "zero_shot").report for r in records]
    rows.append({"strategy": "zero_shot", **evaluate_reports(preds, references)})
    print(f"  [zero_shot] scored on {len(records)} reports.")

    # MedGemma + BiomedCLIP retrieval (few-shot context).
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

    print("\n[1/2] Evaluating the retriever ...")
    retriever_rows = evaluate_retrievers(qa_sample)

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
        "## 1. Retriever evaluation (BiomedCLIP)\n\n"
        + _markdown_table(retriever_rows)
        + "\n## 2. Report generation: MedGemma vs MedGemma + RAG\n\n"
        + _markdown_table(report_rows),
        encoding="utf-8",
    )
    print(f"\nWrote comparison tables -> {out}")


if __name__ == "__main__":
    sys.exit(main())
