"""CLI for Mode 1 - Report Generation.

Usage:
    python scripts/run_report_gen.py --image path/to/xray.jpg
    python scripts/run_report_gen.py --image path/to/xray.jpg --strategy rag_fewshot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cxr.modes.report_generation import ReportGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a radiology report from a chest X-ray.")
    parser.add_argument("--image", required=True, help="Path to a chest X-ray image.")
    parser.add_argument(
        "--strategy",
        default="zero_shot",
        choices=["zero_shot", "rag_fewshot"],
        help="zero_shot = image only; rag_fewshot = inject similar reports as context.",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    retriever = None
    if args.strategy == "rag_fewshot":
        # Lazy import so zero-shot runs need no retrieval dependencies.
        from cxr.models.retrievers import build_retriever

        retriever = build_retriever()
        retriever.load_index()

    result = ReportGenerator().generate(image, strategy=args.strategy, retriever=retriever)

    print(f"\n{'=' * 60}\nStrategy: {result.strategy}\n{'=' * 60}")
    if result.retrieved_examples:
        print(f"(used {len(result.retrieved_examples)} retrieved reference reports)\n")
    print(result.report)


if __name__ == "__main__":
    main()
