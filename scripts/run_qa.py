"""CLI for Mode 2 - RAG-based Question Answering.

Usage:
    python scripts/run_qa.py --question "Is there evidence of pneumonia?"
    python scripts/run_qa.py --question "..." --retriever colpali --image xray.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cxr.models.retrievers import build_retriever  # noqa: E402
from cxr.modes.qa_rag import QAEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the report corpus (RAG).")
    parser.add_argument("--question", required=True, help="Clinical question to answer.")
    parser.add_argument("--retriever", default=None, choices=["biomedclip", "colpali"])
    parser.add_argument("--image", default=None, help="Optional chest X-ray to attach.")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB") if args.image else None
    engine = QAEngine(retriever=build_retriever(args.retriever))
    result = engine.answer(args.question, image=image, top_k=args.top_k)

    print(f"\nQUESTION: {result.question}\n")
    print("RETRIEVED CONTEXT:")
    for hit in result.retrieved:
        snippet = hit.text.replace("\n", " ")[:120]
        print(f"  [{hit.rank}] study={hit.study_id}  score={hit.score:.3f}  {snippet}...")
    print(f"\nANSWER:\n{result.answer}")


if __name__ == "__main__":
    main()
