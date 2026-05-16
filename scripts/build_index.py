"""Build a retrieval index over the report corpus (the QA knowledge base).

Usage:
    python scripts/build_index.py                      # active retriever (config.yaml)
    python scripts/build_index.py --retriever biomedclip --limit 300
    python scripts/build_index.py --retriever colpali   # GPU / Colab only

BiomedCLIP runs fine on an M1; ColPali is heavy and meant for the Colab notebook.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr.config import CONFIG  # noqa: E402
from cxr.data.loader import load_records  # noqa: E402
from cxr.models.retrievers import build_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RAG retrieval index.")
    parser.add_argument("--retriever", default=None, choices=["biomedclip", "colpali"])
    parser.add_argument(
        "--limit", type=int, default=CONFIG.qa_generation.max_reports,
        help="Cap the number of reports indexed.",
    )
    args = parser.parse_args()

    print(f"Loading up to {args.limit} reports ...")
    records = load_records(limit=args.limit)

    retriever = build_retriever(args.retriever)
    print(f"Building '{retriever.name}' index over {len(records)} reports ...")
    start = time.time()
    retriever.build_index(records)
    print(f"Done in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    main()
