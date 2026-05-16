"""Lightweight smoke tests - no model weights, no network, no dataset.

These verify the pure-Python plumbing (config, parsing, column detection,
rendering, table formatting) so wiring bugs surface without a GPU run.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_config_loads():
    from cxr.config import CONFIG

    assert CONFIG.medgemma.backend in {"mlx", "transformers"}
    assert CONFIG.retrieval.top_k > 0
    assert CONFIG.path("data_qa").exists()  # created on access


def test_loader_column_detection():
    from cxr.data.loader import _detect_column

    df = pd.DataFrame({"dicom_id": ["a"], "Findings": ["clear lungs"]})
    assert _detect_column(df, ["image_path", "dicom_id"], "image") == "dicom_id"
    assert _detect_column(df, ["text", "findings"], "report") == "Findings"
    with pytest.raises(KeyError):
        _detect_column(df, ["nonexistent"], "image")


def test_qa_json_parsing_tolerates_code_fences():
    from cxr.data.qa_builder import _parse_json_array

    fenced = '```json\n[{"question": "Q?", "answer": "A.", "qtype": "impression"}]\n```'
    parsed = _parse_json_array(fenced)
    assert parsed[0]["question"] == "Q?"


def test_render_report_page_returns_image():
    from PIL import Image

    from cxr.utils.rendering import render_report_page

    page = render_report_page("FINDINGS: No acute disease.\nIMPRESSION: Normal.")
    assert isinstance(page, Image.Image)
    assert page.size[0] > 0 and page.size[1] > 0


def test_markdown_table_formatting():
    from cxr.evaluation.compare import _markdown_table

    table = _markdown_table([{"retriever": "biomedclip", "mrr": 0.5}])
    assert "| retriever | mrr |" in table
    assert "biomedclip" in table


def test_retriever_factory():
    from cxr.models.retrievers import BiomedCLIPRetriever, ColPaliRetriever, build_retriever

    assert isinstance(build_retriever("biomedclip"), BiomedCLIPRetriever)
    assert isinstance(build_retriever("colpali"), ColPaliRetriever)
    with pytest.raises(ValueError):
        build_retriever("does-not-exist")


def test_retrieval_metrics_on_synthetic_hits():
    """evaluate_retriever should compute Recall@k / MRR from a fake retriever."""
    from cxr.models.retrievers import RetrievalResult
    from cxr.evaluation.retrieval_eval import evaluate_retriever

    class FakeRetriever:
        name = "fake"

        def retrieve(self, query, top_k=4):
            # Source "s1" is always returned at rank 2.
            return [
                RetrievalResult(text="x", score=0.9, study_id="s9", rank=1),
                RetrievalResult(text="y", score=0.8, study_id="s1", rank=2),
            ]

    qa = [{"question": "Q?", "source_study_id": "s1"}]
    metrics = evaluate_retriever(FakeRetriever(), qa, k_values=[1, 3])
    assert metrics["recall@1"] == 0.0   # source at rank 2, not in top-1
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 0.5        # reciprocal of rank 2
