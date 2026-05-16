"""QA dataset construction.

The assignment provides no QA dataset, so we synthesise one from the
radiology reports. For each report we prompt Google Gemini to produce
question-answer pairs that are **grounded strictly in the report text**
(no outside knowledge, no hallucinated findings). The resulting JSON is
used two ways downstream:

  * as the *evaluation set* for the RAG QA mode, and
  * as a transparent, reproducible artefact whose construction is
    documented in the short report (assignment requirement).

Each generated item carries a ``qtype`` so evaluation can break results
down by question category.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cxr.config import CONFIG, get_secret
from cxr.data.loader import iter_records

# Three deliberately distinct question styles keep the set balanced and
# let evaluation report per-category accuracy.
QTYPES = ("factual_yesno", "location_severity", "impression")

_PROMPT = """You are a radiology educator creating a question-answering dataset.

Below is a chest X-ray radiology report. Generate exactly {n} question-answer \
pairs that test understanding of THIS report only.

Strict rules:
- Answers MUST be fully supported by the report text. Never use outside knowledge.
- If the report does not mention something, do not invent it.
- Keep answers concise (1-2 sentences) and clinically worded.
- Produce a mix of these question types: {qtypes}.

REPORT:
\"\"\"
{report}
\"\"\"

Return ONLY a JSON array; each element is an object with keys:
  "question" (string), "answer" (string), "qtype" (one of: {qtype_list})."""


def _client():
    """Create a Gemini client (imported lazily so the dep is optional)."""
    from google import genai

    return genai.Client(api_key=get_secret("GEMINI_API_KEY"))


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse a JSON array, tolerating markdown code fences around it."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in model output: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def generate_qa_for_report(report_text: str, n: int | None = None) -> list[dict[str, Any]]:
    """Generate ``n`` grounded QA pairs for a single report via Gemini."""
    from google.genai import types

    n = n or CONFIG.qa_generation.pairs_per_report
    prompt = _PROMPT.format(
        n=n,
        qtypes=", ".join(QTYPES),
        qtype_list=" | ".join(QTYPES),
        report=report_text,
    )
    client = _client()
    response = client.models.generate_content(
        model=CONFIG.qa_generation.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    items = _parse_json_array(response.text)
    # Keep only well-formed items and normalise the qtype field.
    valid: list[dict[str, Any]] = []
    for item in items:
        if not item.get("question") or not item.get("answer"):
            continue
        qtype = str(item.get("qtype", "")).strip()
        item["qtype"] = qtype if qtype in QTYPES else "other"
        valid.append(item)
    return valid


def build_qa_dataset(
    max_reports: int | None = None,
    out_path: Path | None = None,
    sleep: float = 0.5,
) -> Path:
    """Build the full QA dataset and write it to ``data/qa/qa_dataset.json``.

    ``sleep`` paces requests to stay within the Gemini free-tier rate limit.
    """
    from tqdm import tqdm

    max_reports = max_reports or CONFIG.qa_generation.max_reports
    out_path = out_path or (CONFIG.path("data_qa") / "qa_dataset.json")

    dataset: list[dict[str, Any]] = []
    records = list(iter_records(limit=max_reports))
    for rec in tqdm(records, desc="Generating QA pairs"):
        try:
            pairs = generate_qa_for_report(rec.report_text)
        except Exception as exc:  # noqa: BLE001 - log and continue, don't abort the run
            print(f"  [skip] study {rec.study_id}: {exc}")
            continue
        for pair in pairs:
            dataset.append(
                {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "qtype": pair["qtype"],
                    "source_study_id": rec.study_id,
                    "source_report": rec.report_text,
                }
            )
        time.sleep(sleep)

    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(dataset)} QA pairs from {len(records)} reports -> {out_path}")
    return out_path


def load_qa_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the generated QA dataset from disk."""
    path = path or (CONFIG.path("data_qa") / "qa_dataset.json")
    if not path.exists():
        raise FileNotFoundError(
            f"QA dataset not found at {path}. Run notebook 01 or "
            f"`python -c 'from cxr.data.qa_builder import build_qa_dataset; build_qa_dataset()'`."
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build_qa_dataset()
