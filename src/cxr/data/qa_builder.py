"""QA dataset construction.

The assignment provides no QA dataset, so we synthesise one from the
radiology reports. For each report a language model produces question-
answer pairs **grounded strictly in the report text** (no outside
knowledge, no hallucinated findings). The resulting JSON is used:

  * as the *evaluation set* for the RAG QA mode, and
  * as a transparent, reproducible artefact whose construction is
    documented in the short report (assignment requirement).

Two generation backends share one prompt:

  * ``medgemma`` - local MedGemma (default). No API key, runs on the M1.
  * ``gemini``   - Google Gemini API (needs free-tier quota on the key).

Each generated item carries a ``qtype`` so evaluation can break results
down by question category.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from cxr.config import CONFIG, get_secret

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

_QA_GEN_SYSTEM = "You generate grounded question-answer pairs and reply with JSON only."


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


# ── backends ──────────────────────────────────────────────────────────
def _gemini_generate(prompt: str) -> str:
    """Generate raw text from the Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model=CONFIG.qa_generation.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4, response_mime_type="application/json"
        ),
    )
    return response.text


def _medgemma_generate(prompt: str, medgemma: Any) -> str:
    """Generate raw text from a local MedGemma instance (text-only)."""
    return medgemma.generate(
        prompt=prompt, image=None, system_prompt=_QA_GEN_SYSTEM, max_new_tokens=600
    )


def generate_qa_for_report(
    report_text: str,
    n: int | None = None,
    backend: str | None = None,
    medgemma: Any = None,
) -> list[dict[str, Any]]:
    """Generate ``n`` grounded QA pairs for a single report."""
    n = n or CONFIG.qa_generation.pairs_per_report
    backend = backend or CONFIG.qa_generation.backend
    prompt = _PROMPT.format(
        n=n,
        qtypes=", ".join(QTYPES),
        qtype_list=" | ".join(QTYPES),
        report=report_text,
    )

    if backend == "gemini":
        raw = _gemini_generate(prompt)
    elif backend == "medgemma":
        if medgemma is None:
            from cxr.models.medgemma import MedGemma

            medgemma = MedGemma()
        raw = _medgemma_generate(prompt, medgemma)
    else:
        raise ValueError(f"Unknown QA backend: {backend!r}")

    items = _parse_json_array(raw)
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
    sleep: float = 0.0,
    backend: str | None = None,
) -> Path:
    """Build the full QA dataset and write it to ``data/qa/qa_dataset.json``.

    ``sleep`` paces requests (useful for API rate limits; 0 for local MedGemma).
    """
    from tqdm import tqdm

    from cxr.data.loader import iter_records

    max_reports = max_reports or CONFIG.qa_generation.max_reports
    out_path = out_path or (CONFIG.path("data_qa") / "qa_dataset.json")
    backend = backend or CONFIG.qa_generation.backend

    # For the local backend, load MedGemma once and reuse it for every report.
    medgemma = None
    if backend == "medgemma":
        from cxr.models.medgemma import MedGemma

        medgemma = MedGemma()
        medgemma.load()

    dataset: list[dict[str, Any]] = []
    records = list(iter_records(limit=max_reports))
    for rec in tqdm(records, desc=f"QA pairs ({backend})"):
        try:
            pairs = generate_qa_for_report(rec.report_text, backend=backend, medgemma=medgemma)
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
        if sleep:
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
