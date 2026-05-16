"""Mode 1 - Report Generation.

Takes a chest X-ray image and produces a structured radiology report
(FINDINGS / IMPRESSION). Two strategies are supported so the short report
can compare them:

  * ``zero_shot``   - MedGemma sees only the image.
  * ``rag_fewshot`` - reports of visually similar X-rays are retrieved and
                      injected as in-context examples before generation.

The two strategies share one model (MedGemma); only the prompt differs,
which is exactly what makes them a fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from cxr.config import CONFIG
from cxr.models.medgemma import MedGemma

_STRUCTURED_INSTRUCTION = (
    "Analyse this chest X-ray and write a structured radiology report.\n"
    "Use exactly these two sections:\n"
    "FINDINGS: objective observations (lungs, heart, mediastinum, pleura, bones).\n"
    "IMPRESSION: a concise diagnostic summary.\n"
    "Do not invent findings that are not visible."
)

_FEWSHOT_PREAMBLE = (
    "Reference reports from chest X-rays with similar visual appearance are "
    "given below. Use them only as stylistic and structural guidance - base "
    "your findings on the X-ray actually shown.\n"
)


@dataclass
class ReportResult:
    """Output of one report-generation run."""

    report: str
    strategy: str
    retrieved_examples: list[str] = field(default_factory=list)


class ReportGenerator:
    """Generates structured radiology reports from chest X-rays."""

    def __init__(self, medgemma: MedGemma | None = None) -> None:
        self.medgemma = medgemma or MedGemma()

    def generate(
        self,
        image: Image.Image,
        strategy: str = "zero_shot",
        retriever: object | None = None,
    ) -> ReportResult:
        """Generate a report. ``rag_fewshot`` requires a ``retriever``."""
        if strategy == "zero_shot":
            return self._zero_shot(image)
        if strategy == "rag_fewshot":
            if retriever is None:
                raise ValueError("strategy='rag_fewshot' requires a retriever.")
            return self._rag_fewshot(image, retriever)
        raise ValueError(f"Unknown strategy: {strategy!r}")

    def _zero_shot(self, image: Image.Image) -> ReportResult:
        report = self.medgemma.generate(prompt=_STRUCTURED_INSTRUCTION, image=image)
        return ReportResult(report=report, strategy="zero_shot")

    def _rag_fewshot(self, image: Image.Image, retriever: object) -> ReportResult:
        top_k = CONFIG.retrieval.top_k
        # Retrieve reports of X-rays that look similar to the query image.
        hits = retriever.retrieve(image, top_k=top_k)  # type: ignore[attr-defined]
        examples = [h.text for h in hits]

        example_block = "\n\n".join(
            f"[Reference report {i}]\n{txt}" for i, txt in enumerate(examples, 1)
        )
        prompt = f"{_FEWSHOT_PREAMBLE}\n{example_block}\n\n{_STRUCTURED_INSTRUCTION}"
        report = self.medgemma.generate(prompt=prompt, image=image)
        return ReportResult(
            report=report, strategy="rag_fewshot", retrieved_examples=examples
        )
