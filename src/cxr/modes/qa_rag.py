"""Mode 2 - RAG-based Question Answering.

Pipeline:  question -> retriever -> top-k report context -> MedGemma answer.

The answer is *grounded*: MedGemma is instructed to use only the retrieved
reports and to refuse when they do not contain the answer. An optional
X-ray image can be attached so MedGemma reasons over retrieved text **and**
the image together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from cxr.config import CONFIG
from cxr.models.medgemma import MedGemma
from cxr.models.retrievers import BaseRetriever, RetrievalResult, build_retriever

_QA_SYSTEM = (
    "You are a clinical assistant answering questions about chest X-ray "
    "radiology reports. Answer strictly from the provided context."
)

_QA_TEMPLATE = """CONTEXT - retrieved radiology reports:
{context}

QUESTION: {question}

Instructions:
- Answer using ONLY the context above.
- If the context does not contain the answer, reply exactly:
  "The retrieved reports do not contain enough information to answer this."
- Be concise and clinical (1-3 sentences)."""


@dataclass
class QAResult:
    """Output of one RAG QA query."""

    question: str
    answer: str
    retrieved: list[RetrievalResult] = field(default_factory=list)

    @property
    def context_study_ids(self) -> list[str]:
        """Study IDs of the retrieved reports - used by retrieval evaluation."""
        return [r.study_id for r in self.retrieved]


class QAEngine:
    """Retrieval-augmented QA over the radiology-report knowledge base."""

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        medgemma: MedGemma | None = None,
    ) -> None:
        self.retriever = retriever or build_retriever()
        self.medgemma = medgemma or MedGemma()

    @staticmethod
    def _format_context(hits: list[RetrievalResult]) -> str:
        """Render retrieved reports as a numbered context block."""
        return "\n\n".join(f"[{h.rank}] {h.text}" for h in hits)

    def answer(
        self,
        question: str,
        image: Image.Image | None = None,
        top_k: int | None = None,
    ) -> QAResult:
        """Retrieve context and generate a grounded answer to ``question``."""
        top_k = top_k or CONFIG.retrieval.top_k
        hits = self.retriever.retrieve(question, top_k=top_k)

        prompt = _QA_TEMPLATE.format(
            context=self._format_context(hits) or "(no reports retrieved)",
            question=question,
        )
        answer = self.medgemma.generate(prompt=prompt, image=image, system_prompt=_QA_SYSTEM)
        return QAResult(question=question, answer=answer, retrieved=hits)
