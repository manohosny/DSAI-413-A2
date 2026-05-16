"""Retriever for the RAG QA mode.

The QA pipeline and the evaluation code depend only on the
:class:`BaseRetriever` interface, so the retriever can be swapped without
touching them.

:class:`BiomedCLIPRetriever` is a single-vector, domain-tuned retriever with
a FAISS cosine-similarity index. It is light enough to run live on an 8GB M1
and indexes the **report texts** (the QA knowledge base). It can be queried
with a text question (QA mode) or an X-ray image (report-generation RAG) -
BiomedCLIP's shared image-text space supports both.

The model is switched to inference mode with ``.train(False)`` (== ``.eval()``).
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cxr.config import CONFIG
from cxr.data.loader import CxrRecord


@dataclass
class RetrievalResult:
    """One retrieved report, ranked by similarity to the query."""

    text: str            # report text - the context handed to MedGemma
    score: float
    study_id: str
    rank: int = 0
    image_path: str | None = None


def _torch_device() -> str:
    """Pick the torch device for the retriever.

    MPS is deliberately skipped: BiomedCLIP must coexist with MLX MedGemma,
    and both contending for Metal memory triggers GPU OOM on the 8GB M1.
    CPU is plenty for a single-vector encoder.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════
# Base interface
# ══════════════════════════════════════════════════════════════════════
class BaseRetriever(abc.ABC):
    """Common contract for every retriever."""

    name: str = "base"

    def __init__(self) -> None:
        self.index_dir = CONFIG.path("data_index") / self.name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._meta: list[dict] = []  # per-document metadata, index-aligned

    @abc.abstractmethod
    def build_index(self, records: list[CxrRecord]) -> None:
        """Embed every report and persist a searchable index to disk."""

    @abc.abstractmethod
    def load_index(self) -> None:
        """Load a previously built index from disk."""

    @abc.abstractmethod
    def retrieve(self, query: str | Image.Image, top_k: int = 4) -> list[RetrievalResult]:
        """Return the ``top_k`` most relevant reports for ``query``."""

    def _results_from_meta(
        self, indices: list[int], scores: list[float]
    ) -> list[RetrievalResult]:
        """Build :class:`RetrievalResult` objects from ranked index positions."""
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            meta = self._meta[idx]
            results.append(
                RetrievalResult(
                    text=meta["report"],
                    score=float(score),
                    study_id=meta["study_id"],
                    rank=rank,
                    image_path=meta.get("image_path"),
                )
            )
        return results


# ══════════════════════════════════════════════════════════════════════
# BiomedCLIP - single-vector retriever (local, lightweight)
# ══════════════════════════════════════════════════════════════════════
class BiomedCLIPRetriever(BaseRetriever):
    """Domain-tuned CLIP retriever with a FAISS cosine-similarity index."""

    name = "biomedclip"

    def __init__(self) -> None:
        super().__init__()
        self.model_id = CONFIG.retrieval.biomedclip.model_id
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._index = None
        self._device = _torch_device()

    def _ensure_model(self) -> None:
        """Lazily load the open_clip BiomedCLIP model + tokenizer."""
        if self._model is not None:
            return
        import open_clip

        self._model, self._preprocess = open_clip.create_model_from_pretrained(self.model_id)
        self._tokenizer = open_clip.get_tokenizer(self.model_id)
        self._model = self._model.to(self._device)
        self._model.train(False)  # inference mode (== .eval())

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encode report/question texts into L2-normalised embeddings."""
        import torch

        self._ensure_model()
        vecs = []
        for start in range(0, len(texts), 32):  # batch to bound memory on 8GB
            batch = texts[start : start + 32]
            tokens = self._tokenizer(batch, context_length=256).to(self._device)
            # Explicit no_grad context: thread-safe across Streamlit reruns,
            # unlike a process/thread-global grad-mode flag.
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = torch.nn.functional.normalize(feats, dim=-1)
            vecs.append(feats.detach().cpu().numpy())
        return np.vstack(vecs).astype("float32")

    def _embed_image(self, image: Image.Image) -> np.ndarray:
        """Encode one X-ray into an L2-normalised embedding."""
        import torch

        self._ensure_model()
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.detach().cpu().numpy().astype("float32")

    def build_index(self, records: list[CxrRecord]) -> None:
        import faiss

        texts = [r.report_text for r in records]
        embeddings = self._embed_texts(texts)
        self._meta = [
            {"study_id": r.study_id, "report": r.report_text, "image_path": str(r.image_path)}
            for r in records
        ]
        # Inner product on normalised vectors == cosine similarity.
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

        faiss.write_index(self._index, str(self.index_dir / "index.faiss"))
        np.save(self.index_dir / "embeddings.npy", embeddings)
        (self.index_dir / "meta.json").write_text(json.dumps(self._meta), encoding="utf-8")
        print(f"[BiomedCLIP] indexed {len(records)} reports -> {self.index_dir}")

    def load_index(self) -> None:
        import faiss

        meta_path = self.index_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No BiomedCLIP index at {self.index_dir}. Run scripts/build_index.py."
            )
        self._index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def retrieve(self, query: str | Image.Image, top_k: int = 4) -> list[RetrievalResult]:
        if self._index is None:
            self.load_index()
        if isinstance(query, Image.Image):
            qvec = self._embed_image(query)          # cross-modal: image -> reports
        else:
            qvec = self._embed_texts([str(query)])   # intra-modal: question -> reports
        scores, indices = self._index.search(qvec, top_k)
        return self._results_from_meta(indices[0].tolist(), scores[0].tolist())


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════
_REGISTRY: dict[str, type[BaseRetriever]] = {
    "biomedclip": BiomedCLIPRetriever,
}


def build_retriever(name: str | None = None) -> BaseRetriever:
    """Instantiate a retriever by name (defaults to ``retrieval.active`` in config)."""
    name = (name or CONFIG.retrieval.active).lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown retriever {name!r}. Options: {list(_REGISTRY)}")
    return _REGISTRY[name]()
