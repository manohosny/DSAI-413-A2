"""MedGemma generator wrapper (dual backend).

MedGemma-4B is the vision-language model used for **both** modes:
report generation (image -> report) and RAG answer generation
(question + retrieved context -> grounded answer).

Two backends share one interface so the same calling code runs on a
laptop and in the cloud:

  * ``mlx``          - Apple Silicon. 4-bit quantised (~3GB) via ``mlx-vlm``.
                       The only way to run MedGemma on an 8GB M1, because
                       ``bitsandbytes`` 4-bit is CUDA-only.
  * ``transformers`` - CUDA/CPU (Colab/Kaggle). fp16/bf16 via HF transformers.

The model is loaded lazily on first use and can be unloaded to free the
unified memory an 8GB Mac cannot spare.
"""

from __future__ import annotations

import gc
from typing import Any

from PIL import Image

from cxr.config import CONFIG, get_secret

RADIOLOGIST_SYSTEM = (
    "You are an expert radiologist. Be precise, clinical, and concise. "
    "Only state findings supported by the evidence you are given."
)


class MedGemma:
    """Lazy-loaded MedGemma generator with an MLX or transformers backend."""

    def __init__(self, backend: str | None = None) -> None:
        self.backend = backend or CONFIG.medgemma.backend
        self._model: Any = None
        self._processor: Any = None
        self._cfg: Any = None  # backend-specific extras (e.g. MLX model config)

    # ── lifecycle ─────────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load weights for the configured backend (idempotent)."""
        if self.is_loaded:
            return
        # Authenticate with HF so the gated MedGemma weights can be pulled.
        token = get_secret("HF_TOKEN", required=False)
        if token:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)

        if self.backend == "mlx":
            self._load_mlx()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise ValueError(f"Unknown MedGemma backend: {self.backend!r}")

    def unload(self) -> None:
        """Release the model so an 8GB machine can load a different one."""
        self._model = self._processor = self._cfg = None
        gc.collect()

    def _load_mlx(self) -> None:
        from mlx_vlm import load

        from cxr.config import PROJECT_ROOT

        # Use locally downloaded weights if present, else treat as an HF repo id.
        ref = CONFIG.medgemma.mlx_model_id
        local = PROJECT_ROOT / ref
        model_ref = str(local) if local.exists() else ref

        self._model, self._processor = load(model_ref)
        self._cfg = self._model.config

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_id = CONFIG.medgemma.hf_model_id
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # T4-safe; bf16 needs A100/L4
            device_map="auto",
        )

    # ── generation ────────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        image: Image.Image | None = None,
        system_prompt: str = RADIOLOGIST_SYSTEM,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text from a prompt, optionally conditioned on an X-ray."""
        if not self.is_loaded:
            self.load()
        max_new_tokens = max_new_tokens or CONFIG.medgemma.max_new_tokens
        if self.backend == "mlx":
            return self._generate_mlx(prompt, image, system_prompt, max_new_tokens)
        return self._generate_transformers(prompt, image, system_prompt, max_new_tokens)

    def _generate_mlx(
        self, prompt: str, image: Image.Image | None, system_prompt: str, max_new_tokens: int
    ) -> str:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        # Release cached GPU buffers before a (memory-heavy) multimodal pass -
        # on an 8GB M1 this avoids paging that can trip the Metal GPU watchdog.
        try:
            mx.clear_cache()
        except Exception:  # noqa: BLE001 - older mlx lacks this; harmless
            pass

        images = [image] if image is not None else []
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        formatted = apply_chat_template(
            self._processor, self._cfg, full_prompt, num_images=len(images)
        )
        result = generate(
            self._model,
            self._processor,
            formatted,
            image=images,
            max_tokens=max_new_tokens,
            temperature=CONFIG.medgemma.temperature,
            verbose=False,
        )
        # mlx-vlm's return type varies by version: a GenerationResult (with
        # a .text attribute), a (text, stats) tuple, or a plain string.
        if hasattr(result, "text"):
            text = result.text
        elif isinstance(result, tuple):
            text = result[0]
        else:
            text = result
        return str(text).strip()

    def _generate_transformers(
        self, prompt: str, image: Image.Image | None, system_prompt: str, max_new_tokens: int
    ) -> str:
        import torch

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image is not None:
            user_content.append({"type": "image", "image": image})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        # Decode only the newly generated tokens, not the echoed prompt.
        new_tokens = output[0][input_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()
