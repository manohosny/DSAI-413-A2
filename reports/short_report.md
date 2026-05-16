# Short Report — Multi-Modal Chest X-Ray Intelligence System

**Course:** DSAI 413 — Assignment 2
**System:** Dual-mode (Report Generation + RAG-based QA) chest X-ray AI

> Quantitative tables below are populated by running
> `python -m cxr.evaluation.compare` (or notebook 03). Placeholders marked
> `TBD` are filled from that run. The qualitative analysis is final.

---

## 1. Architecture overview

The system exposes **two independent modes** that share a model layer but
never share control flow:

**Mode 1 — Report Generation** (`image → report`)
A chest X-ray is passed to MedGemma with a radiologist system prompt that
requests a structured `FINDINGS` / `IMPRESSION` report. Two strategies:
- `zero_shot` — MedGemma sees only the image.
- `rag_fewshot` — a retriever finds visually similar X-rays; their reports are
  injected as in-context examples before generation.

**Mode 2 — QA (RAG)** (`question → answer`)
A clinical question is embedded and used to retrieve the top-k most relevant
reports from the corpus (the knowledge base). MedGemma then answers **grounded**
strictly in that retrieved context, with an explicit refusal contract when the
context is insufficient. An X-ray image can optionally be attached.

```
config.yaml ─► config.py ─► { data loader, MedGemma, retrievers }
                                   │
              ┌────────────────────┼─────────────────────┐
         Mode 1 (report)      Mode 2 (RAG QA)        Evaluation
              │                    │                     │
              └──────────► Streamlit app ◄───────────────┘
```

A deliberate engineering constraint shaped the design: the target machine is an
**8GB M1 Mac**. Because `bitsandbytes` 4-bit quantization is CUDA-only, MedGemma
runs locally through **MLX** (4-bit, ~3GB). ColPali is too heavy for 8GB, so it
is indexed in a Colab notebook; BiomedCLIP — small and domain-tuned — powers the
local demo. This "split execution" keeps every assignment requirement runnable.

## 2. Model choices

| Model | Role | Why chosen |
|-------|------|-----------|
| **MedGemma-4B-it** | VLM generator (reports + answers) | Mandatory; medical-domain VLM, strong single-image CXR understanding |
| **ColPali v1.3** | Multi-vector retriever | Mandatory; ColBERT-style late interaction over rendered report pages |
| **BiomedCLIP** | Single-vector retriever | Replaces vanilla CLIP — domain-tuned on 15M PubMed pairs, light enough for 8GB |
| **Gemini 2.0 Flash** | Offline QA-pair generation | Free tier, high-quality grounded generation, no local compute |

ColPali was trained on **document-page screenshots**, not X-ray pixels — hence
the assignment's "might need fine-tune" note. Rather than fine-tune (GPU-heavy,
out of scope for 8GB), reports are **rendered as page images** so ColPali
receives the input modality it expects; the mismatch on raw X-rays is documented
as a comparison finding.

## 3. QA dataset creation

No QA dataset was provided, so one was synthesised from the radiology reports:

1. Each report (the dataset's `text` column) is sent to **Gemini** with a strict
   prompt: generate N question-answer pairs **grounded only in that report**,
   no outside knowledge, no invented findings.
2. Three balanced question types are requested: `factual_yesno`,
   `location_severity`, `impression`.
3. Each pair is stored with its `source_study_id` and `source_report`.

This `source_study_id` is the key design choice: it serves as **ground truth for
retrieval evaluation** — a retrieval is correct when the source report appears in
the top-k. The dataset is therefore both the QA knowledge-base eval set and the
retrieval benchmark, with no manual labelling. Build: `notebooks/01_qa_dataset_creation.ipynb`.

## 4. Comparison results

### 4.1 Retriever comparison (QA mode)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | Avg latency (s) |
|-----------|----------|----------|----------|-----|-----------------|
| BiomedCLIP | TBD | TBD | TBD | TBD | TBD |
| ColPali    | TBD | TBD | TBD | TBD | TBD |

*Expected pattern:* BiomedCLIP is far faster and lighter (single 512-d vector,
FAISS) and benefits from medical-domain pretraining. ColPali's late interaction
captures finer token-level matches and should compete or win on recall once
reports are rendered as pages, at a substantial latency/memory cost.

### 4.2 Report-generation strategy comparison

| Strategy | BLEU | ROUGE-L | BERTScore |
|----------|------|---------|-----------|
| zero_shot   | TBD | TBD | TBD |
| rag_fewshot | TBD | TBD | TBD |

*Expected pattern:* `rag_fewshot` should improve surface metrics (BLEU/ROUGE) by
aligning phrasing to the corpus style; BERTScore gains are usually smaller since
zero-shot MedGemma is already semantically competent. Retrieving an irrelevant
report can also mislead generation — a documented limitation.

## 5. Limitations & insights

- **Hardware** dominated design: 8GB RAM forced MLX quantization, lazy single-model
  loading, and offloading ColPali to Colab.
- **CLIP text-text retrieval** (question → report) is weaker than its cross-modal
  (image → report) use; ColPali's late interaction partly addresses this.
- **Evaluation is automatic** (BLEU/ROUGE/BERTScore, Recall@k/MRR) — these correlate
  imperfectly with clinical correctness; a radiologist review would be the next step.
- Generated reports/answers are **not clinically validated**.
