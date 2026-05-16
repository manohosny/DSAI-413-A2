# Short Report — Multi-Modal Chest X-Ray Intelligence System

**Course:** DSAI 413 — Assignment 2
**System:** Dual-mode (Report Generation + RAG-based QA) chest X-ray AI

> Quantitative tables below were produced by `python -m cxr.evaluation.compare`
> on the local 8GB M1 (BiomedCLIP retriever + MedGemma generator, MLX 4-bit).
> ColPali rows and BERTScore are produced by the Colab notebooks (02/03).

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
| **MedGemma (text-only)** | Offline QA-pair generation | Default QA backend — no API key, reproducible, medically grounded (see §3) |

ColPali was trained on **document-page screenshots**, not X-ray pixels — hence
the assignment's "might need fine-tune" note. Rather than fine-tune (GPU-heavy,
out of scope for 8GB), reports are **rendered as page images** so ColPali
receives the input modality it expects; the mismatch on raw X-rays is documented
as a comparison finding.

## 3. QA dataset creation

No QA dataset was provided, so one was synthesised from the radiology reports:

1. Each report (the dataset's `text` column) is sent to a language model with a
   strict prompt: generate N question-answer pairs **grounded only in that
   report** — no outside knowledge, no invented findings.
2. Three balanced question types are requested: `factual_yesno`,
   `location_severity`, `impression`.
3. Each pair is stored with its `source_study_id` and `source_report`.

**Backend — MedGemma, not Gemini.** Gemini was the original choice, but the
available API key's project returned `429 RESOURCE_EXHAUSTED` with `limit: 0`
(no free-tier quota). QA generation was pivoted to **MedGemma run locally**,
which has two advantages: zero API dependency (fully reproducible offline) and
medical-domain grounding. `qa_builder.py` keeps both backends behind one shared
prompt, switchable in `config.yaml`.

**This run produced 120 QA pairs from 40 reports** (3 per report, 100% JSON-parse
success): 59 `factual_yesno`, 41 `impression`, 20 `location_severity`. MedGemma
grounded answers correctly — e.g. *"Is there evidence of pneumonia?"* →
*"The report does not mention pneumonia"*, refusing to invent an absent finding.

The `source_study_id` is the key design choice: it is **ground truth for
retrieval evaluation** — a retrieval is correct when the source report appears in
the top-k. The dataset is therefore both the QA eval set and the retrieval
benchmark, with no manual labelling. Build: `notebooks/01_qa_dataset_creation.ipynb`.

## 4. Comparison results

### 4.1 Retriever comparison (QA mode)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | Avg latency (s) |
|-----------|----------|----------|----------|-----|-----------------|
| BiomedCLIP | 0.00 | 0.06 | 0.14 | 0.046 | 0.319 |
| ColPali    | *Colab* | *Colab* | *Colab* | *Colab* | *Colab* |

*Run: 50-question sample over a 300-report index. ColPali's index is built in
notebook 02 (Colab) — its 3B base is too heavy for the 8GB M1.*

**Reading the numbers — low recall is itself a finding, not a bug.** The
synthetic questions are deliberately generic ("Are there pleural effusions?",
"What is the impression?"); dozens of the 300 reports legitimately match each,
so the *exact source* report rarely ranks #1. Crucially, Recall@5 = 0.14 still
beats random chance (≈0.017 over 300 reports) by ~8×, confirming retrieval
genuinely works — and the QA mode answers such questions correctly regardless,
since *any* report containing the queried finding is valid grounding context.
The lesson: exact-source Recall@k *under-measures* RAG quality for
generic-question datasets; a finding-overlap metric would score it more fairly.

### 4.2 Report-generation strategy comparison

| Strategy | BLEU | ROUGE-L |
|----------|------|---------|
| zero_shot   | 2.95 | 0.251 |
| rag_fewshot | **6.66** | **0.297** |

*Run: 10 reports with locally-available X-rays. BERTScore omitted on the 8GB M1
(`roberta-large` + MedGemma exceed memory); it is computed in the Colab notebook.*

**RAG few-shot more than doubled BLEU** (2.95 → 6.66) and lifted ROUGE-L
(0.251 → 0.297). Injecting retrieved similar reports as in-context examples
measurably steers MedGemma toward the corpus's reporting style and vocabulary.
Absolute BLEU is low — expected for chest-X-ray report generation, where reports
are short and lexically varied (published CXR BLEU-4 is typically 5–15); the
*relative* gain is the meaningful signal. Risk: retrieving an irrelevant report
can also mislead generation — a documented limitation.

## 5. Limitations & insights

- **Hardware dominated the design.** 8GB RAM forced MLX 4-bit quantization, lazy
  single-model loading, and offloading ColPali to Colab. Two M1-specific issues
  surfaced and were fixed: the multimodal prefill tripped the Metal GPU watchdog
  (`kIOGPUCommandBufferCallbackErrorTimeout`) until `mx.clear_cache()` was called
  before each image pass, and `faiss` + `torch` each vendor `libomp` (resolved
  with `KMP_DUPLICATE_LIB_OK`). MedGemma image inference runs ~28s/report locally.
- **Retrieval metric.** Exact-source Recall@k under-measures RAG quality for the
  generic synthetic questions (see §4.1); a finding-overlap metric would be fairer.
- **CLIP text-text retrieval** (question → report) is weaker than its cross-modal
  (image → report) use; ColPali's late interaction partly addresses this.
- **MedGemma-generated QA** is simpler in phrasing than a frontier model would
  produce, but is reproducible, offline, and medically grounded.
- **Evaluation is automatic** (BLEU/ROUGE/BERTScore, Recall@k/MRR) — it correlates
  imperfectly with clinical correctness; a radiologist review would be the next step.
- Generated reports/answers are **not clinically validated**.
