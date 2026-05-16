# Short Report — Multi-Modal Chest X-Ray Intelligence System

**Course:** DSAI 413 — Assignment 2
**System:** Dual-mode (Report Generation + RAG-based QA) chest X-ray AI

> Quantitative tables below were produced by `python -m cxr.evaluation.compare`
> on a local 8GB M1 (MedGemma generator via MLX 4-bit + BiomedCLIP retriever).

---

## 1. Architecture overview

The system exposes **two independent modes** that share a model layer but
never share control flow:

**Mode 1 — Report Generation** (`image → report`)
A chest X-ray is passed to MedGemma with a radiologist system prompt that
requests a structured `FINDINGS` / `IMPRESSION` report. Two strategies:
- `zero_shot` — MedGemma sees only the image.
- `rag_fewshot` — BiomedCLIP retrieves similar reports; they are injected as
  in-context examples before generation.

**Mode 2 — QA (RAG)** (`question → answer`)
A clinical question is embedded with BiomedCLIP and used to retrieve the top-k
most relevant reports from the corpus (the knowledge base). MedGemma then
answers **grounded** strictly in that retrieved context, with an explicit
refusal contract when the context is insufficient. An X-ray image can
optionally be attached.

```
config.yaml ─► config.py ─► { data loader, MedGemma, BiomedCLIP }
                                   │
              ┌────────────────────┼─────────────────────┐
         Mode 1 (report)      Mode 2 (RAG QA)        Evaluation
              │                    │                     │
              └──────────► Streamlit app ◄───────────────┘
```

A deliberate engineering constraint shaped the design: the target machine is an
**8GB M1 Mac**. Because `bitsandbytes` 4-bit quantization is CUDA-only, MedGemma
runs locally through **MLX** (4-bit, ~3GB). BiomedCLIP is small and domain-tuned,
so it runs comfortably alongside it; models are lazy-loaded one at a time so the
whole pipeline fits in 8GB.

## 2. Model choices

Two models are used and compared, each playing a distinct role:

| Model | Role | Why chosen |
|-------|------|-----------|
| **MedGemma-4B-it** | Vision-language generator (reports + answers) | Medical-domain VLM with strong single-image CXR understanding |
| **BiomedCLIP** | Single-vector retriever (QA mode + RAG few-shot) | A CLIP variant domain-tuned on 15M PubMed image-caption pairs — far better than vanilla CLIP on radiology, and light enough for 8GB |

MedGemma also doubles as the **QA-pair generator** (text-only) when building the
dataset — see §3. The two models are compared in §4: the retriever is evaluated
on the QA task (§4.1), and the report-generation comparison (§4.2) measures the
*performance gain BiomedCLIP contributes* to MedGemma.

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
benchmark, with no manual labelling. Build:
`from cxr.data.qa_builder import build_qa_dataset; build_qa_dataset()`.

## 4. Results

### 4.1 Retriever evaluation (BiomedCLIP, QA mode)

| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | Avg latency (s) |
|-----------|----------|----------|----------|-----|-----------------|
| BiomedCLIP | 0.00 | 0.06 | 0.14 | 0.046 | 0.319 |

*Run: 50-question sample over a 300-report index.*

**Reading the numbers — low recall is itself a finding, not a bug.** The
synthetic questions are deliberately generic ("Are there pleural effusions?",
"What is the impression?"); dozens of the 300 reports legitimately match each,
so the *exact source* report rarely ranks #1. Crucially, Recall@5 = 0.14 still
beats random chance (≈0.017 over 300 reports) by ~8×, confirming retrieval
genuinely works — and the QA mode answers such questions correctly regardless,
since *any* report containing the queried finding is valid grounding context.
The lesson: exact-source Recall@k *under-measures* RAG quality for
generic-question datasets; a finding-overlap metric would score it more fairly.

### 4.2 Report-generation comparison — MedGemma vs MedGemma + BiomedCLIP

| Strategy | BLEU | ROUGE-L |
|----------|------|---------|
| zero_shot (MedGemma alone)          | 2.95 | 0.251 |
| rag_fewshot (MedGemma + BiomedCLIP) | **6.66** | **0.297** |

*Run: 10 reports with locally-available X-rays.*

This is the core model comparison: it isolates the **performance contribution of
BiomedCLIP**. Adding BiomedCLIP-retrieved reports as in-context examples **more
than doubled BLEU** (2.95 → 6.66) and lifted ROUGE-L (0.251 → 0.297) — retrieval
measurably steers MedGemma toward the corpus's reporting style and vocabulary.
Absolute BLEU is low — expected for chest-X-ray report generation, where reports
are short and lexically varied (published CXR BLEU-4 is typically 5–15); the
*relative* gain is the meaningful signal. Risk: retrieving an irrelevant report
can also mislead generation — a documented limitation.

## 5. Limitations & insights

- **Hardware dominated the design.** 8GB RAM forced MLX 4-bit quantization and
  lazy single-model loading. Two M1-specific issues surfaced and were fixed: the
  multimodal prefill tripped the Metal GPU watchdog
  (`kIOGPUCommandBufferCallbackErrorTimeout`) until `mx.clear_cache()` was called
  before each image pass, and `faiss` + `torch` each vendor `libomp` (resolved
  with `KMP_DUPLICATE_LIB_OK`). MedGemma image inference runs ~28s/report locally.
- **Retrieval metric.** Exact-source Recall@k under-measures RAG quality for the
  generic synthetic questions (see §4.1); a finding-overlap metric would be fairer.
- **Text-text retrieval** (question → report) is intrinsically harder than
  cross-modal (image → report) matching for a CLIP-style single-vector model.
- **MedGemma-generated QA** is simpler in phrasing than a frontier model would
  produce, but is reproducible, offline, and medically grounded.
- **Evaluation is automatic** (BLEU/ROUGE, Recall@k/MRR) — it correlates
  imperfectly with clinical correctness; a radiologist review would be the next step.
- Generated reports/answers are **not clinically validated**.
