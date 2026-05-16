# 🩻 Multi-Modal Chest X-Ray Intelligence System

A dual-mode medical AI system for chest X-rays, built for **DSAI 413 – Assignment 2**.

| Mode | Input → Output | Models |
|------|----------------|--------|
| **Report Generation** | X-ray image → structured radiology report | MedGemma-4B (+ optional RAG few-shot) |
| **QA (RAG)** | clinical question → grounded answer | BiomedCLIP retriever + MedGemma generator |

The two modes are fully independent — each has its own pipeline, CLI script,
and Streamlit tab.

> ⚠️ **Research / educational use only.** Outputs are not validated for clinical use.

---

## Architecture

```
              ┌──────────────── MODE 1: REPORT GENERATION ──────────────────┐
  X-ray image ┤ [zero_shot]   image ─────────────────────────► MedGemma ► report │
              │ [rag_fewshot] image ► BiomedCLIP ► reports ──► MedGemma ► report │
              └──────────────────────────────────────────────────────────────┘

              ┌──────────────── MODE 2: QA (RAG) ───────────────────────────┐
  question ───┤ question ► BiomedCLIP ► top-k report context ► MedGemma ► answer │
  (+ opt img) └──────────────────────────────────────────────────────────────┘
```

**Models & roles**

| Model | Role | Runs on |
|-------|------|---------|
| `google/medgemma-4b-it` | Vision-language generator (reports + answers) | M1 via **MLX 4-bit** |
| `microsoft/BiomedCLIP-...` | Single-vector retriever (domain-tuned on PubMed) | M1 (lightweight) |
| MedGemma (text-only) | Offline QA-pair generation — **default**, no API key | M1 |
| Gemini API | Offline QA-pair generation (optional alternative) | API only |

### Designed for an 8GB M1 Mac

`bitsandbytes` 4-bit quantization is CUDA-only, so MedGemma runs locally via
**MLX** (4-bit, ~3GB). BiomedCLIP is a small, domain-tuned CLIP variant that
runs comfortably alongside it. Models are lazy-loaded one at a time so the
whole pipeline — both modes plus the Streamlit demo — fits in 8GB.

---

## Repository structure

```
src/cxr/
  config.py              # central config (config.yaml + .env)
  data/loader.py         # MIMIC-CXR loading, auto-detects CSV columns
  data/qa_builder.py     # QA-pair generation (MedGemma | Gemini backend)
  models/medgemma.py     # MedGemma wrapper (MLX | transformers backends)
  models/retrievers.py   # BiomedCLIP retriever
  modes/report_generation.py   # Mode 1
  modes/qa_rag.py              # Mode 2
  evaluation/            # report_eval, retrieval_eval, compare
app/streamlit_app.py     # dual-mode demo
scripts/                 # download_data, build_index, run_report_gen, run_qa
reports/short_report.md  # architecture + comparison write-up
```

---

## Setup

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes the `cxr` package importable

# 2. Credentials
cp .env.example .env                   # then fill in the values
```

`.env` needs:
- **`HF_TOKEN`** — MedGemma is gated: create a token *and* accept the license at
  <https://huggingface.co/google/medgemma-4b-it>.
- **`GEMINI_API_KEY`** *(optional)* — only needed if `qa_generation.backend` is set
  to `gemini` in `config.yaml`. The default `medgemma` backend needs no API key.
- **`KAGGLE_USERNAME` / `KAGGLE_KEY`** — for the dataset download.

All knobs (models, paths, sample sizes) live in [`config.yaml`](config.yaml).

---

## Usage

```bash
# 1. Download the MIMIC-CXR dataset (prints the detected schema)
python scripts/download_data.py

# 2. Build the QA dataset from the reports (local MedGemma backend)
python -c "from cxr.data.qa_builder import build_qa_dataset; build_qa_dataset()"

# 3. Build the BiomedCLIP retrieval index
python scripts/build_index.py

# 4. Mode 1 — Report Generation
python scripts/run_report_gen.py --image path/to/xray.jpg
python scripts/run_report_gen.py --image path/to/xray.jpg --strategy rag_fewshot

# 5. Mode 2 — QA (RAG)
python scripts/run_qa.py --question "Is there evidence of pleural effusion?"

# 6. Demo app (both modes)
streamlit run app/streamlit_app.py

# 7. Model comparison → reports/comparison_results.md
python -m cxr.evaluation.compare
```

```bash
pytest -q          # lightweight smoke tests (no GPU / no weights needed)
```

---

## QA dataset creation

The assignment provides no QA dataset, so one is synthesised from the report
corpus: for each report, a language model (**MedGemma** by default; Gemini
optional) generates grounded question-answer pairs in three styles (factual
yes/no, location/severity, impression). Each pair stores its `source_study_id`,
which doubles as ground truth for retrieval evaluation.
Full method: [`reports/short_report.md`](reports/short_report.md).

---

## Limitations

- 8GB RAM makes local MedGemma inference slow; the demo lazy-loads one model at a time.
- The synthetic QA questions are generic, so exact-source retrieval recall is
  low by construction (see the comparison write-up).
- Generated reports and answers are **not clinically validated**.
