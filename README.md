# 🩻 Multi-Modal Chest X-Ray Intelligence System

A dual-mode medical AI system for chest X-rays, built for **DSAI 413 – Assignment 2**.

| Mode | Input → Output | Models |
|------|----------------|--------|
| **Report Generation** | X-ray image → structured radiology report | MedGemma-4B (+ optional RAG few-shot) |
| **QA (RAG)** | clinical question → grounded answer | BiomedCLIP / ColPali retriever + MedGemma generator |

The two modes are fully independent — each has its own pipeline, CLI script,
and Streamlit tab.

> ⚠️ **Research / educational use only.** Outputs are not validated for clinical use.

---

## Architecture

```
                ┌──────────────────────── MODE 1: REPORT GENERATION ───────────────────┐
  X-ray image ──┤  [zero_shot]   image ─────────────────────────────► MedGemma ► report │
                │  [rag_fewshot] image ► retriever ► similar reports ► MedGemma ► report │
                └───────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────── MODE 2: QA (RAG) ────────────────────────────┐
  question ─────┤  question ► retriever ► top-k report context ► MedGemma ► answer      │
  (+ opt image) │                         (BiomedCLIP | ColPali)                        │
                └───────────────────────────────────────────────────────────────────────┘
```

**Models & roles**

| Model | Role | Runs on |
|-------|------|---------|
| `google/medgemma-4b-it` | Vision-language generator (reports + answers) | M1 via **MLX 4-bit**; Colab via transformers |
| `microsoft/BiomedCLIP-...` | Single-vector retriever (domain-tuned) | M1 (lightweight) |
| `vidore/colpali-v1.3` | Multi-vector retriever (rendered report pages) | Colab GPU |
| Gemini API | Offline QA-pair generation | API only |

### Why a "split execution" design?

The target machine is an **8GB M1 Mac**. `bitsandbytes` 4-bit quantization is
CUDA-only, so MedGemma runs locally via **MLX** (4-bit, ~3GB). ColPali (3B base,
multi-vector) is too heavy for 8GB, so its index is built in a **Colab notebook**
and downloaded back. BiomedCLIP is small enough to power the live demo. ColPali's
weight/latency on consumer hardware is itself part of the model comparison.

---

## Repository structure

```
src/cxr/
  config.py              # central config (config.yaml + .env)
  data/loader.py         # MIMIC-CXR loading, auto-detects CSV columns
  data/qa_builder.py     # Gemini-based QA-pair generation
  models/medgemma.py     # MedGemma wrapper (MLX | transformers backends)
  models/retrievers.py   # BiomedCLIP + ColPali behind one interface
  modes/report_generation.py   # Mode 1
  modes/qa_rag.py              # Mode 2
  evaluation/            # report_eval, retrieval_eval, compare
  utils/rendering.py     # report text → page image (for ColPali)
app/streamlit_app.py     # dual-mode demo
scripts/                 # download_data, build_index, run_report_gen, run_qa
notebooks/               # 01 QA dataset · 02 ColPali (Colab) · 03 comparison
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
- **`GEMINI_API_KEY`** — free key from <https://aistudio.google.com/apikey> (QA dataset only).
- **`KAGGLE_USERNAME` / `KAGGLE_KEY`** — for the dataset download.

All knobs (models, paths, sample sizes) live in [`config.yaml`](config.yaml).

---

## Usage

```bash
# 1. Download the MIMIC-CXR dataset (prints the detected schema)
python scripts/download_data.py

# 2. Build the QA dataset from the reports (Gemini) — or run notebook 01
python -c "from cxr.data.qa_builder import build_qa_dataset; build_qa_dataset()"

# 3. Build the retrieval index (BiomedCLIP, local)
python scripts/build_index.py --retriever biomedclip

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

**ColPali** is built on Colab — open
[`notebooks/02_colpali_indexing_colab.ipynb`](notebooks/02_colpali_indexing_colab.ipynb),
run it on a T4, download `colpali_index.zip`, and unzip into `data/index/colpali/`.

```bash
pytest -q          # lightweight smoke tests (no GPU / no weights needed)
```

---

## QA dataset creation

The assignment provides no QA dataset, so one is synthesised from the report
corpus: for each report, Gemini generates grounded question-answer pairs in
three styles (factual yes/no, location/severity, impression). Each pair stores
its `source_study_id`, which doubles as ground truth for retrieval evaluation.
Full method: [`reports/short_report.md`](reports/short_report.md).

---

## Limitations

- 8GB RAM makes local MedGemma inference slow; the demo lazy-loads one model at a time.
- ColPali is trained on document screenshots — report pages are rendered for it,
  and it is not optimized for raw X-ray pixels (see the comparison).
- Generated reports and answers are **not clinically validated**.
