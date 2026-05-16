# Video Demo Script (5–15 min)

A suggested running order for the demo recording. Show both modes working and
highlight the comparison insights.

## 0. Intro (~1 min)
- State the task: dual-mode chest X-ray system — Report Generation + RAG QA.
- Show the repo structure and `README.md` architecture diagram.
- Mention the 8GB M1 constraint and the "split execution" design (MLX locally,
  ColPali on Colab).

## 1. Data & QA dataset (~2 min)
- Run `python scripts/download_data.py` (or show it already done) — point out
  the auto-detected CSV schema.
- Open `notebooks/01_qa_dataset_creation.ipynb`; show one report and the
  Gemini-generated grounded QA pairs (3 question types).
- Show `data/qa/qa_dataset.json` and the per-type counts.

## 2. Mode 1 — Report Generation (~3 min)
- `streamlit run app/streamlit_app.py` → **Report Generation** tab.
- Upload a chest X-ray → generate with `zero_shot` → show FINDINGS / IMPRESSION.
- Switch to `rag_fewshot` → generate → expand the retrieved reference reports.
- Briefly contrast the two outputs.

## 3. Mode 2 — QA (RAG) (~3 min)
- **QA** tab → ask a clinical question (e.g. "Is there pleural effusion?").
- Show the retrieved context (study IDs + scores) and the grounded answer.
- Ask an unanswerable question → show the refusal contract in action.
- Switch the retriever to ColPali (if the Colab index is loaded).

## 4. Model comparison (~3 min)
- Show `reports/comparison_results.md` (from `python -m cxr.evaluation.compare`).
- Retriever table: BiomedCLIP vs ColPali — Recall@k, MRR, latency.
- Report-strategy table: zero-shot vs RAG few-shot — BLEU / ROUGE-L / BERTScore.
- Give the key insight: speed/footprint vs late-interaction accuracy trade-off;
  ColPali's document-image training vs raw X-rays.

## 5. Wrap-up (~1 min)
- Recap: two independent modes, ≥2 models compared, end-to-end pipeline.
- State limitations and the not-for-clinical-use disclaimer.
