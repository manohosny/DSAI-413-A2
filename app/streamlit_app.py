"""Streamlit demo - dual-mode Chest X-Ray Intelligence System.

Two tabs, two independent modes:
  * Report Generation - upload an X-ray, get a structured report.
  * QA - ask a clinical question, get a RAG-grounded answer.

Models are cached with ``st.cache_resource`` so the (heavy) MedGemma load
happens once per session - important on an 8GB M1.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the `cxr` package importable without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from cxr.config import CONFIG  # noqa: E402

st.set_page_config(page_title="Chest X-Ray Intelligence", page_icon="🩻", layout="wide")


# ── cached resource loaders ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MedGemma (one-time)...")
def get_medgemma():
    from cxr.models.medgemma import MedGemma

    model = MedGemma()
    model.load()
    return model


@st.cache_resource(show_spinner="Loading retriever index...")
def get_retriever(name: str):
    from cxr.models.retrievers import build_retriever

    retriever = build_retriever(name)
    retriever.load_index()
    return retriever


# ── sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("🩻 CXR Intelligence")
st.sidebar.caption(
    "Multi-modal chest X-ray system. Report Generation and QA run independently."
)
st.sidebar.info(f"MedGemma backend: **{CONFIG.medgemma.backend}**")
st.sidebar.warning("Research/educational use only - not for clinical decisions.")

report_tab, qa_tab = st.tabs(["📝 Report Generation", "💬 QA (RAG)"])


# ══════════════════════════════════════════════════════════════════════
# Mode 1 - Report Generation
# ══════════════════════════════════════════════════════════════════════
with report_tab:
    st.header("Report Generation")
    st.write("Upload a chest X-ray; MedGemma produces a structured radiology report.")

    col_in, col_out = st.columns(2)
    with col_in:
        uploaded = st.file_uploader(
            "Chest X-ray image", type=["png", "jpg", "jpeg"], key="report_img"
        )
        strategy = st.radio(
            "Generation strategy",
            ["zero_shot", "rag_fewshot"],
            help="zero_shot: image only. rag_fewshot: inject similar reports as context.",
        )
        run_report = st.button("Generate report", type="primary", disabled=uploaded is None)

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        with col_in:
            st.image(image, caption="Input X-ray", use_container_width=True)

        if run_report:
            from cxr.modes.report_generation import ReportGenerator

            retriever = None
            if strategy == "rag_fewshot":
                try:
                    retriever = get_retriever(CONFIG.retrieval.active)
                except FileNotFoundError as exc:
                    st.error(f"{exc}")
                    st.stop()

            with col_out:
                with st.spinner("Generating report..."):
                    generator = ReportGenerator(medgemma=get_medgemma())
                    result = generator.generate(image, strategy=strategy, retriever=retriever)
                st.subheader("Generated report")
                st.text_area("Report", result.report, height=320, label_visibility="collapsed")
                if result.retrieved_examples:
                    with st.expander(
                        f"Retrieved reference reports ({len(result.retrieved_examples)})"
                    ):
                        for i, ex in enumerate(result.retrieved_examples, 1):
                            st.markdown(f"**Reference {i}**")
                            st.caption(ex)


# ══════════════════════════════════════════════════════════════════════
# Mode 2 - QA (RAG)
# ══════════════════════════════════════════════════════════════════════
with qa_tab:
    st.header("Question Answering (RAG)")
    st.write(
        "Ask a clinical question. The system retrieves relevant reports and "
        "MedGemma answers grounded in them."
    )

    question = st.text_input(
        "Clinical question", placeholder="e.g. Is there evidence of pleural effusion?"
    )
    qa_image_file = st.file_uploader(
        "Optional X-ray to attach", type=["png", "jpg", "jpeg"], key="qa_img"
    )
    run_qa = st.button("Answer", type="primary", disabled=not question)

    if run_qa:
        try:
            retriever = get_retriever(CONFIG.retrieval.active)
        except FileNotFoundError as exc:
            st.error(f"{exc}")
            st.stop()

        from cxr.modes.qa_rag import QAEngine

        qa_image = Image.open(qa_image_file).convert("RGB") if qa_image_file else None
        with st.spinner("Retrieving context and generating answer..."):
            engine = QAEngine(retriever=retriever, medgemma=get_medgemma())
            result = engine.answer(question, image=qa_image)

        st.subheader("Answer")
        st.success(result.answer)

        with st.expander(f"Retrieved context ({len(result.retrieved)} reports)"):
            for hit in result.retrieved:
                st.markdown(f"**[{hit.rank}]** study `{hit.study_id}` - score {hit.score:.3f}")
                st.caption(hit.text)
