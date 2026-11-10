"""
ESG BERT Automated Intelligence Dashboard
Interactive Web Interface for Environmental, Social, and Governance
Text Classification, Report Auditing, and Hugging Face Dataset Streaming.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path so esg_engine can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from esg_engine.classifier import ESGBertClassifier, PILLAR_MODELS
from esg_engine.dataset_loader import ESGDatasetLoader, HF_ESG_DATASETS, OFFLINE_FALLBACK_DATA
from esg_engine.report_analyzer import ESGReportAnalyzer
from esg_engine.watcher import ESGWatcher

# Page Configuration
st.set_page_config(
    page_title="ESG BERT Automation Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-env {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-soc {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-gov {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-non {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ESGBERT model suite into memory...")
def get_classifier(threshold: float = 0.5):
    return ESGBertClassifier(threshold=threshold, lazy_load=False)


@st.cache_resource
def get_loader():
    return ESGDatasetLoader(use_streaming=True)


# Initialize components
classifier = get_classifier()
report_analyzer = ESGReportAnalyzer(classifier=classifier)
dataset_loader = get_loader()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.title("ESG BERT Studio")
    st.markdown("**Automated Multi-Pillar ESG AI**")
    st.markdown("---")

    st.subheader("⚙️ Classification Engine")
    threshold = st.slider("Pillar Detection Threshold", 0.1, 0.9, 0.5, 0.05)
    classifier.threshold = threshold

    st.markdown("### 🤖 Loaded Models")
    st.markdown(
        f"""
        - 🌱 **E**: `{PILLAR_MODELS['environmental']}`
        - 🤝 **S**: `{PILLAR_MODELS['social']}`
        - 🏛️ **G**: `{PILLAR_MODELS['governance']}`
        """
    )

    st.markdown("### 🌐 Hugging Face Datasets")
    st.markdown(
        f"""
        - [ESGBERT/environment_data](https://huggingface.co/datasets/{HF_ESG_DATASETS['environmental']})
        - [ESGBERT/social_data](https://huggingface.co/datasets/{HF_ESG_DATASETS['social']})
        - [ESGBERT/governance_data](https://huggingface.co/datasets/{HF_ESG_DATASETS['governance']})
        """
    )
    st.markdown("---")
    st.caption("Engine: PyTorch + Transformers + Hugging Face Datasets")

# ----------------- MAIN TITLE -----------------
st.markdown('<div class="main-header">🌿 ESG BERT Automated Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Multi-Pillar ESG Classification, Corporate Report Auditing, and Live Dataset Streaming</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Live Text Classifier",
    "📑 Corporate Report Auditor",
    "🌐 HF Dataset Explorer",
    "📁 Automated Batch Watcher",
])

# =========================================================================
# TAB 1: LIVE TEXT CLASSIFIER
# =========================================================================
with tab1:
    st.subheader("Instant Multi-Pillar ESG Sentence Analysis")
    st.markdown("Enter any corporate statement, ESG pledge, or disclosure sentence to evaluate.")

    sample_options = [
        "Custom Input",
        "🌱 Environmental: We reduced Scope 1 and Scope 2 GHG emissions by 32% through rooftop solar installations.",
        "🤝 Social: We achieved 40% female representation in leadership and implemented zero-accident safety protocols.",
        "🏛️ Governance: The independent audit committee reviews executive compensation clawbacks and compliance reports quarterly.",
        "🔄 Multi-Pillar: We invested $50M in worker safety training and clean renewable microgrids across all facilities.",
        "🏢 Non-ESG: Our operating revenue rose 4.5% driven by improved customer demand and seasonal pricing.",
    ]

    selected_sample = st.selectbox("Or choose a pre-configured sample sentence:", sample_options)

    if selected_sample == "Custom Input":
        default_text = "We achieved a 28% reduction in greenhouse gas emissions and transitioned our vehicle fleet to 100% electric."
    else:
        default_text = selected_sample.split(": ", 1)[-1]

    input_text = st.text_area("Input Statement / Disclosure:", value=default_text, height=110)

    if st.button("🚀 Analyze with ESGBERT", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please provide a text sentence to analyze.")
        else:
            with st.spinner("Classifying across Environmental, Social, and Governance models..."):
                analysis = classifier.analyze_text(input_text)

            st.markdown("---")
            # Result Overview
            col_res1, col_res2, col_res3 = st.columns([1.5, 1.2, 1.3])

            with col_res1:
                st.markdown("#### Primary Verdict")
                dom = analysis.dominant_pillar
                if dom == "environmental":
                    st.markdown('<span class="badge-env">🌱 DOMINANT: ENVIRONMENTAL</span>', unsafe_allow_html=True)
                elif dom == "social":
                    st.markdown('<span class="badge-soc">🤝 DOMINANT: SOCIAL</span>', unsafe_allow_html=True)
                elif dom == "governance":
                    st.markdown('<span class="badge-gov">🏛️ DOMINANT: GOVERNANCE</span>', unsafe_allow_html=True)
                elif dom == "multi-pillar":
                    st.markdown('<span class="badge-soc">🔄 MULTI-PILLAR OVERLAP</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-non">🏢 NON-ESG / GENERAL OPERATIONS</span>', unsafe_allow_html=True)

                st.markdown(f"**Max ESG Confidence:** `{analysis.max_esg_score * 100:.1f}%`")
                st.markdown(f"**Active Pillars:** {', '.join(analysis.active_pillars).title() if analysis.active_pillars else 'None'}")

            with col_res2:
                st.markdown("#### Action Orientation")
                if "Action" in analysis.action_orientation:
                    st.success(f"⚡ {analysis.action_orientation}")
                elif "Aspirational" in analysis.action_orientation:
                    st.info(f"🎯 {analysis.action_orientation}")
                else:
                    st.markdown(f"📄 {analysis.action_orientation}")

                st.caption(f"Quantified Metrics: {'✅ Present' if analysis.has_metrics else '❌ Not detected'}")

            with col_res3:
                st.markdown("#### Identified Sub-Themes")
                if analysis.sub_topics:
                    for tag in analysis.sub_topics:
                        st.markdown(f"- `{tag}`")
                else:
                    st.caption("No specific sub-theme keyword matches.")

            # Gauge / Confidence Bar Charts
            st.markdown("#### 📊 Multi-Pillar Probability Distribution")
            col_e, col_s, col_g = st.columns(3)

            with col_e:
                e_score = analysis.environmental.score
                fig_e = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(e_score * 100, 1),
                    title={'text': "🌱 Environmental", 'font': {'size': 18, 'color': "#166534"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#22c55e"},
                        'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': threshold * 100}
                    }
                ))
                fig_e.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_e, use_container_width=True)
                st.caption(f"Status: **{'Active' if analysis.environmental.is_pillar else 'Inactive'}** ({analysis.environmental.confidence_label})")

            with col_s:
                s_score = analysis.social.score
                fig_s = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(s_score * 100, 1),
                    title={'text': "🤝 Social", 'font': {'size': 18, 'color': "#1e40af"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3b82f6"},
                        'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': threshold * 100}
                    }
                ))
                fig_s.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_s, use_container_width=True)
                st.caption(f"Status: **{'Active' if analysis.social.is_pillar else 'Inactive'}** ({analysis.social.confidence_label})")

            with col_g:
                g_score = analysis.governance.score
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(g_score * 100, 1),
                    title={'text': "🏛️ Governance", 'font': {'size': 18, 'color': "#92400e"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#f59e0b"},
                        'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': threshold * 100}
                    }
                ))
                fig_g.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption(f"Status: **{'Active' if analysis.governance.is_pillar else 'Inactive'}** ({analysis.governance.confidence_label})")


# =========================================================================
# TAB 2: CORPORATE REPORT AUDITOR
# =========================================================================
with tab2:
    st.subheader("Auditor for Corporate ESG Disclosures & Filings")
    st.markdown("Audit long-form corporate sustainability reports, SEC 10-K sections, or sustainability reports.")

    sample_report_path = Path("data/sample_reports/nexus_energy_sustainability_2024.txt")
    loaded_default_text = ""
    if sample_report_path.exists():
        with open(sample_report_path, "r", encoding="utf-8") as f:
            loaded_default_text = f.read()

    doc_col1, doc_col2 = st.columns([3, 1])
    with doc_col1:
        report_title = st.text_input("Document / Company Title:", value="Nexus Global Energy Corp - 2024 Report")
    with doc_col2:
        st.write("")
        st.write("")
        load_sample = st.button("📥 Load Built-in Sample Report", use_container_width=True)

    uploaded_file = st.file_uploader("Upload Report File (.txt, .md):", type=["txt", "md"])

    if uploaded_file is not None:
        report_text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif load_sample or not loaded_default_text:
        report_text = loaded_default_text
    else:
        report_text = loaded_default_text

    doc_input = st.text_area("Report Content:", value=report_text, height=220)

    if st.button("⚡ Run Full Automated ESG Audit", type="primary", use_container_width=True):
        if not doc_input.strip():
            st.error("Please provide report text to audit.")
        else:
            with st.spinner("Segmenting text into sentences and running multi-pillar ESGBERT inference..."):
                audit_summary = report_analyzer.audit_document(doc_input, document_title=report_title)

            # Executive Metrics
            st.markdown("### 📈 Executive Scorecard")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Sentences Analyzed", audit_summary.total_sentences)
            m2.metric("ESG Disclosure Intensity", f"{audit_summary.esg_intensity_pct:.1f}%")
            m3.metric("ESG Sentences Count", audit_summary.esg_sentence_count)
            m4.metric("Action Orientation Rate", f"{audit_summary.action_breakdown.get('action_pct', 0):.1f}%")

            st.markdown("---")

            # Charts Row
            c_chart1, c_chart2 = st.columns(2)

            with c_chart1:
                st.markdown("#### 🎯 ESG Pillar Breakdown")
                pb = audit_summary.pillar_breakdown
                labels = ["🌱 Environmental", "🤝 Social", "🏛️ Governance", "🏢 Non-ESG"]
                values = [
                    pb.get("environmental_count", 0),
                    pb.get("social_count", 0),
                    pb.get("governance_count", 0),
                    pb.get("non_esg_count", 0),
                ]
                colors = ["#22c55e", "#3b82f6", "#f59e0b", "#94a3b8"]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.45,
                    marker=dict(colors=colors),
                )])
                fig_donut.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_donut, use_container_width=True)

            with c_chart2:
                st.markdown("#### ⚡ Disclosure Character (Action vs Boilerplate)")
                ab = audit_summary.action_breakdown
                act_df = pd.DataFrame([
                    {"Category": "Action-Oriented", "Count": ab.get("action_count", 0)},
                    {"Category": "Aspirational", "Count": ab.get("aspirational_count", 0)},
                    {"Category": "Boilerplate / Disclosure", "Count": ab.get("disclosure_count", 0)},
                ])
                fig_bar = px.bar(
                    act_df,
                    x="Category",
                    y="Count",
                    color="Category",
                    color_discrete_map={
                        "Action-Oriented": "#10b981",
                        "Aspirational": "#6366f1",
                        "Boilerplate / Disclosure": "#94a3b8",
                    },
                )
                fig_bar.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)

            # Sub topics chart
            if audit_summary.sub_topic_counts:
                st.markdown("#### 🏷️ Detected Sub-Themes Frequency")
                sub_df = pd.DataFrame([
                    {"Sub-Theme": k, "Count": v}
                    for k, v in list(audit_summary.sub_topic_counts.items())[:8]
                ])
                fig_sub = px.bar(sub_df, x="Count", y="Sub-Theme", orientation="h", color="Count", color_continuous_scale="Viridis")
                fig_sub.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_sub, use_container_width=True)

            # Highlights
            st.markdown("### 🌟 Top Pillar Highlights")
            h_col1, h_col2, h_col3 = st.columns(3)

            with h_col1:
                st.markdown("#### 🌱 Environmental Top Claims")
                for h in audit_summary.top_environmental_highlights:
                    st.info(f"**{h['score']*100:.1f}%** | *{h['action']}*\n\n\"{h['text']}\"")

            with h_col2:
                st.markdown("#### 🤝 Social Top Claims")
                for h in audit_summary.top_social_highlights:
                    st.info(f"**{h['score']*100:.1f}%** | *{h['action']}*\n\n\"{h['text']}\"")

            with h_col3:
                st.markdown("#### 🏛️ Governance Top Claims")
                for h in audit_summary.top_governance_highlights:
                    st.info(f"**{h['score']*100:.1f}%** | *{h['action']}*\n\n\"{h['text']}\"")

            # Downloadable Markdown Audit
            st.markdown("### 📥 Export Audit Report")
            md_content = audit_summary.generate_markdown_report()
            st.download_button(
                label="📄 Download Markdown Executive Report",
                data=md_content,
                file_name=f"{report_title.replace(' ', '_').lower()}_audit.md",
                mime="text/markdown",
            )


# =========================================================================
# TAB 3: HUGGING FACE DATASET EXPLORER
# =========================================================================
with tab3:
    st.subheader("Explore Hugging Face ESGBERT Datasets")
    st.markdown(
        "Stream live sentence rows directly from the official **ESGBERT** datasets on the Hugging Face Hub."
    )

    hf_col1, hf_col2, hf_col3 = st.columns([1.5, 1, 1.5])
    with hf_col1:
        pillar_choice = st.selectbox(
            "Select Hugging Face Dataset:",
            ["environmental", "social", "governance"],
            format_func=lambda x: f"ESGBERT/{x}_data",
        )
    with hf_col2:
        stream_limit = st.slider("Sentences to Stream:", min_value=5, max_value=40, value=15, step=5)
    with hf_col3:
        search_filter = st.text_input("Optional Keyword Filter (e.g., carbon, board, safety):")

    if st.button("🌐 Stream from Hugging Face", type="primary"):
        with st.spinner(f"Streaming live from ESGBERT/{pillar_choice}_data..."):
            if search_filter.strip():
                rows = dataset_loader.search_stream(pillar_choice, search_filter.strip(), max_matches=stream_limit)
                df_stream = pd.DataFrame(rows)
            else:
                rows = dataset_loader.stream_dataset(pillar_choice, limit=stream_limit)
                df_stream = pd.DataFrame(rows)

        if df_stream.empty:
            st.warning("No records matched the filter query.")
        else:
            st.success(f"Streamed {len(df_stream)} records from Hugging Face!")
            st.dataframe(
                df_stream[["pillar", "sentence", "char_count", "word_count"] if "word_count" in df_stream.columns else ["pillar", "sentence"]],
                use_container_width=True,
                height=350,
            )

            # Interactive inspector: choose one sentence from the stream to test immediately
            st.markdown("#### 🔬 Inspect Any Streamed Sentence with ESGBERT:")
            selected_streamed_sent = st.selectbox("Select a sentence to classify:", df_stream["sentence"].tolist())
            if st.button("Classify Selected Sentence"):
                with st.spinner("Running ESGBERT classifier..."):
                    quick_res = classifier.analyze_text(selected_streamed_sent)
                st.json(quick_res.to_dict())


# =========================================================================
# TAB 4: AUTOMATED BATCH WATCHER
# =========================================================================
with tab4:
    st.subheader("Automated Incoming Directory Watcher")
    st.markdown(
        """
        Drop any corporate disclosure files (`.txt`, `.md`, `.csv`) into `data/incoming/`.
        The automated watcher pipeline will scan and audit each file, saving reports into `data/output_reports/`.
        """
    )

    watcher = ESGWatcher(
        incoming_dir="data/incoming",
        output_dir="data/output_reports",
        analyzer=report_analyzer,
    )

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown(f"**Incoming Directory:** `{watcher.incoming_dir.resolve()}`")
        incoming_files = list(watcher.incoming_dir.glob("*.*"))
        st.caption(f"Files currently in queue: {len(incoming_files)}")
        for f in incoming_files:
            st.markdown(f"- 📄 `{f.name}` ({f.stat().st_size} bytes)")

    with col_w2:
        st.markdown(f"**Output Reports Directory:** `{watcher.output_dir.resolve()}`")
        output_files = list(watcher.output_dir.glob("*_esg_audit.md"))
        st.caption(f"Generated Audit Reports: {len(output_files)}")
        for f in output_files:
            st.markdown(f"- ✅ `{f.name}`")

    st.markdown("---")
    if st.button("▶️ Run Batch Process Now", type="primary"):
        with st.spinner("Processing incoming disclosures..."):
            new_reports = watcher.run_batch()
        if new_reports:
            st.success(f"Successfully generated {len(new_reports)} new audit reports!")
            for rep in new_reports:
                st.markdown(f"- Created: `{rep.name}`")
        else:
            st.info("No new unprocessed files found in `data/incoming/`.")
