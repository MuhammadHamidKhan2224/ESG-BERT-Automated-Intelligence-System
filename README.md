# 🌿 ESG BERT Automated Intelligence System

An end-to-end, automated Natural Language Processing pipeline and interactive auditing system for **Environmental, Social, and Governance (ESG)** corporate disclosures.

Powered by the research and models of the **ESGBERT** initiative and linked directly to the three official Hugging Face datasets:
* 🌱 [ESGBERT/environment_data](https://huggingface.co/datasets/ESGBERT/environment_data)
* 🤝 [ESGBERT/social_data](https://huggingface.co/datasets/ESGBERT/social_data)
* 🏛️ [ESGBERT/governance_data](https://huggingface.co/datasets/ESGBERT/governance_data)

---

## 🚀 Key Features

1. **Multi-Pillar ESGBERT Classifier**
   * Orchestrates the three official classifiers:
     * `ESGBERT/EnvironmentalBERT-environmental`
     * `ESGBERT/SocialBERT-social`
     * `ESGBERT/GovernanceBERT-governance`
   * Computes simultaneous multi-pillar probabilities, active pillar flags, and detects multi-pillar overlap vs. non-ESG operational statements.

2. **Sub-Theme & Taxonomy Tagging**
   * Automatically categorizes disclosures into granular sub-themes:
     * **Environmental**: Climate & GHG Emissions, Clean Energy & Efficiency, Biodiversity & Nature, Water & Waste Management.
     * **Social**: Diversity, Equity & Inclusion (DEI), Health & Safety, Labor Rights & Human Capital, Supply Chain & Human Rights, Community Impact.
     * **Governance**: Board Structure & Independence, Executive Compensation, Ethics & Anti-Corruption, Shareholder Rights, Risk & Cybersecurity.

3. **Action vs. Boilerplate Disclosures Detector**
   * Flags whether a corporate statement demonstrates **concrete, measured action** (e.g. past tense verbs, quantified percentages, metric tons, dollar investments) vs. **aspirational pledges** ("aim to", "strive to") vs. **boilerplate disclosures**.

4. **Automated Corporate Document Auditor**
   * Ingests long-form sustainability reports, SEC 10-K/10-Q filings, earnings call transcripts, or press releases.
   * Chunks and scores every sentence, compiling an **Executive ESG Scorecard**, pillar distribution ratios, and automated Markdown/JSON audit reports.

5. **Hugging Face Live Dataset Streaming**
   * Live streaming connector to Hugging Face Hub for `ESGBERT/environment_data`, `social_data`, and `governance_data`.
   * Stream arbitrary sample sizes without having to download multi-gigabyte files locally.
   * Search and filter sentences across the Hugging Face repositories in real time.

6. **Interactive Web Dashboard (`app.py`)**
   * Built with Streamlit and Plotly for executive presentation:
     * **Tab 1: Live Sentence Classifier**: Real-time gauge meters and pillar verdicts.
     * **Tab 2: Report Auditor**: Paste or upload reports, get instant Donut and Bar charts, top highlights, and exportable audit summaries.
     * **Tab 3: Hugging Face Explorer**: Stream live rows from Hugging Face and run on-the-fly model inference.
     * **Tab 4: Batch Watcher**: Monitor input queues and batch process document folders.

7. **Automated Batch Processing & Directory Watcher**
   * Drop files into `data/incoming/` and let the background watcher audit each document and output reports to `data/output_reports/`.

---

## 📁 Project Structure

```
esg-bert-automation/
├── app.py                     # Streamlit Interactive Web Dashboard
├── requirements.txt           # Project dependencies
├── run_dashboard.bat          # One-click Windows launcher for Web App
├── run_cli.bat                # Windows launcher for CLI commands
├── run_tests.bat              # Windows test suite runner
├── esg_engine/
│   ├── __init__.py            # Package initialization
│   ├── classifier.py          # Multi-Pillar ESGBERT Classifier & Sub-theme tagging
│   ├── dataset_loader.py      # Hugging Face Dataset Streamer & Benchmark creator
│   ├── report_analyzer.py     # Document segmentation, scoring & audit report generator
│   ├── watcher.py             # Automated folder monitoring & batch pipeline
│   └── cli.py                 # Command Line Interface
├── data/
│   ├── sample_reports/        # Sample corporate sustainability disclosures
│   ├── incoming/              # Incoming directory monitored by the watcher
│   └── output_reports/        # Generated Markdown & JSON audit scorecards
└── tests/
    └── test_esg_pipeline.py   # Unit and integration test suite
```

---

## ⚡ Quickstart

### 1. Launch the Interactive Web Dashboard
Double-click `run_dashboard.bat` or run:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Single Text Classification via CLI
```bash
python -m esg_engine.cli analyze "We achieved a 28% reduction in Scope 1 greenhouse gas emissions."
```

### 3. Audit a Corporate Sustainability Document
```bash
python -m esg_engine.cli audit data/sample_reports/nexus_energy_sustainability_2024.txt -o data/output_reports/nexus_audit.md
```

### 4. Stream Live Samples from Hugging Face Datasets
```bash
# Stream 5 sentences from ESGBERT/environment_data
python -m esg_engine.cli stream-hf environmental --limit 5

# Stream 5 sentences from ESGBERT/governance_data matching "audit"
python -m esg_engine.cli stream-hf governance --limit 5 --search audit
```

### 5. Start the Automated Directory Watcher
```bash
python -m esg_engine.cli watch --incoming data/incoming --output data/output_reports
```

### 6. Run the Test Suite
```bash
python tests/test_esg_pipeline.py
```

---

## 🐍 Python SDK API Usage

```python
from esg_engine import ESGBertClassifier, ESGReportAnalyzer, ESGDatasetLoader

# 1. Classify an individual statement
classifier = ESGBertClassifier(threshold=0.5)
result = classifier.analyze_text("The board implemented an independent audit oversight charter.")
print("Dominant Pillar:", result.dominant_pillar)
print("Governance Score:", result.governance.score)
print("Sub-topics:", result.sub_topics)

# 2. Audit a complete corporate disclosure document
analyzer = ESGReportAnalyzer(classifier=classifier)
doc_text = open("data/sample_reports/nexus_energy_sustainability_2024.txt").read()
summary = analyzer.audit_document(doc_text, document_title="Nexus Energy 2024")
print("ESG Disclosure Intensity:", summary.esg_intensity_pct, "%")
print("Pillar Breakdown:", summary.pillar_breakdown)

# 3. Stream live Hugging Face dataset rows
loader = ESGDatasetLoader()
stream_rows = loader.stream_dataset("environmental", limit=10)
for row in stream_rows:
    print(row["sentence"])
```

---

## 📚 References & Academic Citation

If you use this system or the underlying ESGBERT models in academic or commercial research, please cite the foundational research paper:

```bibtex
@article{schimanski2024bridging,
  title={Bridging the gap in ESG measurement: Using NLP to quantify environmental, social, and governance communication},
  author={Schimanski, Tobias and Reding, Andrin and Reding, Nicolas and Bingler, Julia and Kraus, Mathias and Leippold, Markus},
  journal={Finance Research Letters},
  volume={61},
  pages={104979},
  year={2024},
  publisher={Elsevier}
}
```
