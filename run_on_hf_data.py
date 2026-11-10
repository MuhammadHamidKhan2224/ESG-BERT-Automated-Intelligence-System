"""
Run ESG BERT Multi-Pillar Pipeline on Live Hugging Face Datasets
==============================================================
Streams real sentences directly from:
  - ESGBERT/environment_data
  - ESGBERT/social_data
  - ESGBERT/governance_data
and evaluates them through the ESGBERT model suite.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Ensure package is found
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from esg_engine.classifier import ESGBertClassifier
from esg_engine.dataset_loader import ESGDatasetLoader, HF_ESG_DATASETS


def main():
    print("=" * 80)
    print("🌐 RUNNING ESGBERT PIPELINE DIRECTLY ON HUGGING FACE DATASETS")
    print("=" * 80)

    loader = ESGDatasetLoader(use_streaming=True)
    classifier = ESGBertClassifier(threshold=0.5, lazy_load=False)

    pillars = ["environmental", "social", "governance"]
    samples_per_pillar = 3

    all_results = []
    correct_matches = 0
    total_evaluated = 0

    for pillar in pillars:
        hf_repo = HF_ESG_DATASETS[pillar]
        print(f"\n📥 Streaming live records from Hugging Face: {hf_repo}...")
        streamed_samples = loader.stream_dataset(pillar, limit=samples_per_pillar, offset=0)

        for idx, item in enumerate(streamed_samples, 1):
            sentence = item["sentence"]
            # Run model inference
            analysis = classifier.analyze_text(sentence)

            is_aligned = (
                analysis.dominant_pillar == pillar
                or pillar in analysis.active_pillars
            )
            if is_aligned:
                correct_matches += 1
            total_evaluated += 1

            record = {
                "dataset_source": hf_repo,
                "ground_truth_pillar": pillar.upper(),
                "sentence": sentence,
                "dominant_pillar": analysis.dominant_pillar.upper(),
                "active_pillars": [p.upper() for p in analysis.active_pillars],
                "confidence_score": round(analysis.max_esg_score * 100, 2),
                "environmental_score": round(analysis.environmental.score * 100, 2),
                "social_score": round(analysis.social.score * 100, 2),
                "governance_score": round(analysis.governance.score * 100, 2),
                "sub_topics": analysis.sub_topics,
                "action_orientation": analysis.action_orientation,
                "has_metrics": analysis.has_metrics,
                "aligned": is_aligned,
            }
            all_results.append(record)

            print(f"\n--- [{pillar.upper()} #{idx}] ---")
            print(f"Sentence: \"{sentence[:110]}...\"" if len(sentence) > 110 else f"Sentence: \"{sentence}\"")
            print(f"-> Predicted Pillar : {analysis.dominant_pillar.upper()} (Max Confidence: {analysis.max_esg_score * 100:.1f}%)")
            print(f"   [E]: {analysis.environmental.score*100:.1f}% | [S]: {analysis.social.score*100:.1f}% | [G]: {analysis.governance.score*100:.1f}%")
            print(f"-> Action Status    : {analysis.action_orientation}")
            if analysis.sub_topics:
                print(f"-> Sub-Themes       : {', '.join(analysis.sub_topics)}")
            print(f"-> Aligned with HF  : {'✅ YES' if is_aligned else '⚠️ Divergent / Overlapping'}")

    # Summary
    alignment_rate = (correct_matches / total_evaluated * 100) if total_evaluated > 0 else 0
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY REPORT ON HUGGING FACE DATA")
    print(f"Total Sentences Streamed & Evaluated : {total_evaluated}")
    print(f"Model-Dataset Alignment Rate        : {alignment_rate:.1f}% ({correct_matches}/{total_evaluated})")
    print("=" * 80)

    # Save to Markdown Report
    output_dir = PROJECT_ROOT / "data" / "output_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "huggingface_live_data_audit.md"

    md_lines = [
        "# 🌐 Live Hugging Face ESGBERT Datasets Audit",
        f"**Evaluated on:** 3 Hugging Face Datasets (`ESGBERT/environment_data`, `ESGBERT/social_data`, `ESGBERT/governance_data`)\n",
        f"### 📈 Performance Metrics",
        f"- **Total Samples Evaluated:** {total_evaluated}",
        f"- **Model Alignment with Dataset Labels:** **{alignment_rate:.1f}%**\n",
        "### 📋 Detailed Sentence Breakdown",
        "| HF Dataset | Sample Sentence | Predicted Dominant | E Score | S Score | G Score | Action Type |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in all_results:
        clean_s = r["sentence"].replace("|", "-").replace("\n", " ")
        if len(clean_s) > 90:
            clean_s = clean_s[:87] + "..."
        md_lines.append(
            f"| `{r['dataset_source'].split('/')[-1]}` | {clean_s} | **{r['dominant_pillar']}** | {r['environmental_score']}% | {r['social_score']}% | {r['governance_score']}% | {r['action_orientation'].split(' ')[0]} |"
        )

    md_lines.append("\n---\n*Generated automatically by ESGBERT Automation Engine.*")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Also save JSON output
    json_file = output_dir / "huggingface_live_data_audit.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n💾 Saved Audit Report to: {report_file}")
    print(f"💾 Saved JSON Results to: {json_file}\n")


if __name__ == "__main__":
    main()
