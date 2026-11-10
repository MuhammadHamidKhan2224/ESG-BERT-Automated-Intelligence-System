"""
Unit and Integration Tests for ESG BERT Automation Engine.
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from esg_engine.classifier import ESGBertClassifier, ESGTextAnalysis
from esg_engine.dataset_loader import ESGDatasetLoader
from esg_engine.report_analyzer import ESGReportAnalyzer
from esg_engine.watcher import ESGWatcher


def test_classifier_environmental():
    classifier = ESGBertClassifier(threshold=0.5)
    text = "We reduced greenhouse gas emissions by 40% and expanded solar photovoltaic installations."
    res = classifier.analyze_text(text)
    assert isinstance(res, ESGTextAnalysis)
    assert res.environmental.score > 0.70
    assert res.dominant_pillar == "environmental"
    assert "environmental" in res.active_pillars
    print("✓ test_classifier_environmental passed.")


def test_classifier_governance():
    classifier = ESGBertClassifier(threshold=0.5)
    text = "An ethical code has been issued to all Group employees."
    res = classifier.analyze_text(text)
    assert isinstance(res, ESGTextAnalysis)
    assert res.governance.score > 0.80
    assert res.dominant_pillar == "governance"
    print("✓ test_classifier_governance passed.")


def test_classifier_non_esg():
    classifier = ESGBertClassifier(threshold=0.5)
    text = "Quarterly revenue increased by 3% due to high seasonal holiday sales volume."
    res = classifier.analyze_text(text)
    assert res.dominant_pillar == "non-esg"
    print("✓ test_classifier_non_esg passed.")


def test_dataset_loader():
    loader = ESGDatasetLoader()
    samples = loader.stream_dataset("environmental", limit=3)
    assert len(samples) > 0
    assert "sentence" in samples[0]
    print(f"✓ test_dataset_loader passed (retrieved {len(samples)} samples).")


def test_report_analyzer():
    analyzer = ESGReportAnalyzer()
    doc = """
    We achieved a 35% cut in greenhouse gas emissions across all factories.
    An ethical code has been issued to all Group employees.
    We increased female representation in engineering leadership roles to 40%.
    Operating revenue grew by 2% in the second quarter.
    """
    summary = analyzer.audit_document(doc, document_title="Test Doc")
    assert summary.total_sentences >= 3
    assert summary.esg_sentence_count >= 2
    assert summary.esg_intensity_pct > 50.0
    md = summary.generate_markdown_report()
    assert "# 🌿 Automated ESG Audit Report" in md
    print("✓ test_report_analyzer passed.")


def test_watcher_batch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_dir = Path(tmp_dir) / "incoming"
        out_dir = Path(tmp_dir) / "reports"
        in_dir.mkdir()
        out_dir.mkdir()

        test_file = in_dir / "sample_test.txt"
        test_file.write_text("We achieved net zero emissions across our manufacturing facilities.")

        watcher = ESGWatcher(incoming_dir=str(in_dir), output_dir=str(out_dir))
        reports = watcher.run_batch()

        assert len(reports) == 1
        assert (out_dir / "sample_test_esg_audit.md").exists()
        assert (out_dir / "sample_test_esg_audit.json").exists()
        print("✓ test_watcher_batch passed.")


if __name__ == "__main__":
    print("Running ESG Pipeline Tests...")
    test_classifier_environmental()
    test_classifier_governance()
    test_classifier_non_esg()
    test_dataset_loader()
    test_report_analyzer()
    test_watcher_batch()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
