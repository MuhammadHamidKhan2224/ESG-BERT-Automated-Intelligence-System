"""
ESG BERT Automated Intelligence Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An automated, multi-pillar Natural Language Processing system for
Environmental, Social, and Governance (ESG) corporate disclosures,
benchmarked on Hugging Face's ESGBERT datasets.
"""

from .classifier import ESGBertClassifier, ESGPillarScore, ESGTextAnalysis
from .dataset_loader import ESGDatasetLoader, HF_ESG_DATASETS
from .report_analyzer import ESGReportAnalyzer, ReportAuditSummary

__all__ = [
    "ESGBertClassifier",
    "ESGPillarScore",
    "ESGTextAnalysis",
    "ESGDatasetLoader",
    "HF_ESG_DATASETS",
    "ESGReportAnalyzer",
    "ReportAuditSummary",
]
