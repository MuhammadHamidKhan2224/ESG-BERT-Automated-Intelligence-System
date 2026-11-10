"""
ESGReportAnalyzer: Comprehensive Corporate Document Auditor.
Parses, segments, and audits long-form corporate disclosures,
sustainability reports, SEC filings, and earning transcripts
using the multi-pillar ESGBERT engine.
"""

import os
import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd

from .classifier import ESGBertClassifier, ESGTextAnalysis


@dataclass
class ReportAuditSummary:
    """High-level scorecard summarizing a corporate document's ESG audit."""
    document_title: str
    analyzed_at: str
    total_sentences: int
    esg_sentence_count: int
    non_esg_sentence_count: int
    esg_intensity_pct: float
    pillar_breakdown: Dict[str, Any]
    action_breakdown: Dict[str, Any]
    sub_topic_counts: Dict[str, int]
    top_environmental_highlights: List[Dict[str, Any]]
    top_social_highlights: List[Dict[str, Any]]
    top_governance_highlights: List[Dict[str, Any]]
    sentences_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_title": self.document_title,
            "analyzed_at": self.analyzed_at,
            "total_sentences": self.total_sentences,
            "esg_sentence_count": self.esg_sentence_count,
            "non_esg_sentence_count": self.non_esg_sentence_count,
            "esg_intensity_pct": round(self.esg_intensity_pct, 2),
            "pillar_breakdown": self.pillar_breakdown,
            "action_breakdown": self.action_breakdown,
            "sub_topic_counts": self.sub_topic_counts,
            "top_environmental_highlights": self.top_environmental_highlights,
            "top_social_highlights": self.top_social_highlights,
            "top_governance_highlights": self.top_governance_highlights,
        }

    def generate_markdown_report(self) -> str:
        """Generates an executive-ready Markdown audit report."""
        pb = self.pillar_breakdown
        ab = self.action_breakdown
        
        md = []
        md.append(f"# 🌿 Automated ESG Audit Report: {self.document_title}")
        md.append(f"**Audit Timestamp:** `{self.analyzed_at}` | **Model Engine:** `ESGBERT Multi-Pillar Suite`\n")
        
        md.append("## 📊 Executive Scorecard")
        md.append(f"- **Total Disclosures Analyzed:** {self.total_sentences} sentences")
        md.append(f"- **ESG Disclosure Intensity:** **{self.esg_intensity_pct:.1f}%** ({self.esg_sentence_count} of {self.total_sentences} sentences)")
        md.append(f"- **Non-ESG / General Operations:** {pb.get('non_esg_share_pct', 0):.1f}% ({self.non_esg_sentence_count} sentences)\n")
        
        md.append("### Pillar Distribution")
        md.append("| Pillar | Count | Share of All Text | Share of ESG Text |")
        md.append("| :--- | :--- | :--- | :--- |")
        md.append(f"| **🌱 Environmental (E)** | {pb.get('environmental_count', 0)} | {pb.get('environmental_share_all_pct', 0):.1f}% | {pb.get('environmental_share_esg_pct', 0):.1f}% |")
        md.append(f"| **🤝 Social (S)** | {pb.get('social_count', 0)} | {pb.get('social_share_all_pct', 0):.1f}% | {pb.get('social_share_esg_pct', 0):.1f}% |")
        md.append(f"| **🏛️ Governance (G)** | {pb.get('governance_count', 0)} | {pb.get('governance_share_all_pct', 0):.1f}% | {pb.get('governance_share_esg_pct', 0):.1f}% |")
        md.append("")

        md.append("### ⚡ Action vs. Boilerplate Breakdown")
        md.append(f"- **Concrete Action & Measured Results:** {ab.get('action_count', 0)} ({ab.get('action_pct', 0):.1f}%)")
        md.append(f"- **Aspirational / Targets & Pledges:** {ab.get('aspirational_count', 0)} ({ab.get('aspirational_pct', 0):.1f}%)")
        md.append(f"- **Boilerplate Disclosures:** {ab.get('disclosure_count', 0)} ({ab.get('disclosure_pct', 0):.1f}%)\n")

        if self.sub_topic_counts:
            md.append("### 🏷️ Top Identified ESG Sub-Themes")
            for topic, count in list(self.sub_topic_counts.items())[:6]:
                md.append(f"- **{topic}**: {count} mention(s)")
            md.append("")

        md.append("## 🔍 Key Pillar Highlights")
        
        md.append("#### 🌱 Top Environmental Disclosures")
        if self.top_environmental_highlights:
            for item in self.top_environmental_highlights:
                md.append(f"> \"{item['text']}\"\n> — *Confidence: {item['score']*100:.1f}% | Action: {item['action']}*")
        else:
            md.append("*No significant environmental statements detected.*")
        md.append("")

        md.append("#### 🤝 Top Social Disclosures")
        if self.top_social_highlights:
            for item in self.top_social_highlights:
                md.append(f"> \"{item['text']}\"\n> — *Confidence: {item['score']*100:.1f}% | Action: {item['action']}*")
        else:
            md.append("*No significant social statements detected.*")
        md.append("")

        md.append("#### 🏛️ Top Governance Disclosures")
        if self.top_governance_highlights:
            for item in self.top_governance_highlights:
                md.append(f"> \"{item['text']}\"\n> — *Confidence: {item['score']*100:.1f}% | Action: {item['action']}*")
        else:
            md.append("*No significant governance statements detected.*")
        md.append("")

        md.append("---\n*Report generated automatically by ESGBERT Automation Engine.*")
        return "\n".join(md)


class ESGReportAnalyzer:
    """
    Automated auditor for corporate sustainability reports and filings.
    """

    def __init__(self, classifier: Optional[ESGBertClassifier] = None):
        self.classifier = classifier or ESGBertClassifier()

    def segment_document(self, text: str) -> List[str]:
        """
        Splits a document into clean, meaningful sentence chunks.
        """
        # Clean lines and paragraphs
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        full_text = " ".join(lines)

        # Split on sentence boundaries (. ! ?) while preserving abbreviations
        raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", full_text)

        cleaned = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Ignore headers, page numbers, or short fragments under 15 characters
            if len(s_clean) >= 15 and any(c.isalpha() for c in s_clean):
                cleaned.append(s_clean)

        return cleaned

    def audit_document(
        self,
        document_text: str,
        document_title: str = "Corporate ESG Disclosure",
        batch_size: int = 16,
    ) -> ReportAuditSummary:
        """
        Executes a complete automated audit of a corporate document.
        """
        sentences = self.segment_document(document_text)
        if not sentences:
            empty_summary = ReportAuditSummary(
                document_title=document_title,
                analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_sentences=0,
                esg_sentence_count=0,
                non_esg_sentence_count=0,
                esg_intensity_pct=0.0,
                pillar_breakdown={},
                action_breakdown={},
                sub_topic_counts={},
                top_environmental_highlights=[],
                top_social_highlights=[],
                top_governance_highlights=[],
                sentences_data=[],
            )
            return empty_summary

        analyses: List[ESGTextAnalysis] = self.classifier.analyze_batch(
            sentences, batch_size=batch_size
        )

        total = len(analyses)
        env_items = []
        soc_items = []
        gov_items = []
        esg_count = 0
        non_esg_count = 0

        action_counts = {
            "Action-Oriented (Concrete Action/Result)": 0,
            "Aspirational (Target / Pledge)": 0,
            "Boilerplate / General Disclosure": 0,
        }
        sub_topics_freq: Dict[str, int] = {}
        sentences_records = []

        for item in analyses:
            sentences_records.append(item.to_dict())
            if item.dominant_pillar != "non-esg":
                esg_count += 1
                action_counts[item.action_orientation] = (
                    action_counts.get(item.action_orientation, 0) + 1
                )
                for topic in item.sub_topics:
                    sub_topics_freq[topic] = sub_topics_freq.get(topic, 0) + 1
            else:
                non_esg_count += 1

            if item.environmental.is_pillar:
                env_items.append((item.environmental.score, item.text, item.action_orientation))
            if item.social.is_pillar:
                soc_items.append((item.social.score, item.text, item.action_orientation))
            if item.governance.is_pillar:
                gov_items.append((item.governance.score, item.text, item.action_orientation))

        env_count = len(env_items)
        soc_count = len(soc_items)
        gov_count = len(gov_items)

        # Sort for top highlights
        env_items.sort(key=lambda x: x[0], reverse=True)
        soc_items.sort(key=lambda x: x[0], reverse=True)
        gov_items.sort(key=lambda x: x[0], reverse=True)

        top_env = [{"score": x[0], "text": x[1], "action": x[2]} for x in env_items[:3]]
        top_soc = [{"score": x[0], "text": x[1], "action": x[2]} for x in soc_items[:3]]
        top_gov = [{"score": x[0], "text": x[1], "action": x[2]} for x in gov_items[:3]]

        sorted_sub_topics = dict(
            sorted(sub_topics_freq.items(), key=lambda item: item[1], reverse=True)
        )

        esg_intensity = (esg_count / total * 100.0) if total > 0 else 0.0
        safe_esg_count = max(esg_count, 1)

        pillar_breakdown = {
            "environmental_count": env_count,
            "environmental_share_all_pct": round(env_count / total * 100.0, 2),
            "environmental_share_esg_pct": round(env_count / safe_esg_count * 100.0, 2),
            "social_count": soc_count,
            "social_share_all_pct": round(soc_count / total * 100.0, 2),
            "social_share_esg_pct": round(soc_count / safe_esg_count * 100.0, 2),
            "governance_count": gov_count,
            "governance_share_all_pct": round(gov_count / total * 100.0, 2),
            "governance_share_esg_pct": round(gov_count / safe_esg_count * 100.0, 2),
            "non_esg_count": non_esg_count,
            "non_esg_share_pct": round(non_esg_count / total * 100.0, 2),
        }

        action_breakdown = {
            "action_count": action_counts.get("Action-Oriented (Concrete Action/Result)", 0),
            "action_pct": round(action_counts.get("Action-Oriented (Concrete Action/Result)", 0) / safe_esg_count * 100.0, 2),
            "aspirational_count": action_counts.get("Aspirational (Target / Pledge)", 0),
            "aspirational_pct": round(action_counts.get("Aspirational (Target / Pledge)", 0) / safe_esg_count * 100.0, 2),
            "disclosure_count": action_counts.get("Boilerplate / General Disclosure", 0),
            "disclosure_pct": round(action_counts.get("Boilerplate / General Disclosure", 0) / safe_esg_count * 100.0, 2),
        }

        summary = ReportAuditSummary(
            document_title=document_title,
            analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_sentences=total,
            esg_sentence_count=esg_count,
            non_esg_sentence_count=non_esg_count,
            esg_intensity_pct=round(esg_intensity, 2),
            pillar_breakdown=pillar_breakdown,
            action_breakdown=action_breakdown,
            sub_topic_counts=sorted_sub_topics,
            top_environmental_highlights=top_env,
            top_social_highlights=top_soc,
            top_governance_highlights=top_gov,
            sentences_data=sentences_records,
        )

        return summary
