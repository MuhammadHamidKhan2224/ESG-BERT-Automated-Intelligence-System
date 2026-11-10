"""
ESGDatasetLoader: Hugging Face Dataset Connector & Streamer.
Interfaces with:
  - ESGBERT/environment_data
  - ESGBERT/governance_data
  - ESGBERT/social_data
"""

import json
import os
from typing import Dict, List, Optional, Generator, Any
import pandas as pd
from datasets import load_dataset


HF_ESG_DATASETS = {
    "environmental": "ESGBERT/environment_data",
    "governance": "ESGBERT/governance_data",
    "social": "ESGBERT/social_data",
}

# Fallback offline samples in case user is offline or hits network rate limit
OFFLINE_FALLBACK_DATA = {
    "environmental": [
        "We achieved a 32% reduction in Scope 1 and Scope 2 greenhouse gas emissions across all manufacturing plants.",
        "Over the past fiscal year, 85% of our global electricity consumption was sourced from certified renewable solar and wind contracts.",
        "Our packaging redesign eliminated 14,000 metric tons of virgin plastic from the distribution network.",
        "The company expanded its zero-waste-to-landfill certification to cover 92% of our primary distribution facilities.",
        "We completed biodiversity impact assessments across 100% of our active mining and extraction concessions.",
    ],
    "social": [
        "We increased the representation of women in senior executive leadership positions to 41% globally.",
        "The company logged a Lost Time Injury Frequency Rate (LTIFR) of 0.28, representing a 20% improvement year-over-year.",
        "All tier-one suppliers were required to undergo third-party audits for forced labor and worker safety compliance.",
        "We invested $15 million in employee upskilling and continuous learning programs during 2024.",
        "The foundation contributed $8.5 million to support clean water and education initiatives in neighboring local communities.",
    ],
    "governance": [
        "The Board of Directors separated the roles of Chief Executive Officer and Independent Board Chair.",
        "Executive annual incentive compensation is tied to measurable ESG performance indicators including carbon reduction targets.",
        "Our confidential whistleblower hotline received 142 inquiries, all of which were reviewed by the independent audit committee.",
        "We updated our enterprise risk management framework to include quarterly cybersecurity and artificial intelligence risk evaluations.",
        "100% of salaried employees completed mandatory annual anti-bribery and compliance training.",
    ],
}


class ESGDatasetLoader:
    """
    Automated loader and streaming processor for ESGBERT Hugging Face datasets.
    """

    def __init__(self, use_streaming: bool = True):
        self.use_streaming = use_streaming

    def stream_dataset(
        self, pillar: str, limit: int = 25, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Streams sentences from the specified Hugging Face dataset.
        
        Args:
            pillar: 'environmental', 'governance', or 'social'
            limit: Number of records to return
            offset: Number of initial records to skip
        """
        pillar_key = pillar.lower().strip()
        dataset_name = HF_ESG_DATASETS.get(pillar_key)

        if not dataset_name:
            raise ValueError(
                f"Unknown pillar '{pillar}'. Must be one of: {list(HF_ESG_DATASETS.keys())}"
            )

        results = []
        try:
            ds = load_dataset(dataset_name, split="train", streaming=self.use_streaming)
            current_idx = 0
            collected = 0

            for row in ds:
                if current_idx < offset:
                    current_idx += 1
                    continue

                sentence = row.get("sentence") or row.get("text") or str(row)
                sentence = str(sentence).strip()
                if len(sentence) > 10:  # Skip trivial fragments
                    results.append({
                        "index": current_idx,
                        "pillar": pillar_key,
                        "dataset": dataset_name,
                        "sentence": sentence,
                        "char_count": len(sentence),
                        "word_count": len(sentence.split()),
                    })
                    collected += 1

                current_idx += 1
                if collected >= limit:
                    break

        except Exception as e:
            # Fallback to curated offline sample if network fails
            print(f"[Warning] Streaming from {dataset_name} encountered an issue: {e}. Using offline cache.")
            fallback_items = OFFLINE_FALLBACK_DATA.get(pillar_key, [])
            for idx, text in enumerate(fallback_items[:limit]):
                results.append({
                    "index": idx,
                    "pillar": pillar_key,
                    "dataset": f"{dataset_name} (Local Cached Sample)",
                    "sentence": text,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                })

        return results

    def search_stream(
        self,
        pillar: str,
        query: str,
        max_matches: int = 10,
        max_scan: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Scans streaming records looking for keyword matches.
        """
        pillar_key = pillar.lower().strip()
        dataset_name = HF_ESG_DATASETS.get(pillar_key)
        q_lower = query.lower().strip()
        matches = []

        try:
            ds = load_dataset(dataset_name, split="train", streaming=True)
            scanned = 0

            for row in ds:
                scanned += 1
                sentence = str(row.get("sentence") or row.get("text") or "")
                if q_lower in sentence.lower():
                    matches.append({
                        "pillar": pillar_key,
                        "sentence": sentence.strip(),
                        "scanned_at_index": scanned,
                        "matched_query": query,
                    })
                    if len(matches) >= max_matches:
                        break

                if scanned >= max_scan:
                    break

        except Exception as e:
            fallback = [
                s for s in OFFLINE_FALLBACK_DATA.get(pillar_key, []) if q_lower in s.lower()
            ]
            for s in fallback[:max_matches]:
                matches.append({
                    "pillar": pillar_key,
                    "sentence": s,
                    "scanned_at_index": 0,
                    "matched_query": query,
                })

        return matches

    def create_balanced_benchmark(
        self, samples_per_pillar: int = 20, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Creates a balanced multi-pillar benchmark DataFrame combining E, S, and G sentences.
        """
        all_rows = []
        for pillar in ["environmental", "social", "governance"]:
            samples = self.stream_dataset(pillar, limit=samples_per_pillar)
            for s in samples:
                all_rows.append({
                    "pillar": pillar,
                    "sentence": s["sentence"],
                    "source": s["dataset"],
                    "word_count": s["word_count"],
                })

        df = pd.DataFrame(all_rows)
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            if output_path.endswith(".csv"):
                df.to_csv(output_path, index=False)
            elif output_path.endswith(".json"):
                df.to_json(output_path, orient="records", indent=2)

        return df
