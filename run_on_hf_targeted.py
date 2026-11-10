"""
Targeted Hugging Face Dataset Query and Live Inference
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from esg_engine.dataset_loader import ESGDatasetLoader
from esg_engine.classifier import ESGBertClassifier


def run_targeted():
    loader = ESGDatasetLoader()
    classifier = ESGBertClassifier()

    queries = [
        ("environmental", "emissions"),
        ("social", "safety"),
        ("governance", "board"),
    ]

    print("=" * 80)
    print("🎯 TARGETED HUGGING FACE DATA STREAMING & LIVE ESGBERT INFERENCE")
    print("=" * 80)

    for pillar, kw in queries:
        print(f"\n🔍 Searching ESGBERT/{pillar}_data for keyword: \"{kw}\"...")
        matches = loader.search_stream(pillar, kw, max_matches=2, max_scan=500)
        for i, m in enumerate(matches, 1):
            s = m["sentence"]
            res = classifier.analyze_text(s)
            print(f"\n[{pillar.upper()} MATCH #{i}]")
            print(f"Sentence: \"{s[:110]}...\"" if len(s) > 110 else f"Sentence: \"{s}\"")
            print(f"-> Predicted Pillar : {res.dominant_pillar.upper()} (Max Score: {res.max_esg_score*100:.1f}%)")
            print(f"   [E]: {res.environmental.score*100:.1f}% | [S]: {res.social.score*100:.1f}% | [G]: {res.governance.score*100:.1f}%")
            print(f"-> Action Status    : {res.action_orientation}")
            if res.sub_topics:
                print(f"-> Sub-Themes       : {', '.join(res.sub_topics)}")


if __name__ == "__main__":
    run_targeted()
