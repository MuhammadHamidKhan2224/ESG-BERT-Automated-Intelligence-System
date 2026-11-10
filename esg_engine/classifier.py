"""
ESGBertClassifier: Multi-Pillar ESG Text Classification Engine.
Integrates the three official ESGBERT models:
  - ESGBERT/EnvironmentalBERT-environmental
  - ESGBERT/SocialBERT-social
  - ESGBERT/GovernanceBERT-governance
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# Suppress noisy library logs if desired
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# Pillar model identifiers on Hugging Face Hub
PILLAR_MODELS = {
    "environmental": "ESGBERT/EnvironmentalBERT-environmental",
    "social": "ESGBERT/SocialBERT-social",
    "governance": "ESGBERT/GovernanceBERT-governance",
}

# Sub-topic keyword taxonomy for deep thematic tagging
SUB_TOPIC_TAXONOMY = {
    "environmental": {
        "Climate & GHG Emissions": [
            "emission", "carbon", "ghg", "greenhouse", "co2", "net-zero", "scope 1", "scope 2", "scope 3", "decarbonization", "warming"
        ],
        "Clean Energy & Efficiency": [
            "renewable", "solar", "wind", "hydro", "energy efficiency", "clean energy", "megawatt", "kwh", "grid"
        ],
        "Biodiversity & Nature": [
            "biodiversity", "conservation", "wildlife", "ecosystem", "forest", "deforestation", "habitat", "nature"
        ],
        "Water & Waste Management": [
            "water", "effluent", "waste", "recycle", "circular", "landfill", "packaging", "plastic", "hazardous"
        ],
    },
    "social": {
        "Diversity, Equity & Inclusion": [
            "diversity", "inclusion", "equity", "gender", "minority", "women", "underrepresented", "bame", "lgbtq"
        ],
        "Health, Safety & Well-being": [
            "health", "safety", "injury", "well-being", "fatality", "mental health", "osha", "workplace safety"
        ],
        "Labor Rights & Human Capital": [
            "labor", "worker", "employee", "union", "fair wage", "training", "development", "retention", "turnover"
        ],
        "Human Rights & Supply Chain": [
            "human rights", "forced labor", "child labor", "supply chain", "supplier audit", "modern slavery"
        ],
        "Community & Social Impact": [
            "community", "philanthropy", "charity", "social impact", "indigenous", "local population"
        ],
    },
    "governance": {
        "Board Structure & Independence": [
            "board", "director", "independence", "governance", "supervisory", "audit committee", "nominating", "chair"
        ],
        "Executive Compensation": [
            "compensation", "executive pay", "remuneration", "clawback", "bonus", "incentive plan", "ceo pay"
        ],
        "Ethics, Anti-Corruption & Compliance": [
            "ethics", "anti-corruption", "bribery", "compliance", "code of conduct", "whistleblower", "fraud"
        ],
        "Shareholder Rights & Transparency": [
            "shareholder", "voting", "proxy", "disclosure", "transparency", "stakeholder engagement"
        ],
        "Risk Oversight & Cybersecurity": [
            "risk management", "internal control", "cybersecurity", "data privacy", "audit", "compliance monitoring"
        ],
    },
}

ACTION_VERBS = [
    "achieved", "reduced", "invested", "implemented", "launched", "installed",
    "decreased", "cut", "funded", "trained", "audited", "established",
    "surpassed", "converted", "deployed", "completed", "eliminated"
]

ASPIRATIONAL_TERMS = [
    "aim to", "target to", "plan to", "seek to", "strive to", "commit to",
    "hope to", "will endeavor", "intend to", "pledge to", "anticipate"
]


@dataclass
class ESGPillarScore:
    """Confidence scores for a single ESG pillar."""
    pillar: str
    is_pillar: bool
    score: float
    confidence_label: str  # e.g., 'High', 'Moderate', 'Low'


@dataclass
class ESGTextAnalysis:
    """Full ESG evaluation of an individual text passage."""
    text: str
    environmental: ESGPillarScore
    social: ESGPillarScore
    governance: ESGPillarScore
    dominant_pillar: str  # 'environmental', 'social', 'governance', 'multi-pillar', or 'non-esg'
    active_pillars: List[str]
    max_esg_score: float
    sub_topics: List[str] = field(default_factory=list)
    action_orientation: str = "Neutral / Disclosure"  # 'Action-Oriented', 'Aspirational', or 'Boilerplate / Disclosure'
    has_metrics: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "dominant_pillar": self.dominant_pillar,
            "active_pillars": self.active_pillars,
            "max_esg_score": round(self.max_esg_score, 4),
            "environmental": {
                "is_pillar": self.environmental.is_pillar,
                "score": round(self.environmental.score, 4),
                "confidence": self.environmental.confidence_label,
            },
            "social": {
                "is_pillar": self.social.is_pillar,
                "score": round(self.social.score, 4),
                "confidence": self.social.confidence_label,
            },
            "governance": {
                "is_pillar": self.governance.is_pillar,
                "score": round(self.governance.score, 4),
                "confidence": self.governance.confidence_label,
            },
            "sub_topics": self.sub_topics,
            "action_orientation": self.action_orientation,
            "has_metrics": self.has_metrics,
        }


class ESGBertClassifier:
    """
    Multi-pillar ESG Classifier orchestrating EnvironmentalBERT,
    SocialBERT, and GovernanceBERT.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        threshold: float = 0.5,
        lazy_load: bool = True,
    ):
        """
        Args:
            device: 'cuda', 'cpu', or None (auto-detect).
            threshold: Probability threshold above which a pillar is declared active.
            lazy_load: If True, models are loaded upon first classification call.
        """
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda" and torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

        self.threshold = threshold
        self.pipelines: Dict[str, Any] = {}
        self.cache: Dict[str, ESGTextAnalysis] = {}

        if not lazy_load:
            self._ensure_models_loaded()

    def _ensure_models_loaded(self):
        """Loads all three ESGBERT models into pipeline cache."""
        for pillar, model_id in PILLAR_MODELS.items():
            if pillar not in self.pipelines:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                pipe = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    top_k=None,
                )
                self.pipelines[pillar] = pipe

    def _score_to_confidence(self, score: float) -> str:
        if score >= 0.85:
            return "High"
        elif score >= 0.60:
            return "Moderate"
        elif score >= 0.50:
            return "Low / Marginal"
        else:
            return "Non-Classified"

    def _detect_sub_topics(self, text: str, active_pillars: List[str]) -> List[str]:
        """Maps text against the sub-topic taxonomy for active pillars."""
        lowered = text.lower()
        detected = []

        pillars_to_check = active_pillars if active_pillars else ["environmental", "social", "governance"]
        for pillar in pillars_to_check:
            for category, keywords in SUB_TOPIC_TAXONOMY.get(pillar, {}).items():
                for kw in keywords:
                    if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                        tag = f"[{pillar[0].upper()}] {category}"
                        if tag not in detected:
                            detected.append(tag)
                        break
        return detected

    def _detect_action_and_metrics(self, text: str) -> Tuple[str, bool]:
        """Detects whether text has concrete actions and quantified metrics."""
        lowered = text.lower()

        # Check for numbers / quantified metrics (% or numbers with units)
        metric_pattern = r"(\d+(\.\d+)?%|\$\d+(\.\d+)?|\b\d{2,}\b\s*(tons|mwh|kwh|mw|ghg|co2e|million|billion|liters|gallons|people|employees))"
        has_metrics = bool(re.search(metric_pattern, lowered))

        has_action = any(re.search(r"\b" + re.escape(verb) + r"\b", lowered) for verb in ACTION_VERBS)
        has_aspirational = any(term in lowered for term in ASPIRATIONAL_TERMS)

        if has_action and (has_metrics or not has_aspirational):
            action_orientation = "Action-Oriented (Concrete Action/Result)"
        elif has_aspirational:
            action_orientation = "Aspirational (Target / Pledge)"
        else:
            action_orientation = "Boilerplate / General Disclosure"

        return action_orientation, has_metrics

    def analyze_text(self, text: str) -> ESGTextAnalysis:
        """
        Analyzes a single text string across all 3 ESG pillars.
        """
        clean_text = text.strip()
        if not clean_text:
            empty_score = ESGPillarScore("none", False, 0.0, "None")
            return ESGTextAnalysis(
                text="",
                environmental=empty_score,
                social=empty_score,
                governance=empty_score,
                dominant_pillar="non-esg",
                active_pillars=[],
                max_esg_score=0.0,
            )

        if clean_text in self.cache:
            return self.cache[clean_text]

        self._ensure_models_loaded()

        pillar_scores: Dict[str, ESGPillarScore] = {}

        for pillar, pipe in self.pipelines.items():
            preds = pipe(clean_text, truncation=True, max_length=512)[0]
            # Find the score for the positive pillar class
            pos_score = 0.0
            for item in preds:
                if item["label"].lower() == pillar:
                    pos_score = float(item["score"])
                    break

            is_active = pos_score >= self.threshold
            conf_label = self._score_to_confidence(pos_score)
            pillar_scores[pillar] = ESGPillarScore(
                pillar=pillar,
                is_pillar=is_active,
                score=pos_score,
                confidence_label=conf_label,
            )

        active = [p for p, s in pillar_scores.items() if s.is_pillar]
        max_score_pillar = max(pillar_scores.items(), key=lambda x: x[1].score)
        max_score = max_score_pillar[1].score

        if not active:
            dominant = "non-esg"
        elif len(active) == 1:
            dominant = active[0]
        else:
            # Overlapping / multi-pillar
            dominant = max_score_pillar[0] if max_score >= 0.70 else "multi-pillar"

        sub_topics = self._detect_sub_topics(clean_text, active)
        action_orientation, has_metrics = self._detect_action_and_metrics(clean_text)

        analysis = ESGTextAnalysis(
            text=clean_text,
            environmental=pillar_scores["environmental"],
            social=pillar_scores["social"],
            governance=pillar_scores["governance"],
            dominant_pillar=dominant,
            active_pillars=active,
            max_esg_score=max_score,
            sub_topics=sub_topics,
            action_orientation=action_orientation,
            has_metrics=has_metrics,
        )

        self.cache[clean_text] = analysis
        return analysis

    def analyze_batch(
        self, texts: List[str], batch_size: int = 16
    ) -> List[ESGTextAnalysis]:
        """
        Analyzes a batch of texts efficiently.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.analyze_text(text))
        return results
