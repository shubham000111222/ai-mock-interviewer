"""
Feedback Engine Module
Evaluates interview answers on:
  1. Content quality  — relevance, depth, structure, keywords
  2. Confidence score — derived from acoustic features + linguistic cues
"""

import os
import re
import json
import math
from typing import Dict, Any, List

# ── Optional: openai ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ── Optional: sentence-transformers for semantic similarity ────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _st_available = True
except ImportError:
    _st_available = False


# ---------------------------------------------------------------------------
# Content Feedback (LLM-based)
# ---------------------------------------------------------------------------
CONTENT_SYSTEM_PROMPT = """You are a senior interview coach who evaluates candidate answers.
Assess the answer based on:
1. Relevance to the question
2. Depth and specificity
3. Use of the STAR method (if behavioral)
4. Mention of relevant technical concepts / keywords
5. Clarity and structure

Return ONLY valid JSON in this exact format:
{
  "score": <int 0-100>,
  "grade": "<A/B/C/D/F>",
  "strengths": ["...", "..."],
  "improvements": ["...", "..."],
  "missing_keywords": ["...", "..."],
  "sample_answer_snippet": "...",
  "overall_comment": "..."
}"""

CONTENT_USER_TEMPLATE = """Question: {question}

Ideal answer hints: {hints}

Candidate's answer:
\"\"\"{transcript}\"\"\"

Evaluate the candidate's answer and return JSON feedback."""


class LLMContentEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        if not _openai_available:
            raise ImportError("Install openai: pip install openai")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def evaluate(
        self,
        question: str,
        transcript: str,
        hints: List[str],
    ) -> Dict[str, Any]:
        user_msg = CONTENT_USER_TEMPLATE.format(
            question=question,
            hints=", ".join(hints),
            transcript=transcript,
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)


class RuleBasedContentEvaluator:
    """
    Lightweight evaluator that works without any API:
    - Keyword overlap with hints
    - Length heuristics
    - Filler word detection
    """

    FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "actually", "sort of"}

    def evaluate(
        self,
        question: str,
        transcript: str,
        hints: List[str],
    ) -> Dict[str, Any]:
        text_lower = transcript.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        word_count = len(words)

        # ── Keyword hits ──────────────────────────────────────────────────────
        keyword_hits = []
        missing_keywords = []
        for hint in hints:
            hint_words = re.findall(r"\b\w+\b", hint.lower())
            if any(hw in text_lower for hw in hint_words if len(hw) > 3):
                keyword_hits.append(hint)
            else:
                missing_keywords.append(hint)

        kw_score = (len(keyword_hits) / max(len(hints), 1)) * 40  # max 40 pts

        # ── Length score ──────────────────────────────────────────────────────
        if word_count < 20:
            length_score = 10
        elif word_count < 60:
            length_score = 20
        elif word_count < 150:
            length_score = 30
        else:
            length_score = 35  # max 35 pts

        # ── Filler word penalty ───────────────────────────────────────────────
        filler_count = sum(text_lower.count(fw) for fw in self.FILLER_WORDS)
        filler_penalty = min(filler_count * 2, 15)

        # ── STAR method heuristic ─────────────────────────────────────────────
        star_keywords = ["situation", "task", "action", "result", "outcome", "impact", "led", "achieved"]
        star_score = 15 if sum(1 for k in star_keywords if k in text_lower) >= 3 else 5  # max 15 pts

        # ── Total ─────────────────────────────────────────────────────────────
        total = max(0, kw_score + length_score + star_score - filler_penalty)
        total = min(100, round(total))

        grade_map = [(90, "A"), (80, "B"), (65, "C"), (50, "D")]
        grade = next((g for threshold, g in grade_map if total >= threshold), "F")

        strengths = []
        improvements = []

        if keyword_hits:
            strengths.append(f"Covered key concepts: {', '.join(keyword_hits[:2])}")
        if word_count >= 60:
            strengths.append("Provided a sufficiently detailed response")
        if any(k in text_lower for k in ["result", "outcome", "impact", "achieved"]):
            strengths.append("Mentioned quantifiable outcomes or results")

        if missing_keywords:
            improvements.append(f"Consider mentioning: {', '.join(missing_keywords)}")
        if word_count < 60:
            improvements.append("Answer is too brief — aim for at least 2-3 sentences")
        if filler_count > 3:
            improvements.append(f"Reduce filler words (detected ~{filler_count} instances)")

        return {
            "score": total,
            "grade": grade,
            "strengths": strengths,
            "improvements": improvements,
            "missing_keywords": missing_keywords,
            "sample_answer_snippet": "Structure your answer using the STAR method for behavioral questions.",
            "overall_comment": f"Score: {total}/100. {'Strong answer!' if total >= 70 else 'Needs more detail and specificity.'}",
        }


# ---------------------------------------------------------------------------
# Confidence Scorer (acoustic + linguistic)
# ---------------------------------------------------------------------------
class ConfidenceScorer:
    """
    Combines acoustic features from SpeechProcessor with linguistic cues
    to produce a 0-100 confidence score and a breakdown.
    """

    HEDGE_WORDS = {
        "maybe", "perhaps", "i think", "i guess", "i suppose",
        "not sure", "kind of", "sort of", "i believe", "i hope",
        "probably", "might", "could be",
    }

    def score(
        self,
        transcript: str,
        acoustic_features: Dict[str, float],
    ) -> Dict[str, Any]:
        text_lower = transcript.lower()

        # ── Sub-scores (all 0-100) ────────────────────────────────────────────
        pause_score = self._pause_score(acoustic_features.get("pause_ratio", 0.2))
        pitch_score = self._pitch_score(
            acoustic_features.get("pitch_std_hz", 30),
            acoustic_features.get("pitch_mean_hz", 150),
        )
        energy_score = self._energy_score(
            acoustic_features.get("energy_mean", 0.04),
            acoustic_features.get("energy_std", 0.015),
        )
        hedge_score = self._hedge_score(text_lower)
        length_score = self._length_score(len(transcript.split()))

        # ── Weighted composite ────────────────────────────────────────────────
        weights = {
            "pause": 0.25,
            "pitch": 0.20,
            "energy": 0.20,
            "hedge": 0.20,
            "length": 0.15,
        }
        composite = (
            weights["pause"] * pause_score
            + weights["pitch"] * pitch_score
            + weights["energy"] * energy_score
            + weights["hedge"] * hedge_score
            + weights["length"] * length_score
        )
        composite = round(min(100, max(0, composite)))

        level = (
            "High" if composite >= 75
            else "Moderate" if composite >= 50
            else "Low"
        )

        return {
            "confidence_score": composite,
            "level": level,
            "breakdown": {
                "fluency_pause": round(pause_score),
                "vocal_variation": round(pitch_score),
                "vocal_energy": round(energy_score),
                "assertive_language": round(hedge_score),
                "answer_completeness": round(length_score),
            },
            "tips": self._tips(composite, acoustic_features, text_lower),
        }

    # ── Sub-score helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _pause_score(pause_ratio: float) -> float:
        # Ideal pause ratio: 10-25%
        if 0.10 <= pause_ratio <= 0.25:
            return 90.0
        elif pause_ratio < 0.10:  # Speaking too fast
            return max(40, 90 - (0.10 - pause_ratio) * 500)
        else:  # Too many pauses
            return max(20, 90 - (pause_ratio - 0.25) * 300)

    @staticmethod
    def _pitch_score(pitch_std: float, pitch_mean: float) -> float:
        # Good variation: std 20-60 Hz
        if 20 <= pitch_std <= 60:
            return 85.0
        elif pitch_std < 20:  # Monotone
            return max(30, 85 - (20 - pitch_std) * 2)
        else:  # Erratic pitch
            return max(40, 85 - (pitch_std - 60) * 0.8)

    @staticmethod
    def _energy_score(energy_mean: float, energy_std: float) -> float:
        # Proxy: moderate energy is good
        # These are raw RMS values — normalize loosely
        if 0.02 <= energy_mean <= 0.1:
            return 85.0
        elif energy_mean < 0.02:
            return max(30, 85 - (0.02 - energy_mean) * 2000)
        else:
            return max(50, 85 - (energy_mean - 0.1) * 300)

    def _hedge_score(self, text_lower: str) -> float:
        hedge_count = sum(text_lower.count(hw) for hw in self.HEDGE_WORDS)
        return max(10, 100 - hedge_count * 10)

    @staticmethod
    def _length_score(word_count: int) -> float:
        if word_count < 20:
            return 30.0
        elif word_count < 50:
            return 55.0
        elif word_count < 100:
            return 75.0
        elif word_count < 200:
            return 90.0
        else:
            return 80.0  # Very long can also be rambling

    @staticmethod
    def _tips(score: float, acoustic: dict, text_lower: str) -> List[str]:
        tips = []
        if acoustic.get("pause_ratio", 0) > 0.35:
            tips.append("You paused frequently — practice your answers to improve fluency.")
        if acoustic.get("pitch_std_hz", 30) < 20:
            tips.append("Your tone was monotone — vary your pitch to sound more engaging.")
        if acoustic.get("energy_mean", 0.04) < 0.02:
            tips.append("Speak louder and with more energy to project confidence.")
        if any(hw in text_lower for hw in ["i guess", "i think", "maybe", "not sure"]):
            tips.append("Replace hedging phrases like 'I guess' with assertive language.")
        if score >= 75:
            tips.append("Great confidence! Maintain this energy throughout the interview.")
        return tips or ["Keep practising to build consistency and confidence."]


# ---------------------------------------------------------------------------
# Unified Feedback Engine
# ---------------------------------------------------------------------------
class FeedbackEngine:
    """
    Orchestrates content evaluation + confidence scoring for a single answer.
    """

    def __init__(self, content_backend: str = "rules"):
        if content_backend == "llm":
            self.content_evaluator = LLMContentEvaluator()
        else:
            self.content_evaluator = RuleBasedContentEvaluator()
        self.confidence_scorer = ConfidenceScorer()

    def evaluate(
        self,
        question: Dict[str, Any],
        transcript: str,
        acoustic_features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Returns combined feedback including content score and confidence score.
        """
        hints = question.get("ideal_answer_hints", [])

        content = self.content_evaluator.evaluate(
            question=question["question"],
            transcript=transcript,
            hints=hints,
        )
        confidence = self.confidence_scorer.score(transcript, acoustic_features)

        # Overall composite
        overall = round(0.6 * content["score"] + 0.4 * confidence["confidence_score"])

        return {
            "question_id": question.get("id"),
            "question_text": question["question"],
            "transcript": transcript,
            "content_feedback": content,
            "confidence_feedback": confidence,
            "overall_score": overall,
            "overall_grade": content["grade"],
        }
