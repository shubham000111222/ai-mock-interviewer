"""
Question Generator Module
Generates role-specific interview questions from a job description
using OpenAI GPT or a local Ollama (Llama) model.
"""

import os
import json
import re
from typing import List, Dict, Any

# ── Optional: openai ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False

# ── Optional: ollama ──────────────────────────────────────────────────────────
try:
    import ollama as _ollama
    _ollama_available = True
except ImportError:
    _ollama_available = False


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert technical recruiter and interview coach.
Your job is to generate highly relevant, thoughtful interview questions
based on a provided job description.

Rules:
- Generate exactly the number of questions requested.
- Categorize them as: Technical, Behavioral, or Situational.
- Vary difficulty: Easy, Medium, Hard.
- Return ONLY valid JSON — no extra text.

JSON format:
[
  {
    "id": 1,
    "category": "Technical",
    "difficulty": "Medium",
    "question": "...",
    "ideal_answer_hints": ["point1", "point2"],
    "follow_up": "..."
  }
]
"""

USER_PROMPT_TEMPLATE = """Job Description:
---
{job_description}
---

Generate {num_questions} interview questions covering:
- {num_technical} Technical questions
- {num_behavioral} Behavioral questions
- {num_situational} Situational questions

Focus heavily on the specific skills, tools, and responsibilities mentioned.
Return ONLY the JSON array.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> List[Dict[str, Any]]:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    # Remove ```json ... ``` fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Core generator classes
# ---------------------------------------------------------------------------
class OpenAIQuestionGenerator:
    """Uses OpenAI GPT-4o-mini (or any model) to generate questions."""

    def __init__(self, model: str = "gpt-4o-mini"):
        if not _openai_available:
            raise ImportError("Install openai: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        job_description: str,
        num_questions: int = 10,
        num_technical: int = 5,
        num_behavioral: int = 3,
        num_situational: int = 2,
    ) -> List[Dict[str, Any]]:
        user_msg = USER_PROMPT_TEMPLATE.format(
            job_description=job_description,
            num_questions=num_questions,
            num_technical=num_technical,
            num_behavioral=num_behavioral,
            num_situational=num_situational,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=3000,
        )
        raw = response.choices[0].message.content
        return _extract_json(raw)


class OllamaQuestionGenerator:
    """Uses a locally running Ollama model (e.g. llama3, mistral)."""

    def __init__(self, model: str = "llama3"):
        if not _ollama_available:
            raise ImportError("Install ollama: pip install ollama")
        self.model = model

    def generate(
        self,
        job_description: str,
        num_questions: int = 10,
        num_technical: int = 5,
        num_behavioral: int = 3,
        num_situational: int = 2,
    ) -> List[Dict[str, Any]]:
        user_msg = USER_PROMPT_TEMPLATE.format(
            job_description=job_description,
            num_questions=num_questions,
            num_technical=num_technical,
            num_behavioral=num_behavioral,
            num_situational=num_situational,
        )
        response = _ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response["message"]["content"]
        return _extract_json(raw)


class MockQuestionGenerator:
    """Fallback generator — returns hardcoded sample questions for demo/testing."""

    def generate(
        self,
        job_description: str,
        num_questions: int = 10,
        **_kwargs,
    ) -> List[Dict[str, Any]]:
        samples = [
            {
                "id": 1,
                "category": "Technical",
                "difficulty": "Medium",
                "question": "Explain the difference between supervised and unsupervised learning with real-world examples.",
                "ideal_answer_hints": ["Labeled vs unlabeled data", "Classification/Regression vs Clustering"],
                "follow_up": "Which approach would you use for anomaly detection and why?",
            },
            {
                "id": 2,
                "category": "Technical",
                "difficulty": "Hard",
                "question": "How would you handle class imbalance in a fraud detection model?",
                "ideal_answer_hints": ["SMOTE / oversampling", "Class weights", "Precision-Recall curve"],
                "follow_up": "What metrics would you optimize and why?",
            },
            {
                "id": 3,
                "category": "Technical",
                "difficulty": "Medium",
                "question": "Walk me through your feature engineering process for a tabular dataset.",
                "ideal_answer_hints": ["EDA first", "Domain knowledge", "Encoding, scaling, interaction features"],
                "follow_up": "How do you avoid data leakage during feature creation?",
            },
            {
                "id": 4,
                "category": "Technical",
                "difficulty": "Easy",
                "question": "What is the bias-variance tradeoff? How does it affect model selection?",
                "ideal_answer_hints": ["Underfitting vs overfitting", "Regularization", "Cross-validation"],
                "follow_up": "Give an example of a high-bias model and a high-variance model.",
            },
            {
                "id": 5,
                "category": "Technical",
                "difficulty": "Hard",
                "question": "Describe how you would deploy a machine learning model to production.",
                "ideal_answer_hints": ["REST API / Docker", "CI/CD pipeline", "Monitoring, drift detection"],
                "follow_up": "How would you monitor model performance post-deployment?",
            },
            {
                "id": 6,
                "category": "Behavioral",
                "difficulty": "Medium",
                "question": "Tell me about a time you had to explain a complex technical concept to a non-technical stakeholder.",
                "ideal_answer_hints": ["STAR method", "Simplification", "Business impact framing"],
                "follow_up": "What would you do differently next time?",
            },
            {
                "id": 7,
                "category": "Behavioral",
                "difficulty": "Medium",
                "question": "Describe a project where you had to work with messy or incomplete data.",
                "ideal_answer_hints": ["Problem identification", "Imputation strategy", "Outcome"],
                "follow_up": "Did you document your data cleaning decisions?",
            },
            {
                "id": 8,
                "category": "Behavioral",
                "difficulty": "Easy",
                "question": "How do you stay current with the latest developments in data science and AI?",
                "ideal_answer_hints": ["Papers, blogs, Kaggle", "Side projects", "Communities"],
                "follow_up": "What is the most recent technique you learned and applied?",
            },
            {
                "id": 9,
                "category": "Situational",
                "difficulty": "Hard",
                "question": "Your model has 95% accuracy in testing but performs poorly in production. What steps do you take?",
                "ideal_answer_hints": ["Distribution shift", "Data leakage check", "Shadow mode testing"],
                "follow_up": "How would you communicate this issue to your manager?",
            },
            {
                "id": 10,
                "category": "Situational",
                "difficulty": "Medium",
                "question": "You are given 2 weeks to build a proof-of-concept recommendation system. How do you approach it?",
                "ideal_answer_hints": ["Scope first", "Collaborative vs content-based", "MVP mindset"],
                "follow_up": "What trade-offs would you make given the time constraint?",
            },
        ]
        return samples[:num_questions]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------
def get_question_generator(backend: str = "mock"):
    """
    Factory to return the appropriate generator.
    backend: 'openai' | 'ollama' | 'mock'
    """
    if backend == "openai":
        return OpenAIQuestionGenerator()
    elif backend == "ollama":
        return OllamaQuestionGenerator()
    else:
        return MockQuestionGenerator()
