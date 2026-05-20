"""LiteLLM-driven LLM explanations.

The provider is chosen entirely by the LLM_MODEL string — litellm routes
on the prefix:
    "groq/llama-3.1-8b-instant"    Groq (free, fast — default)
    "gemini/gemini-2.0-flash"      Google (free tier)
    "gpt-4o-mini"                  OpenAI
    "claude-3-5-haiku-20241022"    Anthropic

litellm reads the matching API key from the environment automatically
(GROQ_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY), so
re-pointing the app at another provider is two env vars and zero code
changes. On any failure (network, timeout, malformed JSON, missing key)
this returns 'Explanation unavailable' so the ranking endpoint never
crashes on the LLM step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import completion

from backend.config import LLM_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Silently drop kwargs a given provider doesn't support (e.g. some models
# reject response_format) instead of raising — keeps the call portable.
litellm.drop_params = True


@dataclass
class Advice:
    """One LLM result: a recruiter assessment plus resume-improvement tips."""

    explanation: str
    improvements: list[str] = field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are an expert technical recruiter and resume coach. Given a job "
    "description and a candidate's resume, return TWO things:\n"
    "1. A concise 2-sentence assessment for the hiring manager — specific "
    "about strengths and clear about gaps.\n"
    "2. 3 to 5 short, concrete, actionable suggestions the candidate could "
    "make to their resume to better match THIS role and raise their match "
    "score — e.g. skills to add, keywords to include, achievements to "
    "quantify. Each suggestion under 20 words, phrased as an imperative.\n"
    "Respond ONLY with strict JSON in the form: "
    '{"explanation": "...", "improvements": ["...", "..."]} '
    "where explanation is one string under 60 words and improvements is a "
    "list of short strings."
)


def _build_user_prompt(
    jd: str,
    resume: str,
    matched: list[str],
    gaps: list[str],
) -> str:
    matched_s = ", ".join(matched) if matched else "(none surfaced)"
    gaps_s = ", ".join(gaps) if gaps else "(none surfaced)"
    return (
        f"JOB DESCRIPTION:\n{jd[:2000]}\n\n"
        f"CANDIDATE RESUME:\n{resume[:3000]}\n\n"
        f"Detected matching skills: {matched_s}\n"
        f"Detected gaps: {gaps_s}\n\n"
        "Return your assessment as JSON now."
    )


def _parse_advice(content: str) -> Advice:
    # Some providers wrap JSON in ```json fences — strip them.
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data: Any = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("explanation"), str):
            raw_tips = data.get("improvements", [])
            tips = [
                str(t).strip()
                for t in raw_tips
                if isinstance(t, (str, int, float)) and str(t).strip()
            ][:5]
            return Advice(explanation=data["explanation"].strip(), improvements=tips)
    except (json.JSONDecodeError, AttributeError):
        pass
    # Couldn't parse JSON — salvage the raw text as the explanation.
    return Advice(explanation=content.strip()[:400] or "Explanation unavailable")


def explain(
    jd: str,
    resume: str,
    matched_skills: list[str],
    gaps: list[str],
) -> Advice:
    """Return an Advice (assessment + improvement tips), or a safe fallback."""
    try:
        response = completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        jd, resume, matched_skills, gaps
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=400,
            timeout=LLM_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return _parse_advice(content)
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        return Advice(explanation="Explanation unavailable")
