"""LiteLLM-driven LLM explanations.

The provider is chosen entirely by the LLM_MODEL string — litellm routes
on the prefix:
    "gemini/gemini-1.5-flash"      Google (free tier, default)
    "gpt-4o-mini"                  OpenAI
    "groq/llama-3.1-8b-instant"    Groq
    "claude-3-5-haiku-20241022"    Anthropic

litellm reads the matching API key from the environment automatically
(GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY), so
re-pointing the app at another provider is two env vars and zero code
changes. On any failure (network, timeout, malformed JSON, missing key)
this returns 'Explanation unavailable' so the ranking endpoint never
crashes on the LLM step.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import litellm
from litellm import completion

from backend.config import LLM_MODEL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Silently drop kwargs a given provider doesn't support (e.g. some models
# reject response_format) instead of raising — keeps the call portable.
litellm.drop_params = True

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter. Given a job description and a "
    "candidate's resume, write a concise 2-sentence assessment for the hiring "
    "manager. Be specific about strengths and clear about gaps. Respond ONLY "
    "with strict JSON in the form: "
    '{"explanation": "..."} '
    "where the value is one string of no more than 60 words."
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


def _parse_explanation(content: str) -> str:
    # Some providers wrap JSON in ```json fences — strip them.
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        data: Any = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("explanation"), str):
            return data["explanation"].strip()
    except (json.JSONDecodeError, AttributeError):
        pass
    return content.strip()[:400] or "Explanation unavailable"


def explain(
    jd: str,
    resume: str,
    matched_skills: list[str],
    gaps: list[str],
) -> str:
    """Return a one-paragraph explanation, or 'Explanation unavailable'."""
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
            max_tokens=200,
            timeout=LLM_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return _parse_explanation(content)
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        return "Explanation unavailable"
