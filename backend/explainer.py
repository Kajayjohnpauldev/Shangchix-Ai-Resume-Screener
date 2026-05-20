"""LangChain-driven LLM explanations.

Dispatches on LLM_PROVIDER: "openai" (gpt-4o-mini default) or "gemini"
(gemini-1.5-flash default). On any failure (network, timeout, malformed
JSON, missing key) returns 'Explanation unavailable' so the ranking
endpoint never crashes on the LLM step.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import LLM_MODEL, LLM_PROVIDER, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert technical recruiter. Given a job description and a "
    "candidate's resume, write a concise 2-sentence assessment for the hiring "
    "manager. Be specific about strengths and clear about gaps. Respond ONLY "
    "with strict JSON in the form: "
    '{"explanation": "..."} '
    "where the value is one string of no more than 60 words."
)

_chat_model: BaseChatModel | None = None


def _get_chat_model() -> BaseChatModel:
    """Build the LangChain chat model lazily; cache the instance."""
    global _chat_model
    if _chat_model is not None:
        return _chat_model

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _chat_model = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.2,
            max_tokens=200,
            timeout=LLM_TIMEOUT_SECONDS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _chat_model = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.2,
            max_output_tokens=200,
            timeout=LLM_TIMEOUT_SECONDS,
            google_api_key=os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Use 'openai' or 'gemini'."
        )
    return _chat_model


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
    # Some Gemini responses wrap JSON in ```json fences — strip them.
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
        chat = _get_chat_model()
        response = chat.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(
                    content=_build_user_prompt(
                        jd, resume, matched_skills, gaps
                    )
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        return _parse_explanation(content)
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        return "Explanation unavailable"
