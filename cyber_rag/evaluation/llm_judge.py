from __future__ import annotations

import json
from functools import wraps

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from openai import RateLimitError

from cyber_rag.config import GenerationConfig, get_question_context_prefix

_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for cybersecurity QA benchmarking.

Task:
- Compare the candidate answer with the reference answer for the given question.
- Score how well the candidate matches the reference on correctness, factual consistency, and completeness.
- Minor wording differences are acceptable if meaning is equivalent.
- Partial credit is allowed: use intermediate values between 0 and 1 when the answer is partly right.

Output rules:
- Return valid JSON only.
- Use schema: {{"accuracy": <number between 0.0 and 1.0>, "reason": "short reason"}}.
- 1.0 = fully correct relative to the reference; 0.0 = wrong, refused to answer, or irrelevant.
"""

_JUDGE_HUMAN_PROMPT = """Question:
{question}

Reference Answer (Ground Truth):
{reference_answer}

Candidate Answer:
{candidate_answer}
"""


def _retry_on_rate_limit(max_retries: int = 5, base_delay: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time

            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt)
                        time.sleep(delay)
                    else:
                        raise
            raise last_exception

        return wrapper

    return decorator


def _build_judge_llm(config: GenerationConfig | None = None):
    judge_config = config or GenerationConfig()

    if not judge_config.api_key:
        raise EnvironmentError("Judge API key is required.")

    if not judge_config.base_url:
        raise EnvironmentError("Judge base URL is required.")

    if judge_config.provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=judge_config.model_name or "gpt-4o-mini",
            api_version=judge_config.api_version or "2024-10-21",
            azure_endpoint=judge_config.base_url.rstrip("/"),
            api_key=judge_config.api_key,
            temperature=0.0,
            max_retries=2,
        )
    if judge_config.provider == "oneapi":
        return ChatOpenAI(
            model=judge_config.model_name or "gpt-4o-mini",
            base_url=judge_config.base_url.rstrip("/"),
            api_key=judge_config.api_key,
            temperature=0.0,
            max_retries=2,
        )

    raise NotImplementedError(f"Unsupported judge provider: {judge_config.provider}")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_judge_output(raw_text: str) -> tuple[float, str]:
    text = str(raw_text).strip()
    reason = ""

    try:
        payload = json.loads(text)
        reason = str(payload.get("reason", "")).strip()

        if "accuracy" in payload and payload["accuracy"] is not None:
            try:
                return _clamp01(float(payload["accuracy"])), reason
            except (TypeError, ValueError):
                pass

        verdict = str(payload.get("verdict", "")).upper()
        if verdict == "CORRECT":
            return 1.0, reason
        if verdict == "INCORRECT":
            return 0.0, reason
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    upper_text = text.upper()
    if "CORRECT" in upper_text and "INCORRECT" not in upper_text:
        return 1.0, text
    if "INCORRECT" in upper_text:
        return 0.0, text
    return 0.0, text


@_retry_on_rate_limit(max_retries=5, base_delay=2.0)
def judge_answer_accuracy(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    judge_config: GenerationConfig | None = None,
) -> tuple[float, str]:
    """Return (accuracy in [0, 1], short reason)."""
    llm = _build_judge_llm(judge_config)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _JUDGE_SYSTEM_PROMPT), ("human", _JUDGE_HUMAN_PROMPT)]
    )
    q_for_judge = question
    prefix = get_question_context_prefix()
    if prefix:
        q_for_judge = f"{prefix}{question}"

    response = llm.invoke(
        prompt.format_messages(
            question=q_for_judge,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
    )
    return _parse_judge_output(str(response.content))


def judge_answer_correctness(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    judge_config: GenerationConfig | None = None,
    threshold: float = 0.5,
) -> tuple[bool, str]:
    """Backward-compatible: True if accuracy >= threshold."""
    acc, reason = judge_answer_accuracy(
        question=question,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        judge_config=judge_config,
    )
    return acc >= threshold, reason
