"""LLM-based request complexity classifier using doubao-lite."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    from claw_router.config import AppConfig

logger = logging.getLogger(__name__)

_PROMPT = """判断以下请求的复杂度。只回答一个词：simple、medium、complex。

simple = 翻译、问候、简单问答、格式转换、短文总结
medium = 代码编写、文本分析、内容生成、数据处理
complex = 架构设计、多步推理、复杂debug、系统分析、长文写作

请求：{text}"""

_VALID = {"simple", "medium", "complex"}


async def classify_complexity(
    client: httpx.AsyncClient,
    config: AppConfig,
    messages: list[dict],
) -> str:
    """Classify request complexity via doubao-lite. Returns 'simple'/'medium'/'complex'.

    Falls back to 'medium' on any error or timeout.
    """
    if not config.classifier_enabled:
        return _regex_fallback(messages)

    ark = config.upstreams.get("ark")
    if not ark:
        return _regex_fallback(messages)

    text = _extract_last_user_text(messages)
    if not text:
        return "simple"

    # Short messages are trivially simple
    if len(text) < 30:
        return "simple"

    try:
        resp = await client.post(
            f"{ark.base}/messages",
            content=json.dumps({
                "model": config.classifier_model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": _PROMPT.format(text=text[:500])}],
            }).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": ark.auth,
                "anthropic-version": "2023-06-01",
            },
            timeout=config.classifier_timeout,
        )
        if resp.status_code >= 400:
            logger.debug(f"[classifier] {resp.status_code}, falling back to regex")
            return _regex_fallback(messages)

        body = resp.json()
        result = body.get("content", [{}])[0].get("text", "").strip().lower()
        # Extract first valid word
        for word in result.split():
            if word in _VALID:
                logger.debug(f"[classifier] complexity={word}")
                return word
        return _regex_fallback(messages)

    except Exception as e:
        logger.debug(f"[classifier] error: {e}, falling back to regex")
        return _regex_fallback(messages)


def _extract_last_user_text(messages: list[dict]) -> str:
    """Get the last user message text."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(parts)
    return ""


def _regex_fallback(messages: list[dict]) -> str:
    """Quick regex-based complexity estimation when LLM classifier unavailable."""
    text = _extract_last_user_text(messages)
    if not text:
        return "simple"

    text_lower = text.lower()
    length = len(text)

    # Long prompts are likely complex
    if length > 2000:
        return "complex"

    complex_signals = (
        "architect", "design system", "分析", "架构", "设计方案", "重构整个",
        "step by step", "trade-off", "compare", "evaluate", "权衡",
        "debug", "调试", "stack trace", "root cause",
    )
    if any(s in text_lower for s in complex_signals):
        return "complex"

    simple_signals = (
        "翻译", "translate", "hello", "你好", "谢谢", "thanks",
        "yes", "no", "ok", "好的", "是的", "总结",
    )
    if any(s in text_lower for s in simple_signals) and length < 200:
        return "simple"

    return "medium"
