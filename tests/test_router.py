"""Tests for request classification and model routing."""

import re
import pytest
from claw_router.router import classify_request, pick_model, parse_model, has_image_content, extract_text, get_fallback_candidates
from claw_router.breaker import CircuitBreaker


SIGNALS = {
    "code": re.compile(
        r'(?:\b(?:code|debug|function|class|import|def |async |await |console\.|print\(|'
        r'implement|refactor|bug|error|exception|stack.?trace|compile|syntax|'
        r'```|typescript|python|javascript|rust|golang|java|html|css|sql)\b|'
        r'写代码|调试|报错|函数|编译|重构)', re.IGNORECASE
    ),
    "reasoning": re.compile(
        r'(?:\b(?:analyze|reason|think|step.by.step|explain why|compare|evaluate|'
        r'pros.and.cons|trade.?off|architecture|design|plan|strategy)\b|'
        r'分析|推理|为什么|比较|权衡|架构|设计)', re.IGNORECASE
    ),
    "fast": re.compile(
        r'(?:\b(?:translate|summarize|hello|hi|hey|thanks|yes|no|ok)\b|'
        r'翻译|总结|你好|谢谢|是的|好的)', re.IGNORECASE
    ),
}

ROUTES = {
    "vision": ["ark:doubao-seed-2.0-pro"],
    "code": ["hub:deepseek", "ark:doubao-seed-2.0-code"],
    "reasoning": ["hub:deepseek-think", "hub:gemini-think"],
    "fast": ["hub:gemini-flash", "hub:glm"],
    "default": ["hub:gemini", "hub:qwen"],
}


def _body(text: str, model: str = "") -> dict:
    b = {"messages": [{"role": "user", "content": text}]}
    if model:
        b["model"] = model
    return b


def _image_body() -> dict:
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]}]}


class TestClassifyRequest:
    def test_code(self):
        assert classify_request(_body("Help me debug this python function"), SIGNALS) == "code"

    def test_reasoning(self):
        assert classify_request(_body("Analyze the architecture of this system"), SIGNALS) == "reasoning"

    def test_fast(self):
        assert classify_request(_body("hello"), SIGNALS) == "fast"

    def test_fast_long_text_becomes_default(self):
        assert classify_request(_body("hello " * 100), SIGNALS) == "default"

    def test_vision(self):
        assert classify_request(_image_body(), SIGNALS) == "vision"

    def test_default(self):
        assert classify_request(_body("Tell me about the weather today"), SIGNALS) == "default"

    def test_chinese_code(self):
        assert classify_request(_body("帮我写代码实现一个排序算法"), SIGNALS) == "code"

    def test_chinese_reasoning(self):
        assert classify_request(_body("分析一下这个方案"), SIGNALS) == "reasoning"


class TestPickModel:
    def test_picks_first_available(self):
        breaker = CircuitBreaker()
        assert pick_model("code", ROUTES, breaker) == "hub:deepseek"

    def test_skips_open_breaker(self):
        breaker = CircuitBreaker()
        breaker.open_until["hub:deepseek"] = 9999999999
        assert pick_model("code", ROUTES, breaker) == "ark:doubao-seed-2.0-code"

    def test_falls_back_to_default(self):
        breaker = CircuitBreaker()
        assert pick_model("unknown", ROUTES, breaker) == "hub:gemini"


class TestParseModel:
    def test_ark_prefix(self):
        assert parse_model("ark:doubao-seed-2.0-pro") == ("ark", "doubao-seed-2.0-pro")

    def test_cli_prefix(self):
        assert parse_model("cli:claude-sonnet-4-6") == ("cli", "claude-sonnet-4-6")

    def test_hub_prefix(self):
        assert parse_model("hub:deepseek") == ("hub", "deepseek")

    def test_guess_cli(self):
        assert parse_model("claude-sonnet-4-6") == ("cli", "claude-sonnet-4-6")

    def test_guess_hub(self):
        assert parse_model("deepseek", hub_names={"deepseek", "kimi"}) == ("hub", "deepseek")

    def test_guess_ark_default(self):
        assert parse_model("some-model") == ("ark", "some-model")


class TestFallbackCandidates:
    def test_returns_all_for_capability(self):
        breaker = CircuitBreaker()
        candidates = get_fallback_candidates("code", ROUTES, breaker)
        assert candidates[0] == "hub:deepseek"
        assert "ark:doubao-seed-2.0-code" in candidates

    def test_appends_default_as_fallback(self):
        breaker = CircuitBreaker()
        candidates = get_fallback_candidates("code", ROUTES, breaker)
        # default models appended after code models
        assert "hub:gemini" in candidates
        assert "hub:qwen" in candidates

    def test_skips_open_circuits(self):
        breaker = CircuitBreaker()
        breaker.open_until["hub:deepseek"] = 9999999999
        candidates = get_fallback_candidates("code", ROUTES, breaker)
        assert "hub:deepseek" not in candidates

    def test_excludes_tried(self):
        breaker = CircuitBreaker()
        candidates = get_fallback_candidates("code", ROUTES, breaker, exclude={"hub:deepseek"})
        assert "hub:deepseek" not in candidates
        assert candidates[0] == "ark:doubao-seed-2.0-code"

    def test_default_no_duplicates(self):
        breaker = CircuitBreaker()
        candidates = get_fallback_candidates("default", ROUTES, breaker)
        assert len(candidates) == len(set(candidates))


class TestHelpers:
    def test_has_image_content(self):
        assert has_image_content([{"content": [{"type": "image_url", "image_url": {"url": "x"}}]}])
        assert not has_image_content([{"content": "just text"}])

    def test_extract_text(self):
        msgs = [
            {"content": "hello"},
            {"content": [{"type": "text", "text": "world"}]},
        ]
        assert extract_text(msgs) == "hello world"
