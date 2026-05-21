"""Tests for LLM complexity classifier."""

import pytest
from claw_router.classifier import _regex_fallback, _extract_last_user_text


class TestRegexFallback:
    def test_empty(self):
        assert _regex_fallback([]) == "simple"

    def test_simple_greeting(self):
        msgs = [{"role": "user", "content": "你好"}]
        assert _regex_fallback(msgs) == "simple"

    def test_simple_short(self):
        msgs = [{"role": "user", "content": "translate this to English"}]
        assert _regex_fallback(msgs) == "simple"

    def test_complex_architecture(self):
        msgs = [{"role": "user", "content": "design system architecture for a distributed queue"}]
        assert _regex_fallback(msgs) == "complex"

    def test_complex_debug(self):
        msgs = [{"role": "user", "content": "debug this stack trace and find root cause"}]
        assert _regex_fallback(msgs) == "complex"

    def test_complex_long_text(self):
        msgs = [{"role": "user", "content": "x " * 1500}]
        assert _regex_fallback(msgs) == "complex"

    def test_medium_default(self):
        msgs = [{"role": "user", "content": "Write a function to sort a list of numbers"}]
        assert _regex_fallback(msgs) == "medium"


class TestExtractLastUserText:
    def test_string_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert _extract_last_user_text(msgs) == "hello"

    def test_list_content(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}]
        assert _extract_last_user_text(msgs) == "hello"

    def test_picks_last_user(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_last_user_text(msgs) == "second"

    def test_empty(self):
        assert _extract_last_user_text([]) == ""

    def test_no_user(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        assert _extract_last_user_text(msgs) == ""
