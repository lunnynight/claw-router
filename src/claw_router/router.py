"""Request classification and model routing."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
    from claw_router.config import AppConfig
    from claw_router.breaker import CircuitBreaker

from claw_router.classifier import classify_complexity

logger = logging.getLogger(__name__)


def has_image_content(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


def extract_text(messages: list[dict]) -> str:
    texts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
    return " ".join(texts)


def classify_request(body: dict, signals: dict[str, re.Pattern]) -> str:
    """Classify a request into a capability category (legacy regex path)."""
    messages = body.get("messages", [])
    if has_image_content(messages):
        return "vision"
    text = extract_text(messages)
    if signals.get("code") and signals["code"].search(text):
        return "code"
    if signals.get("reasoning") and signals["reasoning"].search(text):
        return "reasoning"
    if signals.get("fast") and signals["fast"].search(text) and len(text) < 200:
        return "fast"
    return "default"


_COMPLEXITY_TO_CAPABILITY = {
    "simple": "fast",
    "medium": "default",
    "complex": "reasoning",
}


async def classify_smart(
    body: dict,
    config: AppConfig,
    client: httpx.AsyncClient,
) -> str:
    """Classify request: vision first (instant), then LLM complexity or regex fallback."""
    messages = body.get("messages", [])
    if has_image_content(messages):
        return "vision"

    if config.classifier_enabled:
        complexity = await classify_complexity(client, config, messages)
        cap = _COMPLEXITY_TO_CAPABILITY.get(complexity, "default")
        logger.info(f"[classify] LLM complexity={complexity} -> cap={cap}")
        return cap

    return classify_request(body, config.signals)


def pick_model(capability: str, routes: dict[str, list[str]], breaker: CircuitBreaker) -> str:
    """Pick the best available model for a capability, respecting circuit breaker."""
    candidates = routes.get(capability, routes.get("default", []))
    for model in candidates:
        if not breaker.is_open(model):
            return model
    default_candidates = routes.get("default", [])
    return default_candidates[0] if default_candidates else "hub:gemini"


def get_fallback_candidates(
    capability: str,
    routes: dict[str, list[str]],
    breaker: CircuitBreaker,
    exclude: set[str] | None = None,
) -> list[str]:
    """Get ordered list of fallback candidates, skipping open circuits and exclusions."""
    exclude = exclude or set()
    candidates = list(routes.get(capability, []))
    # Also append default candidates as last resort
    if capability != "default":
        for m in routes.get("default", []):
            if m not in candidates:
                candidates.append(m)
    return [m for m in candidates if not breaker.is_open(m) and m not in exclude]


def parse_model(prefixed_model: str, hub_names: set[str] | None = None) -> tuple[str, str]:
    """Parse 'ark:model' / 'cli:model' / 'hub:model' prefix.

    Returns (upstream_type, model_id).
    """
    if prefixed_model.startswith("ark:"):
        return "ark", prefixed_model[4:]
    elif prefixed_model.startswith("cli:"):
        return "cli", prefixed_model[4:]
    elif prefixed_model.startswith("hub:"):
        return "hub", prefixed_model[4:]

    # No prefix - guess
    cli_prefixes = ("claude-", "gpt-", "claude_", "gpt_")
    if any(prefixed_model.startswith(p) for p in cli_prefixes):
        return "cli", prefixed_model
    if hub_names and prefixed_model in hub_names:
        return "hub", prefixed_model
    return "ark", prefixed_model


async def resolve_target(
    body: dict,
    config: AppConfig,
    breaker: CircuitBreaker,
    client: httpx.AsyncClient,
) -> tuple[str, str, str, str]:
    """Full routing: returns (upstream_type, model_id, prefixed_target, capability).

    The prefixed_target is the canonical 'type:model' string for breaker tracking.
    capability is returned so server.py can get fallback candidates.
    """
    requested_model = body.get("model", "")
    capability = await classify_smart(body, config, client)
    target_model = pick_model(capability, config.routes, breaker)

    if requested_model and requested_model not in ("auto", "router", "claw-router"):
        clean = requested_model.replace("volcengine-plan/", "").removeprefix("openai/")
        clean = config.aliases.get(clean, clean)
        if clean in config.no_vision and has_image_content(body.get("messages", [])):
            target_model = pick_model("vision", config.routes, breaker)
        else:
            target_model = clean
            capability = ""  # explicit model, no fallback chain

    upstream, model_id = parse_model(target_model, set(config.hubs.keys()))
    return upstream, model_id, target_model, capability
