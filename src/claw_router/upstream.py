"""Unified upstream caller - replaces 6 duplicate handlers."""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncIterator

import httpx

from claw_router.breaker import CircuitBreaker
from claw_router.config import AppConfig
from claw_router.protocols import anthropic_to_openai, openai_to_anthropic, anthropic_sse_to_openai_sse

logger = logging.getLogger(__name__)


async def call_upstream(
    client: httpx.AsyncClient,
    config: AppConfig,
    upstream_type: str,
    model_id: str,
    body: dict,
    stream: bool,
    target_model: str,
    breaker: CircuitBreaker,
) -> dict | AsyncIterator[str]:
    """Unified upstream call. Returns dict for sync, async iterator of SSE strings for stream."""

    url, headers, payload = _prepare_request(config, upstream_type, model_id, body, stream)

    timeout = _get_timeout(config, upstream_type)

    if stream:
        return _stream_response(client, url, headers, payload, upstream_type, target_model, breaker, timeout)
    else:
        return await _sync_response(client, url, headers, payload, upstream_type, target_model, breaker, timeout)


def _prepare_request(
    config: AppConfig,
    upstream_type: str,
    model_id: str,
    body: dict,
    stream: bool,
) -> tuple[str, dict, bytes]:
    """Prepare URL, headers, and payload for the upstream request."""

    if upstream_type == "ark":
        upstream_cfg = config.upstreams["ark"]
        url = f"{upstream_cfg.base}/messages"
        anthropic_body = openai_to_anthropic(body, model_id)
        if stream:
            anthropic_body["stream"] = True
        headers = {
            "Content-Type": "application/json",
            "x-api-key": upstream_cfg.auth,
            "anthropic-version": "2023-06-01",
        }
        payload = json.dumps(anthropic_body).encode()

    elif upstream_type == "cli":
        upstream_cfg = config.upstreams["cliproxy"]
        url = f"{upstream_cfg.base}/v1/chat/completions"
        cli_body = dict(body)
        cli_body["model"] = model_id
        if stream:
            cli_body["stream"] = True
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {upstream_cfg.auth}",
        }
        payload = json.dumps(cli_body).encode()

    elif upstream_type == "hub":
        hub_cfg = config.hubs.get(model_id)
        if not hub_cfg:
            raise ValueError(f"Unknown hub: {model_id}")
        # 如果 base 已经带 /v3 或 /v1，不再重复加 /v1
        if hub_cfg.base.rstrip("/").endswith(("/v1", "/v3")):
            url = f"{hub_cfg.base.rstrip('/')}/chat/completions"
        else:
            url = f"{hub_cfg.base}/v1/chat/completions"
        hub_body = dict(body)
        hub_body["model"] = hub_cfg.model
        if stream:
            hub_body["stream"] = True
        headers = {"Content-Type": "application/json"}
        if hub_cfg.auth:
            headers["Authorization"] = f"Bearer {hub_cfg.auth}"
        if hub_cfg.extra_headers:
            headers.update(hub_cfg.extra_headers)
        payload = json.dumps(hub_body).encode()

    else:
        raise ValueError(f"Unknown upstream type: {upstream_type}")

    return url, headers, payload


def _get_timeout(config: AppConfig, upstream_type: str) -> float:
    key = {"ark": "ark", "cli": "cliproxy"}.get(upstream_type)
    if key and key in config.upstreams:
        return config.upstreams[key].timeout
    return 300 if upstream_type == "cli" else 120


async def _sync_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: bytes,
    upstream_type: str,
    target_model: str,
    breaker: CircuitBreaker,
    timeout: float,
) -> dict:
    """Handle sync (non-streaming) upstream call."""
    resp = await client.post(url, content=payload, headers=headers, timeout=timeout)

    if resp.status_code >= 400:
        logger.error(f"[upstream] {resp.status_code} from {target_model}: {resp.text[:300]}")
        breaker.record_failure(target_model)
        raise UpstreamError(resp.status_code, resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"error": {"message": resp.text}})

    breaker.record_success(target_model)

    if upstream_type == "ark":
        return anthropic_to_openai(resp.json(), target_model)
    else:
        return resp.json()


async def _stream_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: bytes,
    upstream_type: str,
    target_model: str,
    breaker: CircuitBreaker,
    timeout: float,
) -> AsyncIterator[str]:
    """Handle streaming upstream call. Yields SSE strings."""

    async def _generate():
        try:
            async with client.stream("POST", url, content=payload, headers=headers, timeout=timeout) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    logger.error(f"[upstream-stream] {resp.status_code}: {body[:200]}")
                    breaker.record_failure(target_model)
                    yield f"data: {json.dumps({'error': {'message': body.decode(errors='replace')[:500]}})}\n\n"
                    return

                if upstream_type == "ark":
                    # Anthropic SSE -> OpenAI SSE conversion
                    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                    buffer = b""
                    async for chunk in resp.aiter_bytes():
                        buffer += chunk
                        while b"\n\n" in buffer:
                            event_data, buffer = buffer.split(b"\n\n", 1)
                            lines = event_data.decode(errors="replace").strip().split("\n")
                            data_line = None
                            for line in lines:
                                if line.startswith("data: "):
                                    data_line = line[6:]

                            if data_line is not None:
                                sse = anthropic_sse_to_openai_sse(data_line, completion_id, target_model)
                                if sse:
                                    yield sse
                else:
                    # OpenAI-compatible: pass through SSE directly
                    async for chunk in resp.aiter_bytes():
                        yield chunk.decode(errors="replace")

                breaker.record_success(target_model)
        except httpx.HTTPStatusError as e:
            logger.error(f"[stream-error] {e}")
            breaker.record_failure(target_model)
        except Exception as e:
            logger.error(f"[stream-error] {e}")
            breaker.record_failure(target_model)

    return _generate()


class UpstreamError(Exception):
    """Raised when upstream returns an error response."""
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Upstream error {status_code}")
