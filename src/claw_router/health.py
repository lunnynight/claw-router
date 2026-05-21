"""Background health checker for upstream endpoints."""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from claw_router.config import AppConfig

logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self, config: AppConfig, interval: int = 30):
        self.config = config
        self.interval = interval
        self.status: dict[str, dict] = {}
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=10)
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()

    async def _loop(self) -> None:
        while True:
            await self._check_all()
            await asyncio.sleep(self.interval)

    async def _check_all(self) -> None:
        client = self._client
        if client is None:
            return
        for name, hub in self.config.hubs.items():
            base = hub.base.rstrip("/")
            if base.endswith(("/v1", "/v3")):
                await self._ping(client, name, f"{base}/models")
            else:
                await self._ping(client, name, f"{base}/v1/models")
        for name, upstream in self.config.upstreams.items():
            await self._ping(client, f"upstream:{name}", upstream.base, follow=True)

    async def _ping(self, client: httpx.AsyncClient, key: str, url: str,
                    follow: bool = False, threshold: int = 400) -> None:
        start = time.monotonic()
        try:
            resp = await client.get(url, follow_redirects=follow)
            latency = (time.monotonic() - start) * 1000
            self.status[key] = {
                "ok": resp.status_code < threshold,
                "status_code": resp.status_code,
                "latency_ms": round(latency, 1),
                "checked_at": time.time(),
            }
        except Exception as e:
            self.status[key] = {
                "ok": False,
                "error": str(e),
                "latency_ms": -1,
                "checked_at": time.time(),
            }

    def get_status(self) -> dict:
        return dict(self.status)
