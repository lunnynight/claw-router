"""FastAPI application with monitoring, logging, and security."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from claw_router import __version__
from claw_router.breaker import CircuitBreaker
from claw_router.config import AppConfig, load_config
from claw_router.dashboard import render_dashboard
from claw_router.health import HealthChecker
from claw_router.logging_config import setup_logging
from claw_router.router import (
    get_fallback_candidates,
    parse_model,
    resolve_target,
)
from claw_router.upstream import UpstreamError, call_upstream

# 初始化日志
setup_logging(json_logs=True, log_level="INFO")
log = structlog.get_logger()

# 安全配置
security = HTTPBearer(auto_error=False)

# Global state
_config: AppConfig | None = None
_breaker = CircuitBreaker()
_health: HealthChecker | None = None
_client: httpx.AsyncClient | None = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify API key from Authorization header."""
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        # 如果未配置 API keys，跳过验证
        return "anonymous"

    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials not in api_keys:
        log.warning("invalid_api_key", provided_key=credentials.credentials[:8] + "...")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""

    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_size:
                log.warning("request_too_large", size=content_length)
                return JSONResponse(
                    status_code=413, content={"error": {"message": "Request too large"}}
                )
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _health
    cfg = get_config()
    _client = httpx.AsyncClient()
    _health = HealthChecker(cfg)
    _health.start()
    log.info("claw_router_started", version=__version__)
    log.info(
        "classifier_config",
        enabled=cfg.classifier_enabled,
        model=cfg.classifier_model,
        fallback_retries=cfg.fallback_max_retries,
    )
    for cap, ms in cfg.routes.items():
        log.info("route_configured", capability=cap, models=" > ".join(ms))
    yield
    log.info("claw_router_stopping")
    await _health.stop()
    await _client.aclose()


app = FastAPI(title="Claw Router", version=__version__, lifespan=lifespan)

# CORS 配置
cors_origins = os.getenv("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins.split(",") if cors_origins != "*" else ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# 请求大小限制
app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)

# Prometheus 监控
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# 速率限制
rate_limit = os.getenv("RATE_LIMIT_PER_MINUTE", "100")
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{rate_limit}/minute"],
    storage_uri="memory://",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/v1/chat/completions")
@limiter.limit(f"{rate_limit}/minute")
async def chat_completions(request: Request, api_key: str = Depends(verify_api_key)):
    # 生成 request_id
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(
        request_id=request_id, api_key=api_key[:8] + "..." if api_key != "anonymous" else "anonymous"
    )

    cfg = get_config()

    try:
        body = await request.json()
    except Exception:
        log.error("invalid_json_request")
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON"}})

    stream = body.get("stream", False)
    upstream, model_id, target_model, capability = await resolve_target(body, cfg, _breaker, _client)

    log.info(
        "request_received",
        method="POST",
        path="/v1/chat/completions",
        requested_model=body.get("model", "none"),
        target_model=target_model,
        upstream=upstream,
        capability=capability,
        stream=stream,
    )

    start_time = time.time()
    tried: set[str] = set()
    last_error: Exception | None = None

    for attempt in range(cfg.fallback_max_retries + 1):
        try:
            result = await call_upstream(
                _client, cfg, upstream, model_id, body, stream, target_model, _breaker
            )
        except UpstreamError as e:
            tried.add(target_model)
            last_error = e
            log.warning(
                "upstream_failed",
                target=target_model,
                status=e.status_code,
                attempt=attempt + 1,
                max_attempts=cfg.fallback_max_retries + 1,
                error=str(e),
            )
            if not capability:
                break
            candidates = get_fallback_candidates(capability, cfg.routes, _breaker, tried)
            if not candidates:
                break
            target_model = candidates[0]
            upstream, model_id = parse_model(target_model, set(cfg.hubs.keys()))
            log.info("fallback_attempt", upstream=upstream, model=model_id)
            continue
        except ValueError as e:
            duration = time.time() - start_time
            log.error(
                "validation_error",
                status=400,
                duration_ms=int(duration * 1000),
                error=str(e),
            )
            return JSONResponse(status_code=400, content={"error": {"message": str(e)}})
        except Exception as e:
            tried.add(target_model)
            last_error = e
            log.error("upstream_unexpected_error", error=str(e), target=target_model)
            if not capability:
                break
            candidates = get_fallback_candidates(capability, cfg.routes, _breaker, tried)
            if not candidates:
                break
            target_model = candidates[0]
            upstream, model_id = parse_model(target_model, set(cfg.hubs.keys()))
            log.info("fallback_attempt", upstream=upstream, model=model_id)
            continue

        duration = time.time() - start_time
        log.info(
            "request_completed",
            status=200,
            duration_ms=int(duration * 1000),
            upstream=upstream,
            model=model_id,
            attempts=attempt + 1,
        )

        if stream:
            from starlette.responses import StreamingResponse

            async def event_generator():
                async for chunk in result:
                    if isinstance(chunk, str):
                        yield chunk.encode()
                    else:
                        yield chunk

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            return JSONResponse(content=result)

    duration = time.time() - start_time
    log.error(
        "all_attempts_exhausted",
        duration_ms=int(duration * 1000),
        tried=sorted(tried),
        error=str(last_error) if last_error else None,
    )
    if isinstance(last_error, UpstreamError):
        return JSONResponse(status_code=last_error.status_code, content=last_error.body)
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": f"All models exhausted ({len(tried)} tried: {', '.join(sorted(tried))})",
                "type": "proxy_error",
            }
        },
    )



@app.get("/v1/models")
@app.get("/models")
async def list_models():
    cfg = get_config()
    models = []
    seen = set()

    for cap, ms in cfg.routes.items():
        for m in ms:
            if m not in seen:
                seen.add(m)
                upstream, model_id = parse_model(m, set(cfg.hubs.keys()))
                models.append(
                    {
                        "id": m,
                        "object": "model",
                        "owned_by": {
                            "ark": "volcengine",
                            "cli": "cliproxy",
                            "hub": "free-llm-hub",
                        }.get(upstream, upstream),
                        "upstream": upstream,
                        "model_id": model_id,
                        "capability": cap,
                        "vision": m not in cfg.no_vision,
                        "circuit_open": _breaker.is_open(m),
                    }
                )

    for hub_name in cfg.hubs:
        hub_id = f"hub:{hub_name}"
        if hub_id not in seen:
            seen.add(hub_id)
            models.append(
                {
                    "id": hub_id,
                    "object": "model",
                    "owned_by": "free-llm-hub",
                    "upstream": "hub",
                    "model_id": cfg.hubs[hub_name].model,
                    "capability": "manual",
                    "circuit_open": _breaker.is_open(hub_id),
                }
            )

    return {"object": "list", "data": models}


@app.get("/status")
async def status():
    cfg = get_config()
    result = {
        "classifier": {
            "enabled": cfg.classifier_enabled,
            "model": cfg.classifier_model,
        },
        "routes": {},
        "hub": {},
    }
    for cap, ms in cfg.routes.items():
        result["routes"][cap] = {m: "OPEN" if _breaker.is_open(m) else "ok" for m in ms}
    for hub_name, hub_cfg in cfg.hubs.items():
        hub_id = f"hub:{hub_name}"
        result["hub"][hub_name] = {
            "base": hub_cfg.base,
            "model": hub_cfg.model,
            "circuit": "OPEN" if _breaker.is_open(hub_id) else "ok",
        }
    return result


@app.get("/health")
async def health():
    health_data = _health.get_status() if _health else {}
    return {
        "ok": True,
        "service": "claw-router",
        "version": __version__,
        "endpoints": health_data,
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    cfg = get_config()
    return render_dashboard(cfg, _breaker, _health)


# --- Admin API ---

_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _check_admin(authorization: str | None) -> JSONResponse | None:
    if not _ADMIN_TOKEN:
        return JSONResponse(status_code=403, content={"error": "ADMIN_TOKEN not configured"})
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Missing Bearer token"})
    if authorization[7:] != _ADMIN_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Invalid token"})
    return None


@app.post("/admin/hubs", status_code=201)
async def admin_add_hub(request: Request, authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    body = await request.json()
    name = body.get("name")
    if not name or not body.get("base") or not body.get("model"):
        return JSONResponse(status_code=400, content={"error": "name, base, model required"})
    cfg = get_config()
    cfg.add_hub(name, body["base"], body["model"], body.get("auth", ""))
    return {"ok": True, "hub": name}


@app.delete("/admin/hubs/{name}")
async def admin_remove_hub(name: str, authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    cfg = get_config()
    if not cfg.remove_hub(name):
        return JSONResponse(status_code=404, content={"error": f"Hub '{name}' not found"})
    return {"ok": True, "removed": name}


@app.patch("/admin/hubs/{name}")
async def admin_update_hub(name: str, request: Request, authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    body = await request.json()
    cfg = get_config()
    if not cfg.update_hub(name, **{k: body.get(k) for k in ("base", "model", "auth")}):
        return JSONResponse(status_code=404, content={"error": f"Hub '{name}' not found"})
    return {"ok": True, "updated": name}


@app.post("/admin/routes/{cap}")
async def admin_add_route(cap: str, request: Request, authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    body = await request.json()
    model = body.get("model")
    if not model:
        return JSONResponse(status_code=400, content={"error": "model required"})
    cfg = get_config()
    cfg.add_route(cap, model)
    return {"ok": True, "cap": cap, "model": model}


@app.delete("/admin/routes/{cap}/{model:path}")
async def admin_remove_route(cap: str, model: str, authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    cfg = get_config()
    if not cfg.remove_route(cap, model):
        return JSONResponse(status_code=404, content={"error": f"Route '{model}' not in '{cap}'"})
    return {"ok": True, "removed": model, "cap": cap}


@app.post("/admin/reload")
async def admin_reload(authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    global _config
    _config = load_config()
    return {"ok": True, "message": "Config reloaded from disk"}


@app.get("/admin/config")
async def admin_get_config(authorization: str | None = Header(None)):
    if err := _check_admin(authorization):
        return err
    cfg = get_config()
    return {
        "upstreams": {n: {"base": u.base, "protocol": u.protocol} for n, u in cfg.upstreams.items()},
        "hubs": {n: {"base": h.base, "model": h.model} for n, h in cfg.hubs.items()},
        "routes": cfg.routes,
        "aliases": cfg.aliases,
        "classifier": {
            "enabled": cfg.classifier_enabled,
            "model": cfg.classifier_model,
            "timeout": cfg.classifier_timeout,
        },
    }
