"""CLI interface for claw-router."""

from __future__ import annotations

import click


@click.group()
def cli():
    """Claw Router - Intelligent LLM API Router"""
    pass


@cli.command()
@click.option("--port", default=3456, help="Listen port")
@click.option("--host", default="0.0.0.0", help="Listen host")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(port: int, host: str, reload: bool):
    """Start the router server."""
    import logging
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    uvicorn.run(
        "claw_router.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


@cli.command()
def status():
    """Show route table and breaker status."""
    from claw_router.config import load_config
    cfg = load_config()
    for cap, models in cfg.routes.items():
        click.echo(f"\n  {cap}:")
        for m in models:
            click.echo(f"    {m}")
    click.echo(f"\n  Aliases: {len(cfg.aliases)}")
    click.echo(f"  Hubs: {len(cfg.hubs)}")


@cli.command()
def health():
    """Ping all hubs and check health."""
    import asyncio
    import httpx

    from claw_router.config import load_config

    cfg = load_config()

    async def _check():
        async with httpx.AsyncClient(timeout=10) as client:
            for name, hub in cfg.hubs.items():
                try:
                    base = hub.base.rstrip("/")
                    models_url = f"{base}/models" if base.endswith(("/v1", "/v3")) else f"{base}/v1/models"
                    resp = await client.get(models_url)
                    click.echo(f"  {name}: {resp.status_code} ({hub.base})")
                except Exception as e:
                    click.echo(f"  {name}: FAIL - {e}")

    asyncio.run(_check())


@cli.command()
def deploy():
    """Deploy to production server (set DEPLOY_TARGET env var)."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent.parent / "deploy" / "deploy.sh"
    if not script.exists():
        click.echo(f"Deploy script not found: {script}")
        sys.exit(1)
    subprocess.run(["bash", str(script)], check=True)


def _admin_call(method: str, path: str, token: str, json_data: dict | None = None):
    """Call admin API endpoint."""
    import httpx

    url = f"http://localhost:3456{path}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = httpx.request(method, url, headers=headers, json=json_data, timeout=10)
    click.echo(resp.json())
    return resp


@cli.command("add-hub")
@click.argument("name")
@click.option("--base", required=True, help="Hub base URL")
@click.option("--model", required=True, help="Model name")
@click.option("--auth", default="", help="Auth token")
@click.option("--admin-token", envvar="ADMIN_TOKEN", required=True, help="Admin token")
def add_hub(name: str, base: str, model: str, auth: str, admin_token: str):
    """Add a hub at runtime."""
    _admin_call("POST", "/admin/hubs", admin_token, {"name": name, "base": base, "model": model, "auth": auth})


@cli.command("remove-hub")
@click.argument("name")
@click.option("--admin-token", envvar="ADMIN_TOKEN", required=True, help="Admin token")
def remove_hub(name: str, admin_token: str):
    """Remove a hub at runtime."""
    _admin_call("DELETE", f"/admin/hubs/{name}", admin_token)


@cli.command("reload")
@click.option("--admin-token", envvar="ADMIN_TOKEN", required=True, help="Admin token")
def reload(admin_token: str):
    """Reload config from disk."""
    _admin_call("POST", "/admin/reload", admin_token)
