"""Configuration loading from YAML + .env"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import threading

import yaml
from dotenv import load_dotenv


@dataclass
class UpstreamConfig:
    name: str
    base: str
    protocol: str  # "openai" or "anthropic"
    auth: str
    timeout: int = 120


@dataclass
class HubConfig:
    name: str
    base: str
    model: str
    auth: str
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    upstreams: dict[str, UpstreamConfig] = field(default_factory=dict)
    hubs: dict[str, HubConfig] = field(default_factory=dict)
    routes: dict[str, list[str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    no_vision: set[str] = field(default_factory=set)
    signals: dict[str, re.Pattern] = field(default_factory=dict)
    classifier_enabled: bool = field(default=False)
    classifier_model: str = field(default="doubao-seed-2.0-lite")
    classifier_timeout: float = field(default=2.0)
    fallback_max_retries: int = field(default=3)
    _config_dir: Path | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_hub(self, name: str, base: str, model: str, auth: str = "") -> None:
        with self._lock:
            self.hubs[name] = HubConfig(name=name, base=base, model=model, auth=auth)
            self._save_hubs()

    def remove_hub(self, name: str) -> bool:
        with self._lock:
            if name not in self.hubs:
                return False
            del self.hubs[name]
            self._save_hubs()
            return True

    def update_hub(self, name: str, **kwargs) -> bool:
        with self._lock:
            if name not in self.hubs:
                return False
            hub = self.hubs[name]
            for k, v in kwargs.items():
                if k in ("base", "model", "auth") and v is not None:
                    setattr(hub, k, v)
            self._save_hubs()
            return True

    def add_route(self, cap: str, model: str) -> None:
        with self._lock:
            if cap not in self.routes:
                self.routes[cap] = []
            if model not in self.routes[cap]:
                self.routes[cap].append(model)
                self._save_routes()

    def remove_route(self, cap: str, model: str) -> bool:
        with self._lock:
            if cap not in self.routes or model not in self.routes[cap]:
                return False
            self.routes[cap].remove(model)
            self._save_routes()
            return True

    def _save_hubs(self) -> None:
        if not self._config_dir:
            return
        data: dict = {}
        if self.upstreams:
            data["upstreams"] = {
                n: {"base": u.base, "protocol": u.protocol, "auth": u.auth, "timeout": u.timeout}
                for n, u in self.upstreams.items()
            }
        if self.hubs:
            data["hubs"] = {
                n: {"base": h.base, "model": h.model, "auth": h.auth}
                for n, h in self.hubs.items()
            }
        (self._config_dir / "hubs.yaml").write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8"
        )

    def _save_routes(self) -> None:
        if not self._config_dir:
            return
        data: dict = {}
        if self.routes:
            data["routes"] = self.routes
        if self.aliases:
            data["aliases"] = self.aliases
        if self.no_vision:
            data["no_vision"] = sorted(self.no_vision)
        if self.signals:
            data["signals"] = {n: p.pattern for n, p in self.signals.items()}
        (self._config_dir / "routes.yaml").write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8"
        )


def _resolve_env(value: str) -> str:
    """Replace ${VAR} or ${VAR:-default} with environment variable value."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r'\$\{(\w+)(?::-([^}]*))?\}', value)
    if match:
        return os.getenv(match.group(1), match.group(2) or "")
    return value


def _find_config_dir() -> Path:
    """Find config directory, checking common locations."""
    candidates = [
        Path.cwd() / "config",
        Path(__file__).resolve().parent.parent.parent / "config",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def load_config(config_dir: Path | None = None, env_file: Path | None = None) -> AppConfig:
    """Load full configuration from YAML files and .env."""
    if config_dir is None:
        config_dir = _find_config_dir()

    # Load .env
    if env_file is None:
        for candidate in [config_dir.parent / ".env", config_dir / ".env", Path.cwd() / ".env"]:
            if candidate.exists():
                env_file = candidate
                break
    if env_file and env_file.exists():
        load_dotenv(env_file)

    cfg = AppConfig()

    # Load hubs.yaml
    hubs_path = config_dir / "hubs.yaml"
    if hubs_path.exists():
        raw = yaml.safe_load(hubs_path.read_text(encoding="utf-8"))

        for name, u in (raw.get("upstreams") or {}).items():
            cfg.upstreams[name] = UpstreamConfig(
                name=name,
                base=u["base"],
                protocol=u.get("protocol", "openai"),
                auth=_resolve_env(u.get("auth", "")),
                timeout=u.get("timeout", 120),
            )

        for name, h in (raw.get("hubs") or {}).items():
            cfg.hubs[name] = HubConfig(
                name=name,
                base=h["base"],
                model=h["model"],
                auth=_resolve_env(h.get("auth", "")),
                extra_headers={k: _resolve_env(v) for k, v in (h.get("extra_headers") or {}).items()},
            )

    # Load routes.yaml
    routes_path = config_dir / "routes.yaml"
    if routes_path.exists():
        raw = yaml.safe_load(routes_path.read_text(encoding="utf-8"))

        cfg.routes = raw.get("routes", {})
        cfg.aliases = raw.get("aliases", {})
        cfg.no_vision = set(raw.get("no_vision", []))

        for name, pattern in (raw.get("signals") or {}).items():
            cfg.signals[name] = re.compile(pattern, re.IGNORECASE)

        classifier = raw.get("classifier") or {}
        cfg.classifier_enabled = classifier.get("enabled", False)
        cfg.classifier_model = classifier.get("model", "doubao-seed-2.0-lite")
        cfg.classifier_timeout = classifier.get("timeout", 2.0)
        cfg.fallback_max_retries = classifier.get("fallback_max_retries", 3)

    cfg._config_dir = config_dir
    return cfg
