"""Tests for admin API and config hot management."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from claw_router.config import AppConfig, HubConfig, load_config


# --- Config mutation tests ---


class TestConfigMutation:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_dir = Path(self.tmpdir) / "config"
        self.config_dir.mkdir()
        # Write minimal hubs.yaml
        (self.config_dir / "hubs.yaml").write_text(yaml.dump({
            "hubs": {"kimi": {"base": "http://localhost:8010", "model": "moonshot-v1", "auth": "tok"}}
        }))
        (self.config_dir / "routes.yaml").write_text(yaml.dump({
            "routes": {"default": ["hub:kimi"]},
            "aliases": {},
        }))
        self.cfg = load_config(config_dir=self.config_dir)

    def test_add_hub(self):
        self.cfg.add_hub("test", "http://localhost:9999", "test-model", "auth123")
        assert "test" in self.cfg.hubs
        assert self.cfg.hubs["test"].model == "test-model"
        # Verify persisted
        raw = yaml.safe_load((self.config_dir / "hubs.yaml").read_text())
        assert "test" in raw["hubs"]

    def test_remove_hub(self):
        assert self.cfg.remove_hub("kimi")
        assert "kimi" not in self.cfg.hubs
        assert not self.cfg.remove_hub("nonexistent")

    def test_update_hub(self):
        assert self.cfg.update_hub("kimi", base="http://new:8080")
        assert self.cfg.hubs["kimi"].base == "http://new:8080"
        assert self.cfg.hubs["kimi"].model == "moonshot-v1"  # unchanged
        assert not self.cfg.update_hub("nonexistent", base="x")

    def test_add_route(self):
        self.cfg.add_route("code", "hub:test")
        assert "hub:test" in self.cfg.routes["code"]
        # Duplicate should be no-op
        self.cfg.add_route("code", "hub:test")
        assert self.cfg.routes["code"].count("hub:test") == 1

    def test_remove_route(self):
        assert self.cfg.remove_route("default", "hub:kimi")
        assert "hub:kimi" not in self.cfg.routes["default"]
        assert not self.cfg.remove_route("default", "nonexistent")

    def test_add_route_new_cap(self):
        self.cfg.add_route("newcap", "hub:kimi")
        assert self.cfg.routes["newcap"] == ["hub:kimi"]
        raw = yaml.safe_load((self.config_dir / "routes.yaml").read_text())
        assert "newcap" in raw["routes"]

    def test_no_persist_without_config_dir(self):
        cfg = AppConfig()
        cfg.add_hub("x", "http://x", "m")  # Should not raise
        assert "x" in cfg.hubs


# --- Admin API tests ---


@pytest.fixture
def admin_client(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "hubs.yaml").write_text(yaml.dump({
        "hubs": {"kimi": {"base": "http://localhost:8010", "model": "moonshot-v1", "auth": ""}}
    }))
    (config_dir / "routes.yaml").write_text(yaml.dump({
        "routes": {"default": ["hub:kimi"]},
        "aliases": {},
    }))

    os.environ["ADMIN_TOKEN"] = "test-secret"
    import claw_router.server as srv
    srv._config = load_config(config_dir=config_dir)
    srv._ADMIN_TOKEN = "test-secret"

    from claw_router.server import app
    client = TestClient(app, raise_server_exceptions=False)
    yield client, srv
    os.environ.pop("ADMIN_TOKEN", None)


def _auth():
    return {"Authorization": "Bearer test-secret"}


class TestAdminAPI:
    def test_no_token_rejected(self, admin_client):
        client, _ = admin_client
        resp = client.get("/admin/config")
        assert resp.status_code == 401

    def test_wrong_token_rejected(self, admin_client):
        client, _ = admin_client
        resp = client.get("/admin/config", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_get_config(self, admin_client):
        client, _ = admin_client
        resp = client.get("/admin/config", headers=_auth())
        assert resp.status_code == 200
        assert "hubs" in resp.json()

    def test_add_hub(self, admin_client):
        client, _ = admin_client
        resp = client.post("/admin/hubs", headers=_auth(), json={
            "name": "new", "base": "http://new:8080", "model": "new-model"
        })
        assert resp.status_code == 201
        # Verify via config endpoint
        cfg_resp = client.get("/admin/config", headers=_auth())
        assert "new" in cfg_resp.json()["hubs"]

    def test_add_hub_missing_fields(self, admin_client):
        client, _ = admin_client
        resp = client.post("/admin/hubs", headers=_auth(), json={"name": "x"})
        assert resp.status_code == 400

    def test_remove_hub(self, admin_client):
        client, _ = admin_client
        resp = client.delete("/admin/hubs/kimi", headers=_auth())
        assert resp.status_code == 200
        resp = client.delete("/admin/hubs/kimi", headers=_auth())
        assert resp.status_code == 404

    def test_update_hub(self, admin_client):
        client, _ = admin_client
        resp = client.patch("/admin/hubs/kimi", headers=_auth(), json={"base": "http://new:9090"})
        assert resp.status_code == 200
        cfg = client.get("/admin/config", headers=_auth()).json()
        assert cfg["hubs"]["kimi"]["base"] == "http://new:9090"

    def test_add_remove_route(self, admin_client):
        client, _ = admin_client
        resp = client.post("/admin/routes/code", headers=_auth(), json={"model": "hub:test"})
        assert resp.status_code == 200
        resp = client.delete("/admin/routes/code/hub:test", headers=_auth())
        assert resp.status_code == 200
        resp = client.delete("/admin/routes/code/hub:test", headers=_auth())
        assert resp.status_code == 404

    def test_reload(self, admin_client):
        client, _ = admin_client
        resp = client.post("/admin/reload", headers=_auth())
        assert resp.status_code == 200

    def test_disabled_without_token(self, admin_client):
        client, srv = admin_client
        srv._ADMIN_TOKEN = ""
        resp = client.get("/admin/config", headers=_auth())
        assert resp.status_code == 403
