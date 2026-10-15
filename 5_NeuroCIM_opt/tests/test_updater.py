from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurocim_opt.updater import GitHubReleaseClient, ReleaseAsset, UpdateError, installed_flavor, is_newer


def test_version_comparison() -> None:
    assert is_newer("v1.1.0", "1.0.9")
    assert is_newer("2.0", "1.99.99")
    assert not is_newer("v1.0.0", "1.0.0")
    assert not is_newer("1.0.0", "1.0.1")


def test_installed_flavor_from_manifest(tmp_path: Path) -> None:
    (tmp_path / "release_manifest.json").write_text(
        json.dumps({"flavor": "cpu"}), encoding="utf-8"
    )
    assert installed_flavor(tmp_path) == "cpu"


def test_latest_update_selects_matching_flavor(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GitHubReleaseClient(token="test-token")
    release = {
        "tag_name": "v1.2.0",
        "body": "notes",
        "assets": [
            {"name": "update-manifest.json", "url": "manifest", "size": 10},
            {"name": "NeuroCIM-Setup-CPU-1.2.0.exe", "url": "cpu", "size": 20},
        ],
    }
    manifest = {
        "flavors": {
            "cpu": {
                "asset": "NeuroCIM-Setup-CPU-1.2.0.exe",
                "sha256": "a" * 64,
            }
        }
    }

    def fake_request(url: str, **_kwargs: object) -> bytes:
        return json.dumps(release).encode()

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr(client, "download_bytes", lambda _asset: json.dumps(manifest).encode())
    update = client.latest_update(flavor="cpu", current="1.0.0")
    assert update is not None
    assert update.version == "1.2.0"
    assert update.installer == ReleaseAsset("NeuroCIM-Setup-CPU-1.2.0.exe", "cpu", 20)
    assert update.files == ((update.installer, "a" * 64),)


def test_latest_update_requires_login() -> None:
    client = GitHubReleaseClient(token=None)
    client.token = None
    with pytest.raises(UpdateError, match="login"):
        client.latest_update(flavor="cpu")
