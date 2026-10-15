"""Authenticated GitHub Release updater for the packaged Windows application."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .version import __version__
from .release_config import GITHUB_DISTRIBUTION_REPOSITORY, GITHUB_OAUTH_CLIENT_ID, GITHUB_OWNER


GITHUB_API = "https://api.github.com"
GITHUB_LOGIN = "https://github.com/login"
DEFAULT_OWNER = GITHUB_OWNER
DEFAULT_REPOSITORY = GITHUB_DISTRIBUTION_REPOSITORY
CREDENTIAL_SERVICE = "NeuroCIM-GitHub"
CREDENTIAL_USER = "release-access-token"


class UpdateError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    api_url: str
    size: int


@dataclass(frozen=True)
class AvailableUpdate:
    version: str
    notes: str
    installer: ReleaseAsset
    sha256: str
    flavor: str
    files: tuple[tuple[ReleaseAsset, str], ...] = ()


def _version_tuple(value: str) -> tuple[int, ...]:
    clean = value.strip().lower().lstrip("v").split("+", 1)[0].split("-", 1)[0]
    try:
        return tuple(int(part) for part in clean.split("."))
    except ValueError as exc:
        raise UpdateError(f"Unsupported release version: {value!r}") from exc


def is_newer(candidate: str, current: str = __version__) -> bool:
    left = _version_tuple(candidate)
    right = _version_tuple(current)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


def installed_flavor(platform_root: Path) -> str:
    candidates = (
        platform_root / "release_manifest.json",
        platform_root.parent / "release_manifest.json",
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        flavor = str(payload.get("flavor", "")).strip().lower()
        if flavor in {"cpu", "gpu"}:
            return flavor
    return "gpu" if _nvidia_available() else "cpu"


def _nvidia_available() -> bool:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip())


class CredentialStore:
    """Keep the GitHub token in Windows Credential Manager through keyring."""

    @staticmethod
    def load() -> str | None:
        try:
            import keyring

            return keyring.get_password(CREDENTIAL_SERVICE, CREDENTIAL_USER)
        except Exception:
            return None

    @staticmethod
    def save(token: str) -> None:
        try:
            import keyring

            keyring.set_password(CREDENTIAL_SERVICE, CREDENTIAL_USER, token)
        except Exception as exc:
            raise UpdateError("Cannot save GitHub login in Windows Credential Manager") from exc

    @staticmethod
    def clear() -> None:
        try:
            import keyring

            keyring.delete_password(CREDENTIAL_SERVICE, CREDENTIAL_USER)
        except Exception:
            pass


class GitHubReleaseClient:
    def __init__(
        self,
        *,
        owner: str = DEFAULT_OWNER,
        repository: str = DEFAULT_REPOSITORY,
        client_id: str | None = None,
        token: str | None = None,
    ) -> None:
        self.owner = owner
        self.repository = repository
        self.client_id = client_id or os.environ.get(
            "NEUROCIM_GITHUB_CLIENT_ID", GITHUB_OAUTH_CLIENT_ID
        )
        self.token = token or CredentialStore.load()

    @staticmethod
    def _request(
        url: str,
        *,
        method: str = "GET",
        token: str | None = None,
        data: bytes | None = None,
        accept: str = "application/vnd.github+json",
        timeout: float = 30,
    ) -> bytes:
        headers = {
            "Accept": accept,
            "User-Agent": f"NeuroCIM/{__version__}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code in {401, 403, 404}:
                raise UpdateError(
                    "GitHub账号尚未获准访问NeuroCIM发布仓库，或登录凭据已经失效。"
                ) from exc
            raise UpdateError(f"GitHub request failed: HTTP {exc.code}") from exc
        except OSError as exc:
            raise UpdateError(f"Cannot reach GitHub: {exc}") from exc

    def begin_device_login(self) -> DeviceAuthorization:
        if not self.client_id:
            raise UpdateError("The release build is missing its GitHub OAuth client ID")
        payload = urlencode({"client_id": self.client_id, "scope": "repo read:user"}).encode()
        raw = self._request(
            f"{GITHUB_LOGIN}/device/code",
            method="POST",
            data=payload,
            accept="application/json",
        )
        value = json.loads(raw)
        return DeviceAuthorization(
            device_code=value["device_code"],
            user_code=value["user_code"],
            verification_uri=value["verification_uri"],
            expires_in=int(value["expires_in"]),
            interval=int(value.get("interval", 5)),
        )

    def finish_device_login(
        self,
        authorization: DeviceAuthorization,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        deadline = time.monotonic() + authorization.expires_in
        interval = authorization.interval
        while time.monotonic() < deadline:
            if cancelled and cancelled():
                raise UpdateError("GitHub login was cancelled")
            payload = urlencode(
                {
                    "client_id": self.client_id,
                    "device_code": authorization.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                }
            ).encode()
            raw = self._request(
                f"{GITHUB_LOGIN}/oauth/access_token",
                method="POST",
                data=payload,
                accept="application/json",
            )
            value = json.loads(raw)
            if value.get("access_token"):
                self.token = str(value["access_token"])
                CredentialStore.save(self.token)
                return self.token
            error = value.get("error")
            if error == "slow_down":
                interval += 5
            elif error not in {"authorization_pending", "slow_down"}:
                raise UpdateError(str(value.get("error_description", error or "GitHub login failed")))
            time.sleep(interval)
        raise UpdateError("GitHub login code expired")

    def latest_update(self, *, flavor: str, current: str = __version__) -> AvailableUpdate | None:
        if not self.token:
            raise UpdateError("GitHub login is required before checking private updates")
        raw = self._request(
            f"{GITHUB_API}/repos/{self.owner}/{self.repository}/releases/latest",
            token=self.token,
        )
        release = json.loads(raw)
        version = str(release.get("tag_name", "")).lstrip("v")
        if not is_newer(version, current):
            return None
        assets = {
            str(item["name"]): ReleaseAsset(
                name=str(item["name"]),
                api_url=str(item["url"]),
                size=int(item.get("size", 0)),
            )
            for item in release.get("assets", [])
        }
        manifest_asset = assets.get("update-manifest.json")
        if manifest_asset is None:
            raise UpdateError("Release does not contain update-manifest.json")
        manifest = json.loads(self.download_bytes(manifest_asset))
        record = manifest.get("flavors", {}).get(flavor)
        if not isinstance(record, dict):
            raise UpdateError(f"Release does not provide a {flavor.upper()} installer")
        file_records = record.get("files")
        if not isinstance(file_records, list):
            file_records = [{"asset": record.get("asset"), "sha256": record.get("sha256")}]
        selected: list[tuple[ReleaseAsset, str]] = []
        for item in file_records:
            if not isinstance(item, dict):
                raise UpdateError("Release file record is invalid")
            asset = assets.get(str(item.get("asset", "")))
            if asset is None:
                raise UpdateError(f"Release file is missing: {item.get('asset')}")
            checksum = str(item.get("sha256", "")).lower()
            if len(checksum) != 64:
                raise UpdateError(f"Release checksum is invalid: {asset.name}")
            selected.append((asset, checksum))
        installer_pair = next(
            ((asset, checksum) for asset, checksum in selected if asset.name.lower().endswith(".exe")),
            None,
        )
        if installer_pair is None:
            raise UpdateError("Release does not contain an installer executable")
        installer, checksum = installer_pair
        return AvailableUpdate(
            version=version,
            notes=str(release.get("body", "")),
            installer=installer,
            sha256=checksum,
            flavor=flavor,
            files=tuple(selected),
        )

    def download_bytes(self, asset: ReleaseAsset) -> bytes:
        if not self.token:
            raise UpdateError("GitHub login is required")
        return self._request(
            asset.api_url,
            token=self.token,
            accept="application/octet-stream",
            timeout=120,
        )

    def download_installer(
        self,
        update: AvailableUpdate,
        destination: Path,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> Path:
        if not self.token:
            raise UpdateError("GitHub login is required")
        destination = destination.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        request = Request(
            update.installer.api_url,
            headers={
                "Accept": "application/octet-stream",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": f"NeuroCIM/{__version__}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        digest = hashlib.sha256()
        completed = 0
        try:
            with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                total = int(response.headers.get("Content-Length", update.installer.size or 0))
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    digest.update(chunk)
                    completed += len(chunk)
                    if progress:
                        progress(completed, total)
            if digest.hexdigest().lower() != update.sha256:
                raise UpdateError("Downloaded installer failed SHA-256 verification")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def download_update_package(
        self,
        update: AvailableUpdate,
        directory: Path,
        *,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> Path:
        """Download an Inno Setup executable and all adjacent disk slices."""

        directory = directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        files = update.files or ((update.installer, update.sha256),)
        installer_path: Path | None = None
        for asset, checksum in files:
            current = AvailableUpdate(
                version=update.version,
                notes=update.notes,
                installer=asset,
                sha256=checksum,
                flavor=update.flavor,
            )
            destination = directory / asset.name
            self.download_installer(
                current,
                destination,
                progress=(
                    (lambda done, total, name=asset.name: progress(name, done, total))
                    if progress
                    else None
                ),
            )
            if asset.name.lower().endswith(".exe"):
                installer_path = destination
        if installer_path is None:
            raise UpdateError("Downloaded update package has no installer executable")
        return installer_path


def launch_installer(path: Path) -> None:
    subprocess.Popen(
        [
            str(path.resolve()),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/CLOSEAPPLICATIONS",
            "/RESTARTAPPLICATIONS",
        ],
        close_fds=True,
    )

