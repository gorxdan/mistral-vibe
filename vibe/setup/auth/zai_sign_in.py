"""Interactive "Continue with Z.ai" browser sign-in for the GLM (zai) provider.

Z.ai's account login is not a token used at inference time: it is a
*bootstrapping* flow that provisions a durable coding-plan API key. We replicate
the flow ZCode (z.ai's desktop client) uses, then hand the resulting key to the
normal API-key persistence path so the rest of the stack (env-var bearer against
the coding endpoint) is unchanged.

The flow, in order:

1. Generate a 32-byte hex ``poll_token`` client-side.
2. ``POST zcode.z.ai/api/v1/oauth/cli/init`` (Bearer poll_token, body
   ``{"provider": "zai"}``) -> ``{flow_id, authorize_url, expires_at,
   poll_interval_sec}``.
3. Open ``authorize_url`` in the browser; the user authorizes their z.ai account.
4. Poll ``GET zcode.z.ai/api/v1/oauth/cli/poll/{flow_id}`` (Bearer poll_token)
   until ``status == "ready"``; read ``zai.access_token``.
5. Exchange that access token for a coding-plan API key (the "biz" dance):
   ``api/auth/z/login`` -> ``getCustomerInfo`` (pick the default org/project) ->
   find-or-create an api key -> copy its secret. The usable key is
   ``{apiKey}.{secret}``.

All endpoints and request shapes are reverse-engineered from the ZCode bundle
(the official z.ai npm packages are API-key configurators only and ship no OAuth
client). They are undocumented and may change without notice; keep this isolated
behind the opt-in zai login path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import secrets
from typing import Any, Final
import webbrowser

import httpx

from vibe.core.logger import logger
from vibe.core.utils.http import build_ssl_context

# --- ZCode OAuth constants (from the unpacked ZCode desktop bundle) -----------
ZCODE_OAUTH_BASE: Final = "https://zcode.z.ai/api/v1"
ZCODE_INIT_URL: Final = f"{ZCODE_OAUTH_BASE}/oauth/cli/init"
ZCODE_POLL_URL: Final = f"{ZCODE_OAUTH_BASE}/oauth/cli/poll"

# --- z.ai "biz" key-provisioning constants -----------------------------------
ZAI_BIZ_HOST: Final = "https://api.z.ai"
ZAI_LOGIN_URL: Final = f"{ZAI_BIZ_HOST}/api/auth/z/login"
# Name of the api key we find-or-create. Distinct from ZCode's own "zcode-api-key"
# so we never clobber a key the user provisioned through ZCode itself.
ZAI_API_KEY_NAME: Final = "chaton-api-key"
# The default org/project z.ai seeds every account with carry these (Chinese)
# names; ZCode prefers them, falling back to the first entry otherwise.
_DEFAULT_ORG_MARKER: Final = "默认机构"  # 默认机构
_DEFAULT_PROJECT_MARKER: Final = "默认项目"  # 默认项目

_HTTP_TIMEOUT: Final = 30.0
_MIN_POLL_INTERVAL: Final = 1.0
_HTTP_ERROR_STATUS: Final = 400
# Poll statuses that mean the flow lapsed or was rejected rather than a transient.
_POLL_LAPSED_STATUSES: Final = frozenset({400, 404, 408})
# Envelope ``code``/``status`` values that signal success across the biz APIs.
_OK_CODES: Final = frozenset({None, 0, 200, "0", "200"})

UrlCallback = Callable[[str], None]
NowFn = Callable[[], datetime]
BrowserOpener = Callable[[str], bool]


class ZaiSignInError(RuntimeError):
    """Raised when the interactive Z.ai sign-in cannot complete."""


@dataclass(frozen=True, slots=True)
class _InitResult:
    flow_id: str
    poll_token: str
    authorize_url: str
    expires_at: datetime
    poll_interval: float


def _unwrap_envelope(payload: Any, *, context: str) -> Any:
    """Validate and unwrap the z.ai ``{code, data, msg}`` response envelope.

    Success is ``code == 0``. Some biz endpoints use ``status`` and/or HTTP-style
    ``200`` instead, so those are accepted too. The unwrapped ``data`` may be a
    dict (account info, a created key) or a list (the api-key listing); both are
    returned as-is. Responses with no envelope are returned whole.
    """
    if not isinstance(payload, dict):
        raise ZaiSignInError(f"{context}: unexpected response shape.")
    code = payload.get("code", payload.get("status"))
    if code not in _OK_CODES:
        msg = payload.get("msg") or payload.get("message") or f"code={code}"
        raise ZaiSignInError(f"{context}: {msg}")
    return payload.get("data", payload)


def _generate_poll_token() -> str:
    """32 random bytes as 64 hex chars, matching ZCode's poll token."""
    return secrets.token_hex(32)


@dataclass
class ZaiSignInService:
    """Drives the full Z.ai sign-in and returns a usable coding-plan API key."""

    now: NowFn = field(default=lambda: datetime.now(UTC))
    poll_interval_override: float | None = None
    sleep: Callable[[float], Any] = field(default=asyncio.sleep)
    open_browser: BrowserOpener = field(default=webbrowser.open)

    async def authenticate(self, *, on_url: UrlCallback | None = None) -> str:
        """Run init -> browser -> poll -> biz dance; return ``{apiKey}.{secret}``."""
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, verify=build_ssl_context()
        ) as client:
            init = await self._init(client)
            if on_url is not None:
                on_url(init.authorize_url)
            self._open_browser(init.authorize_url)
            access_token = await self._poll_until_ready(client, init)
            return await self._provision_api_key(client, access_token)

    def _open_browser(self, url: str) -> None:
        try:
            self.open_browser(url)
        except Exception:
            logger.debug("Failed to open browser for Z.ai sign-in", exc_info=True)

    async def _init(self, client: httpx.AsyncClient) -> _InitResult:
        poll_token = _generate_poll_token()
        try:
            resp = await client.post(
                ZCODE_INIT_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {poll_token}",
                },
                json={"provider": "zai"},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ZaiSignInError(
                f"Sign-in init failed: HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc
        except httpx.RequestError as exc:
            raise ZaiSignInError(f"Sign-in init failed: {exc}") from exc

        data = _unwrap_envelope(resp.json(), context="Sign-in init")
        if not isinstance(data, dict):
            raise ZaiSignInError("Sign-in init returned an unexpected shape.")
        flow_id = data.get("flow_id")
        authorize_url = data.get("authorize_url")
        expires_at_raw = data.get("expires_at")
        interval_raw = data.get("poll_interval_sec")
        if not isinstance(flow_id, str) or not isinstance(authorize_url, str):
            raise ZaiSignInError("Sign-in init returned no flow id or authorize URL.")
        if not isinstance(expires_at_raw, (int, float)):
            raise ZaiSignInError("Sign-in init returned no expiry.")
        # The bundle treats poll_token from the response as authoritative when
        # present, but the client-generated one is what authorizes the poll.
        interval = (
            self.poll_interval_override
            if self.poll_interval_override is not None
            else float(interval_raw)
            if isinstance(interval_raw, (int, float))
            else 3.0
        )
        return _InitResult(
            flow_id=flow_id,
            poll_token=poll_token,
            authorize_url=authorize_url,
            expires_at=datetime.fromtimestamp(float(expires_at_raw), tz=UTC),
            poll_interval=max(_MIN_POLL_INTERVAL, interval),
        )

    async def _poll_until_ready(
        self, client: httpx.AsyncClient, init: _InitResult
    ) -> str:
        url = f"{ZCODE_POLL_URL}/{init.flow_id}"
        headers = {"Authorization": f"Bearer {init.poll_token}"}
        while self.now() < init.expires_at:
            await self.sleep(init.poll_interval)
            try:
                resp = await client.get(url, headers=headers)
            except httpx.RequestError as exc:
                raise ZaiSignInError(f"Sign-in poll failed: {exc}") from exc
            # 400/404/408 mean the flow lapsed or was rejected; treat as failure.
            if resp.status_code in _POLL_LAPSED_STATUSES:
                raise ZaiSignInError("Sign-in was not completed. Please retry.")
            if resp.status_code >= _HTTP_ERROR_STATUS:
                raise ZaiSignInError(
                    f"Sign-in poll failed: HTTP {resp.status_code}: {resp.text[:200]}"
                )
            data = _unwrap_envelope(resp.json(), context="Sign-in poll")
            status = data.get("status") if isinstance(data, dict) else None
            if status == "pending":
                continue
            if status == "failed":
                raise ZaiSignInError("Authorization was denied. Please retry.")
            if status == "ready":
                zai = data.get("zai")
                access_token = (
                    zai.get("access_token") if isinstance(zai, dict) else None
                ) or data.get("token")
                if not isinstance(access_token, str) or not access_token:
                    raise ZaiSignInError("Sign-in succeeded but returned no token.")
                return access_token
            raise ZaiSignInError(f"Sign-in returned an unknown state: {status!r}.")
        raise ZaiSignInError("Sign-in timed out. Please retry.")

    async def _provision_api_key(
        self, client: httpx.AsyncClient, access_token: str
    ) -> str:
        """Exchange the OAuth access token for a durable coding-plan API key."""
        biz_token = await self._biz_login(client, access_token)
        authorization = f"Bearer {biz_token}"
        org_id, project_id = await self._default_org_project(client, authorization)
        api_key = await self._find_or_create_api_key(
            client, authorization, org_id, project_id
        )
        secret = await self._copy_secret(
            client, authorization, org_id, project_id, api_key
        )
        return f"{api_key}.{secret}" if secret else api_key

    async def _biz_login(self, client: httpx.AsyncClient, access_token: str) -> str:
        try:
            resp = await client.post(ZAI_LOGIN_URL, json={"token": access_token})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ZaiSignInError(
                f"Could not exchange the sign-in token: {exc}"
            ) from exc
        body = resp.json()
        token = None
        if isinstance(body, dict):
            token = (
                body.get("access_token")
                or body.get("accessToken")
                or (body.get("data") or {}).get("access_token")
            )
        if not isinstance(token, str) or not token:
            raise ZaiSignInError("Token exchange returned no business token.")
        return token

    async def _biz_get(
        self, client: httpx.AsyncClient, url: str, authorization: str, *, context: str
    ) -> Any:
        try:
            resp = await client.get(url, headers={"Authorization": authorization})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ZaiSignInError(f"{context}: {exc}") from exc
        return _unwrap_envelope(resp.json(), context=context)

    async def _default_org_project(
        self, client: httpx.AsyncClient, authorization: str
    ) -> tuple[str, str]:
        data = await self._biz_get(
            client,
            f"{ZAI_BIZ_HOST}/api/biz/customer/getCustomerInfo",
            authorization,
            context="Could not load your account",
        )
        if not isinstance(data, dict):
            raise ZaiSignInError("Account info had an unexpected shape.")
        orgs = data.get("organizations") or data.get("orgs") or []
        if not isinstance(orgs, list) or not orgs:
            raise ZaiSignInError("No organizations found on this account.")
        org = _pick(orgs, _DEFAULT_ORG_MARKER, ("organizationName", "name"))
        org_id = org.get("organizationId") or org.get("id") or org.get("orgId")
        projects = org.get("projects") or []
        if not isinstance(projects, list) or not projects:
            raise ZaiSignInError("No projects found in the default organization.")
        project = _pick(projects, _DEFAULT_PROJECT_MARKER, ("projectName", "name"))
        project_id = project.get("projectId") or project.get("id")
        if not org_id or not project_id:
            raise ZaiSignInError("Could not resolve the default org/project.")
        return str(org_id), str(project_id)

    def _api_keys_url(self, org_id: str, project_id: str) -> str:
        return (
            f"{ZAI_BIZ_HOST}/api/biz/v1/organization/{org_id}"
            f"/projects/{project_id}/api_keys"
        )

    async def _find_or_create_api_key(
        self,
        client: httpx.AsyncClient,
        authorization: str,
        org_id: str,
        project_id: str,
    ) -> str:
        list_url = self._api_keys_url(org_id, project_id)
        try:
            existing = await self._biz_get(
                client, list_url, authorization, context="list api keys"
            )
            keys = (
                existing
                if isinstance(existing, list)
                else existing.get("list")
                if isinstance(existing, dict)
                else None
            )
            if isinstance(keys, list):
                for key in keys:
                    if isinstance(key, dict) and key.get("name") == ZAI_API_KEY_NAME:
                        api_key = key.get("apiKey") or key.get("api_key")
                        if api_key:
                            return str(api_key)
        except ZaiSignInError:
            logger.debug("Listing z.ai api keys failed; will create one", exc_info=True)

        try:
            resp = await client.post(
                list_url,
                headers={
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                },
                json={"name": ZAI_API_KEY_NAME},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ZaiSignInError(f"Could not create an API key: {exc}") from exc
        created = _unwrap_envelope(resp.json(), context="create api key")
        api_key = (
            created.get("apiKey") or created.get("api_key")
            if isinstance(created, dict)
            else None
        )
        if not api_key:
            raise ZaiSignInError("API key creation returned no key.")
        return str(api_key)

    async def _copy_secret(
        self,
        client: httpx.AsyncClient,
        authorization: str,
        org_id: str,
        project_id: str,
        api_key: str,
    ) -> str:
        url = f"{self._api_keys_url(org_id, project_id)}/copy/{api_key}"
        try:
            data = await self._biz_get(
                client, url, authorization, context="copy api key"
            )
        except ZaiSignInError:
            # The id alone may already be a usable key; degrade gracefully.
            logger.debug("Copying z.ai api key secret failed", exc_info=True)
            return ""
        if not isinstance(data, dict):
            return ""
        secret = data.get("secretKey") or data.get("secret_key") or ""
        return str(secret)


def _pick(
    items: list[Any], marker: str, name_fields: tuple[str, ...]
) -> dict[str, Any]:
    """Pick the entry whose name contains ``marker``, else the first dict entry."""
    dicts = [item for item in items if isinstance(item, dict)]
    for item in dicts:
        name = next((str(item.get(f, "")) for f in name_fields if item.get(f)), "")
        if marker in name:
            return item
    if not dicts:
        raise ZaiSignInError("Account data was empty.")
    return dicts[0]
