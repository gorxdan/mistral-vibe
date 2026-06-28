"""Interactive "Continue with Z.ai" browser sign-in for the GLM (zai) provider.

Z.ai's account login is not a token used at inference time: it is a
*bootstrapping* flow that provisions a durable coding-plan API key. We replicate
the flow ZCode (z.ai's desktop client) uses, then hand the resulting key to the
normal API-key persistence path so the rest of the stack (env-var bearer against
the coding endpoint) is unchanged.

ZCode 3.1.5 uses an OAuth authorization-code round-trip, but Z.ai only
honors its registered ``zcode://`` custom scheme for the post-login redirect.
Setup accepts that callback through a hidden protocol-handler entrypoint when
registered with the desktop, and still supports manual paste as a fallback. The
flow, in order:

1. Generate a 32-byte hex ``state``.
2. Build ``https://chat.z.ai/api/oauth/authorize?redirect_uri=zcode://zai-auth/
   callback&response_type=code&client_id=<zcode client id>&state=<state>`` and
   open it in the browser; the user authorizes their z.ai account.
3. Z.ai redirects to ``zcode://zai-auth/callback?code=<code>&state=<state>``.
   If the hidden handler is registered, it captures the URL for setup. Otherwise
   the user copies the URL (or the ``code``) and pastes it back.
4. ``POST https://zcode.z.ai/api/v1/oauth/token`` (body
   ``{provider, code, redirect_uri, state}``) -> ``{code:0, data:{token, zai:
   {access_token}}}``. Read ``data.zai.access_token``.
5. Exchange that access token for a coding-plan API key (the "biz" dance):
   ``api/auth/z/login`` -> ``getCustomerInfo`` (pick the default org/project) ->
   find-or-create an api key -> copy its secret. The usable key is
   ``{apiKey}.{secret}``.

All endpoints and request shapes are reverse-engineered from the ZCode bundle
(the official z.ai npm packages are API-key configurators only and ship no OAuth
client). They are undocumented and may change without notice; keep this isolated
behind the opt-in zai login path. The earlier ``oauth/cli/init`` + ``poll``
flow ZCode shipped was removed upstream and now returns HTTP 404.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import secrets
from typing import Any, Final
from urllib.parse import ParseResult, parse_qs, urlencode, urlparse
import webbrowser

import httpx

from vibe.core.logger import logger
from vibe.core.utils.http import build_ssl_context

# --- ZCode OAuth constants (from the unpacked ZCode 3.1.5 desktop bundle) -----
ZAI_AUTHORIZE_URL: Final = "https://chat.z.ai/api/oauth/authorize"
ZCODE_TOKEN_URL: Final = "https://zcode.z.ai/api/v1/oauth/token"
ZAI_OAUTH_CLIENT_ID: Final = "client_P8X5CMWmlaRO9gyO-KSqtg"
ZAI_OAUTH_REDIRECT_URI: Final = "zcode://zai-auth/callback"

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
# Envelope ``code``/``status`` values that signal success across the biz APIs.
_OK_CODES: Final = frozenset({None, 0, 200, "0", "200"})
_AUTHORIZE_URL_ERROR: Final = (
    "That is the Z.ai sign-in page URL, not the callback URL. Finish the "
    "browser sign-in, then paste the zcode:// callback URL that contains code=."
)

UrlCallback = Callable[[str], None]
BrowserOpener = Callable[[str], bool]
CodeReceiver = Callable[[str], Awaitable[str]]


class ZaiSignInError(RuntimeError):
    """Raised when the interactive Z.ai sign-in cannot complete."""


@dataclass(frozen=True, slots=True)
class ZaiAuthorizationCallback:
    code: str
    state: str | None


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


def _generate_state() -> str:
    """32 random bytes as 64 hex chars; opaque CSRF state for the OAuth round-trip."""
    return secrets.token_hex(32)


def _build_authorize_url(redirect_uri: str, state: str) -> str:
    query = urlencode({
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "client_id": ZAI_OAUTH_CLIENT_ID,
        "state": state,
    })
    return f"{ZAI_AUTHORIZE_URL}?{query}"


def extract_zai_authorization_code(
    pasted: str, *, expected_state: str | None = None
) -> str:
    callback = parse_zai_authorization_callback(pasted)
    if (
        expected_state is not None
        and callback.state is not None
        and callback.state != expected_state
    ):
        raise ZaiSignInError(
            "Authorization callback state did not match this sign-in attempt. "
            "Please retry Z.ai sign-in."
        )
    return callback.code


def parse_zai_authorization_callback(pasted: str) -> ZaiAuthorizationCallback:
    text = pasted.strip()
    if not text:
        raise ZaiSignInError("No authorization code found in what you pasted.")

    parsed = urlparse(text)
    query = parsed.query
    candidate = query if query else text
    if "=" in candidate:
        params = parse_qs(candidate)
        code = (params.get("code") or params.get("authCode") or [""])[0]
        if code:
            state = (params.get("state") or [None])[0]
            return ZaiAuthorizationCallback(code=code, state=state)
        if _looks_like_authorize_url(parsed, params):
            raise ZaiSignInError(_AUTHORIZE_URL_ERROR)
        raise ZaiSignInError("No authorization code found in what you pasted.")
    if parsed.scheme in {"http", "https", "zcode"}:
        raise ZaiSignInError("No authorization code found in what you pasted.")
    return ZaiAuthorizationCallback(code=text, state=None)


def _extract_code(pasted: str) -> str:
    return extract_zai_authorization_code(pasted)


def _looks_like_authorize_url(
    parsed: ParseResult, params: dict[str, list[str]]
) -> bool:
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == "chat.z.ai"
        and "/oauth/authorize" in parsed.path
    ) or {"response_type", "client_id", "redirect_uri"}.issubset(params)


@dataclass
class ZaiSignInService:
    """Drives the full Z.ai sign-in and returns a usable coding-plan API key."""

    open_browser: BrowserOpener = field(default=webbrowser.open)
    receive_code: CodeReceiver | None = field(default=None)

    async def authenticate(self, *, on_url: UrlCallback | None = None) -> str:
        """Run authorize -> browser -> paste -> token exchange -> biz dance."""
        if self.receive_code is None:
            raise ZaiSignInError(
                "No authorization-code input configured for Z.ai sign-in."
            )
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT, verify=build_ssl_context()
        ) as client:
            state = _generate_state()
            authorize_url = _build_authorize_url(ZAI_OAUTH_REDIRECT_URI, state)
            if on_url is not None:
                on_url(authorize_url)
            self._open_browser(authorize_url)
            code = extract_zai_authorization_code(
                await self.receive_code(authorize_url), expected_state=state
            )
            access_token = await self._exchange_token(
                client, code, ZAI_OAUTH_REDIRECT_URI, state
            )
            return await self._provision_api_key(client, access_token)

    def _open_browser(self, url: str) -> None:
        try:
            self.open_browser(url)
        except Exception:
            logger.debug("Failed to open browser for Z.ai sign-in", exc_info=True)

    async def _exchange_token(
        self, client: httpx.AsyncClient, code: str, redirect_uri: str, state: str
    ) -> str:
        try:
            resp = await client.post(
                ZCODE_TOKEN_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "provider": "zai",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "state": state,
                },
            )
        except httpx.RequestError as exc:
            raise ZaiSignInError(f"Sign-in token exchange failed: {exc}") from exc
        try:
            payload = resp.json()
        except ValueError as exc:
            raise ZaiSignInError(
                f"Sign-in token exchange failed: HTTP {resp.status_code}"
            ) from exc
        data = _unwrap_envelope(payload, context="Sign-in token exchange")
        if not isinstance(data, dict):
            raise ZaiSignInError("Sign-in token exchange returned an unexpected shape.")
        zai = data.get("zai")
        access_token = (
            zai.get("access_token") if isinstance(zai, dict) else None
        ) or data.get("token")
        if not isinstance(access_token, str) or not access_token:
            raise ZaiSignInError("Sign-in succeeded but returned no Z.ai access token.")
        return access_token

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
