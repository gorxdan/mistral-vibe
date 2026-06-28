from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import respx

from vibe.setup.auth.zai_sign_in import (
    ZAI_API_KEY_NAME,
    ZAI_OAUTH_CLIENT_ID,
    ZAI_OAUTH_REDIRECT_URI,
    ZaiSignInError,
    ZaiSignInService,
    _extract_code,
    extract_zai_authorization_code,
)

_TOKEN = "https://zcode.z.ai/api/v1/oauth/token"
_LOGIN = "https://api.z.ai/api/auth/z/login"
_CUSTOMER = "https://api.z.ai/api/biz/customer/getCustomerInfo"
_API_KEYS = "https://api.z.ai/api/biz/v1/organization/org1/projects/proj1/api_keys"


async def _canned_code(_: str) -> str:
    return "oauth-code"


def _service(**kwargs: Any) -> ZaiSignInService:
    return ZaiSignInService(
        open_browser=lambda _: True, receive_code=_canned_code, **kwargs
    )


def _mock_token(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(_TOKEN).mock(
        return_value=httpx.Response(
            200, json={"code": 0, "data": {"zai": {"access_token": "oauth-tok"}}}
        )
    )


def _mock_customer(respx_mock: respx.MockRouter) -> None:
    respx_mock.get(_CUSTOMER).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "organizations": [
                        {
                            "organizationName": "默认机构",
                            "organizationId": "org1",
                            "projects": [
                                {"projectName": "默认项目", "projectId": "proj1"}
                            ],
                        }
                    ]
                },
            },
        )
    )


@pytest.mark.asyncio
@respx.mock
async def test_full_flow_creates_key_and_joins_secret(
    respx_mock: respx.MockRouter,
) -> None:
    token_route = _mock_token(respx_mock)
    login = respx_mock.post(_LOGIN).mock(
        return_value=httpx.Response(200, json={"access_token": "biz-tok"})
    )
    _mock_customer(respx_mock)
    list_route = respx_mock.get(_API_KEYS).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": []})
    )
    create_route = respx_mock.post(_API_KEYS).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"apiKey": "key-id"}})
    )
    copy_route = respx_mock.get(f"{_API_KEYS}/copy/key-id").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"secretKey": "sec"}})
    )

    seen_urls: list[str] = []
    api_key = await _service().authenticate(on_url=seen_urls.append)

    assert api_key == "key-id.sec"
    assert len(seen_urls) == 1
    authorize = urlparse(seen_urls[0])
    assert authorize.scheme == "https"
    assert authorize.netloc == "chat.z.ai"
    assert authorize.path == "/api/oauth/authorize"
    q = parse_qs(authorize.query)
    assert q["client_id"] == [ZAI_OAUTH_CLIENT_ID]
    assert q["response_type"] == ["code"]
    assert q["redirect_uri"] == [ZAI_OAUTH_REDIRECT_URI]
    assert len(q["state"][0]) == 64
    token_body = json.loads(token_route.calls.last.request.read())
    assert token_body["provider"] == "zai"
    assert token_body["code"] == "oauth-code"
    assert token_body["redirect_uri"] == ZAI_OAUTH_REDIRECT_URI
    assert token_body["state"] == q["state"][0]
    assert json.loads(login.calls.last.request.read())["token"] == "oauth-tok"
    assert list_route.calls.last.request.headers["Authorization"] == "Bearer biz-tok"
    assert (
        json.loads(create_route.calls.last.request.read())["name"] == ZAI_API_KEY_NAME
    )
    assert copy_route.called


@pytest.mark.asyncio
@respx.mock
async def test_full_flow_accepts_pasted_redirect_url(
    respx_mock: respx.MockRouter,
) -> None:
    token_route = _mock_token(respx_mock)
    respx_mock.post(_LOGIN).mock(
        return_value=httpx.Response(200, json={"access_token": "biz-tok"})
    )
    _mock_customer(respx_mock)
    respx_mock.get(_API_KEYS).mock(
        return_value=httpx.Response(
            200, json={"code": 0, "data": [{"name": ZAI_API_KEY_NAME, "apiKey": "k"}]}
        )
    )
    respx_mock.get(f"{_API_KEYS}/copy/k").mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"secretKey": "s"}})
    )

    async def paste_url(authorize_url: str) -> str:
        state = parse_qs(urlparse(authorize_url).query)["state"][0]
        return f"zcode://zai-auth/callback?code=the-real-code&state={state}"

    api_key = await ZaiSignInService(
        open_browser=lambda _: True, receive_code=paste_url
    ).authenticate()

    assert api_key == "k.s"
    token_body = json.loads(token_route.calls.last.request.read())
    assert token_body["code"] == "the-real-code"


@pytest.mark.asyncio
@respx.mock
async def test_pasted_authorize_url_without_code_does_not_exchange_token(
    respx_mock: respx.MockRouter,
) -> None:
    token_route = respx_mock.post(_TOKEN)

    async def paste_authorize_url(_: str) -> str:
        return (
            "https://chat.z.ai/auth/oauth/authorize?response_type=code"
            f"&client_id={ZAI_OAUTH_CLIENT_ID}"
            "&redirect_uri=zcode%3A%2F%2Fzai-auth%2Fcallback"
            "&state=abc"
        )

    with pytest.raises(ZaiSignInError, match="sign-in page URL"):
        await ZaiSignInService(
            open_browser=lambda _: True, receive_code=paste_authorize_url
        ).authenticate()

    assert not token_route.called


@pytest.mark.asyncio
@respx.mock
async def test_reuses_existing_named_key_and_tolerates_copy_failure(
    respx_mock: respx.MockRouter,
) -> None:
    _mock_token(respx_mock)
    respx_mock.post(_LOGIN).mock(
        return_value=httpx.Response(200, json={"access_token": "biz-tok"})
    )
    _mock_customer(respx_mock)
    respx_mock.get(_API_KEYS).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": [{"name": ZAI_API_KEY_NAME, "apiKey": "existing-key"}],
            },
        )
    )
    create_route = respx_mock.post(_API_KEYS)
    respx_mock.get(f"{_API_KEYS}/copy/existing-key").mock(
        return_value=httpx.Response(500, json={"code": 1, "msg": "boom"})
    )

    api_key = await _service().authenticate()

    assert api_key == "existing-key"
    assert not create_route.called


@pytest.mark.asyncio
@respx.mock
async def test_token_exchange_envelope_error_raises(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_TOKEN).mock(
        return_value=httpx.Response(200, json={"code": 1, "msg": "nope"})
    )

    with pytest.raises(ZaiSignInError, match="nope"):
        await _service().authenticate()


@pytest.mark.asyncio
@respx.mock
async def test_token_exchange_http_error_raises(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_TOKEN).mock(
        return_value=httpx.Response(500, json={"code": 2007, "msg": "http error"})
    )

    with pytest.raises(ZaiSignInError, match="http error"):
        await _service().authenticate()


@pytest.mark.asyncio
async def test_authenticate_requires_receive_code() -> None:
    svc = ZaiSignInService(open_browser=lambda _: True)

    with pytest.raises(ZaiSignInError, match="configured"):
        await svc.authenticate()


@pytest.mark.parametrize(
    ("pasted", "expected"),
    [
        ("zcode://zai-auth/callback?code=abc&state=s", "abc"),
        ("code=abc&state=s", "abc"),
        ("authCode=zzz&state=s", "zzz"),
        ("  abc  ", "abc"),
    ],
)
def test_extract_code_parses_paste_forms(pasted: str, expected: str) -> None:
    assert _extract_code(pasted) == expected


def test_extract_code_empty_raises() -> None:
    with pytest.raises(ZaiSignInError, match="No authorization code"):
        _extract_code("   ")


def test_extract_code_authorize_url_without_code_raises() -> None:
    with pytest.raises(ZaiSignInError, match="sign-in page URL"):
        _extract_code(
            "https://chat.z.ai/auth/oauth/authorize?response_type=code"
            f"&client_id={ZAI_OAUTH_CLIENT_ID}"
            "&redirect_uri=zcode%3A%2F%2Fzai-auth%2Fcallback"
            "&state=abc"
        )


def test_extract_code_rejects_mismatched_state() -> None:
    with pytest.raises(ZaiSignInError, match="state did not match"):
        extract_zai_authorization_code(
            "zcode://zai-auth/callback?code=abc&state=old", expected_state="new"
        )
