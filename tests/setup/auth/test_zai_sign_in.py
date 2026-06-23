from __future__ import annotations

import httpx
import pytest
import respx

from vibe.setup.auth.zai_sign_in import (
    ZAI_API_KEY_NAME,
    ZaiSignInError,
    ZaiSignInService,
)

_FAR_FUTURE_EPOCH = 9_999_999_999  # year 2286, well past any test run

_INIT = "https://zcode.z.ai/api/v1/oauth/cli/init"
_POLL = "https://zcode.z.ai/api/v1/oauth/cli/poll/flow1"
_LOGIN = "https://api.z.ai/api/auth/z/login"
_CUSTOMER = "https://api.z.ai/api/biz/customer/getCustomerInfo"
_API_KEYS = "https://api.z.ai/api/biz/v1/organization/org1/projects/proj1/api_keys"


async def _noop_sleep(_: float) -> None:
    return None


def _service(**kwargs: object) -> ZaiSignInService:
    return ZaiSignInService(
        poll_interval_override=0.0,
        sleep=_noop_sleep,
        open_browser=lambda _url: True,
        **kwargs,  # type: ignore[arg-type]
    )


def _mock_init(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_INIT).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "flow_id": "flow1",
                    "authorize_url": "https://chat.z.ai/api/oauth/authorize?x=1",
                    "expires_at": _FAR_FUTURE_EPOCH,
                    "poll_interval_sec": 0,
                },
            },
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
    _mock_init(respx_mock)
    poll = respx_mock.get(_POLL).mock(
        side_effect=[
            httpx.Response(200, json={"code": 0, "data": {"status": "pending"}}),
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"status": "ready", "zai": {"access_token": "oauth-tok"}},
                },
            ),
        ]
    )
    login = respx_mock.post(_LOGIN).mock(
        return_value=httpx.Response(200, json={"access_token": "biz-tok"})
    )
    _mock_customer(respx_mock)
    # No existing keys -> service must create one.
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

    import json

    assert api_key == "key-id.sec"
    assert seen_urls == ["https://chat.z.ai/api/oauth/authorize?x=1"]
    assert poll.call_count == 2
    # OAuth token is exchanged for a biz token, used as the bearer downstream.
    assert json.loads(login.calls.last.request.read())["token"] == "oauth-tok"
    assert list_route.calls.last.request.headers["Authorization"] == "Bearer biz-tok"
    assert (
        json.loads(create_route.calls.last.request.read())["name"] == ZAI_API_KEY_NAME
    )
    assert copy_route.called


@pytest.mark.asyncio
@respx.mock
async def test_reuses_existing_named_key_and_tolerates_copy_failure(
    respx_mock: respx.MockRouter,
) -> None:
    _mock_init(respx_mock)
    respx_mock.get(_POLL).mock(
        return_value=httpx.Response(
            200,
            json={
                "code": 0,
                "data": {"status": "ready", "zai": {"access_token": "oauth-tok"}},
            },
        )
    )
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
    # Copy fails -> service degrades to the bare key id.
    respx_mock.get(f"{_API_KEYS}/copy/existing-key").mock(
        return_value=httpx.Response(500, json={"code": 1, "msg": "boom"})
    )

    api_key = await _service().authenticate()

    assert api_key == "existing-key"
    assert not create_route.called


@pytest.mark.asyncio
@respx.mock
async def test_poll_failed_status_raises(respx_mock: respx.MockRouter) -> None:
    _mock_init(respx_mock)
    respx_mock.get(_POLL).mock(
        return_value=httpx.Response(200, json={"code": 0, "data": {"status": "failed"}})
    )

    with pytest.raises(ZaiSignInError, match="denied"):
        await _service().authenticate()


@pytest.mark.asyncio
@respx.mock
async def test_init_business_error_raises(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_INIT).mock(
        return_value=httpx.Response(200, json={"code": 1, "msg": "nope"})
    )

    with pytest.raises(ZaiSignInError, match="nope"):
        await _service().authenticate()
