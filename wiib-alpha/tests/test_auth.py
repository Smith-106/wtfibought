import httpx
import pytest

from wiib_alpha.auth import Auth, AuthError, Authenticator, Forbidden, UpstreamError, pick_lang


def sim(calls: list) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("satoken"))
        token = request.headers.get("satoken")
        if token == "admin":
            return httpx.Response(200, json={"code": 0, "msg": "", "data": {"id": 1}})
        if token == "user":
            return httpx.Response(200, json={"code": 0, "msg": "", "data": {"id": 7}})
        if token == "expired":
            return httpx.Response(200, json={"code": 401, "msg": "登录态失效"})
        if token == "down":
            return httpx.Response(500)
        return httpx.Response(401)

    return httpx.MockTransport(handle)


@pytest.fixture
def calls() -> list:
    return []


@pytest.fixture
def auth(calls) -> Authenticator:
    return Authenticator("http://sim", httpx.AsyncClient(transport=sim(calls)))


async def test_admin_passes_and_is_cached(auth, calls):
    assert await auth("admin", "en-US") == Auth(1, "admin", "en")
    assert await auth("admin", None) == Auth(1, "admin", "zh")
    assert calls == ["admin"]


async def test_non_admin_is_forbidden(auth):
    with pytest.raises(Forbidden):
        await auth("user", None)


@pytest.mark.parametrize("token", [None, "", "bad", "expired"])
async def test_missing_or_invalid_token_is_401(auth, token):
    with pytest.raises(AuthError):
        await auth(token, None)


async def test_sim_down_is_upstream_error(auth):
    with pytest.raises(UpstreamError):
        await auth("down", None)


def test_pick_lang():
    assert pick_lang("en") == "en" and pick_lang("EN-GB") == "en" and pick_lang("zh-CN") == "zh" and pick_lang(None) == "zh"
