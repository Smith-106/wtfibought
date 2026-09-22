import httpx
import pytest

from wiib_gen.platform import AuthError, Platform, PlatformError


def _platform(handler, lang="zh"):
    return Platform("tok-1", lang, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _ok(data):
    return lambda req: httpx.Response(200, json={"code": 0, "msg": None, "data": data})


async def test_current_user_id():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["satoken"] = req.headers.get("satoken")
        seen["lang"] = req.headers.get("X-Lang")
        return httpx.Response(200, json={"code": 0, "data": {"id": 7, "username": "a"}})

    async with _platform(handler) as p:
        assert await p.current_user_id() == 7
    assert seen["url"].endswith("/api/auth/current")
    assert seen["satoken"] == "tok-1"
    assert seen["lang"] == "zh"


async def test_mine_null_when_no_trader():
    async with _platform(_ok(None)) as p:
        assert await p.mine() is None


async def test_mine_returns_view():
    view = {"customPrompt": "只做多", "spec": {"leverageMin": 3}, "wakeWindow": None}
    async with _platform(_ok(view)) as p:
        assert (await p.mine())["customPrompt"] == "只做多"


async def test_default_instructions():
    async with _platform(_ok("【方法】...")) as p:
        assert await p.default_instructions() == "【方法】..."


async def test_prompt_template_posts_body():
    seen = {}

    def handler(req):
        seen["body"] = httpx.Response(200, content=req.content).json()
        return httpx.Response(200, json={"code": 0, "data": "系统提示词全文"})

    async with _platform(handler) as p:
        text = await p.prompt_template("15m", "BTCUSDT,ETHUSDT", {"leverageMin": 3}, "00:00-08:00")
    assert text == "系统提示词全文"
    assert seen["body"] == {
        "intervalCode": "15m",
        "symbols": "BTCUSDT,ETHUSDT",
        "spec": {"leverageMin": 3},
        "wakeWindow": "00:00-08:00",
    }


async def test_result_code_401_is_auth_error():
    handler = lambda req: httpx.Response(200, json={"code": 401, "msg": "登录已过期", "data": None})
    async with _platform(handler) as p:
        with pytest.raises(AuthError, match="登录已过期"):
            await p.mine()


async def test_http_401_is_auth_error():
    handler = lambda req: httpx.Response(401, text="Unauthorized")
    async with _platform(handler) as p:
        with pytest.raises(AuthError):
            await p.current_user_id()


async def test_non_zero_code_carries_msg():
    handler = lambda req: httpx.Response(200, json={"code": 500, "msg": "trader 不存在", "data": None})
    async with _platform(handler) as p:
        with pytest.raises(PlatformError, match="trader 不存在"):
            await p.default_instructions()


async def test_connect_error_is_platform_error_without_url():
    def handler(req):
        raise httpx.ConnectError("boom", request=req)

    async with _platform(handler) as p:
        with pytest.raises(PlatformError, match="暂时不可用") as caught:
            await p.mine()
    assert "http" not in str(caught.value)


async def test_http_5xx_is_platform_error():
    handler = lambda req: httpx.Response(502, text="Bad Gateway")
    async with _platform(handler) as p:
        with pytest.raises(PlatformError, match="HTTP 502"):
            await p.mine()


async def test_lang_header_skipped_when_absent():
    seen = {}

    def handler(req):
        seen["lang"] = req.headers.get("X-Lang")
        return httpx.Response(200, json={"code": 0, "data": "【方法】..."})

    async with _platform(handler, lang=None) as p:
        await p.default_instructions()
    assert seen["lang"] is None
