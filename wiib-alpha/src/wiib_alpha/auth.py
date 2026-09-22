"""鉴权：页面的 satoken 原样转给 sim 换 userId，只放行管理员（userId=1，与 Java RequireAdminAspect 同值）。

sa-token 只认请求头；60 秒缓存少打 sim。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

ADMIN_USER_ID = 1
_TTL = 60.0


class AuthError(RuntimeError):
    """token 失效/缺失 → 401"""


class Forbidden(RuntimeError):
    """登录了但不是管理员 → 403"""


class UpstreamError(RuntimeError):
    """sim 没起或返回不对 → 502"""


@dataclass(frozen=True)
class Auth:
    user_id: int
    token: str
    lang: str


class Authenticator:
    def __init__(self, sim_base_url: str, client: httpx.AsyncClient | None = None):
        self._url = sim_base_url.rstrip("/") + "/api/auth/current"
        self._client = client if client is not None else httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, tuple[int, float]] = {}

    async def __call__(self, token: str | None, lang: str | None) -> Auth:
        if not token:
            raise AuthError("未登录")
        user_id = await self._user_id(token, lang)
        if user_id != ADMIN_USER_ID:
            raise Forbidden("只有管理员能用")
        return Auth(user_id, token, pick_lang(lang))

    async def _user_id(self, token: str, lang: str | None) -> int:
        hit = self._cache.get(token)
        now = time.monotonic()
        if hit and hit[1] > now:
            return hit[0]
        headers = {"satoken": token}
        if lang:
            headers["X-Lang"] = lang
        try:
            resp = await self._client.get(self._url, headers=headers)
        except httpx.TransportError as exc:
            raise UpstreamError("sim 暂时不可用") from exc
        if resp.status_code == 401:
            raise AuthError("登录态失效")
        if resp.status_code >= 400:
            raise UpstreamError(f"sim 返回 HTTP {resp.status_code}")
        body = resp.json()
        if body.get("code") == 401:
            raise AuthError(body.get("msg") or "登录态失效")
        if body.get("code") != 0:
            raise UpstreamError(body.get("msg") or f"sim 返回 code={body.get('code')}")
        user_id = int(body["data"]["id"])
        self._cache[token] = (user_id, now + _TTL)
        return user_id

    async def aclose(self) -> None:
        await self._client.aclose()


def pick_lang(raw: str | None) -> str:
    """只认 zh/en，其它一律按 zh"""
    return "en" if (raw or "").strip().lower().startswith("en") else "zh"
