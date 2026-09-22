"""调 sim / agent 的只读接口。

页面的 satoken 原样转发（sa-token 只认请求头），X-Lang 跟着走好让上游的错误文案对上界面语言。
所有响应都是 {code, msg, data}：code==0 取 data，code==401 或 HTTP 401 抛 AuthError，其余抛 PlatformError。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import config

logger = logging.getLogger(__name__)


class AuthError(RuntimeError):
    """token 失效/缺失；调用方转成 HTTP 401"""


class PlatformError(RuntimeError):
    """上游业务失败，带上游 msg 原文"""


def _unwrap(resp: httpx.Response) -> Any:
    if resp.status_code == 401:
        raise AuthError("登录态失效")
    if resp.status_code >= 400:
        raise PlatformError(f"上游返回 HTTP {resp.status_code}")
    body = resp.json()
    code = body.get("code")
    if code == 401:
        raise AuthError(body.get("msg") or "登录态失效")
    if code != 0:
        raise PlatformError(body.get("msg") or f"上游返回 code={code}")
    return body.get("data")


class Platform:
    """一个请求周期一个实例，拿着这次的 satoken 与界面语言"""

    def __init__(self, token: str, lang: str | None = None, client: httpx.AsyncClient | None = None):
        self._headers = {"satoken": token}
        if lang:
            self._headers["X-Lang"] = lang
        self._client = client if client is not None else httpx.AsyncClient(timeout=15.0)
        # 外面传进来的客户端归外面关
        self._own_client = client is None

    async def __aenter__(self) -> "Platform":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    async def _get(self, url: str) -> Any:
        return await self._send("GET", url)

    async def _post(self, url: str, body: dict[str, Any]) -> Any:
        return await self._send("POST", url, body)

    async def _send(self, method: str, url: str, body: dict[str, Any] | None = None) -> Any:
        # sim / agent 没起或超时也算上游失败；地址只进日志，不下发给前端
        try:
            resp = await self._client.request(method, url, headers=self._headers, json=body)
        except httpx.TransportError as exc:
            logger.warning("上游请求失败 %s %s：%s", method, url, exc)
            raise PlatformError("上游服务暂时不可用，请稍后再试") from exc
        return _unwrap(resp)

    async def current_user_id(self) -> int:
        data = await self._get(f"{config().sim_base_url}/api/auth/current")
        return int(data["id"])

    async def mine(self) -> dict[str, Any] | None:
        """主人视图；没建 trader 时 data 是 null"""
        return await self._get(f"{config().agent_base_url}/api/ai/trader/mine")

    async def default_instructions(self) -> str:
        return await self._get(f"{config().agent_base_url}/api/ai/trader/default-instructions")

    async def prompt_template(self, interval_code: str, symbols: str, spec: dict[str, Any] | None,
                              wake_window: str | None) -> str:
        """平台系统提示词全文，与唤醒组装同一份"""
        return await self._post(
            f"{config().agent_base_url}/api/ai/trader/prompt-template",
            {"intervalCode": interval_code, "symbols": symbols, "spec": spec, "wakeWindow": wake_window},
        )
