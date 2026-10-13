"""Binance 公开数据仓 data.binance.vision：USDⓈ-M 合约 15m K 线与资金费率，按月下 zip，不要 key、无频控、无地域限制。

K 线 CSV 与 REST 同 12 列：open_time,open,high,low,close,volume,close_time,quote_volume,count,
taker_buy_volume,taker_buy_quote_volume,ignore。费率 CSV 三列：calc_time,funding_interval_hours,last_funding_rate。
月度文件要到下月初才出：当月与上月 404 就改走日度文件；费率没有日度文件，这两个月改走 fapi 的 /fapi/v1/fundingRate
（公开接口，不要 key）。更早的 404 是那时还没上市。
老文件没有头行、2025 年后的时间戳可能是微秒，解析时一并归一。
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INTERVAL = "15m"
BAR_MS = 15 * 60_000
_RETRIES = 3

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
_INT_COLUMNS = ["open_time", "close_time", "trades"]
_FLOAT_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
FUNDING_COLUMNS = ["funding_time", "funding_rate"]
# CSV 按位置对到我们的列名，K 线最后一列 ignore 不要
_KLINE_CSV = [*KLINE_COLUMNS, "ignore"]
_FUNDING_CSV = ["funding_time", "interval_hours", "funding_rate"]
# 超过 15 位就是微秒
_MICRO_THRESHOLD = 100_000_000_000_000


class BinanceError(RuntimeError):
    """拉行情失败"""


def empty_klines() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in KLINE_COLUMNS})
    return df.astype({c: "int64" for c in _INT_COLUMNS})


def empty_funding() -> pd.DataFrame:
    return pd.DataFrame({"funding_time": pd.Series(dtype="int64"), "funding_rate": pd.Series(dtype="float64")})


class Binance:
    def __init__(self, base_url: str, client: httpx.Client | None = None, fapi_url: str = "https://fapi.binance.com"):
        self._base = base_url.rstrip("/")
        self._fapi = fapi_url.rstrip("/")
        self._client = client if client is not None else httpx.Client(timeout=60.0)

    def klines(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """[start_ms, end_ms) 内已收盘的 15m bar，按 open_time 升序"""
        frames: list[pd.DataFrame] = []
        for month in _months(start_ms, end_ms):
            csv = self._zip(f"/data/futures/um/monthly/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{month:%Y-%m}.zip")
            if csv is not None:
                frames.append(_parse(csv, _KLINE_CSV, ["open_time", "close_time"]))
            elif _recent(month, end_ms):
                for day in _days(month, start_ms, end_ms):
                    csv = self._zip(f"/data/futures/um/daily/klines/{symbol}/{INTERVAL}/{symbol}-{INTERVAL}-{day:%Y-%m-%d}.zip")
                    if csv is not None:
                        frames.append(_parse(csv, _KLINE_CSV, ["open_time", "close_time"]))
        if not frames:
            return empty_klines()
        df = pd.concat(frames, ignore_index=True)[KLINE_COLUMNS]
        df[_INT_COLUMNS] = df[_INT_COLUMNS].astype("int64")
        df[_FLOAT_COLUMNS] = df[_FLOAT_COLUMNS].astype("float64")
        df = df[(df["open_time"] >= start_ms) & (df["close_time"] < end_ms)]
        return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)

    def funding(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """[start_ms, end_ms) 内已结算的资金费率；月度文件还没出的月份走 REST，fapi 地址留空就不补"""
        frames: list[pd.DataFrame] = []
        for month in _months(start_ms, end_ms):
            csv = self._zip(f"/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month:%Y-%m}.zip")
            if csv is not None:
                frames.append(_parse(csv, _FUNDING_CSV, ["funding_time"]))
            elif self._fapi and _recent(month, end_ms):
                frames.append(self._funding_rest(symbol, max(_ms(month), start_ms), min(_ms(_next_month(month)), end_ms)))
        if not frames:
            return empty_funding()
        df = pd.concat(frames, ignore_index=True)[FUNDING_COLUMNS]
        df = df.astype({"funding_time": "int64", "funding_rate": "float64"})
        df = df[(df["funding_time"] >= start_ms) & (df["funding_time"] < end_ms)]
        return df.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)

    def _funding_rest(self, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        """一页最多 1000 条；一个月最多 186 条（4 小时结算的币），一页够"""
        rows = self._rest("/fapi/v1/fundingRate",
                          {"symbol": symbol, "startTime": start_ms, "endTime": end_ms - 1, "limit": 1000})
        if not rows:
            return empty_funding()
        return pd.DataFrame({"funding_time": [int(r["fundingTime"]) for r in rows],
                             "funding_rate": [float(r["fundingRate"]) for r in rows]})

    def _zip(self, path: str) -> bytes | None:
        """下载 zip 解出里面唯一的 CSV；404 是没这个文件"""
        resp = self._get(self._base + path, None)
        if resp is None:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            return zf.read(zf.namelist()[0])

    def _rest(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        resp = self._get(self._fapi + path, params)
        if resp is None:
            raise BinanceError(f"fapi {path} 返回 404")
        return resp.json()

    def _get(self, url: str, params: dict[str, Any] | None) -> httpx.Response | None:
        """带重试的 GET；404 返回 None"""
        for attempt in range(_RETRIES):
            if attempt:
                time.sleep(2 ** attempt)
            try:
                resp = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                logger.warning("%s 网络错误，第 %d 次：%s", url, attempt + 1, exc)
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None
            # 429 是限流，5xx 是对面抖，都值得再试；其它直接报
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning("%s HTTP %d，第 %d 次", url, resp.status_code, attempt + 1)
                continue
            raise BinanceError(f"{url} 返回 HTTP {resp.status_code}")
        raise BinanceError(f"{url} 连试 {_RETRIES} 次都失败")


def _parse(csv: bytes, names: list[str], time_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(csv), header=None, names=names, dtype=str)
    # 头行第一格是列名不是数字
    df = df[df[names[0]].str.isdigit()].apply(pd.to_numeric)
    for c in time_cols:
        t = df[c].to_numpy(dtype="int64")
        df[c] = np.where(t > _MICRO_THRESHOLD, t // 1000, t)
    return df


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _next_month(month: datetime) -> datetime:
    return (month.replace(day=28) + timedelta(days=4)).replace(day=1)


def _months(start_ms: int, end_ms: int) -> list[datetime]:
    """[start, end) 覆盖到的月份，UTC 月初"""
    cur = _utc(start_ms).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = _utc(end_ms)
    out = []
    while cur < end:
        out.append(cur)
        cur = _next_month(cur)
    return out


def _recent(month: datetime, end_ms: int) -> bool:
    """当月或上月：月度文件可能还没出"""
    end = _utc(end_ms)
    return (end.year - month.year) * 12 + end.month - month.month <= 1


def _days(month: datetime, start_ms: int, end_ms: int) -> list[datetime]:
    """这个月里落在 [start, end) 的日子，UTC 零点"""
    cur = max(month, _utc(start_ms).replace(hour=0, minute=0, second=0, microsecond=0))
    end = min(_next_month(month), _utc(end_ms))
    out = []
    while cur < end:
        out.append(cur)
        cur += timedelta(days=1)
    return out
