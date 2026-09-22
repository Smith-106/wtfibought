"""一币一张表：K 线 + 资金费率 + BTC 基准 + 强平桶 + 标准答案。

index 是 bar 收盘时刻（ms），全部特征列在那一刻都已知；标准答案 fwd_ret_* 看未来，DSL 不许引用。
raw/ 放拉下来的原始数据，panel/ 放装配好的表，都是 parquet。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import liquidation
from .binance import BAR_MS, Binance

logger = logging.getLogger(__name__)

BENCHMARK = "BTCUSDT"
DAY_MS = 86_400_000
# 标准答案三档：h 小时 → 根 15m bar
HORIZON_BARS = {6: 24, 12: 48, 24: 96}
LIQ_COLUMNS = ["liq_long", "liq_short", "liq_count"]
# 费率 8 小时结算一次，一个半周期没新结算就当断了
FUNDING_STALE_MS = 12 * 3_600_000
# DSL 里能用 $ 引用的列
FEATURE_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
                   "taker_buy_base", "taker_buy_quote", "ret", "funding_rate", "btc_close", "btc_ret",
                   *LIQ_COLUMNS, "hour"]
TARGET_COLUMNS = [f"fwd_ret_{bars}" for bars in HORIZON_BARS.values()]


def assemble(klines: pd.DataFrame, funding: pd.DataFrame, btc_klines: pd.DataFrame,
             liq: pd.DataFrame | None = None, liq_start: int | None = None,
             outage: frozenset[int] = frozenset()) -> pd.DataFrame:
    """liq 是这个币的桶；liq_start 是全部币里最早的一桶，之前没数据不是没强平；outage 是断档的桶 key"""
    df = klines.drop_duplicates("close_time").set_index("close_time").sort_index()
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    # 收盘时刻最近一次已结算的费率；离上次结算太久还没新的是数据断了（月度文件只到上个整月），记 NaN 不硬填
    df["funding_rate"] = _asof(df.index, funding, "funding_time", "funding_rate")
    settled = _asof(df.index, funding, "funding_time", "funding_time")
    df.loc[df.index.to_numpy() - settled > FUNDING_STALE_MS, "funding_rate"] = np.nan
    # BTC 按收盘时刻对齐：取 ≤ 该时刻最近的一根，与 Java benchmarkReturns 同口径
    df["btc_close"] = _asof(df.index, btc_klines, "close_time", "close")
    df["btc_ret"] = np.log(df["btc_close"] / df["btc_close"].shift(1))
    _liq_columns(df, liq, liq_start, outage)
    # bar 开始时刻的 UTC 小时：00:45~01:00 那根算 0 点的（closeTime 是边界减 1ms，加回去再退一根）
    df["hour"] = ((df.index.to_numpy() + 1 - BAR_MS) // 3_600_000) % 24
    for bars in HORIZON_BARS.values():
        df[f"fwd_ret_{bars}"] = np.log(df["close"].shift(-bars) / df["close"])
    return df


def _asof(index: pd.Index, source: pd.DataFrame | None, time_col: str, value_col: str) -> np.ndarray:
    """每个收盘时刻取 source 里时间 ≤ 它的最近一条，没有就是 NaN"""
    if source is None or source.empty:
        return np.full(len(index), np.nan)
    src = pd.DataFrame({"t": source[time_col].to_numpy(dtype="int64"),
                        "v": source[value_col].to_numpy(dtype="float64")}).sort_values("t")
    left = pd.DataFrame({"t": index.to_numpy(dtype="int64")})
    return pd.merge_asof(left, src, on="t", direction="backward")["v"].to_numpy(dtype="float64")


def _liq_columns(df: pd.DataFrame, liq: pd.DataFrame | None, liq_start: int | None,
                 outage: frozenset[int]) -> None:
    """有数据的时候没强平就是 0；数据开始前和采集断掉的桶是 NaN"""
    if liq is None or liq_start is None:
        for c in LIQ_COLUMNS:
            df[c] = np.nan
        return
    joined = df[[]].join(liq.drop_duplicates("close_time").set_index("close_time")[LIQ_COLUMNS], how="left")
    times = df.index.to_numpy(dtype="int64")
    known = (times >= liq_start) & ~np.isin(times, list(outage))
    for c in LIQ_COLUMNS:
        values = joined[c].to_numpy(dtype="float64", copy=True)
        values[known & np.isnan(values)] = 0.0
        values[~known] = np.nan
        df[c] = values


def describe(panel: pd.DataFrame) -> dict[str, Any]:
    liq_known = panel.index[panel["liq_count"].notna()]
    times = panel.index.to_numpy(dtype="int64")
    return {
        "rows": int(len(panel)),
        "from": int(panel.index.min()) if len(panel) else None,
        "to": int(panel.index.max()) if len(panel) else None,
        "liqFrom": int(liq_known.min()) if len(liq_known) else None,
        # 相邻两根不是正好 15m 的地方：交易所维护缺 bar 时切窗按行号会漂，先把数字亮出来
        "gaps": int((np.diff(times) != BAR_MS).sum()) if len(times) > 1 else 0,
    }


class PanelStore:
    def __init__(self, data_dir: str | Path):
        self.raw = Path(data_dir) / "raw"
        self.panels = Path(data_dir) / "panel"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.panels.mkdir(parents=True, exist_ok=True)

    def refresh(self, binance: Binance, symbols: tuple[str, ...], history_days: int,
                conninfo: dict[str, Any] | None = None, tz: str = "UTC",
                now_ms: int | None = None) -> dict[str, dict[str, Any]]:
        """增量拉 K 线与费率，重读强平桶，装配全部币；返回每币的 describe"""
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        earliest = now_ms - history_days * DAY_MS
        klines: dict[str, pd.DataFrame] = {}
        funding: dict[str, pd.DataFrame] = {}
        for s in dict.fromkeys((*symbols, BENCHMARK)):
            klines[s] = self._grow(f"klines_{s}", "close_time", lambda a, b, s=s: binance.klines(s, a, b), earliest, now_ms)
            if s in symbols:
                funding[s] = self._grow(f"funding_{s}", "funding_time", lambda a, b, s=s: binance.funding(s, a, b),
                                        earliest, now_ms)
        liq = liquidation.load(conninfo, symbols, tz) if conninfo else None
        liq_start = None if liq is None or liq.empty else int(liq["close_time"].min())
        outage = liquidation.outage_buckets(liq) if liq is not None else frozenset()
        status = {}
        for s in symbols:
            sym_liq = None if liq is None else liq[liq["symbol"] == s]
            panel = assemble(klines[s], funding[s], klines[BENCHMARK], sym_liq, liq_start, outage)
            panel.to_parquet(self.panels / f"{s}.parquet")
            status[s] = describe(panel)
            logger.info("panel %s：%s", s, status[s])
        return status

    def _grow(self, name: str, time_col: str, fetch: Callable[[int, int], pd.DataFrame],
              earliest: int, now_ms: int) -> pd.DataFrame:
        """有缓存就从最后一条之后接着拉，没有就从 earliest 拉；想拉更深的历史删掉 raw 重来"""
        path = self.raw / f"{name}.parquet"
        old = pd.read_parquet(path) if path.exists() else None
        start = earliest if old is None or old.empty else int(old[time_col].max()) + 1
        new = fetch(start, now_ms)
        if old is None:
            df = new
        elif new.empty:
            df = old
        else:
            df = pd.concat([old, new], ignore_index=True)
        df = df.drop_duplicates(time_col).sort_values(time_col).reset_index(drop=True)
        df.to_parquet(path)
        return df

    def load(self, symbol: str) -> pd.DataFrame:
        return pd.read_parquet(self.panels / f"{symbol}.parquet")

    def status(self) -> dict[str, dict[str, Any]]:
        return {p.stem: describe(pd.read_parquet(p)) for p in sorted(self.panels.glob("*.parquet"))}
