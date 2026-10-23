"""参照因子：5 个手写基线（与 Java ContinuousFactorSignals 同口径）+ 截面上已知的老效应 + Qlib Alpha158 的模板按 15m 窗口改写。

参照集只做三件事：冗余检查的靠山、结构原创的比对、校准时看"教科书因子在 15m 加密上活几个"。不可入库。
冗余检查用的参照网格：每条参照逐币求值，只留主档网格上的在池行，每个时刻内转成截面秩，落在当前那版面板目录的 reference_grid.parquet。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data.panel import FEATURE_COLUMNS, HORIZON_BARS, POOL_COLUMN, PanelStore
from .data.pool import Pool, on_grid
from .dsl import Expr, parse

# 参照网格只在 12h 的网格点上算
GRID_BARS = HORIZON_BARS[12]

# 基线窗口：12h / 1h / 24h / 24h / 12h，beta 48h，与 Java ContinuousFactorParams.forBarMillis 同
BASELINES: dict[str, tuple[str, str]] = {
    "risk_adj_momentum": ("Sum($ret, 48) / Std($ret, 48)", "12 小时累计收益除以同窗波动，趋势延续"),
    "short_reversal": ("-(Sum($ret, 4) / Std($ret, 4))", "1 小时冲过头会回来"),
    "volume_zscore": ("($volume - Mean($volume, 96)) / Std($volume, 96)", "异常放量后面跟着大动作"),
    "amihud": ("Mean(Abs($ret) / ($close * $volume), 96)", "同样的钱推得动更大价格的时候更不流动"),
    "residual_momentum": (
        "Sum($ret - Ref(Cov($ret, $btc_ret, 192) / Var($btc_ret, 192), 1) * $btc_ret, 48)",
        "剔掉 BTC 带动的部分，剩下这个币自己的动量"),
}

# 截面上早有文献的效应，7 天窗口为主：候选和它们高度相关，就是老效应换了个列名。名字别以数字结尾，按窗口筛的地方会当成 Alpha158
KNOWN_EFFECTS: dict[str, tuple[str, str]] = {
    "size_quote": ("Log(Sum($quote_volume, 672))", "规模、流动性：近 7 天成交额"),
    "size_oi": ("Log($oi_value)", "规模：持仓名义"),
    "turnover_oi": ("Sum($quote_volume, 672) / $oi_value", "换手：近 7 天成交额 ÷ 持仓名义"),
    "vol_7d": ("Std($ret, 672)", "波动：近 7 天收益的标准差"),
    "lottery_7d": ("Max($ret, 672)", "彩票：近 7 天最大的一根涨幅"),
    "mom_7d": ("Sum($ret, 672)", "动量 / 反转：近 7 天累计收益"),
    "mom_21d": ("Sum($ret, 2016)", "动量：近 3 周累计收益"),
    "beta_btc": ("Cov($ret, $btc_ret, 672) / Var($btc_ret, 672)", "市场 beta：近 7 天对 BTC 的 beta"),
    "carry_7d": ("Mean($funding_rate, 672)", "carry：近 7 天资金费率水平"),
    "basis_7d": ("Mean($premium, 672)", "基差：近 7 天溢价率水平"),
}

# Alpha158 的日频窗口 5/10/20/30/60 换成 1h/4h/12h/24h/48h
ALPHA158_WINDOWS = (4, 16, 48, 96, 192)
# 24h 反转：去老效应时和老效应一起回归掉；它的截面秩就是 −Sum($ret, 96) 的秩
REVERSAL = "ROC96"

_KBAR = {
    "KMID": "($close - $open) / $open",
    "KLEN": "($high - $low) / $open",
    "KMID2": "($close - $open) / ($high - $low + 1e-12)",
    "KUP": "($high - Greater($open, $close)) / $open",
    "KUP2": "($high - Greater($open, $close)) / ($high - $low + 1e-12)",
    "KLOW": "(Less($open, $close) - $low) / $open",
    "KLOW2": "(Less($open, $close) - $low) / ($high - $low + 1e-12)",
    "KSFT": "(2 * $close - $high - $low) / $open",
    "KSFT2": "(2 * $close - $high - $low) / ($high - $low + 1e-12)",
}

# {w} 是窗口；IdxMax/IdxMin 语义与 Qlib 差一个 w − x 的仿射，冗余检查不受影响
_ROLLING = {
    "ROC": "Ref($close, {w}) / $close",
    "MA": "Mean($close, {w}) / $close",
    "STD": "Std($close, {w}) / $close",
    "BETA": "Slope($close, {w}) / $close",
    "RSQR": "Rsquare($close, {w})",
    "RESI": "Resi($close, {w}) / $close",
    "MAX": "Max($high, {w}) / $close",
    "MIN": "Min($low, {w}) / $close",
    "QTLU": "Quantile($close, {w}, 0.8) / $close",
    "QTLD": "Quantile($close, {w}, 0.2) / $close",
    "RANK": "Rank($close, {w})",
    "RSV": "($close - Min($low, {w})) / (Max($high, {w}) - Min($low, {w}) + 1e-12)",
    "IMAX": "IdxMax($high, {w}) / {w}",
    "IMIN": "IdxMin($low, {w}) / {w}",
    "IMXD": "(IdxMax($high, {w}) - IdxMin($low, {w})) / {w}",
    "CORR": "Corr($close, Log($volume + 1), {w})",
    "CORD": "Corr($close / Ref($close, 1), Log($volume / Ref($volume, 1) + 1), {w})",
    "CNTP": "Mean(Gt($close, Ref($close, 1)), {w})",
    "CNTN": "Mean(Lt($close, Ref($close, 1)), {w})",
    "CNTD": "Mean(Gt($close, Ref($close, 1)), {w}) - Mean(Lt($close, Ref($close, 1)), {w})",
    "SUMP": "Sum(Greater($close - Ref($close, 1), 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-12)",
    "SUMN": "Sum(Greater(Ref($close, 1) - $close, 0), {w}) / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-12)",
    "SUMD": "(Sum(Greater($close - Ref($close, 1), 0), {w}) - Sum(Greater(Ref($close, 1) - $close, 0), {w}))"
            " / (Sum(Abs($close - Ref($close, 1)), {w}) + 1e-12)",
    "VMA": "Mean($volume, {w}) / ($volume + 1e-12)",
    "VSTD": "Std($volume, {w}) / ($volume + 1e-12)",
    "WVMA": "Std(Abs($close / Ref($close, 1) - 1) * $volume, {w})"
            " / (Mean(Abs($close / Ref($close, 1) - 1) * $volume, {w}) + 1e-12)",
    "VSUMP": "Sum(Greater($volume - Ref($volume, 1), 0), {w}) / (Sum(Abs($volume - Ref($volume, 1)), {w}) + 1e-12)",
    "VSUMN": "Sum(Greater(Ref($volume, 1) - $volume, 0), {w}) / (Sum(Abs($volume - Ref($volume, 1)), {w}) + 1e-12)",
    "VSUMD": "(Sum(Greater($volume - Ref($volume, 1), 0), {w}) - Sum(Greater(Ref($volume, 1) - $volume, 0), {w}))"
             " / (Sum(Abs($volume - Ref($volume, 1)), {w}) + 1e-12)",
}


def alpha158() -> dict[str, str]:
    out = dict(_KBAR)
    for name, template in _ROLLING.items():
        for w in ALPHA158_WINDOWS:
            out[f"{name}{w}"] = template.format(w=w)
    return out


def expressions() -> dict[str, str]:
    """全部参照：基线、老效应在前"""
    out = {name: expr for name, (expr, _) in {**BASELINES, **KNOWN_EFFECTS}.items()}
    out.update(alpha158())
    return out


def parsed() -> dict[str, Expr]:
    return {name: parse(text, FEATURE_COLUMNS) for name, text in expressions().items()}


@dataclass(frozen=True)
class Grid:
    """参照因子在搜索段主档网格上的截面秩：行与 pool.search 里落在主档网格上的行一一对应"""
    names: tuple[str, ...]
    ranks: np.ndarray  # (行数, 条数) float32

    @classmethod
    def empty(cls) -> "Grid":
        """参照集自己评估时不查冗余，对比集给空的"""
        return cls((), np.empty((0, 0), dtype="float32"))


def grid(store: PanelStore, symbols: Sequence[str], progress: dict[str, Any],
         exprs: dict[str, Expr] | None = None) -> pd.DataFrame:
    """参照逐币求值（一币读一次），只留主档网格上的在池行，每个时刻内转成秩 (0, 1]；列 t、symbol 和各参照名（float32）"""
    exprs = parsed() if exprs is None else exprs
    columns = sorted(set().union(*(e.columns() for e in exprs.values())))
    parts = []
    for s in symbols:
        progress["symbol"] = s
        df = store.load(s, [*columns, POOL_COLUMN])
        t = df.index.to_numpy(dtype="int64")
        keep = (df[POOL_COLUMN].to_numpy() == 1.0) & on_grid(t, GRID_BARS)
        parts.append(pd.DataFrame({"t": t[keep], "symbol": s,
                                   **{name: e.evaluate(df).to_numpy(dtype="float32")[keep] for name, e in exprs.items()}}))
        progress["done"] += 1
    progress["symbol"] = None
    table = pd.concat(parts, ignore_index=True)
    names = list(exprs)
    ranked = table.groupby("t")[names].rank(pct=True).astype("float32")
    return pd.concat([table[["t", "symbol"]], ranked], axis=1)


def load_grid(store: PanelStore) -> pd.DataFrame | None:
    """落盘的参照网格；比面板旧，或参照集加减过条目（列对不上），都当没有，得重算"""
    table = store.load_reference_grid()
    if table is None or set(table.columns) - {"t", "symbol"} != set(parsed()):
        return None
    return table


def align(table: pd.DataFrame, pool: Pool) -> Grid:
    """落盘的参照网格按 (t, 币) 对到 pool.search 的主档行上；对不上的（网格算完之后才进池的币）是 NaN"""
    names = [c for c in table.columns if c not in ("t", "symbol")]
    code = {s: i for i, s in enumerate(pool.symbols)}
    known = table[table["symbol"].isin(list(code))]
    known = known.assign(sym=known["symbol"].map(code).astype("int32")).drop(columns="symbol")
    rows = pool.search
    keys = rows.loc[on_grid(rows["t"].to_numpy(), GRID_BARS), ["t", "sym"]].astype({"sym": "int32"})
    merged = keys.merge(known, on=["t", "sym"], how="left")
    return Grid(tuple(names), merged[names].to_numpy(dtype="float32"))
