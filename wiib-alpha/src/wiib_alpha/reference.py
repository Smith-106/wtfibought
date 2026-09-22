"""参照因子：5 个手写基线（与 Java ContinuousFactorSignals 同口径）+ Qlib Alpha158 的模板按 15m 窗口改写。

参照集只做三件事：冗余检查的靠山、结构原创的比对、校准时看"教科书因子在 15m 加密上活几个"。不可入库。
"""

from __future__ import annotations

import pandas as pd

from .data.panel import FEATURE_COLUMNS
from .dsl import Expr, parse

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

# Alpha158 的日频窗口 5/10/20/30/60 换成 1h/4h/12h/24h/48h
ALPHA158_WINDOWS = (4, 16, 48, 96, 192)

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
    """全部参照：基线在前"""
    out = {name: expr for name, (expr, _) in BASELINES.items()}
    out.update(alpha158())
    return out


def parsed() -> dict[str, Expr]:
    return {name: parse(text, FEATURE_COLUMNS) for name, text in expressions().items()}


def values(panels: dict[str, pd.DataFrame], exprs: dict[str, Expr] | None = None) -> dict[str, pd.DataFrame]:
    """一币一张，列是参照因子名；给 evaluate 的 references 用"""
    exprs = parsed() if exprs is None else exprs
    return {s: pd.DataFrame({name: e.evaluate(p) for name, e in exprs.items()}, index=p.index)
            for s, p in panels.items()}
