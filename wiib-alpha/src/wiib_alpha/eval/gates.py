"""硬门：拿 evaluate 的结果逐门判，每门给数字、阈值、过没过、一句原因。

G5 的 BH-FDR 是同批候选一起判的，单条先按原始排列分位判，批次汇总时 apply_fdr 再改判。
替身 p 不做 BH：它是离散的，分辨率只有 1/(k+1)，多重检验的账记在 G5 上。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .params import Thresholds
from .stats import bh_reject


def judge(ev: dict[str, Any], t: Thresholds, primary: int = 48,
          known_canonicals: Iterable[str] = ()) -> list[dict[str, Any]]:
    h = ev["horizons"]
    main = h[primary]["ic"]
    mean_ic = main["mean"]
    surrogate = main.get("surrogate", {})
    sp = surrogate.get("p", np.nan)
    null_mean = surrogate.get("nullMean")
    others = [b for b in h if b != primary]
    same_sign_h = [np.sign(h[b]["ic"]["mean"]) == np.sign(mean_ic) for b in others
                   if np.isfinite(h[b]["ic"]["mean"])]
    segs = main["segments"]
    seg_same = sum(1 for v in segs if np.isfinite(v) and np.isfinite(mean_ic) and np.sign(v) == np.sign(mean_ic))
    pooled = h[primary]["leg"]["pooled"]
    net = pooled.get("netReturn", np.nan)
    perm = pooled.get("permPct", 0.0)
    cx = ev["complexity"]
    red = ev["redundancy"]
    longest = max(h)
    samples = h[longest]["ic"]["samples"]

    gates = [
        _gate("G0", "前视自检", ev["lookaheadFree"], ev["lookaheadFree"], True,
              "截断重算与全量一致" if ev["lookaheadFree"] else "截断重算结果变了，算子偷看了未来"),
        _gate("G1", "最小样本", samples >= t.min_samples, {"samples": samples}, {"min": t.min_samples},
              f"{longest} 根档非重叠 OOS 点 {samples}"),
        # 替身保留了大半 IC，说明真值主要是持久量偏差；裸价格真值只比替身极端一点点，p 停在 1/21 就骗过了 pMax
        _gate("G2", "IC",
              _finite(mean_ic) and abs(mean_ic) >= t.ic_min and _finite(main["icir"]) and abs(main["icir"]) >= t.icir_min
              and main["sameSignSymbols"] >= t.min_symbols_same_sign and _finite(sp) and sp <= t.ic_p
              and _finite(null_mean) and abs(null_mean) <= t.surrogate_null_ratio * abs(mean_ic),
              {"meanIc": mean_ic, "icir": main["icir"], "sameSignSymbols": main["sameSignSymbols"],
               "months": main["months"], "surrogateP": sp, "surrogateK": surrogate.get("k", 0),
               "surrogateMean": null_mean},
              {"icMin": t.ic_min, "icirMin": t.icir_min, "sameSignSymbols": t.min_symbols_same_sign, "pMax": t.ic_p,
               "nullRatio": t.surrogate_null_ratio},
              f"主档月度 IC 均值 {fmt(mean_ic)}，ICIR {fmt(main['icir'])}，{main['sameSignSymbols']} 个币同号，"
              f"替身 IC 均值 {fmt(surrogate.get('nullMean'))}，p={fmt(sp)}（分辨率 1/{surrogate.get('k', 0) + 1}）"),
        _gate("G3", "四段稳定", seg_same >= t.min_segments, {"segments": segs, "sameSign": seg_same},
              {"min": t.min_segments}, f"四段 IC {[fmt(v) for v in segs]}，{seg_same} 段与总体同号"),
        _gate("G4", "衰减", bool(same_sign_h) and all(same_sign_h),
              {b: h[b]["ic"]["mean"] for b in h}, "三档同号",
              "三档 IC " + "、".join(f"{b}:{fmt(h[b]['ic']['mean'])}" for b in h)),
        _gate("G5", "可交易", _finite(net) and net > 0 and perm >= t.perm_pct,
              {"netReturn": net, "permPct": perm, "hitRate": pooled.get("hitRate"), "periods": pooled.get("periods")},
              {"netReturn": "> 0", "permPct": t.perm_pct, "fdrQ": t.fdr_q},
              f"主档非重叠净收益 {fmt(net)}，排列分位 {fmt(perm)}"),
        _gate("G6", "冗余", red["maxAbs"] < t.max_redundancy, red, {"max": t.max_redundancy},
              f"与 {red['with']} 的 |ρ| = {fmt(red['maxAbs'])}" if red["with"] else "没有参照因子"),
        _gate("G7", "复杂度",
              cx["nodes"] <= t.max_nodes and cx["windows"] <= t.max_windows and cx["constants"] <= t.max_constants,
              cx, {"nodes": t.max_nodes, "windows": t.max_windows, "constants": t.max_constants},
              f"节点 {cx['nodes']}，窗口 {cx['windows']}，常量 {cx['constants']}"),
        origin_gate(ev["canonical"], known_canonicals),
        gain_gate(ev["gain"], primary),
    ]
    return gates


def origin_gate(canonical: str, known: Iterable[str]) -> dict[str, Any]:
    """G8：规范形不能和已知的重合。同批阶段再对排在前面的候选查一次"""
    seen = canonical in set(known)
    return _gate("G8", "结构原创", not seen, canonical, "不与已知重合", "规范形与已知因子重合" if seen else "规范形是新的")


def gain_gate(gain: dict[str, Any] | None, primary: int) -> dict[str, Any]:
    """G9：加进组合后主档 OOS IC 要提升。同批阶段排在别的候选后面会重算一次。落过库的键是字符串，先归一"""
    if gain is None:
        return _gate("G9", "增量", True, None, {"gain": "> 0"}, "没算对组合的增量")
    g = {int(k): v for k, v in gain["horizons"].items()}[primary]
    return _gate("G9", "增量", _finite(g["gain"]) and g["gain"] > 0, gain, {"gain": "> 0"},
                 f"加进组合后主档 OOS IC {fmt(g['base'])} → {fmt(g['with'])}（增量 {fmt(g['gain'])}），"
                 f"{gain['months']} 个月里 {gain['monthsBetter']} 个月更好")


def apply_fdr(evals: list[dict[str, Any]], q: float, iterations: int, extra_tests: int = 0) -> None:
    """同批候选的排列 p 做 BH，没扛住的把 G5 改成不过；就地改 gates 与 passed。

    extra_tests 是同批里被评审打回改掉的旧版本数：各按 p=1 占一个位置，只把 m 撑大。
    """
    perm_ps = [max(1.0 - _gate_value(ev, "G5", "permPct", 0.0), 1.0 / iterations) for ev in evals]
    for ev, keep in zip(evals, bh_reject(perm_ps + [1.0] * extra_tests, q)):
        _flip(ev, "G5", keep)
        ev["passed"] = all(g["passed"] for g in ev["gates"])


def passed(gates: list[dict[str, Any]]) -> bool:
    return all(g["passed"] for g in gates)


def _flip(ev: dict[str, Any], gate_id: str, keep: bool) -> None:
    gate = next(g for g in ev["gates"] if g["id"] == gate_id)
    if gate["passed"] and not keep:
        gate["passed"] = False
        gate["note"] += "；同批 BH-FDR 没扛住"


def _gate_value(ev: dict[str, Any], gate_id: str, key: str, default: float) -> float:
    gate = next(g for g in ev["gates"] if g["id"] == gate_id)
    value = gate["value"].get(key)
    return float(value) if value is not None and np.isfinite(value) else default


def _gate(id_: str, name: str, ok: Any, value: Any, threshold: Any, note: str) -> dict[str, Any]:
    return {"id": id_, "name": name, "passed": bool(ok), "value": value, "threshold": threshold, "note": note}


def _finite(v: Any) -> bool:
    return v is not None and np.isfinite(v)


def fmt(v: Any) -> str:
    return "NaN" if v is None or not np.isfinite(v) else f"{v:.4f}"
