"""校准脚本：把三类已知因子过一遍全部门，看门本身对不对。

  uv run python -m wiib_alpha.calibrate --synthetic            随机行情：验假阳率与泄漏捕获
  uv run python -m wiib_alpha.calibrate --data-dir ./data      真数据：看基线与参照集能活几个

验收：随机因子不做同批 BH 的单条通过率 ≤ 10%（这才是门本身的判别力），泄漏因子 100% 被 G0 抓住；
同批 BH 按生产的 K 分批再算一遍只做参考。基线与参照集的门表打印出来人读。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from . import ops, reference
from .config import DEFAULT_SYMBOLS
from .data.binance import BAR_MS, KLINE_COLUMNS
from .data.panel import FEATURE_COLUMNS, PanelStore, assemble
from .dsl import DslError, Expr, parse
from .eval.evaluate import evaluate
from .eval.gates import apply_fdr, judge, passed
from .eval.params import EvalParams, Thresholds

RANDOM_COLUMNS = ("close", "volume", "ret", "quote_volume", "taker_buy_base", "high", "low", "funding_rate")
RANDOM_ROLLING = ("Mean", "Std", "Sum", "Max", "Min", "Rank", "Delta", "EMA", "Slope", "Skew", "WMA")
RANDOM_WINDOWS = (4, 16, 48, 96, 192)
# Lead 只在这里临时注册，正式算子表里没有
LEAKY = {
    "lead_ret": "Lead($ret, 1)",
    "oracle_12h": "Lead(Sum($ret, 48), 48)",
    "hidden_lead": "Sum($ret, 4) - Lead($ret, 2) * 0.1",
}
GATE_IDS = [f"G{i}" for i in range(9)]


# ========== 数据 ==========


def synthetic_panels(symbols: tuple[str, ...], days: int, seed: int) -> dict[str, pd.DataFrame]:
    """随机游走行情，列结构与真 panel 一样；没有强平数据"""
    n = days * 96
    rng = np.random.default_rng(seed)
    base = 1_700_000_000_000
    klines: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    for s in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
        open_ = np.concatenate([[close[0]], close[:-1]])
        spread = np.abs(rng.normal(0, 0.002, n))
        volume = rng.lognormal(5, 0.6, n)
        taker = volume * rng.uniform(0.3, 0.7, n)
        rows = np.column_stack([
            base + np.arange(n) * BAR_MS, open_, np.maximum(open_, close) * (1 + spread),
            np.minimum(open_, close) * (1 - spread), close, volume, base + np.arange(1, n + 1) * BAR_MS - 1,
            volume * close, rng.integers(10, 500, n), taker, taker * close,
        ])
        klines[s] = pd.DataFrame(rows, columns=KLINE_COLUMNS).astype({"open_time": "int64", "close_time": "int64", "trades": "int64"})
        times = klines[s]["close_time"].to_numpy()[31::32]
        funding[s] = pd.DataFrame({"funding_time": times, "funding_rate": rng.normal(0, 1e-4, len(times))})
    return {s: assemble(klines[s], funding[s], klines[symbols[0]]) for s in symbols}


def real_panels(data_dir: str, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    store = PanelStore(data_dir)
    return {s: store.load(s) for s in symbols}


# ========== 三类因子 ==========


def random_expression(rng: np.random.Generator, panel: pd.DataFrame) -> Expr:
    """随机拼一条表达式；拼出来全 NaN 或不动的重拼"""
    def leaf() -> str:
        return f"${rng.choice(RANDOM_COLUMNS)}"

    def node(depth: int) -> str:
        if depth == 0 or rng.random() < 0.25:
            return leaf()
        kind = rng.integers(0, 3)
        if kind == 0:
            return f"{rng.choice(RANDOM_ROLLING)}({node(depth - 1)}, {rng.choice(RANDOM_WINDOWS)})"
        if kind == 1:
            return f"({node(depth - 1)} {rng.choice(['+', '-', '*', '/'])} {node(depth - 1)})"
        return f"Corr({node(depth - 1)}, {node(depth - 1)}, {rng.choice(RANDOM_WINDOWS)})"

    while True:
        try:
            expr = parse(node(3), FEATURE_COLUMNS)
        except DslError:
            continue
        out = expr.evaluate(panel).to_numpy()
        finite = np.isfinite(out)
        if finite.mean() > 0.5 and np.nanstd(out) > 0:
            return expr


def run(name: str, exprs: dict[str, Expr], panels: dict[str, pd.DataFrame], params: EvalParams,
        thresholds: Thresholds, refs: dict[str, pd.DataFrame] | None, known: set[str]) -> list[dict[str, Any]]:
    out = []
    print(f"\n== {name}：{len(exprs)} 条 ==", flush=True)
    for i, (label, expr) in enumerate(exprs.items(), 1):
        t = time.perf_counter()
        ev = evaluate(expr, panels, params, refs)
        ev["name"] = label
        ev["gates"] = judge(ev, thresholds, params.primary, known)
        ev["passed"] = passed(ev["gates"])
        out.append(ev)
        print(f"  [{i}/{len(exprs)}] {label} {time.perf_counter() - t:.1f}s", flush=True)
    return out


# ========== 打印 ==========


def table(evals: list[dict[str, Any]], primary: int) -> str:
    head = (f"{'name':22s} " + " ".join(f"{g:>2s}" for g in GATE_IDS)
            + f" {'IC12':>8s} {'null':>7s} {'p':>5s} {'ICIR':>7s} {'seg':>3s} {'perm':>6s} {'net':>8s} {'red':>5s}  verdict")
    lines = [head, "-" * len(head)]
    for ev in evals:
        gates = {g["id"]: g for g in ev["gates"]}
        ic = ev["horizons"][primary]["ic"]
        sur = ic.get("surrogate", {})
        leg = ev["horizons"][primary]["leg"]["pooled"]
        lines.append(
            f"{ev['name'][:22]:22s} " + " ".join(f"{'Y' if gates[g]['passed'] else '.':>2s}" for g in GATE_IDS)
            + f" {ic['mean']:8.4f} {sur.get('nullMean', float('nan')):7.3f} {sur.get('p', float('nan')):5.2f}"
            + f" {ic['icir']:7.2f} {gates['G3']['value']['sameSign']:3d}"
            + f" {leg.get('permPct', 0):6.3f} {leg.get('netReturn', float('nan')):8.3f} {ev['redundancy']['maxAbs']:5.2f}"
            + f"  {'PASS' if ev['passed'] else 'fail'}"
        )
    return "\n".join(lines)


def gate_counts(evals: list[dict[str, Any]]) -> str:
    counts = {g: sum(1 for ev in evals if next(x for x in ev["gates"] if x["id"] == g)["passed"]) for g in GATE_IDS}
    return "  ".join(f"{g}:{c}/{len(evals)}" for g, c in counts.items())


def _window_of(name: str) -> str:
    """Alpha158 名字末尾的窗口数字；KBAR 那 9 个没有窗口，返回空串"""
    return name[len(name.rstrip("0123456789")):]


# ========== 入口 ==========


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="因子硬门校准")
    ap.add_argument("--synthetic", action="store_true", help="用随机游走行情，不读 data 目录")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--random", type=int, default=100, help="随机因子条数")
    ap.add_argument("--k", type=int, default=3, help="同批 BH 的批大小，按生产一轮的候选数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--skip-alpha158", action="store_true", help="只跑 5 个基线，不跑 Alpha158")
    ap.add_argument("--alpha158-windows", default="", help="只跑这些窗口的 Alpha158，逗号分隔；空 = 全部")
    ap.add_argument("--surrogates", type=int, default=20, help="替身个数")
    args = ap.parse_args(argv)

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    params, thresholds = EvalParams(surrogates=args.surrogates), Thresholds()
    print(f"参数 {params}\n阈值 {thresholds}")
    t0 = time.perf_counter()
    panels = synthetic_panels(symbols, args.days, args.seed) if args.synthetic else real_panels(args.data_dir, symbols)
    print(f"面板 {len(panels)} 币，每币 {min(len(p) for p in panels.values())}~{max(len(p) for p in panels.values())} 行，"
          f"{time.perf_counter() - t0:.1f}s")

    ref_exprs = reference.parsed()
    if args.skip_alpha158:
        ref_exprs = {k: v for k, v in ref_exprs.items() if k in reference.BASELINES}
    elif args.alpha158_windows:
        keep = {w.strip() for w in args.alpha158_windows.split(",") if w.strip()}
        ref_exprs = {k: v for k, v in ref_exprs.items()
                     if k in reference.BASELINES or _window_of(k) in keep | {""}}
    t0 = time.perf_counter()
    ref_values = reference.values(panels, ref_exprs)
    known = {e.canonical() for e in ref_exprs.values()}
    print(f"参照集 {len(ref_exprs)} 条算好，{time.perf_counter() - t0:.1f}s")

    # 参照集自己不查冗余、不做 BH：它们是尺子不是候选
    ref_evals = run("参照集", ref_exprs, panels, params, thresholds, None, set())
    baselines = [ev for ev in ref_evals if ev["name"] in reference.BASELINES]
    alpha = [ev for ev in ref_evals if ev["name"] not in reference.BASELINES]

    rng = np.random.default_rng(args.seed)
    first = panels[symbols[0]]
    random_exprs = {f"rand{i:02d}": random_expression(rng, first) for i in range(args.random)}
    random_evals = run("随机因子", random_exprs, panels, params, thresholds, ref_values, known)
    single_passes = sum(ev["passed"] for ev in random_evals)
    # 同批 BH 按生产的 K 分批做，和一轮的候选数一样
    for start in range(0, len(random_evals), args.k):
        apply_fdr(random_evals[start:start + args.k], thresholds.fdr_q, params.iterations)

    ops.OPERATORS["Lead"] = ops.Op("rolling", lambda x, n: x.shift(-n))
    try:
        leaky_exprs = {k: parse(v, FEATURE_COLUMNS) for k, v in LEAKY.items()}
        leaky_evals = run("泄漏因子", leaky_exprs, panels, params, thresholds, ref_values, known)
    finally:
        del ops.OPERATORS["Lead"]

    print("\n===== 基线 =====")
    print(table(baselines, params.primary))
    if alpha:
        print(f"\n===== Alpha158 参照集（{len(alpha)} 条）=====")
        print("各门通过数：" + gate_counts(alpha))
        survivors = [ev for ev in alpha if ev["passed"]]
        print(f"全过的 {len(survivors)} 条：" + ", ".join(ev["name"] for ev in survivors))
        print(table(sorted(alpha, key=lambda e: -abs(e["horizons"][params.primary]["ic"]["mean"]))[:15], params.primary))
        print("（按主档 |IC| 排的前 15 条）")
    print("\n===== 随机因子 =====")
    print(table(random_evals, params.primary))
    for ev in random_evals:
        print(f"  {ev['name']}: {ev['expr']}")
    print("\n===== 泄漏因子 =====")
    print(table(leaky_evals, params.primary))

    n = max(1, len(random_evals))
    single_rate = single_passes / n
    batch_rate = sum(ev["passed"] for ev in random_evals) / n
    g2_rate = sum(1 for ev in random_evals if next(g for g in ev["gates"] if g["id"] == "G2")["passed"]) / n
    caught = sum(1 for ev in leaky_evals if not ev["lookaheadFree"]) / max(1, len(leaky_evals))
    print(f"\n随机因子：单条通过率 {single_rate:.0%}（要求 ≤ 10%）  G2 单项通过率 {g2_rate:.0%}（名义 5%）  "
          f"按 K={args.k} 同批 BH 后 {batch_rate:.0%}")
    print(f"泄漏被 G0 抓住 {caught:.0%}（要求 100%）")
    ok = single_rate <= 0.10 and caught == 1.0
    print("校准结论：" + ("门在工作" if ok else "门有问题，别接 LLM"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
