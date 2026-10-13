"""挖掘图的提示词：生成角色（提假设、写表达式）与评审角色。

DSL 语法、算子表、列表、窗口换算表从代码里拼，加算子不用改提示词。提示词固定中文，假设与评语的语言按界面语言指定。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from . import ops
from .data.panel import FEATURE_COLUMNS
from .dsl import MAX_WINDOW

_LANG_NAME = {"zh": "中文", "en": "英文"}

OPERATOR_DOCS = {
    "Abs": "绝对值", "Sign": "符号 -1/0/1", "Log": "自然对数，非正数为 NaN",
    "Pow": "x 的 y 次方", "Greater": "两者取大", "Less": "两者取小",
    "Gt": "x > y 出 1 否则 0", "Lt": "x < y 出 1 否则 0", "If": "If(c, a, b)：c 非零取 a 否则取 b",
    "Ref": "n 根 bar 之前的值", "Delta": "当前值减 n 根之前的值",
    "Mean": "窗内均值", "Sum": "窗内求和", "Std": "窗内样本标准差", "Var": "窗内样本方差",
    "Max": "窗内最大", "Min": "窗内最小", "Med": "窗内中位数", "Mad": "窗内平均绝对偏差",
    "Rank": "当前值在窗内的分位 (0,1]", "Skew": "窗内偏度", "Kurt": "窗内超额峰度",
    "EMA": "span=w 的指数均线", "WMA": "线性加权均线，最新一根权重最大",
    "Quantile": "Quantile(x, w, q)：窗内 q 分位数", "IdxMax": "窗内最大值离现在几根 bar", "IdxMin": "窗内最小值离现在几根 bar",
    "Slope": "窗内对 bar 序号回归的斜率", "Rsquare": "该回归的 R²", "Resi": "当前点相对该回归线的残差",
    "Corr": "Corr(x, y, w)：窗内相关系数", "Cov": "Cov(x, y, w)：窗内协方差",
}

COLUMN_DOCS = {
    "open": "开盘价", "high": "最高价", "low": "最低价", "close": "收盘价",
    "volume": "成交量（币）", "quote_volume": "成交额（USDT）", "trades": "成交笔数",
    "taker_buy_base": "主动买入量（币）", "taker_buy_quote": "主动买入额（USDT）",
    "ret": "本根 bar 的对数收益 log(close/上一根 close)",
    "funding_rate": "最近一次已结算的资金费率，8 小时结算一次，期间不变",
    "btc_close": "同一时刻 BTC 收盘价，BTC 自己的面板上是空的", "btc_ret": "同一时刻 BTC 的对数收益，BTC 自己的面板上是空的",
    "liq_long": "本根 bar 内多头被强平的名义额（USDT）", "liq_short": "本根 bar 内空头被强平的名义额（USDT）",
    "liq_count": "本根 bar 内强平笔数", "hour": "bar 开始时刻的 UTC 小时 0~23",
}

WINDOW_TABLE = "4=1h、16=4h、48=12h、96=24h、192=48h、672=7d"

FAMILIES = ["动量", "反转", "突破", "波动", "量能", "流动性", "资金费率 carry", "强平失衡", "taker 买卖失衡", "时段效应"]


# ========== 生成角色 ==========


def system_gen(facts: Mapping[str, Any], lang: str) -> str:
    columns = "\n".join(f"- ${c}：{COLUMN_DOCS.get(c, '')}" for c in FEATURE_COLUMNS)
    operators = "\n".join(f"- {name}：{OPERATOR_DOCS.get(name, '')}" for name in ops.OPERATORS)
    liq = f"强平三列从 {facts['liqFrom']} 起才有数" if facts.get("liqFrom") else "强平三列暂时没有数据"
    return f"""你在给一套加密永续合约的因子挖掘系统提假设、写因子表达式。

数据：{', '.join(facts['symbols'])} 的 15 分钟 K 线，{facts['from']} 到 {facts['to']}，每币约 {facts['rows']} 根。{liq}。
{facts['holdoutFrom']} 起是锁定的 holdout：硬门、评审、你看到的所有数字都只来自它之前的数据；过门的候选会在 holdout 上单独算一次 IC 给人审批看，结果不会反馈给你。
标准答案：因子在 t 时刻的值，与 t 之后 6/12/24 小时的对数涨跌幅算 Spearman 相关（IC），主档 12 小时。只做时序，不做横截面。

可用的列（表达式里写成 $列名）：
{columns}

表达式语法：Python 表达式写法，只允许 + - * / 与负号、数字字面量、$列、下面这些函数。窗口单位是 bar（15 分钟），必须是 1~{MAX_WINDOW} 的整数字面量，换算：{WINDOW_TABLE}。
{operators}

评估是代码硬门，不是人看：
- G0 前视自检；G1 样本量；G2 主档 IC 与替身检验；G3 四段同号；G4 三档同号；G5 扣 {facts['costBps']:g} bps 成本后能赚且排列检验显著；G6 与参照集和库里的因子相关 |ρ| < 0.7；G7 节点 ≤ 40、窗口 ≤ 6、常量 ≤ 6（解析时就会拒）；G8 换个窗口的翻版不算新；G9 加进库里因子的组合之后，主档样本外 IC 要有提升；同批里排在更强候选后面的，还会算一次相对它们的增量写给审批的人看，不是门。
- 替身检验会把裸价格、长窗口均线比、区间位置这类持久量在月内 IC 上的偏差识别出来并拒掉。用收益、比值、z-score、秩这类平稳量。
- 资金费率 8 小时才变一次，对它做短窗差分几乎全是 0。强平列历史短，只用它的因子多半过不了样本量门。
- 参数越少越好，先有一句能说清的经济理由，再写公式。

假设与说明用{_LANG_NAME[lang]}写，因子名用短英文 slug。"""


def propose_message(k: int, library: Sequence[Mapping[str, Any]], tried: Sequence[Mapping[str, Any]],
                    note: str, lang: str) -> str:
    lib = "\n".join(f"- {f['name']}：{f['expr']}（{f['hypothesis']}）" for f in library) or "（空）"
    past = "\n".join(f"- [{t['outcome']}] {t['hypothesis']}｜{t['expr'] or '未写出'}｜{t['reason'] or ''}"
                     for t in tried) or "（无）"
    extra = f"\n本轮方向要求：{note}\n" if note.strip() else ""
    return f"""库里已有的因子（别重复，也别换个窗口翻版）：
{lib}

已经试过的假设与结果（别再提）：
{past}
{extra}
可选的方向：{'、'.join(FAMILIES)}，只是启发不是限制。

提出 {k} 条彼此不同的假设。只输出一个 JSON 数组，不要代码块、不要任何别的字：
[{{"name": "短英文 slug", "family": "方向", "hypothesis": "一两句话（用{_LANG_NAME[lang]}）：什么现象、为什么能预测未来 12 小时的涨跌"}}]"""


def implement_message(cand: Mapping[str, Any], feedback: str | None) -> str:
    head = f"假设 [{cand['name']}]：{cand['hypothesis']}"
    fix = f"\n上一版的问题：{feedback}\n请改正。" if feedback else ""
    return f"""{head}{fix}

把这条假设写成一条表达式。只输出一个 JSON 对象，不要代码块、不要任何别的字：
{{"expr": "表达式", "note": "一句话说明公式怎么对应假设"}}"""


# ========== 评审角色 ==========


def system_review(lang: str) -> str:
    return f"""你是加密永续合约因子研究的评审。数字已经由代码算过并过了硬门，你不对数字投票。你只判三件事：
1. 表达式和假设对不对得上：公式算的是不是假设说的那个东西，方向、窗口、归一化有没有跑偏。
2. 经济上讲不讲得通：这条假设在 15 分钟 K 线、6~24 小时持有期的永续合约上有没有一个说得出口的机制。
3. 硬门看不见的问题：常量凑数、把一个已知因子换皮、依赖历史很短的列、结果实际上由某个退化项主导。

裁决：pass = 可以给人审批入库；revise = 假设对但公式要改，给出具体改法；reject = 假设本身不成立或就是换皮。
评语用{_LANG_NAME[lang]}写。"""


def review_message(cand: Mapping[str, Any], brief: Mapping[str, Any]) -> str:
    return f"""假设 [{cand['name']}]：{cand['hypothesis']}
表达式：{cand['expr']}
公式说明：{cand.get('note') or '（无）'}

代码算出的数字（供参考，你不对它们投票）：
{json.dumps(brief, ensure_ascii=False, indent=1)}

只输出一个 JSON 对象，不要代码块、不要任何别的字：
{{"verdict": "pass|revise|reject", "comment": "一两句话", "suggestion": "revise 时给具体改法，否则留空"}}"""


def json_retry() -> str:
    return "刚才那段不是合法 JSON。重来一次，只输出 JSON 本身，不要代码块、不要任何别的字。"


# ========== 解析与摘要 ==========


def parse_object(text: str) -> dict[str, Any] | None:
    return _first(text, r"\{.*\}", dict)


def parse_list(text: str) -> list[Any] | None:
    return _first(text, r"\[.*\]", list)


def _first(text: str, pattern: str, kind: type) -> Any:
    """模型爱裹代码块或多说两句，取里头的 JSON；贪婪匹配不行再试非贪婪"""
    for candidate in (pattern, pattern.replace(".*", ".*?")):
        found = re.search(candidate, text or "", re.S)
        if not found:
            continue
        try:
            obj = json.loads(found.group())
        except json.JSONDecodeError:
            continue
        return obj if isinstance(obj, kind) else None
    return None


def eval_brief(ev: Mapping[str, Any], gates: Sequence[Mapping[str, Any]], primary: int) -> dict[str, Any]:
    """给评审和页面看的精简数字：三档 IC、主档替身与方向腿、每门一句；算过增量、holdout 的再各带一段。落过库的 eval 键是字符串，先归一"""
    sur = {int(k): v for k, v in ev["horizons"].items()}[primary]["ic"].get("surrogate", {})
    out = {
        "horizons": horizons_brief(ev["horizons"]),
        "surrogate": {"p": _r(sur.get("p")), "nullMean": _r(sur.get("nullMean")), "k": sur.get("k")},
        "redundancy": {"maxAbs": _r(ev["redundancy"]["maxAbs"]), "with": ev["redundancy"]["with"]},
        "complexity": ev["complexity"],
        "gates": [{"id": g["id"], "passed": g["passed"], "note": g["note"]} for g in gates],
    }
    if ev.get("gain"):
        out["gain"] = {_hours(bars): _r(h["gain"]) for bars, h in ev["gain"]["horizons"].items()}
    if ev.get("holdout"):
        out["holdout"] = holdout_brief(ev["holdout"])
    return out


def horizons_brief(horizons: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    """三档的 IC 与方向腿，单因子和组合共用"""
    out = {}
    for bars, h in horizons.items():
        ic = h["ic"]
        out[_hours(bars)] = {
            "meanIc": _r(ic["mean"]), "icir": _r(ic["icir"]), "months": ic["months"],
            "sameSignSymbols": ic["sameSignSymbols"], "segments": [_r(v) for v in ic["segments"]],
            "netReturn": _r(h["leg"]["pooled"].get("netReturn")), "permPct": _r(h["leg"]["pooled"].get("permPct")),
            "hitRate": _r(h["leg"]["pooled"].get("hitRate")), "exposure": _r(h["leg"]["pooled"].get("exposure")),
            "turnover": _r(h["leg"]["pooled"].get("turnover")),
        }
    return out


def holdout_brief(held: Mapping[str, Any]) -> dict[str, Any]:
    return {"since": held["since"], "horizons": {
        _hours(bars): {"meanIc": _r(h["mean"]), "icir": _r(h["icir"]), "months": h["months"],
                       "sameSignSymbols": h["sameSignSymbols"], "samples": h["samples"]}
        for bars, h in held["horizons"].items()}}


def _hours(bars: Any) -> str:
    return f"{int(bars) // 4}h"


def _r(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else round(f, 4)
