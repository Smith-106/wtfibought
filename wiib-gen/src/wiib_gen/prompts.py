"""生成器自己的提示词与题库。

系统提示词固定中文拼出来（平台模板与默认指令按用户的 agent 语言原样嵌进去），
正文与 notes 用哪门语言由输出规则那段指定，不靠系统提示词的语言带。
题库只有 zh/en 两份，按界面语言取。
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Mapping, Sequence

# 指令正文上限，与 Java 侧 TraderService 的校验同数
MAX_CHARS = 4000

# 正文与 notes 的分界：模型单独起一行写它
NOTES_MARK = "=====NOTES====="

# 风格 → 单笔风险（占权益 %），顺序与题库里的 styles 标签一一对上
_RISK_PCT = [("conservative", 0.5), ("balanced", 1.0), ("aggressive", 2.0)]

# 15m 上比这个还小的止损距离基本落在噪音里；别的周期按时间的平方根缩放
NOISE_STOP_PCT = 0.3
_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}

# 护栏认的论点标签与两门语言的叫法，顺序同 TradeGuard.PLAY_TYPES；OTHER 不当主打法选项
_PLAYS = [("BREAKOUT", "突破", "Breakout"), ("PULLBACK", "顺势回调", "Pullback"), ("REVERSAL", "反转", "Reversal"),
          ("TREND_FOLLOW", "趋势跟随", "Trend following"), ("RANGE", "区间震荡", "Range"),
          ("NEWS", "消息面", "News-driven"), ("FUNDING", "资金费率", "Funding-rate")]


def pick_lang(raw: str | None) -> str:
    """只认 zh/en，其它一律按 zh"""
    return "en" if (raw or "").strip().lower().startswith("en") else "zh"


# ========== 长度口径：与 Java String.length() / JS .length 一样按 UTF-16 单元算 ==========


def ulen(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def utrunc(text: str, limit: int) -> str:
    """截到 limit 个 UTF-16 单元，不把 emoji 这种代理对切成半个"""
    units = 0
    for i, ch in enumerate(text):
        width = 2 if ord(ch) >= 0x10000 else 1
        if units + width > limit:
            return text[:i]
        units += width
    return text


# ========== 题库 ==========

_BANK: dict[str, dict[str, Any]] = {
    "zh": {
        "mode": ("你已经有一份交易指令了，这次怎么弄？", ["在现有指令上改", "从零写一份"]),
        "direction": ("方向上你想让它怎么做？", ["只做多", "多空都做", "只做空"]),
        "holding": ("一笔单大概拿多久？", ["日内几根 K 线", "波段 1~3 天", "更长"]),
        "style": ("风格偏哪边？", ["保守", "均衡", "激进"]),
        "play": ("主打法想用哪种？", [zh for _, zh, _ in _PLAYS]),
        "free": ("还想调整什么？", []),
        "styles": ["保守", "均衡", "激进"],
    },
    "en": {
        "mode": ("You already have a trading instruction. What now?", ["Edit the existing one", "Start from scratch"]),
        "direction": ("Which side should it trade?", ["Long only", "Both sides", "Short only"]),
        "holding": ("How long should a trade be held?", ["Intraday, a few candles", "Swing, 1-3 days", "Longer"]),
        "style": ("Which style fits you?", ["Conservative", "Balanced", "Aggressive"]),
        "play": ("Which play should be the main one?", [en for _, _, en in _PLAYS]),
        "free": ("Anything else you want to change?", []),
        "styles": ["Conservative", "Balanced", "Aggressive"],
    },
}


def _item(lang: str, key: str, allow_free: bool = True) -> dict[str, Any]:
    text, options = _BANK[lang][key]
    return {"key": key, "text": text, "options": list(options), "allow_free": allow_free}


def _screen(kind: str, items: list[dict[str, Any]], skip: bool = False, finish: bool = False) -> dict[str, Any]:
    return {"kind": kind, "items": items, "allow_skip": skip, "allow_finish": finish}


def mode_question(lang: str) -> dict[str, Any]:
    return _screen("mode", [_item(lang, "mode", allow_free=False)])


def base_question(lang: str) -> dict[str, Any]:
    return _screen("base", [_item(lang, k) for k in ("direction", "holding", "style")])


def play_question(lang: str) -> dict[str, Any]:
    return _screen("play", [_item(lang, "play")], skip=True, finish=True)


def free_question(lang: str) -> dict[str, Any]:
    """模型出不了题时的兜底：一道自由输入"""
    return _screen("detail", [_item(lang, "free")], skip=True, finish=True)


def detail_screen(item: dict[str, Any]) -> dict[str, Any]:
    return _screen("detail", [item], skip=True, finish=True)


def is_incremental(answer: str, lang: str) -> bool:
    """选了「在现有指令上改」"""
    return answer.strip() == _BANK[lang]["mode"][1][0]


# ========== 风险表 ==========


def risk_table(spec: Mapping[str, Any], lang: str) -> list[dict[str, Any]]:
    """三档风格各算一行止损距离可达范围。

    模板公式：风险 = 止损距离% × 杠杆 × 保证金占比 → 止损距离 = 风险 / (杠杆 × 占比)。
    占比用小数，杠杆与占比都取到两端算出整段范围。
    """
    lev_min, lev_max = float(spec["leverageMin"]), float(spec["leverageMax"])
    m_min, m_max = float(spec["marginPctMin"]) / 100, float(spec["marginPctMax"]) / 100
    rows = []
    for (_key, risk), label in zip(_RISK_PCT, _BANK[lang]["styles"], strict=True):
        rows.append({
            "style": label,
            "risk_pct": risk,
            "stop_min": _round2(risk / (lev_max * m_max)),
            "stop_max": _round2(risk / (lev_min * m_min)),
        })
    return rows


def _round2(value: float) -> float:
    """四舍五入到两位；内置 round 是银行家舍入，0.125 会被抹成 0.12"""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def noise_stop_pct(interval_code: str) -> float:
    """这个周期上止损距离的噪音线（%）"""
    minutes = int(interval_code[:-1]) * _UNIT_MINUTES[interval_code[-1]]
    return _round2(NOISE_STOP_PCT * math.sqrt(minutes / 15))


# ========== 生成指令的提示词 ==========

_LANG_NAME = {"zh": "中文", "en": "英文"}

_PLAY_TAGS = "、".join(f"{tag} {zh}" for tag, zh, _ in _PLAYS) + "、OTHER 其它"

_HARNESS_RULES = f"""harness 的硬现实（指令必须绕开，不能装看不见）：
- 挂不了一排单：网格、马丁这类「铺一层层限价单等回调」的玩法做不了。最接近的可行写法是分批建仓 + 分批止盈。
- 限价方向是死的：买入限价必须低于现价，卖出限价必须高于现价；反了会当场成交，不算挂单。想立刻进场只能用市价单。
- 挂单也占仓位数：单仓模式下一张没成交的限价单就占掉了唯一的坑，再开新仓会被护栏拒。
- 最多 3 个可交易币，且币种由主人在配置页定，指令改不了。
- 每根 K 线才醒一次，单轮最多 12 次模型调用（最后一次要留给写结论）。不要写「持续盯盘」「每分钟检查」「一小时内复查」这类节奏。
- 开仓必须带论点标签、失效条件、止损价、止盈价，缺一样护栏直接拒；想让利润奔跑只能先把止盈设远再调。失效条件只能是市场条件，不能是盈亏数字。数据引用护栏不查，指令里照样要求写。
- 论点标签只认这几个：{_PLAY_TAGS}。指令里提到打法就用这些名字（带英文标签），别造新词。
- 没有「暂停交易」这种工具：回撤到几个点就停手、连亏几笔就休息，只能写成纪律条文由交易员自己遵守，系统不会强制执行。
- 指令正文不超过 {MAX_CHARS} 字。"""


def _spec_block(facts: Mapping[str, Any]) -> str:
    spec = facts["spec"]
    lines = [
        "当前仓位规格（主人在配置页定的硬参数，指令改不了，只能在区间内挑）：",
        f"- 杠杆区间 {spec['leverageMin']}~{spec['leverageMax']} 倍",
        f"- 单笔开新仓的保证金占权益 {spec['marginPctMin']}%~{spec['marginPctMax']}%",
        "- 可同时持有多个仓位" if spec["allowMultiPosition"] else "- 全账户同时只能有一个仓位（挂单也占坑）",
        "- 允许同币多空双开" if spec["allowHedge"] else "- 同一个币不能同时做多做空",
        f"- 可交易标的 {facts['symbols']}，每根 {facts['interval_code']} K 线唤醒一次",
    ]
    if facts.get("wake_window"):
        lines.append(f"- 只在 {facts['wake_window']}（北京时间）内唤醒，时段外不做例行唤醒")
    return "\n".join(lines)


def _risk_block(facts: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"- {r['style']}：单笔风险 {r['risk_pct']}% 权益 → 止损距离可取 {r['stop_min']}% ~ {r['stop_max']}%"
        for r in facts["risk_table"]
    )
    return (
        "风险档位（代码按上面的规格用「风险 = 止损距离% × 杠杆 × 保证金占比」反算好了，直接引用，不要自己算）：\n"
        f"{rows}\n"
        f"{facts['interval_code']} 周期上，止损距离低于 {noise_stop_pct(facts['interval_code'])}% 基本落在噪音里，随时被扫。"
        "某一档的整段范围都低于它，就在 notes 里明说这档在当前规格下不可行，并指出哪一档可行。"
    )


def _output_block(facts: Mapping[str, Any]) -> str:
    body_lang = _LANG_NAME[facts["output_lang"]]
    note_lang = _LANG_NAME[facts["dialog_lang"]]
    return f"""输出规则：
1. 先写指令正文，然后单独一行 {NOTES_MARK}，再写 notes。这一行只出现一次，正文里不许出现。
2. 正文用{body_lang}写，notes 用{note_lang}写。
3. 正文不超过 {MAX_CHARS} 字，只写指令本身：不要标题、不要前言、不要解释自己在干什么。
4. notes 写两件事：这一版相对上一版改了什么；用户要求里 harness 做不到的，最接近的可行写法是什么。
5. 用户要求做不到时，正文里落成最接近的可行写法，不要照抄做不到的说法，也不要闷声丢掉。"""


def system_prompt(facts: Mapping[str, Any]) -> str:
    """一轮一轮不变的那部分：角色 + 平台模板 + 底稿 + harness 规矩 + 规格数字 + 输出规则"""
    parts = [
        "你在给一个 AI 永续合约交易员写「主人交易指令」。这段指令会被平台原样注入下面那份系统提示词的"
        "「主人的交易指令」段，交易员的方法、风格与纪律以它为准。\n"
        "指令只管方法、风格、纪律三件事；环境、工具、仓位规格区间、护栏、收尾格式都由系统强制执行。",
        "===== 平台系统提示词（原文，只读） =====\n"
        f"{facts['template']}\n"
        "===== 平台系统提示词结束 =====\n"
        "上面这份里的环境、工具、仓位规格区间、护栏、收尾格式一律系统强制，指令覆盖不了。"
        "你写的指令里不要重复它们，也不要试图改它们。",
        "===== 平台默认指令（底稿与格式参考） =====\n"
        f"{facts['default_instructions']}\n"
        "===== 默认指令结束 =====\n"
        "格式照它来：【方法】【持仓管理】【仓位】【观望与警报】四段，段可增可删；每条一行，给得出可执行的判据。",
        _HARNESS_RULES,
        _spec_block(facts),
        _risk_block(facts),
    ]
    if facts.get("mode") == "incremental" and facts.get("existing_prompt"):
        parts.append(
            "本次是在主人现有指令上改：以下面这份为基础改，不要推倒重写，没被要求动的段落原样留着。\n"
            "===== 现有指令 =====\n"
            f"{facts['existing_prompt']}\n"
            "===== 现有指令结束 ====="
        )
    parts.append(_output_block(facts))
    return "\n\n".join(parts)


def user_message(answers: Sequence[Mapping[str, str]], last_text: str,
                 fresh: Sequence[Mapping[str, str]]) -> str:
    """每轮变的那部分：全部问答 + 上一版正文 + 本轮新答"""
    parts = ["已有的问答：\n" + _qa_lines(answers)] if answers else ["还没有问答。"]
    if last_text:
        parts.append(f"上一版指令正文：\n{last_text}")
    if fresh:
        parts.append("本轮新答：\n" + _qa_lines(fresh))
    parts.append("按上面的规则给出新一版。")
    return "\n\n".join(parts)


def _qa_lines(answers: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(f"- [{a['key']}] {a['question']} → {a['answer']}" for a in answers)


def shrink_message() -> str:
    return (f"上面这版正文超过 {MAX_CHARS} 字了。把它压到 {MAX_CHARS} 字以内重写一遍，"
            f"砍重复和铺垫，判据和数字一个不能少。格式不变：正文 + 单独一行 {NOTES_MARK} + notes。")


def truncated_note(lang: str) -> str:
    if lang == "en":
        return f"(Body still exceeded {MAX_CHARS} chars after a rewrite and was cut at {MAX_CHARS}; the tail is missing.)"
    return f"（重写后正文仍超 {MAX_CHARS} 字，已在 {MAX_CHARS} 字处截断，尾部内容缺失。）"


# ========== 让模型出细节题 ==========

DETAIL_SYSTEM = """你在帮用户打磨一份 AI 永续合约交易员的交易指令。现在只做一件事：挑下一个该问的细节，出一道单选题。
题目要冲着当前这版指令里还模糊的地方去（退出方式、加仓规则、回撤停手、事件回避、止损锚定、时段偏好等），
选项都得是这个 harness 里做得到的写法：挂不了一排单、限价方向固定、单仓模式下挂单占坑、最多 3 个币、每根 K 线才醒一次、没有暂停交易工具。"""


def detail_message(facts: Mapping[str, Any], answers: Sequence[Mapping[str, str]],
                   current_text: str, asked: Sequence[str]) -> str:
    return f"""已问过的 key（不要再问这些）：{", ".join(asked)}

已有的问答：
{_qa_lines(answers)}

当前这版指令正文：
{current_text}

出一道新的细节题，只输出一个 JSON 对象，不要代码块、不要任何别的字：
{{"key": "短英文 slug", "text": "题目（用{_LANG_NAME[facts['dialog_lang']]}写）", "options": ["选项1", "选项2", "选项3"]}}
options 给 3~4 个，互斥、具体、每个都是能直接写进指令的做法。"""


def json_retry() -> str:
    return "刚才那段不是合法 JSON。重来一次，只输出那个 JSON 对象本身，不要代码块、不要任何别的字。"
