"""评估参数与硬门阈值。

EvalParams 是 Java EvalParams.defaults() 的 5m 数字换算到 15m；Thresholds 是硬门默认值，阶段 A 校准后定稿。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dsl import COMPLEXITY

BARS_PER_HOUR = 4
HOURS_PER_YEAR = 8760


@dataclass(frozen=True)
class EvalParams:
    test_size: int = 192        # 2 天
    embargo: int = 4            # 1 小时
    min_train: int = 2880       # 30 天
    iterations: int = 1000      # 排列次数
    seed: int = 42
    cost_bps: float = 10.0      # 每笔往返
    primary: int = 48           # 主档 12h
    lookahead_cuts: int = 20    # 前视自检截断点数
    min_points: int = 8         # 一个月/一段少于这么多非重叠点不算 IC
    redundancy_stride: int = 4  # 算冗余时每几行取一行
    segments: int = 4
    surrogates: int = 100       # 替身个数，p 的分辨率是 1/(k+1)；0 = 不做，G2 直接不过
    surrogate_block: int = 8    # 替身打乱的块长，2 小时
    holdout_days: int = 90      # 最近这么多天锁成 holdout：门和模型都看不到，过门后只算一次 IC 给人看

    @property
    def cost(self) -> float:
        return self.cost_bps / 1e4


@dataclass(frozen=True)
class Thresholds:
    min_samples: int = 300              # G1：24h 档非重叠 OOS 点（全部币合计）
    ic_min: float = 0.02                # G2：主档 |月度 IC 均值|
    icir_min: float = 0.3               # G2：主档 ICIR
    ic_p: float = 0.05                  # G2：替身检验 p 值上限
    surrogate_null_ratio: float = 0.5   # G2：替身 IC 均值最多是真值的一半，替身保留大半 IC 说明是持久量偏差不是信号
    min_symbols_same_sign: int = 5      # G2：与总体同号的币数
    min_segments: int = 3               # G3：四段里同号的段数
    perm_pct: float = 0.95              # G5：排列分位
    fdr_q: float = 0.10                 # G5：同批 BH-FDR，只对排列 p 做（替身 p 是离散的，做不了）
    max_redundancy: float = 0.7         # G6
    max_nodes: int = COMPLEXITY["nodes"]        # G7：解析时已经卡住，这里只做报告
    max_windows: int = COMPLEXITY["windows"]
    max_constants: int = COMPLEXITY["constants"]


def periods_per_year(horizon_bars: int) -> float:
    return HOURS_PER_YEAR / (horizon_bars / BARS_PER_HOUR)
