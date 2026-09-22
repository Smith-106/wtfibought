import numpy as np
import pandas as pd
import pytest

from wiib_alpha.data.binance import BAR_MS

PLANTED_N = 6000


def planted_panel(seed: int, n: int = PLANTED_N) -> pd.DataFrame:
    """volume 是埋进去的 AR(1) 预测信号，trades 是纯噪音；列够 evaluate 用"""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = 0.98 * s[i - 1] + rng.normal(0, 1.0)
    r = 0.0004 * np.roll(s, 1) + rng.normal(0, 0.003, n)
    close = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame({"close": close, "ret": r, "volume": s, "trades": rng.normal(0, 1, n),
                       "liq_long": np.nan, "liq_short": np.nan, "liq_count": np.nan},
                      index=np.arange(n) * BAR_MS + 1_735_689_600_000)
    df.index.name = "close_time"
    for h in (24, 48, 96):
        df[f"fwd_ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])
    return df


@pytest.fixture(scope="session")
def planted_panels() -> dict[str, pd.DataFrame]:
    return {s: planted_panel(i) for i, s in enumerate(("AAA", "BBB", "CCC"))}
