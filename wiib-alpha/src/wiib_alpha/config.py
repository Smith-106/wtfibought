"""环境变量配置：进程起来读一次，之后只读。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# feed 订阅的 7 个加密永续，与 wiib-feed application.yml 的 symbols 同一份
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "DOGEUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "ZECUSDT")


@dataclass(frozen=True)
class ModelSpec:
    """一个角色的模型：协议分叉与 wiib-gen 同，baseUrl 不带版本后缀"""
    protocol: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    effort: str = ""


@dataclass(frozen=True)
class Config:
    # 两个角色必须是不同模型，否则评审只会附和
    gen: ModelSpec
    review: ModelSpec
    binance_vision_url: str
    symbols: tuple[str, ...]
    history_days: int
    data_dir: str
    db_path: str
    # 方向腿每笔往返成本，bps
    cost_bps: float
    # force_order.trade_time 是不带时区的本地时间，按这个时区换成 UTC；与容器 TZ 同值
    force_order_tz: str
    sim_base_url: str
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str = field(repr=False)

    @property
    def pg_conninfo(self) -> dict[str, object]:
        """psycopg.connect 的关键字参数"""
        return {
            "host": self.pg_host,
            "port": self.pg_port,
            "dbname": self.pg_db,
            "user": self.pg_user,
            "password": self.pg_password,
        }


def _model(prefix: str) -> ModelSpec:
    return ModelSpec(
        protocol=os.getenv(f"{prefix}_PROTOCOL", "openai"),
        base_url=os.getenv(f"{prefix}_BASE_URL", ""),
        model=os.getenv(f"{prefix}_MODEL", ""),
        api_key=os.getenv(f"{prefix}_API_KEY", ""),
        effort=os.getenv(f"{prefix}_EFFORT", ""),
    )


@lru_cache(maxsize=1)
def config() -> Config:
    symbols = tuple(s.strip().upper() for s in os.getenv("ALPHA_SYMBOLS", "").split(",") if s.strip())
    return Config(
        gen=_model("ALPHA_GEN"),
        review=_model("ALPHA_REVIEW"),
        binance_vision_url=os.getenv("BINANCE_VISION_URL", "https://data.binance.vision").rstrip("/"),
        symbols=symbols or DEFAULT_SYMBOLS,
        history_days=int(os.getenv("ALPHA_HISTORY_DAYS", "730")),
        data_dir=os.getenv("ALPHA_DATA_DIR", "./data"),
        db_path=os.getenv("ALPHA_DB_PATH", "./data/alpha.sqlite"),
        cost_bps=float(os.getenv("ALPHA_COST_BPS", "10")),
        force_order_tz=os.getenv("TZ", "Asia/Shanghai"),
        sim_base_url=os.getenv("SIM_BASE_URL", "http://localhost:8080").rstrip("/"),
        pg_host=os.getenv("PG_HOST", "localhost"),
        pg_port=int(os.getenv("PG_PORT", "5432")),
        pg_db=os.getenv("PG_DB", "wiib"),
        pg_user=os.getenv("PG_USER", ""),
        pg_password=os.getenv("PG_PASSWORD", ""),
    )
