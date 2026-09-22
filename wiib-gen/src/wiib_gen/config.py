"""环境变量配置：进程起来读一次，之后只读。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Config:
    sim_base_url: str
    agent_base_url: str
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    # repr 关掉：整个 Config 被 print/log 出去时不带密码与密钥
    pg_password: str = field(repr=False)
    # base64 的 32 字节，解 user_llm_endpoint.api_key_enc 用；与 Java 侧同一个变量
    key_secret: str = field(repr=False)
    gen_db_path: str

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


@lru_cache(maxsize=1)
def config() -> Config:
    return Config(
        sim_base_url=os.getenv("SIM_BASE_URL", "http://localhost:8080").rstrip("/"),
        agent_base_url=os.getenv("AGENT_BASE_URL", "http://localhost:8082").rstrip("/"),
        pg_host=os.getenv("PG_HOST", "localhost"),
        pg_port=int(os.getenv("PG_PORT", "5432")),
        pg_db=os.getenv("PG_DB", "wiib"),
        pg_user=os.getenv("PG_USER", ""),
        pg_password=os.getenv("PG_PASSWORD", ""),
        key_secret=os.getenv("WIIB_TRADER_KEY_SECRET", ""),
        gen_db_path=os.getenv("GEN_DB_PATH", "./data/gen.sqlite"),
    )
