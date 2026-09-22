"""SQLite：运行、候选、因子库、已试假设、事件回放、全局试验计数。与 LangGraph checkpoint 同一个文件。

评估结果里有 NaN 和 numpy 标量，落库和下发前一律 clean 成纯 JSON。
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import aiosqlite
import numpy as np

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS run(
        id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, status TEXT NOT NULL, k INTEGER NOT NULL,
        note TEXT NOT NULL DEFAULT '', lang TEXT NOT NULL, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS candidate(
        id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL, name TEXT NOT NULL,
        hypothesis TEXT NOT NULL, family TEXT, expr TEXT, canonical TEXT, note TEXT, status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0, eval_json TEXT, gates_json TEXT, review_json TEXT, reason TEXT,
        decision TEXT, updated_at INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_candidate_run ON candidate(run_id, seq)",
    """CREATE TABLE IF NOT EXISTS run_event(
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, event TEXT NOT NULL, data_json TEXT NOT NULL,
        created_at INTEGER NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS idx_event_run ON run_event(run_id, id)",
    """CREATE TABLE IF NOT EXISTS tried(
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, hypothesis TEXT NOT NULL, expr TEXT,
        outcome TEXT NOT NULL, reason TEXT, created_at INTEGER NOT NULL, UNIQUE(run_id, hypothesis))""",
    """CREATE TABLE IF NOT EXISTS factor(
        id TEXT PRIMARY KEY, name TEXT NOT NULL, hypothesis TEXT NOT NULL, expr TEXT NOT NULL, canonical TEXT NOT NULL,
        columns_json TEXT NOT NULL, eval_json TEXT NOT NULL, gates_json TEXT NOT NULL, review_json TEXT,
        approved_at INTEGER NOT NULL, run_id TEXT NOT NULL, recheck_json TEXT)""",
    "CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY CHECK (id = 1), count INTEGER NOT NULL)",
    "INSERT OR IGNORE INTO trials(id, count) VALUES (1, 0)",
    """CREATE TABLE IF NOT EXISTS reference_eval(
        name TEXT PRIMARY KEY, expr TEXT NOT NULL, passed INTEGER NOT NULL, eval_json TEXT NOT NULL,
        gates_json TEXT NOT NULL, updated_at INTEGER NOT NULL)""",
]

_JSON_COLUMNS = {"eval_json": "eval", "gates_json": "gates", "review_json": "review", "data_json": "data",
                 "columns_json": "columns", "recheck_json": "recheck"}


def clean(obj: Any) -> Any:
    """NaN/inf → None，numpy 标量 → Python，递归"""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False)


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    @classmethod
    async def open(cls, path: str) -> "Store":
        db = await aiosqlite.connect(path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        for statement in SCHEMA:
            await db.execute(statement)
        await db.commit()
        return cls(db)

    async def close(self) -> None:
        await self.db.close()

    # ========== 运行 ==========

    async def create_run(self, run_id: str, k: int, note: str, lang: str) -> None:
        await self.db.execute("INSERT INTO run(id, created_at, status, k, note, lang) VALUES (?, ?, 'running', ?, ?, ?)",
                              (run_id, now_ms(), k, note, lang))
        await self.db.commit()

    async def set_run_status(self, run_id: str, status: str, error: str | None = None) -> None:
        await self.db.execute("UPDATE run SET status = ?, error = ? WHERE id = ?", (status, error, run_id))
        await self.db.commit()

    async def claim_run(self, run_id: str, expect: str, new: str) -> bool:
        """状态从 expect 改成 new，一条 UPDATE 完成检查与置位；两个请求同时来只有一个能抢到"""
        cur = await self.db.execute("UPDATE run SET status = ?, error = NULL WHERE id = ? AND status = ?",
                                    (new, run_id, expect))
        await self.db.commit()
        return cur.rowcount == 1

    async def run(self, run_id: str) -> dict[str, Any] | None:
        return await self._one("SELECT * FROM run WHERE id = ?", (run_id,))

    async def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM run ORDER BY created_at DESC LIMIT ?", (limit,))

    async def running(self) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM run WHERE status = 'running'", ())

    # ========== 候选 ==========

    async def upsert_candidate(self, c: dict[str, Any]) -> None:
        await self.db.execute(
            """INSERT INTO candidate(id, run_id, seq, name, hypothesis, family, expr, canonical, note, status, attempts,
                                     eval_json, gates_json, review_json, reason, decision, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET expr = excluded.expr, canonical = excluded.canonical, note = excluded.note,
                   status = excluded.status, attempts = excluded.attempts, eval_json = excluded.eval_json,
                   gates_json = excluded.gates_json, review_json = excluded.review_json, reason = excluded.reason,
                   decision = excluded.decision, updated_at = excluded.updated_at""",
            (c["id"], c["run_id"], c["seq"], c["name"], c["hypothesis"], c.get("family"), c.get("expr"),
             c.get("canonical"), c.get("note"), c["status"], c.get("attempts", 0),
             dumps(c["eval"]) if c.get("eval") else None, dumps(c["gates"]) if c.get("gates") else None,
             dumps(c["review"]) if c.get("review") else None, c.get("reason"), c.get("decision"), now_ms()))
        await self.db.commit()

    async def candidates(self, run_id: str) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM candidate WHERE run_id = ? ORDER BY seq", (run_id,))

    # ========== 事件 ==========

    async def add_event(self, run_id: str, event: str, data: Any) -> int:
        cur = await self.db.execute("INSERT INTO run_event(run_id, event, data_json, created_at) VALUES (?, ?, ?, ?)",
                                    (run_id, event, dumps(data), now_ms()))
        await self.db.commit()
        return int(cur.lastrowid or 0)

    async def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM run_event WHERE run_id = ? AND id > ? ORDER BY id", (run_id, after))

    # ========== 已试假设 ==========

    async def add_tried(self, run_id: str, hypothesis: str, expr: str | None, outcome: str, reason: str | None) -> None:
        """同一轮同一条假设只记一次，commit 重跑不留重复"""
        await self.db.execute(
            "INSERT OR REPLACE INTO tried(run_id, hypothesis, expr, outcome, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, hypothesis, expr, outcome, reason, now_ms()))
        await self.db.commit()

    async def tried(self, limit: int = 200) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM tried ORDER BY id DESC LIMIT ?", (limit,))

    # ========== 因子库 ==========

    async def add_factor(self, f: dict[str, Any]) -> None:
        """按 id 覆盖写：commit 节点重跑不会撞主键"""
        await self.db.execute(
            """INSERT INTO factor(id, name, hypothesis, expr, canonical, columns_json, eval_json, gates_json, review_json,
                                  approved_at, run_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name = excluded.name, hypothesis = excluded.hypothesis, expr = excluded.expr,
                   canonical = excluded.canonical, columns_json = excluded.columns_json, eval_json = excluded.eval_json,
                   gates_json = excluded.gates_json, review_json = excluded.review_json, run_id = excluded.run_id""",
            (f["id"], f["name"], f["hypothesis"], f["expr"], f["canonical"], dumps(f["columns"]), dumps(f["eval"]),
             dumps(f["gates"]), dumps(f.get("review")), f["approved_at"], f["run_id"]))
        await self.db.commit()

    async def factors(self) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM factor ORDER BY approved_at DESC", ())

    async def factor(self, factor_id: str) -> dict[str, Any] | None:
        return await self._one("SELECT * FROM factor WHERE id = ?", (factor_id,))

    async def set_recheck(self, factor_id: str, recheck: Any) -> None:
        await self.db.execute("UPDATE factor SET recheck_json = ? WHERE id = ?", (dumps(recheck), factor_id))
        await self.db.commit()

    # ========== 参照集评估 ==========

    async def set_reference_eval(self, name: str, expr: str, ev: Any, gates: Any, passed: bool) -> None:
        await self.db.execute(
            """INSERT INTO reference_eval(name, expr, passed, eval_json, gates_json, updated_at) VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET expr = excluded.expr, passed = excluded.passed,
                   eval_json = excluded.eval_json, gates_json = excluded.gates_json, updated_at = excluded.updated_at""",
            (name, expr, int(passed), dumps(ev), dumps(gates), now_ms()))
        await self.db.commit()

    async def reference_evals(self) -> list[dict[str, Any]]:
        return await self._all("SELECT * FROM reference_eval ORDER BY name", ())

    # ========== 全局试验计数 ==========

    async def bump_trials(self, n: int) -> int:
        await self.db.execute("UPDATE trials SET count = count + ? WHERE id = 1", (n,))
        await self.db.commit()
        return await self.trials()

    async def trials(self) -> int:
        row = await self._one("SELECT count FROM trials WHERE id = 1", ())
        return int(row["count"]) if row else 0

    # ========== 内部 ==========

    async def _one(self, sql: str, args: tuple) -> dict[str, Any] | None:
        async with self.db.execute(sql, args) as cur:
            row = await cur.fetchone()
        return _decode(row) if row else None

    async def _all(self, sql: str, args: tuple) -> list[dict[str, Any]]:
        async with self.db.execute(sql, args) as cur:
            rows = await cur.fetchall()
        return [_decode(r) for r in rows]


def _decode(row: aiosqlite.Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key in _JSON_COLUMNS:
            out[_JSON_COLUMNS[key]] = json.loads(value) if value else None
        else:
            out[key] = value
    return out
