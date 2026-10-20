import httpx
import pytest
from conftest import POOL_SYMBOLS
from fakes import FakeFapi, FakeSim
from test_api import ADMIN, client, data, prepared, wait_until  # noqa: F401  fixtures

from wiib_alpha import main as m
from wiib_alpha.data import binance as b
from wiib_alpha.data.binance import DAY_MS
from wiib_alpha.data.panel import FEATURE_COLUMNS, PanelStore
from wiib_alpha.dsl import parse
from wiib_alpha.trade import executor, scheduler

# 2025-03-01 00:00 UTC：出信号的是 2 月最后一根，币池按 2 月（测试的池子表有 2025-01 ~ 03）
DAY = 1_740_787_200_000
NOW = DAY + 20 * 60_000


@pytest.fixture(autouse=True)
def no_rest_gap(monkeypatch):
    monkeypatch.setattr(b, "REST_GAP_S", 0.0)


async def add_factor(fid: str, name: str, expr: str) -> None:
    e = parse(expr, FEATURE_COLUMNS)
    await m.app.state.store.add_factor({"id": fid, "name": name, "hypothesis": "h", "expr": expr, "canonical": e.canonical(),
                                        "columns": sorted(e.columns()), "eval": {"horizons": {}}, "gates": [],
                                        "approved_at": 1, "run_id": "r"})


@pytest.fixture
async def trader(client, tmp_path, monkeypatch):
    """归档写到 2-27 0 点，假 fapi 往后给到 now；库里一条埋了信号的因子，交易员勾着它；调度的墙钟跟着假 fapi 的 now 走"""
    fake = FakeFapi([*POOL_SYMBOLS, "BTCUSDT"], NOW)
    monkeypatch.setattr(scheduler, "now_ms", lambda: fake.now)
    fake.write_raw(PanelStore(tmp_path), DAY - 28 * DAY_MS, DAY - 2 * DAY_MS)
    market = FakeSim(fake.mark)
    m.app.state.fapi, m.app.state.sim = fake.binance(), market.sim()
    await add_factor("f-vol", "vol4", "Mean($volume, 4)")
    await m.app.state.store.set_trade_factor("f-vol", True)
    return fake, market


async def test_two_days_of_rebalancing_and_the_pages_that_show_it(client, trader):
    fake, market = trader
    store = m.app.state.store
    await scheduler.rebalance(m.app, m._library, NOW)
    run = await store.trade_run("2025-03-01")
    assert run["status"] == "done", run["error"]
    assert (run["signal_ms"], run["user_id"], run["equity_before"]) == (DAY - 1, 42, 10000)
    assert [f["name"] for f in run["factors"]] == ["vol4"] and run["factors"][0]["weight"] > 0
    # 24 个币：每条腿多空各进头尾 1/5，4 个
    assert all(len(run["legs"][leg][side]) == 4 for leg in ("combo", "carry") for side in ("long", "short"))
    # live 尾巴从归档之后补到现在，缺口两天多，K 线 1000 根一窗；币池加上 BTC 基准。下完单每个币再拉一根 00:15 的（limit 99）
    klines = [p for path, p in fake.log if path == "/fapi/v1/klines"]
    assert {p["symbol"] for p in klines if p["limit"] == "1000"} == {*POOL_SYMBOLS, "BTCUSDT"}
    assert {p["limit"] for p in klines} == {"1000", "99"}
    # 第一天全是开仓：4 倍全仓市价，数量 = |净权重| × 净值 ÷ 标记价、按步长向零取整
    coins = {c["symbol"]: c for c in run["targets"]["coins"]}
    assert market.calls and all(a == "open" and body["leverage"] == 4 and body["marginMode"] == "CROSS"
                                for a, body in market.calls)
    for _, body in market.calls:
        c = coins[body["symbol"]]
        assert body["clientRequestId"] == f"ft-20250301-{body['symbol']}-{3 if c['weight'] > 0 else 4}"
        assert 0 <= abs(c["weight"]) * 10000 / c["price"] - body["quantity"] < 0.001
        assert c["weight"] == pytest.approx(c["combo"] + c["carry"])
        # 00:15 那根的收盘价记下来对成交价
        assert c["refPrice"] == pytest.approx(fake.close(body["symbol"], DAY))
    assert sum(1 for o in run["orders"] if o["ok"]) == len(market.calls) == len(coins)
    assert run["equity_after"] == pytest.approx(10000)
    # 算完目标就把目标里的币登记给 sim（feed 据此订行情）
    assert market.watched == [sorted(coins)]

    # 页面：账户、持仓（带两条腿各占多少）、调仓记录与详情
    account = data(await client.get("/api/alpha/trader/account", headers=ADMIN))
    assert account["userId"] == 42 and account["equity"] == pytest.approx(10000) and account["available"] == 10000
    positions = data(await client.get("/api/alpha/trader/positions", headers=ADMIN))
    assert positions["day"] == "2025-03-01" and len(positions["items"]) == len(coins)
    for p in positions["items"]:
        c = coins[p["symbol"]]
        assert p["side"] == ("LONG" if c["weight"] > 0 else "SHORT")
        assert p["legs"] == {"combo": pytest.approx(c["combo"] * 10000), "carry": pytest.approx(c["carry"] * 10000)}
    detail = data(await client.get("/api/alpha/trader/runs/2025-03-01", headers=ADMIN))
    assert detail["orderCount"] == len(coins) and detail["failedCount"] == 0 and detail["factorNames"] == ["vol4"]
    assert set(detail["targets"]["signals"]) == {"combo", "carry"} and detail["targets"]["notTrading"] == []
    assert (await client.get("/api/alpha/trader/runs/2099-01-01", headers=ADMIN)).status_code == 404

    # 第二天：价格动了、信号动了；先平后开，上一期留在缓冲区里的币不换
    fake.now += DAY_MS
    fake.halted = {"X00"}
    market.calls.clear()
    await scheduler.rebalance(m.app, m._library, NOW + DAY_MS)
    second = await store.trade_run("2025-03-02")
    assert second["status"] == "done", second["error"]
    actions = [a for a, _ in market.calls]
    assert "close" in actions and actions == sorted(actions, key=lambda a: a != "close")
    assert all(body["clientRequestId"].startswith("ft-20250302-") for _, body in market.calls)
    # 不在交易的币不出目标，持着的平掉；登记的只有目标里的币，要平的 sim 按仓位自己算
    assert second["targets"]["notTrading"] == ["X00"] and "X00" not in second["legs"]["combo"]["long"]
    assert market.watched[1] == sorted(c["symbol"] for c in second["targets"]["coins"] if c["weight"])
    assert "X00" not in market.watched[1]
    runs = data(await client.get("/api/alpha/trader/runs", headers=ADMIN))
    assert [r["day"] for r in runs["items"]] == ["2025-03-02", "2025-03-01"] and runs["job"]["running"] is False

    # sim 不通：页面拿到 502
    m.app.state.sim = executor.Sim("http://sim", "tok", httpx.Client(transport=httpx.MockTransport(
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("连不上")))))
    assert (await client.get("/api/alpha/trader/account", headers=ADMIN)).status_code == 502


async def test_a_month_missing_from_the_pool_table_fails_the_day_without_trading(client, trader):
    fake, market = trader
    # 4 月底：池子表只到 3 月
    await scheduler.rebalance(m.app, m._library, DAY + 60 * DAY_MS + 20 * 60_000)
    run = await m.app.state.store.trade_run("2025-04-30")
    assert run["status"] == "failed" and "2025-04" in run["error"] and run["user_id"] is None
    assert market.calls == [] and fake.log == []
    assert m.app.state.runs.jobs["trader"]["error"] == run["error"]
    # 没下单就没有这一期的多空集合，下一期的缓冲不拿它算
    assert await m.app.state.store.last_trade_legs() is None


async def test_a_retry_that_reaches_0100_before_ordering_skips_the_day(client, trader):
    fake, market = trader
    # 00:59 开始的重试，补完数据、算完目标已经 01:01
    fake.now = DAY + 61 * 60_000
    await scheduler.rebalance(m.app, m._library, DAY + 59 * 60_000)
    run = await m.app.state.store.trade_run("2025-03-01")
    assert run["status"] == "failed" and "01:00" in run["error"] and market.calls == []


async def test_tick_rebalances_between_0005_and_0100_and_records_equity_every_hour(client, trader, monkeypatch):
    store = m.app.state.store
    ran: list[int] = []
    # 每次调仓的结局：状态和下出去的单
    outcomes = [("failed", []), ("done", [{"ok": True}]), ("failed", [{"ok": False}])]

    async def rebalance(app, library, now):
        ran.append(now)
        status, orders = outcomes.pop(0)
        await store.add_trade_run({"day": scheduler.day_label(now), "signal_ms": 0, "started_at": now,
                                   "finished_at": now, "status": status, "user_id": 42, "orders": orders})

    monkeypatch.setattr(scheduler, "rebalance", rebalance)
    t0 = DAY + 5 * 60_000
    # 00:05 之前、刷新中都不调
    await scheduler.tick(m.app, m._library, t0 - 60_000)
    m.app.state.runs.jobs["refresh"]["running"] = True
    await scheduler.tick(m.app, m._library, t0)
    m.app.state.runs.jobs["refresh"]["running"] = False
    assert ran == [] and await store.trade_equity() == []
    # 一笔单都没下出去就失败了：下一分钟再来；调成了当天不再调。调过仓有了账户：每个小时记一次净值
    await scheduler.tick(m.app, m._library, t0)
    await scheduler.tick(m.app, m._library, t0 + 60_000)
    await scheduler.tick(m.app, m._library, t0 + 120_000)
    assert ran == [t0, t0 + 60_000]
    # 下过单的失败（单可能已成交）当天不重跑
    day2 = t0 + DAY_MS
    await scheduler.tick(m.app, m._library, day2)
    await scheduler.tick(m.app, m._library, day2 + 60_000)
    assert ran == [t0, t0 + 60_000, day2]
    # 01:00 之后还没调过：跳过这一天，不拿 0 点的信号隔几小时再下
    day3 = DAY + 2 * DAY_MS + 61 * 60_000
    await scheduler.tick(m.app, m._library, day3)
    assert ran == [t0, t0 + 60_000, day2] and await store.trade_run(scheduler.day_label(day3)) is None
    curve = data(await client.get("/api/alpha/trader/equity", headers=ADMIN))
    assert [p["ts"] for p in curve["items"]] == [t0 + 60_000, day2, day3]
    assert curve["items"][0] == {"ts": t0 + 60_000, "equity": 10000, "balance": 10000, "long": 0, "short": 0}
    assert curve["initialBalance"] == 10000


async def test_factor_selection_and_compare(client, trader):
    await add_factor("f-noise", "noise", "$trades")
    await add_factor("f-liq", "liq", "Mean($liq_count, 4)")
    items = data(await client.get("/api/alpha/trader/factors", headers=ADMIN))["items"]
    assert {f["id"]: (f["selected"], f["missing"]) for f in items} == {
        "f-vol": (True, []), "f-noise": (False, []), "f-liq": (False, ["liq_count"])}
    # live 尾巴补不出来的列：不能勾
    resp = await client.put("/api/alpha/trader/factors/f-liq", json={"selected": True}, headers=ADMIN)
    assert resp.status_code == 409 and "$liq_count" in resp.json()["msg"]
    assert (await client.put("/api/alpha/trader/factors/nope", json={"selected": True}, headers=ADMIN)).status_code == 404
    assert data(await client.put("/api/alpha/trader/factors/f-noise", json={"selected": True}, headers=ADMIN))["selected"]
    assert data(await client.put("/api/alpha/trader/factors/f-noise", json={"selected": False}, headers=ADMIN)) == {
        "id": "f-noise", "selected": False}

    async def compare(fid: str) -> dict:
        assert data(await client.post(f"/api/alpha/trader/compare/{fid}", headers=ADMIN)) == {"started": True}

        async def done():
            return not data(await client.get("/api/alpha/trader/factors", headers=ADMIN))["compare"]["running"]

        await wait_until(done, 60)
        job = data(await client.get("/api/alpha/trader/factors", headers=ADMIN))["compare"]
        assert job["error"] is None and job["factorId"] == fid
        return job["result"]

    # 去掉埋了信号的那条：只剩 carry，搜索段变差，不给 holdout
    out = await compare("f-vol")
    assert out["action"] == "remove" and out["base"]["factors"] == ["vol4"] and out["alt"]["factors"] == []
    assert out["better"] is False and "holdout" not in out["base"] and "holdout" not in out["alt"]
    assert out["alt"]["search"]["sharpe"] < out["base"]["search"]["sharpe"]
    # 没勾时加上它：搜索段更好，两组的 holdout 才一起给
    await client.put("/api/alpha/trader/factors/f-vol", json={"selected": False}, headers=ADMIN)
    out = await compare("f-vol")
    assert out["action"] == "add" and out["better"] is True
    assert out["base"]["holdout"]["periods"] == out["alt"]["holdout"]["periods"] > 0
    assert (await client.post("/api/alpha/trader/compare/nope", headers=ADMIN)).status_code == 404


async def test_trader_is_admin_only_and_holds_off_refresh(client):
    assert data(await client.get("/api/alpha/trader/account", headers=ADMIN))["userId"] is None
    assert data(await client.get("/api/alpha/trader/positions", headers=ADMIN)) == {"items": [], "day": None}
    for path in ("account", "positions", "equity", "runs", "factors"):
        assert (await client.get(f"/api/alpha/trader/{path}", headers={"satoken": "user"})).status_code == 403
        assert (await client.get(f"/api/alpha/trader/{path}")).json()["code"] == 401
    assert (await client.post("/api/alpha/trader/compare/x", headers={"satoken": "user"})).status_code == 403
    jobs = m.app.state.runs.jobs
    for name, word in (("trader", "调仓"), ("compare", "对比")):
        jobs[name]["running"] = True
        resp = await client.post("/api/alpha/data/refresh", headers=ADMIN)
        assert resp.status_code == 409 and word in resp.json()["msg"]
        jobs[name]["running"] = False
    # 刷新中不起对比
    jobs["refresh"]["running"] = True
    resp = await client.post("/api/alpha/trader/compare/x", headers=ADMIN)
    assert resp.status_code == 409 and "刷新" in resp.json()["msg"]
    jobs["refresh"]["running"] = False
