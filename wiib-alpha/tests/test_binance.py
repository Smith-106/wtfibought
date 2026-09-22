import io
import re
import zipfile
from datetime import datetime, timezone

import httpx
import pytest

from wiib_alpha.data import binance as b

BAR = b.BAR_MS
DAY = 86_400_000
H8 = 8 * 3_600_000


def ms(y: int, m: int, d: int, h: int = 0) -> int:
    return int(datetime(y, m, d, h, tzinfo=timezone.utc).timestamp() * 1000)


LISTED = ms(2026, 7, 1)          # 7 月上市，之前没文件
MONTHLY_END = ms(2026, 9, 1)     # 月度文件出到 8 月
DAILY_END = ms(2026, 9, 20)      # 日度文件出到 09-19
NOW = ms(2026, 9, 21, 9)

_PATH = re.compile(r"/data/futures/um/(monthly|daily)/(klines|fundingRate)/X(?:/15m)?/X-(?:15m|fundingRate)-([\d-]+)\.zip$")


def zipped(lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def kline_lines(a: int, b_: int, header: bool, micro: bool) -> list[str]:
    """[a, b_) 内的 bar；close = 第几根，好核对；7 月的老文件没头行，8 月的时间戳是微秒"""
    scale = 1000 if micro else 1
    rows = [f"{t * scale},99.0,101.0,98.0,{t // BAR},10.0,{(t + BAR - 1) * scale},1000.0,7,4.0,400.0,0"
            for t in range(a, b_, BAR)]
    return (["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
             "taker_buy_quote_volume,ignore"] if header else []) + rows


def funding_lines(a: int, b_: int) -> list[str]:
    return ["calc_time,funding_interval_hours,last_funding_rate"] + [f"{t},8,0.0001" for t in range(a, b_, H8)]


def make_fake(log: list[str]):
    def fake_vision(request: httpx.Request) -> httpx.Response:
        log.append(request.url.path)
        m = _PATH.match(request.url.path)
        if not m:
            return httpx.Response(404)
        kind, what, stamp = m.groups()
        parts = [int(p) for p in stamp.split("-")]
        if kind == "monthly":
            y, mo = parts
            a, b_ = ms(y, mo, 1), ms(y + mo // 12, mo % 12 + 1, 1)
            if b_ > MONTHLY_END:
                return httpx.Response(404)
        else:
            y, mo, d = parts
            a = ms(y, mo, d)
            b_ = a + DAY
            if b_ > DAILY_END:
                return httpx.Response(404)
        if a < LISTED:
            return httpx.Response(404)
        if what == "fundingRate":
            return httpx.Response(200, content=zipped(funding_lines(a, b_)))
        return httpx.Response(200, content=zipped(kline_lines(a, b_, header=mo != 7, micro=mo == 8)))
    return fake_vision


def client(log: list[str] | None = None) -> b.Binance:
    return b.Binance("https://v", httpx.Client(transport=httpx.MockTransport(make_fake(log if log is not None else []))))


def test_klines_monthly_then_daily_for_recent_months():
    log: list[str] = []
    df = client(log).klines("X", LISTED, NOW)
    # 7、8 月走月度，9 月月度 404 改走日度，09-20 起没文件
    assert df["open_time"].tolist() == list(range(LISTED, DAILY_END, BAR))
    assert df["close"].tolist() == [t // BAR for t in range(LISTED, DAILY_END, BAR)]
    assert df["close_time"].iloc[-1] == DAILY_END - 1
    assert df["open_time"].dtype == "int64" and df["trades"].dtype == "int64" and df["close"].dtype == "float64"
    assert sum("/daily/" in p for p in log) == 21


def test_klines_unlisted_months_do_not_fall_back_to_daily():
    log: list[str] = []
    df = client(log).klines("X", ms(2026, 5, 1), NOW)
    assert df["open_time"].iloc[0] == LISTED
    assert not any("/daily/" in p for p in log if "2026-05" in p or "2026-06" in p)


def test_klines_incremental_start_only_touches_needed_days():
    log: list[str] = []
    start = ms(2026, 9, 10, 6)
    df = client(log).klines("X", start, NOW)
    assert df["open_time"].iloc[0] == start and df["open_time"].iloc[-1] == DAILY_END - BAR
    assert not any("2026-09-09" in p for p in log) and not any("/monthly/klines/X/15m/X-15m-2026-08" in p for p in log)


def test_funding_monthly_only():
    df = client().funding("X", LISTED, NOW)
    assert df["funding_time"].tolist() == list(range(LISTED, MONTHLY_END, H8))
    assert df["funding_rate"].iloc[0] == 0.0001 and df["funding_time"].dtype == "int64"


def test_empty_range_gives_typed_empty():
    df = client().klines("X", ms(2020, 1, 1), ms(2020, 2, 1))
    assert df.empty and df["close_time"].dtype == "int64"
    assert client().funding("X", ms(2020, 1, 1), ms(2020, 2, 1)).empty


def test_client_error_raises():
    api = b.Binance("https://v", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(403))))
    with pytest.raises(b.BinanceError):
        api.klines("X", LISTED, NOW)
