import pytest

from wiib_alpha import config as c


@pytest.fixture(autouse=True)
def fresh_config():
    c.config.cache_clear()
    yield
    c.config.cache_clear()


def test_blank_lists_fall_back_to_defaults(monkeypatch):
    # .env.example 里取消注释但没填：和没设一样，排除名单不能变成空的
    monkeypatch.setenv("ALPHA_EXCLUDE_SYMBOLS", " ")
    monkeypatch.setenv("ALPHA_SYMBOLS", "")
    cfg = c.config()
    assert cfg.exclude_symbols == c.DEFAULT_EXCLUDE and cfg.symbols == c.DEFAULT_SYMBOLS


def test_lists_are_upper_cased_and_trimmed(monkeypatch):
    monkeypatch.setenv("ALPHA_EXCLUDE_SYMBOLS", "usdcusdt, fdusdusdt ,")
    assert c.config().exclude_symbols == ("USDCUSDT", "FDUSDUSDT")
