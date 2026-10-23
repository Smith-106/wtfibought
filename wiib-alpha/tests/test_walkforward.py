import numpy as np
import pytest

from wiib_alpha.eval.walkforward import TrainStats, Window, windows


def test_windows_hand_case():
    # gap = 2 + 1 = 3；test 从 13 起每 5 根一块，28 + 5 > 30 停
    assert windows(30, 5, 2, 1, 10) == [Window(10, 13, 18), Window(15, 18, 23), Window(20, 23, 28)]


def test_windows_invariants():
    total, test_size, h, embargo, min_train = 1000, 100, 12, 12, 200
    wins = windows(total, test_size, h, embargo, min_train)
    assert wins
    prev_end = -1
    for w in wins:
        assert w.train_end + h <= w.test_start            # 最后一个训练样本的答案够不到 test
        assert w.test_start - w.train_end == h + embargo  # 间隔恰好 purge + embargo
        assert w.train_end >= min_train
        assert w.test_start >= prev_end                   # test 块不重叠且前进
        assert w.test_end <= total and w.test_end - w.test_start == test_size
        prev_end = w.test_end


def test_windows_too_small():
    assert windows(100, 50, 12, 12, 200) == []
    assert windows(0, 5, 1, 1, 1) == [] and windows(10, 0, 1, 1, 1) == []


def test_train_stats_match_direct_numpy():
    rng = np.random.default_rng(3)
    f = rng.normal(5000.0, 10.0, 500)
    y = 0.3 * (f - 5000.0) + rng.normal(0, 1.0, 500)
    f[[10, 200]] = np.nan
    y[[50, 300]] = np.nan
    st = TrainStats(f, y)
    for end in (2, 60, 250, 500):
        mean, std, corr = st.at(end)
        ff, yy = f[:end], y[:end]
        ok = np.isfinite(ff)
        assert mean == pytest.approx(ff[ok].mean())
        assert std == pytest.approx(ff[ok].std(ddof=1))
        both = ok & np.isfinite(yy)
        assert corr == pytest.approx(np.corrcoef(ff[both], yy[both])[0, 1])
    assert TrainStats(f, -y).at(500)[2] == pytest.approx(-st.at(500)[2])


def test_train_stats_degenerate():
    st = TrainStats(np.array([1.0, np.nan]), np.array([0.1, 0.2]))
    assert np.isnan(st.at(0)[0]) and np.isnan(st.at(0)[2])
    mean, std, corr = st.at(2)
    assert np.isnan(mean) and np.isnan(std) and np.isnan(corr)
    flat = TrainStats(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    mean, std, corr = flat.at(3)
    assert mean == 2.0 and np.isnan(std) and np.isnan(corr)
