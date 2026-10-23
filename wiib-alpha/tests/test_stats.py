import numpy as np
import pytest

from wiib_alpha.eval.stats import (annualized_sharpe, bh_reject, compound, grouped_permutation_percentile,
                                   permutation_percentile, spearman)


def test_grouped_permutation_shuffles_within_groups_only():
    # 两个币各自 [1,0,0] / [0.05,-0.01,-0.01]：各组的 1 落回槽 0 的概率 1/3，两组都落回才不小于真实 → 1 - 1/9
    pos = [1, 0, 0, 1, 0, 0]
    ret = [0.05, -0.01, -0.01, 0.05, -0.01, -0.01]
    pct = grouped_permutation_percentile(pos, ret, [0, 0, 0, 1, 1, 1], 20000, 42)
    assert pct == pytest.approx(8 / 9, abs=0.02)
    assert grouped_permutation_percentile([], [], [], 10, 1) == 0.0


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 2], [1, 2]))
    assert spearman([1, 2, np.nan, 4, 5], [1, 2, 3, np.nan, 5]) == pytest.approx(1.0)
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_permutation_matches_closed_form():
    # 仓位 [1,0,0]，收益 [0.05,-0.01,-0.01]：那个 1 落到 3 个槽等概率，只有落回槽 0 不小于真实值 → 2/3
    pct = permutation_percentile([1, 0, 0], [0.05, -0.01, -0.01], 20000, 42)
    assert pct == pytest.approx(2 / 3, abs=0.02)


def test_permutation_edges():
    assert permutation_percentile([], [], 100, 1) == 0.0
    assert permutation_percentile([0, 0, 0], [0.1, 0.2, 0.3], 100, 1) == 0.0
    # 最差的时机：所有洗牌都不会更差
    assert permutation_percentile([1, 0, 0], [-0.05, 0.01, 0.01], 500, 1) == 0.0


def test_bh_reject():
    assert bh_reject([0.01, 0.02, 0.03, 0.5], 0.1) == [True, True, True, False]
    assert bh_reject([0.5, 0.01, 0.03, 0.02], 0.1) == [False, True, True, True]
    assert bh_reject([0.5, 0.6], 0.1) == [False, False]
    assert bh_reject([], 0.1) == []
    # 第 2 名 0.045 > 2/5·0.1 = 0.04，只留第 1 名
    assert bh_reject([0.001, 0.045, 0.5, 0.6, 0.7], 0.1) == [True, False, False, False, False]


def test_sharpe_and_compound():
    r = np.array([0.01, -0.01, 0.02])
    assert annualized_sharpe(r, 730) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(730))
    assert np.isnan(annualized_sharpe([0.01], 730)) and np.isnan(annualized_sharpe([0.01, 0.01], 730))
    assert compound([0.1, 0.1]) == pytest.approx(0.21)
    assert compound([]) == 0.0
