import itertools
import json

import numpy as np
import pandas as pd
import pytest

from portagent.tracker import top3, rare_events
from portagent.report.diagnostics import explanations


def frames(n=135):
    rng = np.random.default_rng(456)
    index = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    r = rng.normal(0, .01, (n, 4))
    r[70::17] *= 5
    return {a: pd.DataFrame({"close": 100 * np.exp(r[:, i].cumsum())}, index=index)
            for i, a in enumerate("ABCD")}


def test_count_pmf_matches_enumeration_and_preserves_marginals():
    p = np.array([.05, .3, .6, .8])
    expected = np.zeros(5)
    for outcomes in itertools.product([0, 1], repeat=4):
        expected[sum(outcomes)] += np.prod(np.where(outcomes, p, 1 - p))
    np.testing.assert_allclose(top3.poisson_binomial(p), expected)
    pmf, marginals = top3.count_distribution(p)
    np.testing.assert_allclose(marginals, p, atol=1e-11)
    assert pmf.sum() == pytest.approx(1)
    assert np.arange(5) @ pmf == pytest.approx(p.sum())
    assert np.arange(5) ** 2 @ pmf > np.arange(5) ** 2 @ expected
    with pytest.raises(ValueError):
        top3.poisson_binomial([float("nan")])


def test_learning_correlated_outcomes_prefers_common_factor():
    lw = np.zeros(4)
    for i in range(50):
        lw = top3.learn_dependence(lw, [.2] * 6, [int(i % 5 == 0)] * 6)
    assert np.argmax(lw) == 3
    with pytest.raises(ValueError):
        top3.learn_dependence(lw, [.2] * 6, [0, 1])


def test_event_uses_lagged_volatility_not_shock_inflated_sigma():
    close = frames()["A"].close
    changed = close.copy()
    changed.iloc[-1] *= 3
    a = top3.event_features(close, threshold=2.5, halflife=20, window=100)
    b = top3.event_features(changed, threshold=2.5, halflife=20, window=100)
    assert a["sigma"].iloc[-1] == b["sigma"].iloc[-1]
    assert b["outcomes"].iloc[-1] == 1
    pd.testing.assert_series_equal(a["outcomes"].iloc[:-1], b["outcomes"].iloc[:-1])
    assert a["outcomes"].iloc[:20].isna().all()


def test_detector_excludes_future_and_handles_missing_trading_dates():
    full = frames(170)
    full["D"] = full["D"].iloc[::2]
    day = full["A"].index[150]
    args = dict(day=day, threshold=2.5, halflife=20, window=100)
    earlier = {a: f.loc[f.index <= day] for a, f in full.items()}
    a, b = top3.detect(full, **args), top3.detect(earlier, **args)
    assert a == b
    json.dumps(a, allow_nan=False)
    assert a["dependence"]["fit_end"] <= day
    assert a["validation"]["end"] <= day
    assert top3.detect({"A": full["A"].iloc[:20]}, **args)["status"] == "insufficient_history"


def test_selective_uses_original_top3_without_fourth_backfill():
    result = top3.attention_lists(["B", "A", "C", "D"], [4, 4, 2, 1],
                                  [.1, .1, .6, .9], [.2, .2, .2, .2], [3])
    assert result["top3"] == ["A", "B", "C"]
    assert result["count_top3"] == ["A", "B"]
    assert result["selective_top3"] == ["C"]
    assert top3.attention_lists(["A"], [1], [.1], [.05], [0])["selective_top3"] == []


def test_contributions_explain_exact_existing_formula():
    close = pd.DataFrame({a: f.close for a, f in frames().items()})
    returns = np.log(close / close.shift()).dropna()
    explanation = rare_events.explain_last(returns, close, window=60)
    scores = rare_events.asset_scores(returns, close, window=60)
    contributions = explanation.filter(like="contribution_").sum(axis=1) + explanation.bias
    np.testing.assert_allclose(1 / (1 + np.exp(-contributions)), scores.iloc[-1])
    report = {"tracker": {"warn_threshold": .5, "rows": [
        {"asset": a, "score": round(float(scores.iloc[-1][a]), 4), "warned": True}
        for a in close]}}
    assert all(r["exact"] for r in explanations(report, explanation).values())
    explanation["score"] += .1
    assert all(not r["exact"] and not r["components"] for r in explanations(report, explanation).values())
