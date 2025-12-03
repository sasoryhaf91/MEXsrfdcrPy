# tests/test_metrics.py

import numpy as np
import pandas as pd
import pytest

from MEXsrfdcrPy.metrics import (
    kge,
    nse,
    regression_metrics,
    aggregate_and_score,
)


def test_kge_perfect_match_is_one():
    """KGE should be 1.0 for a perfect match."""
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    val = kge(y_true, y_pred)
    assert val == pytest.approx(1.0, rel=1e-6)


def test_nse_perfect_match_is_one():
    """NSE should be 1.0 for a perfect match."""
    y_true = [0.0, 1.0, 2.0, 3.0]
    y_pred = [0.0, 1.0, 2.0, 3.0]
    val = nse(y_true, y_pred)
    assert val == pytest.approx(1.0, rel=1e-6)


def test_kge_reasonable_for_biased_series():
    """
    KGE should be finite and < 1 when there is bias and correlation < 1.
    We don't test the exact closed form, just that it behaves sensibly.
    """
    y_true = np.arange(1, 11, dtype=float)
    # add bias and some noise
    rng = np.random.default_rng(0)
    y_pred = y_true * 1.2 + 0.5 * rng.normal(size=y_true.size)

    val = kge(y_true, y_pred)
    assert np.isfinite(val)
    assert val < 1.0  # worse than perfect


def test_nse_zero_variance_returns_nan():
    """NSE is undefined (NaN) when variance of y_true is zero."""
    y_true = [5.0, 5.0, 5.0, 5.0]
    y_pred = [5.0, 5.0, 5.0, 5.0]
    val = nse(y_true, y_pred)
    assert np.isnan(val)


def test_kge_too_few_points_returns_nan():
    """KGE should return NaN when length < 2."""
    y_true = [1.0]
    y_pred = [1.0]
    val = kge(y_true, y_pred)
    assert np.isnan(val)


def test_metrics_shape_mismatch_raises():
    """All metrics should fail when shapes do not match."""
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0]
    with pytest.raises(ValueError):
        kge(y_true, y_pred)
    with pytest.raises(ValueError):
        nse(y_true, y_pred)
    with pytest.raises(ValueError):
        regression_metrics(y_true, y_pred)


def test_regression_metrics_perfect_match():
    """Perfect match should give MAE=0, RMSE=0, R2=1, KGE≈1, NSE≈1."""
    y_true = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_pred = [0.0, 1.0, 2.0, 3.0, 4.0]

    m = regression_metrics(y_true, y_pred)

    assert set(m.keys()) == {"MAE", "RMSE", "R2", "KGE", "NSE"}
    assert m["MAE"] == pytest.approx(0.0, abs=1e-12)
    assert m["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert m["R2"] == pytest.approx(1.0, rel=1e-6)
    assert m["KGE"] == pytest.approx(1.0, rel=1e-6)
    assert m["NSE"] == pytest.approx(1.0, rel=1e-6)


def test_regression_metrics_empty_inputs_return_nan():
    """Empty inputs should return all-NaN metrics, not crash."""
    m = regression_metrics([], [])
    for v in m.values():
        assert np.isnan(v)


def test_aggregate_and_score_monthly_sum_perfect():
    """
    aggregate_and_score: for a daily perfect match, monthly aggregation
    should also yield 'perfect' metrics (MAE=0, RMSE=0, R2=1, KGE≈1, NSE≈1).
    """
    dates = pd.date_range("2000-01-01", periods=10, freq="D")
    y_true = np.ones(10, dtype=float)  # daily value = 1
    y_pred = np.ones(10, dtype=float)

    df = pd.DataFrame(
        {
            "date": dates,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    metrics, agg_df = aggregate_and_score(
        df,
        date_col="date",
        y_col="y_true",
        yhat_col="y_pred",
        freq="M",
        agg="sum",
    )

    # One monthly row
    assert len(agg_df) == 1
    assert agg_df["y_true"].iloc[0] == pytest.approx(10.0)
    assert agg_df["y_pred"].iloc[0] == pytest.approx(10.0)

    assert metrics["MAE"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["R2"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["KGE"] == pytest.approx(1.0, rel=1e-6)
    assert metrics["NSE"] == pytest.approx(1.0, rel=1e-6)


def test_aggregate_and_score_handles_empty_dataframe():
    """Empty or all-NaN input should return NaN metrics and not crash."""
    df = pd.DataFrame({"date": [], "y_true": [], "y_pred": []})

    metrics, agg_df = aggregate_and_score(df)

    assert agg_df.empty
    for v in metrics.values():
        assert np.isnan(v)
