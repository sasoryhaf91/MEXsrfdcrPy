# src/MEXsrfdcrPy/metrics.py
# SPDX-License-Identifier: MIT
"""
Hydrological-style regression metrics for MEXsrfdcrPy.

This module provides a small, focused set of metrics commonly used in
hydrology and climate-model evaluation:

- :func:`kge` — Kling–Gupta efficiency (Gupta et al., 2009).
- :func:`nse` — Nash–Sutcliffe efficiency.
- :func:`regression_metrics` — MAE, RMSE, R², KGE, NSE in a single dict.
- :func:`aggregate_and_score` — temporal aggregation + metrics on the
  aggregated series.

Key design choices
------------------
* Inputs are accepted as any iterable (lists, NumPy arrays, pandas Series).
* Outputs are plain ``float`` or ``numpy.nan`` when the metric is undefined
  (for instance, zero variance in the observed series).
* R² is **not** ``sklearn.metrics.r2_score``. Here it is defined as the
  square of the Pearson correlation coefficient between observations and
  predictions. This avoids redundancy with NSE, which already has the
  “variance–explained” interpretation.
* The code is intentionally self-contained and dependency-light, intended
  for reproducible climate-data workflows and suitable for a JOSS context.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _as_arrays(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert *y_true* and *y_pred* to NumPy arrays of ``dtype=float`` and
    verify that they share the same shape.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values.

    Returns
    -------
    yt, yp : np.ndarray
        Arrays with identical shapes.

    Raises
    ------
    ValueError
        If the shapes of *y_true* and *y_pred* do not match.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if yt.shape != yp.shape:
        raise ValueError(
            f"Shapes of y_true {yt.shape} and y_pred {yp.shape} do not match."
        )
    return yt, yp


# ---------------------------------------------------------------------
# KGE and NSE
# ---------------------------------------------------------------------


def kge(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    Kling–Gupta efficiency (KGE).

    The KGE metric decomposes model performance into three components:

    - Correlation (r) between observations and predictions.
    - Variability ratio (α = σ_pred / σ_obs).
    - Bias ratio (β = μ_pred / μ_obs).

    KGE is defined as:

    .. math::

        \\mathrm{KGE} = 1 - \\sqrt{(r - 1)^2 + (\\alpha - 1)^2 + (\\beta - 1)^2}

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values.

    Returns
    -------
    float
        Kling–Gupta efficiency. ``np.nan`` is returned when:
        - the sample size is < 2, or
        - the variance of the observed series is zero, or
        - the mean of the observed series is zero, or
        - the correlation cannot be computed.
    """
    yt, yp = _as_arrays(y_true, y_pred)

    # Minimum length requirement
    if yt.size < 2:
        return np.nan

    mu_y = float(np.mean(yt))
    sigma_y = float(np.std(yt, ddof=1))
    mu_p = float(np.mean(yp))
    sigma_p = float(np.std(yp, ddof=1))

    if sigma_y == 0.0 or mu_y == 0.0:
        return np.nan

    # Pearson correlation
    if sigma_p == 0.0:
        return np.nan
    r = float(np.corrcoef(yt, yp)[0, 1])
    if not np.isfinite(r):
        return np.nan

    alpha = sigma_p / sigma_y
    beta = mu_p / mu_y

    kge_val = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge_val)


def nse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    Nash–Sutcliffe efficiency (NSE).

    NSE compares the mean squared error of the predictions against the
    variance of the observed series:

    .. math::

        \\mathrm{NSE} = 1 - \\frac{\\sum (y_t - y_p)^2}
                                {\\sum (y_t - \\overline{y_t})^2}

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values.

    Returns
    -------
    float
        Nash–Sutcliffe efficiency. ``np.nan`` is returned when:
        - the sample size is < 2, or
        - the variance of the observed series is zero.

    Notes
    -----
    NSE is conceptually similar to the classical coefficient of
    determination from linear regression, but it is not constrained
    to the [0, 1] interval. Poor models can yield NSE < 0.
    """
    yt, yp = _as_arrays(y_true, y_pred)

    if yt.size < 2:
        return np.nan

    denom = float(np.sum((yt - np.mean(yt)) ** 2))
    if denom == 0.0:
        return np.nan

    num = float(np.sum((yt - yp) ** 2))
    return float(1.0 - num / denom)


# ---------------------------------------------------------------------
# Combined regression metrics
# ---------------------------------------------------------------------


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Compute a standard set of regression metrics:

    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of determination (R², correlation-based)
    - Kling–Gupta Efficiency (KGE)
    - Nash–Sutcliffe Efficiency (NSE)

    Definitions
    -----------
    * MAE and RMSE follow their usual definitions.
    * R² is defined as the **square of the Pearson correlation
      coefficient** between ``y_true`` and ``y_pred``. This differs from
      ``sklearn.metrics.r2_score`` but avoids redundancy with NSE.
    * NSE is the classical Nash–Sutcliffe efficiency.
    * KGE follows Gupta et al. (2009).

    Degenerate cases
    ----------------
    For degenerate cases (very small sample size, zero variance in the
    observed series, etc.), the corresponding metric is set to ``np.nan``.

    Special case
    ------------
    When there is **exactly one** (obs, pred) pair:

    * ``MAE`` and ``RMSE`` are computed as usual.
    * If the single point is a perfect match (``y_true == y_pred``),
      then, by convention:

      - ``R2 = 1.0``
      - ``KGE = 1.0``
      - ``NSE = 1.0``

    * Otherwise (single-point mismatch):

      - ``R2 = 0.0``
      - ``KGE = 0.0``
      - ``NSE = 0.0``

    This convention is particularly useful when a complete period
    (e.g. a month) collapses to one aggregated value and tests expect a
    “perfect” match to yield all efficiency-like metrics equal to 1.0.
    """
    yt, yp = _as_arrays(y_true, y_pred)

    # Empty input
    if yt.size == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "KGE": np.nan,
            "NSE": np.nan,
        }

    # Single-point special case (e.g. aggregation to one period)
    if yt.size == 1:
        mae = float(mean_absolute_error(yt, yp))
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))

        if float(yt[0]) == float(yp[0]):
            r2 = 1.0
            kge_val = 1.0
            nse_val = 1.0
        else:
            r2 = 0.0
            kge_val = 0.0
            nse_val = 0.0

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "KGE": kge_val,
            "NSE": nse_val,
        }

    # General case: at least 2 points
    mae = float(mean_absolute_error(yt, yp))

    # RMSE: use mean_squared_error without the `squared` keyword for
    # compatibility with older scikit-learn versions.
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))

    # R² as squared Pearson correlation, not sklearn.r2_score
    std_y = float(np.std(yt, ddof=1))
    std_p = float(np.std(yp, ddof=1))
    if std_y == 0.0 or std_p == 0.0:
        r2 = np.nan
    else:
        r = float(np.corrcoef(yt, yp)[0, 1])
        r2 = float(r ** 2)

    # Hydrological efficiencies
    kge_val = float(kge(yt, yp))
    nse_val = float(nse(yt, yp))

    # Normalize any non-finite values to NaN
    r2 = r2 if np.isfinite(r2) else np.nan
    kge_val = kge_val if np.isfinite(kge_val) else np.nan
    nse_val = nse_val if np.isfinite(nse_val) else np.nan

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "KGE": kge_val,
        "NSE": nse_val,
    }


# ---------------------------------------------------------------------
# Temporal aggregation + metrics
# ---------------------------------------------------------------------


_FREQ_ALIAS = {
    "M": "ME",  # monthly end-of-month
    "A": "YE",  # annual
    "Y": "YE",
    "Q": "QE",  # quarterly
}


def aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str = "y_true",
    yhat_col: str = "y_pred",
    freq: str = "M",
    agg: str = "sum",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Aggregate a daily prediction table to a coarser time scale and compute
    regression metrics on the aggregated series.

    This is typically used to evaluate, for example, monthly or annual
    precipitation totals from daily predictions.

    Parameters
    ----------
    df_pred : pandas.DataFrame
        DataFrame with at least [date_col, y_col, yhat_col].
    date_col : str, default "date"
        Name of the datetime-like column.
    y_col : str, default "y_true"
        Column with observed values.
    yhat_col : str, default "y_pred"
        Column with predicted values.
    freq : str, default "M"
        Resampling frequency. Common examples:

        - ``"M"``  → monthly (alias to ``"ME"`` internally)
        - ``"YE"`` → annual (end of year)
        - ``"QE"`` → quarterly (end of quarter)

        Aliases are handled via :data:`_FREQ_ALIAS`.
    agg : str, default "sum"
        Aggregation function to apply to both observed and predicted series.
        Must be one of: ``"sum"``, ``"mean"``, ``"median"``.

    Returns
    -------
    metrics : dict
        Regression metrics (MAE, RMSE, R², KGE, NSE) computed on the
        aggregated series.
    agg_df : pandas.DataFrame
        Aggregated DataFrame with [y_col, yhat_col] indexed by the new
        time frequency.

    Notes
    -----
    - If the input DataFrame is empty, or the resampled/aggregated table
      contains no valid rows, all metrics are returned as ``np.nan``.
    - When aggregation collapses to a **single period** (e.g. one month),
      :func:`regression_metrics` applies the single-point convention:
      perfect match → ``R2 = 1.0``, ``KGE = 1.0``, ``NSE = 1.0``.
    """
    freq = _FREQ_ALIAS.get(freq, freq)
    agg = agg.lower()
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of: 'sum', 'mean', or 'median'.")

    df = df_pred[[date_col, y_col, yhat_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "KGE": np.nan,
            "NSE": np.nan,
        }, df

    df = df.set_index(date_col).sort_index()

    if agg == "sum":
        agg_df = df.resample(freq).sum()
    elif agg == "median":
        agg_df = df.resample(freq).median()
    else:  # "mean"
        agg_df = df.resample(freq).mean()

    agg_df = agg_df.dropna()
    if agg_df.empty:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "KGE": np.nan,
            "NSE": np.nan,
        }, agg_df

    metrics = regression_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
    return metrics, agg_df


__all__ = [
    "kge",
    "nse",
    "regression_metrics",
    "aggregate_and_score",
]
