# SPDX-License-Identifier: MIT
"""
High-level pipeline utilities for LOSO-based climate reconstruction.

This module implements end-to-end helpers around spatial Random Forest
models for daily climate reconstruction:

- Station selection and neighbor discovery (KDTree-based).
- One-pass pre-processing for Leave-One-Station-Out (LOSO) evaluation.
- Per-station LOSO training and prediction (observed days only).
- Full-series reconstruction for a given station with LOSO logic.
- Network-wide evaluation at daily / monthly / annual scales.
- Batch export of reconstructed series with a manifest table.
- Plot utilities to compare in-situ observations vs. RF vs. external products.

The functions assume the canonical column names used across the codebase::

    station | date | latitude | longitude | altitude | prec | tmin | tmax | evap

You can pass alternative names via parameters if needed.

Metrics
-------
All evaluation helpers rely on :func:`MEXsrfdcrPy.metrics.regression_metrics`,
which returns a consistent dictionary with the keys

``"MAE"``, ``"RMSE"``, ``"R2"``, ``"KGE"`` and ``"NSE"``.

In this package, ``R2`` is defined as the **square of the Pearson correlation
coefficient** between observations and predictions, whereas ``NSE`` is the
classical Nash–Sutcliffe efficiency. This avoids redundancy between the two
and follows common practice in hydrology-oriented model evaluation.

Dependencies
------------
Core:
    - numpy
    - pandas
    - scikit-learn

Optional:
    - matplotlib (only for plotting)
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KDTree

from .metrics import regression_metrics

try:  # pragma: no cover - tqdm is optional, fallback is trivial
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------
# Warning policy
# ---------------------------------------------------------------------


def set_warning_policy(silence: bool = True) -> None:
    """
    Configure a conservative warning policy for the pipeline.

    Parameters
    ----------
    silence:
        If ``True`` (default), silence most noisy warnings that are not
        actionable in a typical workflow, such as pandas ``FutureWarning``
        and scikit-learn metric edge cases.
    """
    warnings.resetwarnings()
    if silence:
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            from sklearn.exceptions import UndefinedMetricWarning

            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        except Exception:  # pragma: no cover
            pass


# Apply a safe default at import time
set_warning_policy(True)


# ---------------------------------------------------------------------
# Small I/O helpers (internal)
# ---------------------------------------------------------------------


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create the parent directory for *path* if needed (no-op on None)."""
    if not path:
        return
    d = os.path.dirname(str(path)) or "."
    os.makedirs(d, exist_ok=True)


def _save_df(
    df: pd.DataFrame,
    path: Optional[str],
    *,
    parquet_compression: str = "snappy",
) -> Optional[str]:
    """
    Save a DataFrame to CSV, Parquet or Feather depending on the file extension.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Output path (``.csv``, ``.parquet`` or ``.feather``). If ``None``,
        nothing is written and ``None`` is returned.
    parquet_compression:
        Compression codec when writing Parquet files.

    Returns
    -------
    str or None
        The output path, or ``None`` if no file was written.
    """
    if path is None:
        return None
    _ensure_parent_dir(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".parquet":
        df.to_parquet(path, index=False, compression=parquet_compression)
    elif ext == ".feather":
        df.to_feather(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return path


def _save_json(obj: dict, path: Optional[str]) -> Optional[str]:
    """
    Save a Python dictionary as a pretty-printed JSON file.

    Parameters
    ----------
    obj:
        Object to serialise (must be JSON-serialisable).
    path:
        Output path. If ``None``, nothing is written.

    Returns
    -------
    str or None
        The output path, or ``None`` if no file was written.
    """
    if path is None:
        return None
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _resolve_columns(df: pd.DataFrame, cols) -> List[str]:
    """
    Resolve column names from strings or lists in a case-insensitive way.

    This helper allows the user to pass aliases (e.g. a variable name
    from an external product) without worrying about case or minor
    variants.

    Parameters
    ----------
    df:
        Input DataFrame.
    cols:
        A string or list-like of candidate column names.

    Returns
    -------
    list of str
        Columns that are actually present in *df*.
    """
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [cols]

    lower_map = {c.lower(): c for c in df.columns}
    aliases = {"pretotcorr": "prectotcorr"}  # common alias example

    out: List[str] = []
    for raw in cols:
        if raw is None:
            continue
        key = aliases.get(str(raw).lower(), str(raw).lower())
        if key in lower_map:
            out.append(lower_map[key])
    return out


def ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Return a shallow copy with a timezone-naive datetime column.

    Parameters
    ----------
    df:
        Input DataFrame.
    date_col:
        Column containing date-like values.

    Returns
    -------
    DataFrame
        Copy of *df* with ``date_col`` converted to ``datetime64[ns]``,
        dropped rows where conversion failed, and any timezone removed.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if isinstance(out[date_col].dtype, pd.DatetimeTZDtype):
        out[date_col] = out[date_col].dt.tz_localize(None)
    return out.dropna(subset=[date_col])


def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Append basic time features (year, month, day-of-year) and optionally
    cyclic encodings of the day-of-year.

    Parameters
    ----------
    df:
        Input DataFrame containing a datetime column.
    date_col:
        Name of the datetime column.
    add_cyclic:
        If ``True``, add ``doy_sin`` and ``doy_cos`` features.

    Returns
    -------
    DataFrame
        Copy of *df* with additional columns ``year``, ``month``, ``doy``,
        and, if requested, ``doy_sin``, ``doy_cos``.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"] / 365.25)
    return out


def _safe_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Thin wrapper around :func:`regression_metrics` for backwards compatibility.

    This helper guarantees that a metrics dictionary with the keys
    ``"MAE"``, ``"RMSE"``, ``"R2"``, ``"KGE"`` and ``"NSE"`` is always returned,
    even when the input has too few points or a constant target
    (in those cases the values are returned as NaN).
    """
    return regression_metrics(y_true, y_pred)


_FREQ_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE"}


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
    Resample a prediction DataFrame and compute regression metrics.

    This is a higher-level helper used in multiple places within the
    pipeline to summarise performance at monthly or annual scales.

    Parameters
    ----------
    df_pred:
        Input DataFrame with at least the columns ``date_col``,
        ``y_col`` (true values) and ``yhat_col`` (predictions).
    date_col:
        Name of the datetime column.
    y_col, yhat_col:
        Names of the columns containing the observed and predicted values.
    freq:
        Resampling frequency. Common values are ``'M'`` for monthly,
        ``'A'``/``'Y'`` for annual, and ``'Q'`` for quarterly. Internally
        some aliases are normalised to avoid pandas deprecation warnings.
    agg:
        Aggregation operation to apply before scoring. One of
        ``'sum'``, ``'mean'`` or ``'median'``.

    Returns
    -------
    metrics:
        Dictionary with the keys ``"MAE"``, ``"RMSE"``, ``"R2"``, ``"KGE"`` and
        ``"NSE"``. When there is not enough information to compute a metric
        (e.g. a single aggregated point), the corresponding value is NaN.
    agg_df:
        The resampled DataFrame used to compute the metrics.
    """
    freq = _FREQ_ALIAS.get(freq, freq)
    aggfunc = {"sum": "sum", "mean": "mean", "median": "median"}[agg]

    metrics_empty = {
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "KGE": np.nan,
        "NSE": np.nan,
    }

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return metrics_empty.copy(), df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: aggfunc, yhat_col: aggfunc}).dropna()
    if agg_df.empty:
        return metrics_empty.copy(), agg_df

    m = _safe_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
    return m, agg_df


# ---------------------------------------------------------------------
# Station selection & neighbors
# ---------------------------------------------------------------------


def select_stations(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    prefix: Optional[Iterable[str] | str] = None,
    station_ids: Optional[Iterable[int]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[int], bool]] = None,
) -> List[int]:
    """
    Select station identifiers using simple rules (prefix, regex, etc.).

    Parameters
    ----------
    data:
        Input DataFrame with a station identifier column.
    id_col:
        Name of the station identifier column.
    prefix:
        String or iterable of strings. All stations whose ID starts with
        any of these prefixes are selected.
    station_ids:
        Explicit list of station IDs to include.
    regex:
        Regular expression applied to the string representation of station
        IDs. Matching stations are selected.
    custom_filter:
        Callable that receives an integer station ID and returns ``True``
        if the station should be included.

    Returns
    -------
    list of int
        Sorted list of station identifiers. If no rule selects any station,
        all unique IDs in *data* are returned.
    """
    ids = data[id_col].dropna().astype(int).unique().tolist()
    chosen: set[int] = set()

    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            chosen.update([i for i in ids if str(i).startswith(str(p))])

    if station_ids is not None:
        chosen.update([int(i) for i in station_ids])

    if regex is not None:
        pat = re.compile(regex)
        chosen.update([i for i in ids if pat.match(str(i))])

    if custom_filter is not None:
        chosen.update([i for i in ids if custom_filter(i)])

    if not chosen:
        chosen = set(ids)
    return sorted(chosen)


def build_station_kneighbors(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    k: int = 100,
) -> Dict[int, List[int]]:
    """
    Build a dictionary of spatial neighbors for each station using a KDTree
    on spherical coordinates.

    Parameters
    ----------
    data:
        Input DataFrame with station positions.
    id_col:
        Station identifier column.
    lat_col, lon_col:
        Latitude and longitude columns (degrees).
    k:
        Number of neighbors to query per station (excluding self).

    Returns
    -------
    dict
        Mapping ``station_id -> list_of_neighbor_ids``.
    """
    centroids = (
        data.groupby(id_col)[[lat_col, lon_col]]
        .median()
        .rename(columns={lat_col: "lat", lon_col: "lon"})
        .reset_index()
    )
    X = np.deg2rad(centroids[["lat", "lon"]].values)
    tree = KDTree(X, metric="euclidean")
    _, idx = tree.query(X, k=min(k + 1, len(centroids)))
    neighbors: Dict[int, List[int]] = {}
    ids = centroids[id_col].tolist()
    for i, st in enumerate(ids):
        neigh_idx = idx[i][1:]  # drop self
        neighbors[st] = centroids.iloc[neigh_idx][id_col].tolist()
    return neighbors


def neighbor_correlation_table(
    data: pd.DataFrame,
    station_id: int,
    neighbor_ids: Iterable[int],
    *,
    id_col: str = "station",
    date_col: str = "date",
    value_col: str = "prec",
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_overlap: int = 60,
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> pd.DataFrame:
    """
    Compute pairwise correlation between one station and a set of neighbors.

    Parameters
    ----------
    data:
        Long-format DataFrame with at least ``id_col``, ``date_col`` and
        ``value_col``.
    station_id:
        Station identifier of the focal station.
    neighbor_ids:
        Iterable of neighbor station IDs.
    id_col, date_col, value_col:
        Column names.
    start, end:
        Optional date window. If provided, restrict the analysis to
        ``[start, end]`` (inclusive).
    min_overlap:
        Minimum number of overlapping days required to compute a correlation.
        If the overlap is smaller, the correlation is reported as NaN.
    save_table_path:
        Optional path where the resulting table will be written.
    parquet_compression:
        Compression codec when saving Parquet files.

    Returns
    -------
    DataFrame
        Table with columns ``neighbor``, ``corr`` and ``n_overlap``.
    """
    if not neighbor_ids:
        res = pd.DataFrame(columns=["neighbor", "corr", "n_overlap"])
        _save_df(res, save_table_path, parquet_compression=parquet_compression)
        return res

    df = data[[id_col, date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    ids_needed = set([int(station_id)] + [int(x) for x in neighbor_ids])
    df = df[df[id_col].isin(ids_needed)]

    wide = df.pivot_table(index=date_col, columns=id_col, values=value_col, aggfunc="mean")
    try:
        wide.columns = wide.columns.astype(int)
    except Exception:  # pragma: no cover
        pass

    if station_id not in wide.columns:
        res = pd.DataFrame(columns=["neighbor", "corr", "n_overlap"])
        _save_df(res, save_table_path, parquet_compression=parquet_compression)
        return res

    s_main = wide[station_id]
    rows = []
    for nid in neighbor_ids:
        if nid not in wide.columns:
            rows.append({"neighbor": int(nid), "corr": np.nan, "n_overlap": 0})
            continue
        pair = pd.concat([s_main, wide[nid]], axis=1, join="inner").dropna()
        n = int(len(pair))
        if n < min_overlap:
            corr = np.nan
        else:
            a, b = pair.iloc[:, 0].values, pair.iloc[:, 1].values
            if np.std(a) == 0 or np.std(b) == 0:
                corr = np.nan
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
        rows.append({"neighbor": int(nid), "corr": corr, "n_overlap": n})

    res = (
        pd.DataFrame(rows)
        .sort_values("corr", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    _save_df(res, save_table_path, parquet_compression=parquet_compression)
    return res


# ---------------------------------------------------------------------
# Target sampling helper
# ---------------------------------------------------------------------


def sample_target_for_training(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_cols: List[str],
    station_id: int,
    include_target_pct: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample a fraction of the target station's observed rows for inclusion
    in the training set.

    This helper is used in LOSO variants where the user wants to allow a
    small fraction of local information to enter the model (e.g. to test
    the sensitivity to "data leakage").

    Parameters
    ----------
    df:
        Input DataFrame.
    id_col, date_col, target_col:
        Column names.
    feature_cols:
        List of feature column names that must be non-NaN for a row to be
        eligible.
    station_id:
        Station identifier whose rows are being sampled.
    include_target_pct:
        Fraction of valid rows to sample, expressed in percent (0–100).
    random_state:
        Seed for the internal RNG.

    Returns
    -------
    DataFrame
        Sampled subset of *df* (possibly empty).
    """
    pct = max(0.0, min(float(include_target_pct), 100.0))
    if pct <= 0.0:
        return df.iloc[0:0].copy()
    df_t = df[df[id_col] == station_id].copy()
    ok = (~df_t[target_col].isna()) & (~df_t[feature_cols].isna().any(axis=1))
    df_t = df_t.loc[ok]
    if df_t.empty:
        return df_t
    n = int(np.ceil(len(df_t) * (pct / 100.0)))
    return df_t.sample(n=n, random_state=random_state)


# ---------------------------------------------------------------------
# Core LOSO (observed days)
# ---------------------------------------------------------------------


def loso_train_predict_station(
    data: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    feature_cols: Optional[List[str]] = None,
    add_cyclic: bool = False,
    model=None,
    rf_params: Optional[Dict] = None,
    agg_for_metrics: str = "sum",
    start: Optional[str] = None,
    end: Optional[str] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_predictions_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Leave-One-Station-Out training and prediction for a single station
    (observed days only).

    Parameters
    ----------
    data:
        Long-format DataFrame containing all stations.
    station_id:
        Station that will be left out for testing.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names in *data*.
    feature_cols:
        Optional explicit list of feature columns. If ``None``, a standard
        set ``[lat, lon, alt, year, month, doy]`` plus optional cyclic
        features is used.
    add_cyclic:
        If ``True``, add ``doy_sin`` and ``doy_cos`` to the feature set.
    model:
        Pre-instantiated regressor. If ``None``, a
        :class:`RandomForestRegressor` is created from ``rf_params``.
    rf_params:
        Parameters for :class:`sklearn.ensemble.RandomForestRegressor` when
        ``model`` is ``None``.
    agg_for_metrics:
        Aggregation function used for monthly and annual metrics
        (``'sum'``, ``'mean'`` or ``'median'``).
    start, end:
        Optional period restriction for the test station.
    include_target_pct:
        Percentage of valid rows from the target station to *also* include
        in the training set (to explore sensitivity to leakage).
    include_target_seed:
        RNG seed for the target sampling.
    save_predictions_path:
        Optional path where the test predictions are saved.
    save_metrics_path:
        Optional path where the metric dictionary is saved as JSON.
    parquet_compression:
        Compression codec for Parquet output if used.

    Returns
    -------
    out:
        DataFrame with columns ``[date, station, y_true, y_pred]``.
    metrics:
        Dictionary with keys ``"daily"``, ``"monthly"``, ``"annual"``,
        each containing a metrics dictionary with the keys
        ``"MAE"``, ``"RMSE"``, ``"R2"``, ``"KGE"``, ``"NSE"``.
    model:
        Fitted model (same object that was passed in, or a new RF instance).
    feature_cols:
        The feature column list actually used.
    """
    if model is None:
        rf_params = rf_params or dict(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )
        model = RandomForestRegressor(**rf_params)

    df = ensure_datetime(data, date_col)
    df = add_time_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feature_cols = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feature_cols += ["doy_sin", "doy_cos"]

    test_df = df[df[id_col] == station_id].copy()
    if start or end:
        if not test_df.empty:
            lo = pd.to_datetime(start) if start else test_df[date_col].min()
            hi = pd.to_datetime(end) if end else test_df[date_col].max()
            test_df = test_df[(test_df[date_col] >= lo) & (test_df[date_col] <= hi)]
    if test_df.empty:
        raise ValueError("Target station has no rows in the requested period.")

    train_rest = df[df[id_col] != station_id].copy()
    sample_t = sample_target_for_training(
        df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        feature_cols=feature_cols,
        station_id=station_id,
        include_target_pct=include_target_pct,
        random_state=include_target_seed,
    )
    train_df = pd.concat([train_rest, sample_t], axis=0, copy=False)

    train_df = train_df.dropna(subset=feature_cols + [target_col])
    test_df = test_df.dropna(subset=feature_cols + [target_col])
    if train_df.empty:
        raise ValueError("Training set is empty after filtering.")

    X_train = train_df[feature_cols].to_numpy(copy=False)
    y_train = train_df[target_col].to_numpy(copy=False)
    X_test = test_df[feature_cols].to_numpy(copy=False)
    y_test = test_df[target_col].to_numpy(copy=False)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    out = test_df[[date_col, id_col]].copy()
    out.rename(columns={id_col: "station"}, inplace=True)
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out = out.sort_values(date_col).reset_index(drop=True)

    daily = _safe_metrics(out["y_true"], out["y_pred"])
    monthly, _ = aggregate_and_score(
        out,
        date_col=date_col,
        y_col="y_true",
        yhat_col="y_pred",
        freq="M",
        agg=agg_for_metrics,
    )
    annual, _ = aggregate_and_score(
        out,
        date_col=date_col,
        y_col="y_true",
        yhat_col="y_pred",
        freq="YE",
        agg=agg_for_metrics,
    )
    metrics = {"daily": daily, "monthly": monthly, "annual": annual}

    _save_df(out, save_predictions_path, parquet_compression=parquet_compression)
    _save_json(metrics, save_metrics_path)
    return out, metrics, model, feature_cols


# ---------------------------------------------------------------------
# Full series LOSO
# ---------------------------------------------------------------------


def loso_predict_full_series(
    data: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_series_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Reconstruct the full daily series for one station using LOSO logic.

    The model is trained on all other stations (plus an optional fraction of
    the target station itself) and then used to reconstruct the target for
    the entire period ``[start, end]``.

    Parameters
    ----------
    data:
        Long-format DataFrame containing all stations.
    station_id:
        Station to reconstruct.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names in *data*.
    start, end:
        Date range (inclusive) for the reconstructed series.
    rf_params:
        Parameters for :class:`RandomForestRegressor`.
    add_cyclic:
        If ``True``, use cyclic encodings of day-of-year.
    feature_cols:
        Optional explicit feature list. If ``None``, the default
        ``[lat, lon, alt, year, month, doy]`` (+ cyclic features) is used.
    include_target_pct, include_target_seed:
        See :func:`sample_target_for_training`.
    save_series_path:
        Optional path where the full reconstructed series is written.
    save_metrics_path:
        Optional path to save the metric dictionary as JSON.
    parquet_compression:
        Parquet compression codec.

    Returns
    -------
    full_df:
        DataFrame with ``[date, station, y_pred_full, y_true]`` where
        ``y_true`` is present only on days with observations.
    metrics:
        Dictionary with keys ``"daily"``, ``"monthly"``, ``"annual"``.
    model:
        Fitted :class:`RandomForestRegressor`.
    feature_cols:
        Feature list actually used.
    """
    df = ensure_datetime(data, date_col)

    st = df[df[id_col] == station_id]
    if st.empty:
        raise ValueError(f"No rows for station {station_id}.")
    lat0 = st[lat_col].median()
    lon0 = st[lon_col].median()
    alt0 = st[alt_col].median()

    train_df = add_time_features(df.copy(), date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feature_cols = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feature_cols += ["doy_sin", "doy_cos"]

    train_rest = train_df[train_df[id_col] != station_id]
    sample_t = sample_target_for_training(
        train_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        feature_cols=feature_cols,
        station_id=station_id,
        include_target_pct=include_target_pct,
        random_state=include_target_seed,
    )
    train_df = pd.concat([train_rest, sample_t], axis=0, copy=False).dropna(
        subset=feature_cols + [target_col]
    )
    if train_df.empty:
        raise ValueError("Training set is empty after filtering.")

    model = RandomForestRegressor(**rf_params)
    model.fit(train_df[feature_cols], train_df[target_col])

    dates = pd.date_range(
        start=pd.to_datetime(start),
        end=pd.to_datetime(end),
        freq="D",
    )
    synth = pd.DataFrame(
        {
            date_col: dates,
            "station": station_id,
            lat_col: lat0,
            lon_col: lon0,
            alt_col: alt0,
        }
    )
    synth = add_time_features(synth, date_col, add_cyclic=add_cyclic)

    y_pred_full = model.predict(synth[feature_cols])
    full_df = synth[[date_col, "station"]].copy()
    full_df["y_pred_full"] = y_pred_full

    obs = (
        df[df[id_col] == station_id][[date_col, target_col]]
        .rename(columns={target_col: "y_true"})
    )
    full_df = full_df.merge(obs, on=date_col, how="left")

    comp = full_df.dropna(subset=["y_true"]).copy()
    comp["y_pred"] = comp["y_pred_full"]
    daily = _safe_metrics(comp["y_true"], comp["y_pred"])
    monthly, _ = aggregate_and_score(
        comp,
        date_col=date_col,
        y_col="y_true",
        yhat_col="y_pred",
        freq="M",
        agg="sum",
    )
    annual, _ = aggregate_and_score(
        comp,
        date_col=date_col,
        y_col="y_true",
        yhat_col="y_pred",
        freq="YE",
        agg="sum",
    )
    metrics = {"daily": daily, "monthly": monthly, "annual": annual}

    _save_df(full_df, save_series_path, parquet_compression=parquet_compression)
    _save_json(metrics, save_metrics_path)
    return full_df, metrics, model, feature_cols


def loso_predict_full_series_neighbors(
    data: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    k_neighbors: int = 50,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_series_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Convenience wrapper for LOSO full-series reconstruction using spatial neighbors.

    This helper mimics the typical workflow of earlier "fast" helpers:

    1. Build (or reuse) a KDTree-based neighbor map.
    2. Restrict the training pool to the K nearest neighbors of the
       target station plus the station itself.
    3. Call :func:`loso_predict_full_series` on this local subset.

    Parameters
    ----------
    data:
        Long-format DataFrame containing all stations.
    station_id:
        Station to reconstruct.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names in *data*.
    start, end:
        Date range (inclusive) for the reconstructed series.
    k_neighbors:
        Number of spatial neighbors to use for the training pool.
    neighbor_map:
        Optional pre-computed neighbor dictionary as returned by
        :func:`build_station_kneighbors`. If ``None``, it is computed
        internally from *data*.
    rf_params, add_cyclic, feature_cols:
        Passed directly to :func:`loso_predict_full_series`.
    include_target_pct, include_target_seed:
        See :func:`sample_target_for_training`.
    save_series_path, save_metrics_path, parquet_compression:
        See :func:`loso_predict_full_series`.

    Returns
    -------
    full_df:
        DataFrame with ``[date, station, y_pred_full, y_true]``.
    metrics:
        Dictionary with keys ``"daily"``, ``"monthly"``, ``"annual"``.
    model:
        Fitted :class:`RandomForestRegressor`.
    feature_cols:
        Feature list actually used by the model.
    """
    # 1) Neighbor map if not provided
    if neighbor_map is None:
        neighbor_map = build_station_kneighbors(
            data,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=k_neighbors,
        )

    # 2) Local subset = neighbors + target
    neigh_ids = neighbor_map.get(station_id, [])
    local_ids = neigh_ids + [station_id]
    data_local = data[data[id_col].isin(local_ids)].copy()

    # 3) Delegate to the core full-series function
    return loso_predict_full_series(
        data_local,
        station_id=station_id,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        rf_params=rf_params,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
        save_series_path=save_series_path,
        save_metrics_path=save_metrics_path,
        parquet_compression=parquet_compression,
    )


def loso_predict_full_series_fast(
    data: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: int = 50,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_series_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Backwards-compatible convenience wrapper for full-series LOSO with neighbors.

    This function is intentionally thin: it simply delegates to
    :func:`loso_predict_full_series_neighbors` with the same arguments,
    so that older workflows that used a ``*_fast`` helper remain easy to
    read and use.

    Parameters
    ----------
    See :func:`loso_predict_full_series_neighbors`.

    Returns
    -------
    See :func:`loso_predict_full_series_neighbors`.
    """
    return loso_predict_full_series_neighbors(
        data,
        station_id=station_id,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
        rf_params=rf_params,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
        save_series_path=save_series_path,
        save_metrics_path=save_metrics_path,
        parquet_compression=parquet_compression,
    )


# ---------------------------------------------------------------------
# EVALUATION — Classic
# ---------------------------------------------------------------------


def evaluate_all_stations(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    rf_params: Dict = dict(
        n_estimators=300,
        max_depth=30,
        random_state=42,
        n_jobs=-1,
    ),
    agg_for_metrics: str = "sum",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    order_by: Tuple[str, bool] = ("RMSE_d", True),
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> pd.DataFrame:
    """
    Evaluate LOSO performance for all stations in *data*.

    This is a straightforward loop over stations calling
    :func:`loso_train_predict_station` for each one.

    Parameters
    ----------
    data:
        Input DataFrame.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names.
    rf_params:
        Parameters for the underlying RandomForest model.
    agg_for_metrics:
        Aggregation function used for monthly and yearly metrics.
    start, end:
        Global date window for all stations.
    add_cyclic, feature_cols:
        See :func:`loso_train_predict_station`.
    order_by:
        Tuple ``(column_name, ascending)`` used to sort the final table.
        If the column is not present, no sorting is applied.
    include_target_pct, include_target_seed:
        See :func:`sample_target_for_training`.
    save_table_path:
        Optional CSV/Parquet/Feather path.
    parquet_compression:
        Parquet compression codec.

    Returns
    -------
    DataFrame
        One row per station, with extended metric columns:

        - ``MAE_d``, ``RMSE_d``, ``R2_d``, ``KGE_d``, ``NSE_d``
        - ``MAE_m``, ``RMSE_m``, ``R2_m``, ``KGE_m``, ``NSE_m``
        - ``MAE_y``, ``RMSE_y``, ``R2_y``, ``KGE_y``, ``NSE_y``
    """
    results = []
    stations = data[id_col].dropna().unique()

    for stid in stations:
        try:
            out_df, metrics, _m, _f = loso_train_predict_station(
                data,
                station_id=int(stid),
                id_col=id_col,
                date_col=date_col,
                lat_col=lat_col,
                lon_col=lon_col,
                alt_col=alt_col,
                target_col=target_col,
                rf_params=rf_params,
                agg_for_metrics=agg_for_metrics,
                start=start,
                end=end,
                add_cyclic=add_cyclic,
                feature_cols=feature_cols,
                include_target_pct=include_target_pct,
                include_target_seed=include_target_seed,
            )
            results.append(
                {
                    "station": int(stid),
                    "n_rows": len(out_df),
                    # daily
                    "MAE_d": metrics["daily"]["MAE"],
                    "RMSE_d": metrics["daily"]["RMSE"],
                    "R2_d": metrics["daily"]["R2"],
                    "KGE_d": metrics["daily"]["KGE"],
                    "NSE_d": metrics["daily"]["NSE"],
                    # monthly
                    "MAE_m": metrics["monthly"]["MAE"],
                    "RMSE_m": metrics["monthly"]["RMSE"],
                    "R2_m": metrics["monthly"]["R2"],
                    "KGE_m": metrics["monthly"]["KGE"],
                    "NSE_m": metrics["monthly"]["NSE"],
                    # annual
                    "MAE_y": metrics["annual"]["MAE"],
                    "RMSE_y": metrics["annual"]["RMSE"],
                    "R2_y": metrics["annual"]["R2"],
                    "KGE_y": metrics["annual"]["KGE"],
                    "NSE_y": metrics["annual"]["NSE"],
                }
            )
        except Exception:
            results.append(
                {
                    "station": int(stid),
                    "n_rows": 0,
                    # daily
                    "MAE_d": np.nan,
                    "RMSE_d": np.nan,
                    "R2_d": np.nan,
                    "KGE_d": np.nan,
                    "NSE_d": np.nan,
                    # monthly
                    "MAE_m": np.nan,
                    "RMSE_m": np.nan,
                    "R2_m": np.nan,
                    "KGE_m": np.nan,
                    "NSE_m": np.nan,
                    # annual
                    "MAE_y": np.nan,
                    "RMSE_y": np.nan,
                    "R2_y": np.nan,
                    "KGE_y": np.nan,
                    "NSE_y": np.nan,
                }
            )

    df_out = pd.DataFrame(results)
    if order_by and not df_out.empty:
        col, asc = order_by
        if col in df_out.columns:
            df_out = df_out.sort_values(col, ascending=asc).reset_index(drop=True)

    _save_df(df_out, save_table_path, parquet_compression=parquet_compression)
    return df_out


# ---------------------------------------------------------------------
# EVALUATION — FAST (single pre-processing, optional neighbors, buffered logging)
# ---------------------------------------------------------------------


def _append_rows_to_csv(
    rows: List[Dict],
    path: str,
    *,
    header_written_flag: Dict[str, bool],
) -> None:
    """Append rows to a CSV file, writing headers only once."""
    if not rows or path is None:
        return
    df_tmp = pd.DataFrame(rows)
    first = not os.path.exists(path) and not header_written_flag.get(path, False)
    df_tmp.to_csv(path, mode="a", index=False, header=first)
    header_written_flag[path] = True


def _preprocess_for_loso_fast(
    data: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    add_cyclic: bool,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    var_cols: Optional[Iterable[str]],
    target_col: str,
    start: Optional[str],
    end: Optional[str],
    feature_cols: Optional[List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One-pass pre-processing used by :func:`evaluate_all_stations_fast`.

    The function:
        - casts the date column to datetime and removes timezones,
        - optionally clips to a given period,
        - adds time features,
        - resolves additional covariates via :func:`_resolve_columns`,
        - defines the final feature list.
    """
    df = data.copy()

    # datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if isinstance(df[date_col].dtype, pd.DatetimeTZDtype):
        df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.dropna(subset=[date_col])

    # clip
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # time features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["doy"] = df[date_col].dt.dayofyear
    if add_cyclic:
        df["doy_sin"] = np.sin(2.0 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2.0 * np.pi * df["doy"] / 365.25)

    # optional covariates
    resolved_vars = _resolve_columns(df, var_cols)

    # features
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
        feats += resolved_vars
    else:
        feats = list(feature_cols)

    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    df = df[keep]
    return df, feats


def evaluate_all_stations_fast(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    var_col: Optional[str] = None,
    var_cols: Optional[Iterable[str]] = None,
    target_col: str = "prec",
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[int]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[int], bool]] = None,
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    rf_params: Dict = dict(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    agg_for_metrics: str = "sum",
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    show_progress: bool = True,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> pd.DataFrame:
    """
    Faster LOSO evaluation across many stations using a single pre-processing
    pass and optional neighbor restriction.

    Parameters
    ----------
    data:
        Input DataFrame.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names.
    var_col, var_cols:
        Optional additional covariate columns. ``var_col`` is a simple
        alias string; if provided, it is wrapped into a list and merged
        into ``var_cols``.
    prefix, station_ids, regex, custom_filter:
        Station selection rules. See :func:`select_stations`.
    start, end:
        Global analysis period.
    rf_params:
        Parameters for :class:`RandomForestRegressor`.
    agg_for_metrics:
        Aggregation function for monthly/annual metrics.
    add_cyclic, feature_cols:
        Time encoding options. See :func:`add_time_features`.
    k_neighbors, neighbor_map:
        Optional spatial restriction: each station can be trained only on
        its neighbors defined in ``neighbor_map`` or computed via
        :func:`build_station_kneighbors`.
    log_csv:
        Optional CSV file where per-station rows are appended as the
        evaluation progresses.
    flush_every:
        Number of stations after which the buffered log is flushed.
    show_progress:
        If ``True``, wrap the station iterator with :func:`tqdm`.
    include_target_pct, include_target_seed:
        Fraction of test-station rows to include in training to explore
        leakage scenarios.
    min_station_rows:
        If provided, stations with fewer valid rows than this threshold
        are silently skipped.
    save_table_path:
        Optional path where the final summary table is saved.
    parquet_compression:
        Parquet compression codec.

    Returns
    -------
    DataFrame
        Per-station summary including extended metrics:

        - ``MAE_d``, ``RMSE_d``, ``R2_d``, ``KGE_d``, ``NSE_d``
        - ``MAE_m``, ``RMSE_m``, ``R2_m``, ``KGE_m``, ``NSE_m``
        - ``MAE_y``, ``RMSE_y``, ``R2_y``, ``KGE_y``, ``NSE_y``
    """
    if var_cols is None and var_col is not None:
        var_cols = [var_col]

    t_all0 = time.time()

    # one pre-processing pass
    df, feats = _preprocess_for_loso_fast(
        data,
        id_col=id_col,
        date_col=date_col,
        add_cyclic=add_cyclic,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        var_cols=var_cols,
        target_col=target_col,
        start=start,
        end=end,
        feature_cols=feature_cols,
    )

    # valid mask
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # station selection
    stations = select_stations(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    # optional filter by min valid rows per station
    if min_station_rows is not None:
        valid_counts = df.loc[valid_mask_global, [id_col]].groupby(id_col).size().astype(int)
        before = len(stations)
        stations = [
            s for s in stations if int(valid_counts.get(s, 0)) >= int(min_station_rows)
        ]
        if show_progress:
            tqdm.write(
                f"Filtered by min_station_rows={min_station_rows}: "
                f"{before} → {len(stations)} stations"
            )

    # neighbors
    if k_neighbors is not None and neighbor_map is None:
        neighbor_map = build_station_kneighbors(
            df,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=k_neighbors,
        )

    header_flag: Dict[str, bool] = {}
    rows: List[Dict] = []
    pending: List[Dict] = []
    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = time.time()

        is_target = df[id_col] == sid
        st_block = df.loc[is_target, [lat_col, lon_col, alt_col]]
        lat_med = float(st_block[lat_col].median()) if not st_block.empty else np.nan
        lon_med = float(st_block[lon_col].median()) if not st_block.empty else np.nan
        alt_med = float(st_block[alt_col].median()) if not st_block.empty else np.nan

        # train pool (neighbors optional)
        if k_neighbors is not None:
            neigh = neighbor_map.get(sid, [])
            is_train_pool = df[id_col].isin(neigh) & (~is_target)
        else:
            is_train_pool = ~is_target

        test_mask = is_target & valid_mask_global
        test_df = df.loc[test_mask]
        if test_df.empty:
            sec = time.time() - t0
            row = {
                "station": sid,
                "n_rows": 0,
                "seconds": sec,
                # daily
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "KGE_d": np.nan,
                "NSE_d": np.nan,
                # monthly
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "KGE_m": np.nan,
                "NSE_m": np.nan,
                # annual
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
                "KGE_y": np.nan,
                "NSE_y": np.nan,
                "include_target_pct": float(include_target_pct),
                lat_col: lat_med,
                lon_col: lon_med,
                alt_col: alt_med,
            }
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid rows (skipped)")
            continue

        train_pool = df.loc[is_train_pool & valid_mask_global]

        # optional inclusion of target rows in training
        pct = max(0.0, min(float(include_target_pct), 100.0))
        if pct > 0.0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            idx_sample = test_df.sample(
                n=n_take,
                random_state=include_target_seed,
            ).index
            train_df = pd.concat([train_pool, df.loc[idx_sample]], axis=0, copy=False)
        else:
            train_df = train_pool

        if train_df.empty:
            sec = time.time() - t0
            row = {
                "station": sid,
                "n_rows": 0,
                "seconds": sec,
                # daily
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "KGE_d": np.nan,
                "NSE_d": np.nan,
                # monthly
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "KGE_m": np.nan,
                "NSE_m": np.nan,
                # annual
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
                "KGE_y": np.nan,
                "NSE_y": np.nan,
                "include_target_pct": float(include_target_pct),
                lat_col: lat_med,
                lon_col: lon_med,
                alt_col: alt_med,
            }
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: empty train (skipped)")
            continue

        # fit & predict
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model = RandomForestRegressor(**rf_params)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        df_pred = pd.DataFrame(
            {
                date_col: test_df[date_col].values,
                "y_true": y_test,
                "y_pred": y_hat,
            }
        )
        daily = _safe_metrics(df_pred["y_true"], df_pred["y_pred"])
        monthly, _ = aggregate_and_score(
            df_pred,
            date_col=date_col,
            y_col="y_true",
            yhat_col="y_pred",
            freq="M",
            agg=agg_for_metrics,
        )
        annual, _ = aggregate_and_score(
            df_pred,
            date_col=date_col,
            y_col="y_true",
            yhat_col="y_pred",
            freq="YE",
            agg=agg_for_metrics,
        )

        sec = time.time() - t0
        row = {
            "station": sid,
            "n_rows": int(len(df_pred)),
            "seconds": sec,
            # daily
            "MAE_d": daily["MAE"],
            "RMSE_d": daily["RMSE"],
            "R2_d": daily["R2"],
            "KGE_d": daily["KGE"],
            "NSE_d": daily["NSE"],
            # monthly
            "MAE_m": monthly["MAE"],
            "RMSE_m": monthly["RMSE"],
            "R2_m": monthly["R2"],
            "KGE_m": monthly["KGE"],
            "NSE_m": monthly["NSE"],
            # annual
            "MAE_y": annual["MAE"],
            "RMSE_y": annual["RMSE"],
            "R2_y": annual["R2"],
            "KGE_y": annual["KGE"],
            "NSE_y": annual["NSE"],
            "include_target_pct": float(include_target_pct),
            lat_col: lat_med,
            lon_col: lon_med,
            alt_col: alt_med,
        }
        rows.append(row)
        pending.append(row)

        if log_csv and len(pending) >= flush_every:
            _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
            pending = []

        if show_progress:
            tqdm.write(
                f"Station {sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)"
            )

    if log_csv and pending:
        _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)

    result_df = pd.DataFrame(rows)
    _save_df(result_df, save_table_path, parquet_compression=parquet_compression)

    if show_progress:
        total_sec = time.time() - t_all0
        tqdm.write(
            f"Done. {len(stations)} stations in {total_sec:.1f}s "
            f"(avg {total_sec / max(1, len(stations)):.2f}s/station)."
        )
    return result_df


# ---------------------------------------------------------------------
# EXPORT — per-station + batch
# ---------------------------------------------------------------------


def export_full_series_station(
    data: pd.DataFrame,
    station_id: int,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    out_dir: str = "/kaggle/working/series",
    file_format: str = "parquet",
    parquet_compression: str = "snappy",
    csv_index: bool = False,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    save_metrics_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], str]:
    """
    Export the full reconstructed series for a single station to disk.

    Parameters
    ----------
    data:
        Input DataFrame.
    station_id:
        Station to reconstruct/export.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names.
    start, end:
        Reconstruction period for the output file.
    train_start, train_end:
        Optional training period restriction for the local dataset.
    rf_params, add_cyclic, feature_cols:
        See :func:`loso_predict_full_series`.
    k_neighbors, neighbor_map:
        Optional neighbor restriction for training.
    out_dir:
        Directory where the output file will be written.
    file_format:
        ``"parquet"`` or ``"csv"``.
    parquet_compression:
        Compression codec for Parquet.
    csv_index:
        Whether to include an index column in CSV output.
    include_target_pct, include_target_seed:
        See :func:`sample_target_for_training`.
    save_metrics_path:
        Optional path to save the metric dictionary as JSON.

    Returns
    -------
    full_df:
        Reconstructed series DataFrame.
    metrics:
        Metric dictionary returned by :func:`loso_predict_full_series`.
    out_path:
        Path of the file written to *out_dir*.
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if k_neighbors is not None:
        if neighbor_map is None:
            neighbor_map = build_station_kneighbors(
                df,
                id_col=id_col,
                lat_col=lat_col,
                lon_col=lon_col,
                k=k_neighbors,
            )
        neigh_ids = neighbor_map.get(station_id, [])
        df_train = df[df[id_col].isin(neigh_ids)]
        df_target = df[df[id_col] == station_id]
        df_local = pd.concat([df_train, df_target], axis=0, copy=False)
    else:
        df_local = df

    if train_start or train_end:
        lo = pd.to_datetime(train_start) if train_start else df_local[date_col].min()
        hi = pd.to_datetime(train_end) if train_end else df_local[date_col].max()
        df_local = df_local[(df_local[date_col] >= lo) & (df_local[date_col] <= hi)]

    full_df, metrics, _model, _feats = loso_predict_full_series(
        df_local,
        station_id,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        rf_params=rf_params,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    base = f"loso_fullseries_{station_id}_{start.replace('-','')}_{end.replace('-','')}"
    if file_format.lower() == "parquet":
        out_path = os.path.join(out_dir, base + ".parquet")
        full_df.to_parquet(out_path, index=False, compression=parquet_compression)
    elif file_format.lower() == "csv":
        out_path = os.path.join(out_dir, base + ".csv")
        full_df.to_csv(out_path, index=csv_index)
    else:
        raise ValueError("file_format must be 'parquet' or 'csv'.")

    _save_json(metrics, save_metrics_path)
    return full_df, metrics, out_path


def export_full_series_batch(
    data: pd.DataFrame,
    *,
    station_ids: Optional[Iterable[int]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[int], bool]] = None,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    out_dir: str = "/kaggle/working/series",
    file_format: str = "parquet",
    parquet_compression: str = "snappy",
    csv_index: bool = False,
    manifest_path: Optional[str] = "/kaggle/working/series_manifest.csv",
    show_progress: bool = True,
    min_overlap_corr: int = 60,
    combine_output_path: Optional[str] = None,
    combine_format: str = "csv",
    combine_parquet_compression: str = "snappy",
    combine_schema: str = "input_like",
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
) -> pd.DataFrame:
    """
    Batch export of reconstructed full series for many stations.

    Parameters
    ----------
    data:
        Input DataFrame.
    station_ids, prefix, regex, custom_filter:
        Station selection rules. See :func:`select_stations`.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names.
    start, end:
        Reconstruction period (common to all stations).
    train_start, train_end:
        Optional training period restriction.
    rf_params, add_cyclic, feature_cols:
        See :func:`loso_predict_full_series`.
    k_neighbors, neighbor_map:
        Optional neighbor-based restriction for the training pool.
    out_dir:
        Directory where individual per-station files are written.
    file_format:
        Output format for per-station files (``"parquet"`` or ``"csv"``).
    parquet_compression, csv_index:
        I/O options for per-station files.
    manifest_path:
        Optional CSV manifest where per-station metadata and paths
        are recorded.
    show_progress:
        Whether to display a progress bar.
    min_overlap_corr:
        Currently unused placeholder (kept for backwards compatibility).
    combine_output_path:
        If not ``None``, a combined file is written that stacks all
        reconstructed series using a chosen schema.
    combine_format:
        Format for the combined file (``"csv"`` or ``"parquet"``).
    combine_parquet_compression:
        Compression codec when writing the combined Parquet file.
    combine_schema:
        Either ``"input_like"`` (columns similar to the input table) or
        ``"compact"`` (subset of columns plus predicted and observed values).
    include_target_pct, include_target_seed:
        See :func:`sample_target_for_training`.

    Returns
    -------
    DataFrame
        Manifest with one row per station, including:

        - path to the per-station file,
        - evaluation metrics (daily / monthly / annual),
        - coverage and run time information.
    """
    stations = select_stations(
        data,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    if k_neighbors is not None and neighbor_map is None:
        neighbor_map = build_station_kneighbors(
            data,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=k_neighbors,
        )

    header_written = False
    combine_parts: List[pd.DataFrame] = []
    if combine_output_path is not None:
        os.makedirs(os.path.dirname(combine_output_path), exist_ok=True)
        if combine_format.lower() == "csv" and os.path.exists(combine_output_path):
            os.remove(combine_output_path)

    rows = []
    it = tqdm(stations, desc="Exporting full series", unit="st") if show_progress else stations

    for sid in it:
        t0 = time.time()
        try:
            if k_neighbors is not None:
                neigh_ids = neighbor_map.get(sid, [])
                df_train = data[data[id_col].isin(neigh_ids)]
                df_test = data[data[id_col] == sid]
                df_local = pd.concat([df_train, df_test], axis=0, copy=False)
            else:
                df_local = data

            full_df, metrics, _model, _feats = loso_predict_full_series(
                df_local,
                sid,
                id_col=id_col,
                date_col=date_col,
                lat_col=lat_col,
                lon_col=lon_col,
                alt_col=alt_col,
                target_col=target_col,
                start=start,
                end=end,
                rf_params=rf_params,
                add_cyclic=add_cyclic,
                feature_cols=feature_cols,
                include_target_pct=include_target_pct,
                include_target_seed=include_target_seed,
            )

            os.makedirs(out_dir, exist_ok=True)
            base = f"loso_fullseries_{sid}_{start.replace('-','')}_{end.replace('-','')}"
            if file_format.lower() == "parquet":
                path = os.path.join(out_dir, base + ".parquet")
                full_df.to_parquet(path, index=False, compression=parquet_compression)
            elif file_format.lower() == "csv":
                path = os.path.join(out_dir, base + ".csv")
                full_df.to_csv(path, index=csv_index)
            else:
                raise ValueError("file_format must be 'parquet' or 'csv'.")

            # Optional combined output
            if combine_output_path is not None:
                if combine_schema == "input_like":
                    part = pd.DataFrame(
                        {
                            id_col: sid,
                            lat_col: full_df[lat_col].median() if lat_col in full_df else np.nan,
                            lon_col: full_df[lon_col].median() if lon_col in full_df else np.nan,
                            alt_col: full_df[alt_col].median() if alt_col in full_df else np.nan,
                            date_col: full_df[date_col].values,
                        }
                    )
                    filled = full_df["y_true"].where(
                        full_df["y_true"].notna(),
                        full_df["y_pred_full"],
                    )
                    part[target_col] = filled.values
                    part = part[[id_col, lat_col, lon_col, alt_col, date_col, target_col]]
                elif combine_schema == "compact":
                    part = full_df[[date_col, "station", "y_pred_full", "y_true"]].copy()
                    part[target_col] = part["y_true"].where(
                        part["y_true"].notna(),
                        part["y_pred_full"],
                    )
                    part = part[["station", "y_pred_full", "y_true", date_col, target_col]]
                else:
                    raise ValueError("combine_schema must be 'input_like' or 'compact'.")

                if combine_format.lower() == "csv":
                    part.to_csv(
                        combine_output_path,
                        mode="a",
                        index=False,
                        header=(not header_written),
                    )
                    header_written = True
                elif combine_format.lower() == "parquet":
                    combine_parts.append(part)
                else:
                    raise ValueError("combine_format must be 'csv' or 'parquet'.")

            sec = time.time() - t0
            cov = float(full_df["y_true"].notna().mean()) * 100.0
            rows.append(
                {
                    "station": sid,
                    "path": path,
                    "seconds": sec,
                    "coverage_pct": cov,
                    # daily
                    "MAE_d": metrics["daily"]["MAE"],
                    "RMSE_d": metrics["daily"]["RMSE"],
                    "R2_d": metrics["daily"]["R2"],
                    "KGE_d": metrics["daily"]["KGE"],
                    "NSE_d": metrics["daily"]["NSE"],
                    # monthly
                    "MAE_m": metrics["monthly"]["MAE"],
                    "RMSE_m": metrics["monthly"]["RMSE"],
                    "R2_m": metrics["monthly"]["R2"],
                    "KGE_m": metrics["monthly"]["KGE"],
                    "NSE_m": metrics["monthly"]["NSE"],
                    # annual
                    "MAE_y": metrics["annual"]["MAE"],
                    "RMSE_y": metrics["annual"]["RMSE"],
                    "R2_y": metrics["annual"]["R2"],
                    "KGE_y": metrics["annual"]["KGE"],
                    "NSE_y": metrics["annual"]["NSE"],
                }
            )
        except Exception as e:
            sec = time.time() - t0
            rows.append(
                {
                    "station": sid,
                    "path": None,
                    "seconds": sec,
                    "coverage_pct": np.nan,
                    "MAE_d": np.nan,
                    "RMSE_d": np.nan,
                    "R2_d": np.nan,
                    "KGE_d": np.nan,
                    "NSE_d": np.nan,
                    "MAE_m": np.nan,
                    "RMSE_m": np.nan,
                    "R2_m": np.nan,
                    "KGE_m": np.nan,
                    "NSE_m": np.nan,
                    "MAE_y": np.nan,
                    "RMSE_y": np.nan,
                    "R2_y": np.nan,
                    "KGE_y": np.nan,
                    "NSE_y": np.nan,
                    "error": str(e),
                }
            )

    if combine_output_path is not None and combine_format.lower() == "parquet" and len(combine_parts) > 0:
        big = pd.concat(combine_parts, axis=0, ignore_index=True)
        big.to_parquet(
            combine_output_path,
            index=False,
            compression=combine_parquet_compression,
        )

    manifest = pd.DataFrame(rows)
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
    return manifest


# ---------------------------------------------------------------------
# Plot helper — OBS vs RF vs external (e.g., NASA)
# ---------------------------------------------------------------------


def plot_compare_obs_rf_nasa(
    data: pd.DataFrame,
    *,
    station_id: int,
    id_col: str = "station",
    date_col: str = "date",
    obs_col: Optional[str] = "tmax",
    nasa_col: Optional[str] = None,
    rf_df: Optional[pd.DataFrame] = None,
    rf_date_col: str = "date",
    rf_value_col: Optional[str] = "y_pred_full",
    rf_label: Optional[str] = "RF",
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = "D",
    agg: Optional[str] = "mean",
    smooth: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = (12, 5),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_loc: Optional[str] = "best",
    grid: Optional[bool] = True,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    date_fmt: Optional[str] = None,
    save_to: Optional[str] = None,
    obs_style: Optional[Dict] = None,
    nasa_style: Optional[Dict] = None,
    rf_style: Optional[Dict] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Dict[str, Dict[str, float]]]:
    """
    Compare observed series vs. RF predictions vs. an external product
    (e.g. NASA) for a single station.

    The function returns the matplotlib Figure/Axes objects plus a small
    metrics dictionary containing pairwise scores (obs vs RF, obs vs NASA).

    Metrics shown in the legend labels are MAE, RMSE, R² and KGE. The full
    metrics dictionary also includes NSE.

    Parameters
    ----------
    data:
        Input DataFrame containing at least the observation column.
    station_id:
        Station to plot.
    id_col, date_col:
        Column names for station ID and dates.
    obs_col:
        Name of the in-situ observation column (e.g. ``"tmax"``).
    nasa_col:
        Optional column with an external gridded product to compare against.
    rf_df:
        Optional DataFrame containing RF predictions for the same station.
    rf_date_col:
        Date column name in *rf_df*.
    rf_value_col:
        Value column name in *rf_df* (predictions).
    rf_label:
        Label used in the legend for the RF series.
    start, end:
        Optional clipping period for all series.
    resample:
        Optional resampling frequency (e.g. ``"D"``, ``"M"``). If
        ``None``, no resampling is applied.
    agg:
        Aggregation operation on resampling (``"mean"``, ``"sum"``,
        ``"median"``).
    smooth:
        Optional integer window for rolling mean smoothing.
    figsize:
        Figure size.
    title, ylabel:
        Plot title and y-axis label. If ``title`` is ``None``, a default
        one is constructed from ``station_id`` and ``obs_col``.
    legend_loc:
        Legend location for matplotlib.
    grid:
        Whether to draw a light grid.
    xlim, ylim:
        Optional axis limits.
    date_fmt:
        Optional date formatter string for the x-axis (e.g. ``"%Y"``).
    save_to:
        Optional path where the figure is saved (PNG/PNG, etc.).
    obs_style, nasa_style, rf_style:
        Optional dictionaries with matplotlib style overrides for each
        series.

    Returns
    -------
    fig, ax, metrics
        The matplotlib Figure and Axes, plus a dict with keys ``"rf"``
        and/or ``"nasa"`` (depending on availability), each containing
        the metrics dictionary with keys ``"MAE"``, ``"RMSE"``, ``"R2"``,
        ``"KGE"``, ``"NSE"``.
    """

    def _ensure_dt(s: pd.Series) -> pd.Series:
        if not np.issubdtype(s.dtype, np.datetime64):
            s = pd.to_datetime(s, errors="coerce")
        if isinstance(s.dtype, pd.DatetimeTZDtype):
            s = s.dt.tz_localize(None)
        return s

    def _clip_period(df: pd.DataFrame, cdate: str) -> pd.DataFrame:
        out = df.copy()
        out[cdate] = _ensure_dt(out[cdate])
        if start or end:
            lo = pd.to_datetime(start) if start else out[cdate].min()
            hi = pd.to_datetime(end) if end else out[cdate].max()
            out = out[(out[cdate] >= lo) & (out[cdate] <= hi)]
        return out

    def _prep_series(s: pd.Series) -> pd.Series:
        out = s.copy().sort_index()
        if resample is not None:
            op = (agg or "mean").lower()
            if op == "sum":
                out = out.resample(resample).sum()
            elif op == "median":
                out = out.resample(resample).median()
            else:
                out = out.resample(resample).mean()
        if smooth and isinstance(smooth, int) and smooth > 1:
            out = out.rolling(smooth, min_periods=1, center=True).mean()
        return out

    base = data.copy()
    if id_col in base.columns:
        base = base[base[id_col] == station_id]
    if base.empty:
        raise ValueError(f"No data for station {station_id}.")
    base = _clip_period(base, date_col)

    series: Dict[str, pd.Series] = {}
    labels: Dict[str, str] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    # Observed
    if obs_col is not None and obs_col in base.columns:
        obs = base[[date_col, obs_col]].dropna().rename(columns={obs_col: "obs"})
        obs[date_col] = _ensure_dt(obs[date_col])
        obs = obs.set_index(date_col).sort_index()
        if not obs.empty:
            series["obs"] = obs["obs"]
            labels["obs"] = "Observed"

    # External (e.g. NASA)
    if nasa_col is not None and nasa_col in base.columns:
        nasa = base[[date_col, nasa_col]].dropna().rename(columns={nasa_col: "nasa"})
        nasa[date_col] = _ensure_dt(nasa[date_col])
        nasa = nasa.set_index(date_col).sort_index()
        if not nasa.empty:
            series["nasa"] = nasa["nasa"]
            labels["nasa"] = f"{nasa_col}"

    # RF predictions
    if rf_df is not None and rf_value_col is not None and rf_value_col in rf_df.columns:
        rf = rf_df.copy()
        rf[rf_date_col] = _ensure_dt(rf[rf_date_col])
        rf = _clip_period(rf, rf_date_col)
        rf = rf[[rf_date_col, rf_value_col]].dropna()
        rf = rf.set_index(rf_date_col).sort_index()
        if not rf.empty:
            series["rf"] = rf[rf_value_col]
            labels["rf"] = rf_label or "RF"

    if len(series) == 0:
        raise ValueError("Nothing to plot (all series are None or empty).")

    # Pairwise metrics vs. obs
    if "obs" in series:
        from numpy import nan

        def _pair_metrics(a: pd.Series, b: pd.Series) -> Dict[str, float]:
            pair = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(pair) == 0:
                return {"MAE": nan, "RMSE": nan, "R2": nan, "KGE": nan, "NSE": nan}
            return regression_metrics(pair.iloc[:, 0].values, pair.iloc[:, 1].values)

        if "nasa" in series:
            metrics["nasa"] = _pair_metrics(series["obs"], series["nasa"])
        if "rf" in series:
            metrics["rf"] = _pair_metrics(series["obs"], series["rf"])

    # Resampling / smoothing
    for k in list(series.keys()):
        series[k] = _prep_series(series[k])

    # Default styles
    obs_style = dict(
        {"marker": "o", "ms": 3, "alpha": 0.7, "color": "#37474F", "ls": ""},
        **(obs_style or {}),
    )
    nasa_style = dict(
        {"lw": 2.0, "alpha": 0.9, "color": "#B71C1C"},
        **(nasa_style or {}),
    )
    rf_style = dict(
        {"lw": 2.0, "alpha": 0.9, "color": "#1E88E5"},
        **(rf_style or {}),
    )

    fig, ax = plt.subplots(figsize=figsize or (12, 5))

    if "obs" in series:
        ax.plot(
            series["obs"].index,
            series["obs"].values,
            label=labels["obs"],
            **obs_style,
        )
    if "nasa" in series:
        lbl = labels["nasa"]
        if "nasa" in metrics:
            m = metrics["nasa"]
            lbl = (
                f"{lbl} — MAE={m['MAE']:.2f} "
                f"RMSE={m['RMSE']:.2f} "
                f"R²={m['R2']:.2f} "
                f"KGE={m['KGE']:.2f}"
            )
        ax.plot(
            series["nasa"].index,
            series["nasa"].values,
            label=lbl,
            **nasa_style,
        )
    if "rf" in series:
        lbl = labels["rf"]
        if "rf" in metrics:
            m = metrics["rf"]
            lbl = (
                f"{lbl} — MAE={m['MAE']:.2f} "
                f"RMSE={m['RMSE']:.2f} "
                f"R²={m['R2']:.2f} "
                f"KGE={m['KGE']:.2f}"
            )
        ax.plot(
            series["rf"].index,
            series["rf"].values,
            label=lbl,
            **rf_style,
        )

    if title is None:
        ylab = ylabel if ylabel is not None else (obs_col.upper() if obs_col else "Value")
        title = f"Station {station_id} — {ylab}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if grid:
        ax.grid(ls=":", alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if date_fmt is not None:
        import matplotlib.dates as mdates

        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))

    ax.legend(frameon=False, loc=legend_loc or "best")

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")

    return fig, ax, metrics


# ---------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------

__all__ = [
    # core & full series
    "loso_train_predict_station",
    "loso_predict_full_series",
    "loso_predict_full_series_neighbors",
    "loso_predict_full_series_fast",
    # evaluation
    "evaluate_all_stations",
    "evaluate_all_stations_fast",
    # export
    "export_full_series_station",
    "export_full_series_batch",
    # helpers
    "select_stations",
    "build_station_kneighbors",
    "neighbor_correlation_table",
    "plot_compare_obs_rf_nasa",
    # utility (intended for users)
    "ensure_datetime",
    "add_time_features",
    "aggregate_and_score",
    "set_warning_policy",
]
