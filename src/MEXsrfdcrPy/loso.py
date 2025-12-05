# SPDX-License-Identifier: MIT
"""
LOSO (Leave-One-Station-Out) Toolkit
====================================

This module provides a compact, production-ready implementation of LOSO
validation for daily climate records reconstruction with Random Forests. It
supports:

- Strict LOSO or partial inclusion of the target station in training via
  ``include_target_pct``.
- Station-wise training/prediction on observed days and full-series prediction.
- Fast evaluation across many stations with optional neighbor-based acceleration.
- Flexible feature space: coordinates + calendar + optional covariates.
- Robust metrics (MAE/RMSE/R²) with safe R² and resampling aliases.
- Station filtering by minimum valid rows (``min_station_rows``).
- Neighbor discovery (KDTree) and station–neighbor correlation tables.
- Batch export of full daily series (per-station files + optional combined
  output).
- Comparison plots (Observed vs RF vs external/NASA) helper.

Runtime dependencies
--------------------
- numpy
- pandas
- scikit-learn
- matplotlib

Optional
--------
- geopandas, folium (only for maps in downstream modules)
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import DatetimeTZDtype
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KDTree

try:
    # Optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x


# ---------------------------------------------------------------------
# Warning policy (silence harmless warnings by default)
# ---------------------------------------------------------------------


def set_warning_policy(silence: bool = True) -> None:
    """
    Configure warning filters for this module.

    Parameters
    ----------
    silence : bool, default True
        If True, silences:
        - Most Future/Deprecation warnings from pandas/scikit-learn.
        - The noisy ``is_sparse is deprecated`` DeprecationWarning emitted
          by ``sklearn.utils.validation``.
        - ``UndefinedMetricWarning`` (e.g., R² with zero variance).

    Notes
    -----
    The filters are intentionally narrow for the sklearn deprecation:
    they match by message content rather than module only, to survive
    changes in the internal module path, while avoiding silencing
    unrelated deprecations.
    """
    # Reset then re-apply filters so that notebook/global config does not
    # override our intent.
    warnings.resetwarnings()

    if not silence:
        return

    # Generic future deprecations from pandas / sklearn interplay.
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Specific sklearn / pandas bridge deprecation about is_sparse.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*is_sparse is deprecated and will be removed in a future version.*",
    )

    # Some environments phrase the message slightly differently; keep a
    # second, more generic pattern.
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r".*pd\.SparseDtype.*",
    )

    # Optionally, silence deprecations coming from pandas algorithms
    # (this covers dtype-related warnings that are not actionable here).
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"pandas\.core\.algorithms",
    )

    # Metrics warnings (e.g. R² undefined when y_true has zero variance).
    try:
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    except Exception:  # pragma: no cover
        pass


# Apply default warning policy on import.
set_warning_policy(True)


# ---------------------------------------------------------------------
# Small I/O helpers
# ---------------------------------------------------------------------


def _ensure_parent_dir(path: Optional[str]) -> None:
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
    Save a DataFrame to CSV/Parquet/Feather/Excel depending on the extension.

    Parameters
    ----------
    df : DataFrame
        DataFrame to save.
    path : str or None
        Output path. If None, nothing is saved and None is returned.
    parquet_compression : {"snappy", "gzip", ...}, default "snappy"
        Compression for Parquet output.

    Returns
    -------
    str or None
        The path used, or None if ``path`` is None.
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
    elif ext == ".xlsx":
        try:
            df.to_excel(path, index=False)
        except Exception as e:  # pragma: no cover - optional dependency
            raise ValueError(
                "Saving to .xlsx requires 'openpyxl' or 'xlsxwriter'."
            ) from e
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    return path


def _save_json(obj: dict, path: Optional[str]) -> Optional[str]:
    """
    Save a JSON-serializable object to disk.

    Parameters
    ----------
    obj : dict
        JSON-serializable object.
    path : str or None
        Output path. If None, nothing is saved and None is returned.

    Returns
    -------
    str or None
        The path used, or None if ``path`` is None.
    """
    if path is None:
        return None
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------


def _resolve_columns(df: pd.DataFrame, cols) -> List[str]:
    """
    Internal helper: resolve column names case-insensitively, with aliases.

    Parameters
    ----------
    df : DataFrame
    cols : str or iterable of str

    Returns
    -------
    list of str
        Only the column names that actually exist in ``df``.
    """
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [cols]

    lower_map = {c.lower(): c for c in df.columns}
    aliases = {
        "pretotcorr": "prectotcorr",
    }

    resolved: List[str] = []
    for raw in cols:
        if raw is None:
            continue
        key = str(raw).lower()
        key = aliases.get(key, key)
        if key in lower_map:
            resolved.append(lower_map[key])
    return resolved


def resolve_columns(df: pd.DataFrame, cols) -> List[str]:
    """
    Public variant of ``_resolve_columns``.

    Resolve column names case-insensitively and ignore missing names.

    Parameters
    ----------
    df : DataFrame
    cols : str or iterable of str

    Returns
    -------
    list of str
        The subset of names that exist in the DataFrame.

    Examples
    --------
    >>> resolve_columns(df, "PRECTOTCORR")
    >>> resolve_columns(df, ["tmin", "tmax"])
    """
    return _resolve_columns(df, cols)


# ---------------------------------------------------------------------
# Generic datetime / time-features utilities
# ---------------------------------------------------------------------


def ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Ensure ``date_col`` is timezone-naive ``datetime64[ns]`` and drop invalid
    dates.

    Parameters
    ----------
    df : DataFrame
    date_col : str, default "date"

    Returns
    -------
    DataFrame
        Copy of ``df`` with a cleaned datetime column and invalid rows removed.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if isinstance(out[date_col].dtype, DatetimeTZDtype):
        out[date_col] = out[date_col].dt.tz_localize(None)
    return out.dropna(subset=[date_col])


def add_time_features(
    df: pd.DataFrame,
    date_col: str = "date",
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Add calendar features (year, month, day-of-year) and optionally cyclic
    sines/cosines for day-of-year.

    Parameters
    ----------
    df : DataFrame
    date_col : str, default "date"
    add_cyclic : bool, default False
        If True, adds ``doy_sin`` and ``doy_cos`` columns.

    Returns
    -------
    DataFrame
        Copy of the input with added time features.
    """
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


# ---------------------------------------------------------------------
# Metrics and aggregation (safe R²; resampling aliases)
# ---------------------------------------------------------------------

_FREQ_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE"}


def _safe_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> Dict[str, float]:
    """
    Compute MAE, RMSE and R² in a robust way.

    - R² is set to NaN when the variance of ``y_true`` is zero or when
      there are fewer than 2 observations.

    Parameters
    ----------
    y_true, y_pred : array-like

    Returns
    -------
    dict
        Keys: ``{"MAE", "RMSE", "R2"}``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if y_true.size >= 2 and float(np.var(y_true)) > 0.0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


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
    Aggregate daily predictions to a lower frequency and compute metrics.

    Parameters
    ----------
    df_pred : DataFrame
        Must contain ``[date_col, y_col, yhat_col]``.
    date_col : str, default "date"
    y_col : str, default "y_true"
    yhat_col : str, default "y_pred"
    freq : str, default "M"
        Resampling code. Aliases are applied to avoid pandas deprecation
        warnings, e.g. "M" -> "ME", "A"/"Y" -> "YE", "Q" -> "QE".
    agg : {"sum", "mean", "median"}, default "sum"
        Aggregation function.

    Returns
    -------
    metrics : dict
        MAE, RMSE, R² at the aggregated frequency.
    agg_df : DataFrame
        Aggregated series with the same columns.
    """
    freq = _FREQ_ALIAS.get(freq, freq)
    aggfunc = {"sum": "sum", "mean": "mean", "median": "median"}[agg]

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: aggfunc, yhat_col: aggfunc}).dropna()
    if agg_df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, agg_df

    m = _safe_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
    return m, agg_df


# ---------------------------------------------------------------------
# Station selection and neighbors
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
    Select station ids using OR semantics across several filters.

    Parameters
    ----------
    data : DataFrame
    id_col : str, default "station"
    prefix : str or iterable of str, optional
        Match stations whose string representation starts with any of these
        prefixes.
    station_ids : iterable of int, optional
        Explicit station ids to include.
    regex : str, optional
        Regular expression applied to the string representation of station ids.
    custom_filter : callable, optional
        Function receiving an integer station id and returning True/False.

    Returns
    -------
    list of int
        Sorted list of selected station ids. If no filter is provided,
        all stations are returned.
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
    Return k nearest neighbors for each station using median lat/lon.

    Parameters
    ----------
    data : DataFrame
    id_col, lat_col, lon_col : str
        Column names for station id and coordinates.
    k : int, default 100
        Number of neighbors (excluding the station itself).

    Returns
    -------
    dict
        Mapping ``station_id -> list of neighbor station_ids``.
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
        neigh_idx = idx[i][1:]  # exclude itself
        neighbors[st] = centroids.iloc[neigh_idx][id_col].tolist()
    return neighbors


def neighbor_correlation_table(
    data: pd.DataFrame,
    station_id: int,
    *,
    neighbor_ids: Iterable[int],
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
    Compute daily Pearson correlations between a target station and its
    neighbors over overlapping days.

    Parameters
    ----------
    data : DataFrame
    station_id : int
    neighbor_ids : iterable of int
    id_col, date_col, value_col : str
    start, end : str or None
        Optional date limits for the analysis window.
    min_overlap : int, default 60
        Minimum number of daily overlaps required to report a correlation.
    save_table_path : str or None
        Optional path to save the resulting table.
    parquet_compression : str, default "snappy"

    Returns
    -------
    DataFrame
        Columns: ``["neighbor", "corr", "n_overlap"]``.
    """
    if not neighbor_ids:
        res = pd.DataFrame(columns=["neighbor", "corr", "n_overlap"])
        _save_df(res, save_table_path, parquet_compression=parquet_compression)
        return res

    df = data[[id_col, date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if isinstance(df[date_col].dtype, DatetimeTZDtype):
        df[date_col] = df[date_col].dt.tz_localize(None)

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    ids_needed = set([int(station_id)] + [int(x) for x in neighbor_ids])
    df = df[df[id_col].isin(ids_needed)]

    wide = df.pivot_table(
        index=date_col,
        columns=id_col,
        values=value_col,
        aggfunc="mean",
    )
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
    date_col: str,  # kept for API symmetry (not used here but may be useful)
    target_col: str,
    feature_cols: List[str],
    station_id: int,
    include_target_pct: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample a percentage of the target-station rows to be *leaked* into the
    training set.

    Only rows where all features and the target are valid (non-missing) are
    eligible for sampling.

    Parameters
    ----------
    df : DataFrame
    id_col : str
    date_col : str
    target_col : str
    feature_cols : list of str
    station_id : int
    include_target_pct : float
        Percentage of eligible rows to include (0–100).
    random_state : int, default 42

    Returns
    -------
    DataFrame
        Sampled rows. If ``include_target_pct <= 0`` or there are no valid
        rows, an empty DataFrame is returned.
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
# Core LOSO (observed days only)
# ---------------------------------------------------------------------


def loso_train_predict_station(
    data: pd.DataFrame,
    station_id: int,
    *,
    # columns
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    # features
    feature_cols: Optional[List[str]] = None,
    add_cyclic: bool = False,
    # model
    model=None,
    rf_params: Optional[Dict] = None,
    # metrics
    agg_for_metrics: str = "sum",
    # period
    start: Optional[str] = None,
    end: Optional[str] = None,
    # leakage
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # I/O
    save_predictions_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Classic LOSO for a single station on observed days only.

    Training data:
        All stations except ``station_id`` plus, optionally, a percentage of
        target-station rows (``include_target_pct``).

    Test data:
        Observed days of the target station within the requested period.

    Parameters
    ----------
    data : DataFrame
    station_id : int
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
    feature_cols : list of str or None
        If None, defaults to::
            [lat_col, lon_col, alt_col, "year", "month", "doy"] (+ cyclic terms).
    add_cyclic : bool, default False
    model : scikit-learn regressor or None, default None
        If None, a new RandomForestRegressor is created from ``rf_params``.
    rf_params : dict or None
    agg_for_metrics : {"sum", "mean", "median"}
        Aggregation for monthly and annual metrics.
    start, end : str or None
        Optional test window.
    include_target_pct : float, default 0.0
        Percentage (0–100) of target-station rows to inject into training.
    include_target_seed : int, default 42
    save_predictions_path, save_metrics_path : str or None
        Optional paths where predictions/metrics are saved.

    Returns
    -------
    out : DataFrame
        Columns: [date_col, "station", "y_true", "y_pred"].
    metrics : dict
        ``{"daily": {...}, "monthly": {...}, "annual": {...}}``.
    model : object
        Fitted model.
    feature_cols : list of str
        The feature columns actually used.
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

    with warnings.catch_warnings():
        # Extra safety: silence is_sparse warnings inside fit/predict even if
        # global filters were changed externally.
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*is_sparse is deprecated and will be removed in a future version.*",
        )
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
# Full series LOSO (y_pred_full + optional y_true)
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
    Train (LOSO or partial leakage) and predict a full daily series [start, end]
    for a single station.

    Metrics are computed only over the overlap with observed data.

    Parameters
    ----------
    data : DataFrame
    station_id : int
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
    start, end : str
        Full prediction horizon.
    rf_params : dict
        Parameters for RandomForestRegressor.
    add_cyclic : bool, default False
    feature_cols : list of str or None
    include_target_pct : float, default 0.0
    include_target_seed : int, default 42
    save_series_path, save_metrics_path : str or None

    Returns
    -------
    full_df : DataFrame
        Columns: [date, "station", "y_pred_full", "y_true?"].
    metrics : dict
    model : object
    feature_cols : list of str
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
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*is_sparse is deprecated and will be removed in a future version.*",
        )
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

    obs = df[df[id_col] == station_id][[date_col, target_col]].rename(
        columns={target_col: "y_true"}
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


# ---------------------------------------------------------------------
# FAST EVALUATION (many stations)
# ---------------------------------------------------------------------


def _append_rows_to_csv(
    rows: List[Dict],
    path: str,
    *,
    header_written_flag: Dict[str, bool],
) -> None:
    """
    Append rows to a CSV file, writing the header only once.

    Parameters
    ----------
    rows : list of dict
    path : str
    header_written_flag : dict
        Mutable state tracking whether the header has been written.
    """
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
    One-pass preprocessing for massive LOSO runs: datetime cleaning,
    clipping, time features, covariates resolution and feature list.

    Returns
    -------
    df : DataFrame
        Preprocessed table.
    feats : list of str
        Feature columns to be used for training.
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if isinstance(df[date_col].dtype, DatetimeTZDtype):
        df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["doy"] = df[date_col].dt.dayofyear
    if add_cyclic:
        df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)

    resolved_vars = resolve_columns(df, var_cols)

    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
        feats += resolved_vars
    else:
        feats = list(feature_cols)

    keep = sorted(
        set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats)
    )
    df = df[keep]
    return df, feats


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
    Evaluate LOSO performance for all stations (naive loop, no neighbor
    restriction).

    Parameters
    ----------
    data : DataFrame
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
    rf_params : dict
    agg_for_metrics : {"sum", "mean", "median"}
    start, end : str
    add_cyclic : bool
    feature_cols : list of str or None
    order_by : (str, bool), default ("RMSE_d", True)
        Column name and ascending flag for sorting.
    include_target_pct : float
    include_target_seed : int
    save_table_path : str or None

    Returns
    -------
    DataFrame
        Row per station with daily/monthly/annual metrics.
    """
    results: List[Dict] = []
    stations = data[id_col].dropna().unique()

    for st in stations:
        try:
            out_df, metrics, _, _ = loso_train_predict_station(
                data,
                station_id=int(st),
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
                    "station": int(st),
                    "n_rows": len(out_df),
                    "MAE_d": metrics["daily"]["MAE"],
                    "RMSE_d": metrics["daily"]["RMSE"],
                    "R2_d": metrics["daily"]["R2"],
                    "MAE_m": metrics["monthly"]["MAE"],
                    "RMSE_m": metrics["monthly"]["RMSE"],
                    "R2_m": metrics["monthly"]["R2"],
                    "MAE_y": metrics["annual"]["MAE"],
                    "RMSE_y": metrics["annual"]["RMSE"],
                    "R2_y": metrics["annual"]["R2"],
                }
            )
        except Exception:
            results.append(
                {
                    "station": int(st),
                    "n_rows": 0,
                    "MAE_d": np.nan,
                    "RMSE_d": np.nan,
                    "R2_d": np.nan,
                    "MAE_m": np.nan,
                    "RMSE_m": np.nan,
                    "R2_m": np.nan,
                    "MAE_y": np.nan,
                    "RMSE_y": np.nan,
                    "R2_y": np.nan,
                }
            )

    df_out = pd.DataFrame(results)
    if order_by and not df_out.empty:
        col, asc = order_by
        if col in df_out.columns:
            df_out = df_out.sort_values(col, ascending=asc).reset_index(drop=True)

    _save_df(df_out, save_table_path, parquet_compression=parquet_compression)
    return df_out


def evaluate_all_stations_fast(
    data: pd.DataFrame,
    *,
    # columns
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    # optional covariates (string or list)
    var_col: Optional[str] = None,
    var_cols: Optional[Iterable[str]] = None,
    target_col: str = "prec",
    # selection
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[int]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[int], bool]] = None,
    # period
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    # model
    rf_params: Dict = dict(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    agg_for_metrics: str = "sum",
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    # neighbors
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    # logging / I/O
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    show_progress: bool = True,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # station filter
    min_station_rows: Optional[int] = None,
    # output
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> pd.DataFrame:
    """
    Fast LOSO evaluation with single-pass preprocessing and optional
    neighbor restriction.

    Parameters
    ----------
    data : DataFrame
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
    var_col, var_cols : str or iterable of str, optional
        Additional covariate columns.
    prefix, station_ids, regex, custom_filter :
        See :func:`select_stations`.
    start, end : str
        Time window for both training and testing.
    rf_params : dict
    agg_for_metrics : {"sum", "mean", "median"}
    add_cyclic : bool
    feature_cols : list of str or None
    k_neighbors : int or None
    neighbor_map : dict or None
    log_csv : str or None
        If provided, appends station-level results as they are computed.
    flush_every : int
    show_progress : bool
    include_target_pct : float
    include_target_seed : int
    min_station_rows : int or None
        Minimum number of valid rows (features + target) required per station.
    save_table_path : str or None

    Returns
    -------
    DataFrame
        Station-level metrics table.
    """
    # Back-compat: var_col -> var_cols
    if var_cols is None and var_col is not None:
        var_cols = [var_col]

    t_all0 = time.time()

    # Single preprocessing pass
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

    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)
    stations = select_stations(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    if min_station_rows is not None:
        valid_counts = (
            df.loc[valid_mask_global, [id_col]].groupby(id_col).size().astype(int)
        )
        before_n = len(stations)
        stations = [
            s
            for s in stations
            if int(valid_counts.get(s, 0)) >= int(min_station_rows)
        ]
        if show_progress:
            tqdm.write(
                f"Filtered by min_station_rows={min_station_rows}: "
                f"{before_n} → {len(stations)} stations"
            )

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
    iterator = (
        tqdm(stations, desc="Evaluating stations", unit="st")
        if show_progress
        else stations
    )

    for sid in iterator:
        t0 = time.time()

        is_target = df[id_col] == sid
        st_block = df.loc[is_target, [lat_col, lon_col, alt_col]]
        lat_med = float(st_block[lat_col].median()) if not st_block.empty else np.nan
        lon_med = float(st_block[lon_col].median()) if not st_block.empty else np.nan
        alt_med = float(st_block[alt_col].median()) if not st_block.empty else np.nan

        if k_neighbors is not None:
            neigh = (neighbor_map or {}).get(sid, [])
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
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
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
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
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

        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model = RandomForestRegressor(**rf_params)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".*is_sparse is deprecated and will be removed in a future version.*",
            )
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
            "MAE_d": daily["MAE"],
            "RMSE_d": daily["RMSE"],
            "R2_d": daily["R2"],
            "MAE_m": monthly["MAE"],
            "RMSE_m": monthly["RMSE"],
            "R2_m": monthly["R2"],
            "MAE_y": annual["MAE"],
            "RMSE_y": annual["RMSE"],
            "R2_y": annual["R2"],
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
                f"(train={len(train_df):,}  test={len(test_df):,} "
                f"incl={pct:.1f}%)"
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
# Export of full-series (per station and batch)
# ---------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    Generate a full daily series for one station and save it to disk.

    Parameters
    ----------
    data : DataFrame
    station_id : int
    ...
    out_dir : str, default "/kaggle/working/series"
    file_format : {"parquet", "csv"}

    Returns
    -------
    full_df : DataFrame
    metrics : dict
    path : str
        Path where the series was saved.
    """
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if isinstance(df[date_col].dtype, DatetimeTZDtype):
        df[date_col] = df[date_col].dt.tz_localize(None)

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

    _ensure_dir(out_dir)
    base = f"loso_fullseries_{station_id}_{start.replace('-', '')}_{end.replace('-', '')}"
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
    combine_output_path: Optional[str] = None,
    combine_format: str = "csv",
    combine_parquet_compression: str = "snappy",
    combine_schema: str = "input_like",
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
) -> pd.DataFrame:
    """
    Export full daily series for many stations (per-station files + manifest),
    and optionally create a combined file.

    Parameters
    ----------
    data : DataFrame
    ...
    combine_schema : {"input_like", "compact"}
        - "input_like": (station, coords, date, target)
        - "compact": (station, date, y_pred_full, y_true, target)

    Returns
    -------
    manifest : DataFrame
        One row per station with timing and metric information.
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

    rows: List[Dict] = []
    it = (
        tqdm(stations, desc="Exporting full series", unit="st")
        if show_progress
        else stations
    )

    for sid in it:
        t0 = time.time()
        try:
            if k_neighbors is not None:
                neigh_ids = (neighbor_map or {}).get(sid, [])
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

            _ensure_dir(out_dir)
            base = f"loso_fullseries_{sid}_{start.replace('-', '')}_{end.replace('-', '')}"
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
                st_coords = df_local[df_local[id_col] == sid]
                lat0 = st_coords[lat_col].median() if lat_col in st_coords else np.nan
                lon0 = st_coords[lon_col].median() if lon_col in st_coords else np.nan
                alt0 = st_coords[alt_col].median() if alt_col in st_coords else np.nan

                if combine_schema == "input_like":
                    part = pd.DataFrame(
                        {
                            id_col: sid,
                            lat_col: lat0,
                            lon_col: lon0,
                            alt_col: alt0,
                            date_col: full_df[date_col].values,
                        }
                    )
                    filled = full_df["y_true"].where(
                        full_df["y_true"].notna(),
                        full_df["y_pred_full"],
                    )
                    part[target_col] = filled.values
                    part = part[
                        [id_col, lat_col, lon_col, alt_col, date_col, target_col]
                    ]
                elif combine_schema == "compact":
                    part = full_df[
                        [date_col, "station", "y_pred_full", "y_true"]
                    ].copy()
                    part[target_col] = part["y_true"].where(
                        part["y_true"].notna(),
                        part["y_pred_full"],
                    )
                    part = part[["station", "y_pred_full", "y_true", date_col, target_col]]
                else:  # pragma: no cover
                    raise ValueError(
                        "combine_schema must be 'input_like' or 'compact'."
                    )

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
                else:  # pragma: no cover
                    raise ValueError("combine_format must be 'csv' or 'parquet'.")

            sec = time.time() - t0
            cov = float(full_df["y_true"].notna().mean()) * 100.0
            rows.append(
                {
                    "station": sid,
                    "path": path,
                    "seconds": sec,
                    "coverage_pct": cov,
                    "MAE_d": metrics["daily"]["MAE"],
                    "RMSE_d": metrics["daily"]["RMSE"],
                    "R2_d": metrics["daily"]["R2"],
                    "MAE_m": metrics["monthly"]["MAE"],
                    "RMSE_m": metrics["monthly"]["RMSE"],
                    "R2_m": metrics["monthly"]["R2"],
                    "MAE_y": metrics["annual"]["MAE"],
                    "RMSE_y": metrics["annual"]["RMSE"],
                    "R2_y": metrics["annual"]["R2"],
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
                    "MAE_m": np.nan,
                    "RMSE_m": np.nan,
                    "R2_m": np.nan,
                    "MAE_y": np.nan,
                    "RMSE_y": np.nan,
                    "R2_y": np.nan,
                    "error": str(e),
                }
            )

    if (
        combine_output_path is not None
        and combine_format.lower() == "parquet"
        and len(combine_parts) > 0
    ):
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
# Fast full-series LOSO (neighbors + single-pass prep)
# ---------------------------------------------------------------------


def loso_predict_full_series_fast(
    data: pd.DataFrame,
    station_id: int,
    *,
    # columns
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    # periods
    start: str = "1961-01-01",  # full-series prediction window
    end: str = "2023-12-31",
    train_start: Optional[str] = None,  # optional training window
    train_end: Optional[str] = None,
    # model
    rf_params: Dict = dict(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    ),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    var_cols: Optional[Iterable[str]] = None,  # extra covariates
    # neighbors
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    # target leakage control
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # I/O
    save_series_path: Optional[str] = None,
    save_metrics_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], object, List[str]]:
    """
    Fast LOSO full-series prediction for one station.

    - Trains on all stations except the target (optionally only its k nearest
      neighbors).
    - Optionally includes a percentage of target rows in training
      (``include_target_pct``).
    - Predicts a continuous daily series [start, end] at the station's median
      coordinates.
    - Returns ``(full_df, metrics, fitted_model, feature_cols)``.

    Parameters
    ----------
    data : DataFrame
    station_id : int
    ...
    """
    # 1) Ensure datetime, and optionally clip a training window
    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if isinstance(df[date_col].dtype, DatetimeTZDtype):
        df[date_col] = df[date_col].dt.tz_localize(None)
    df = df.dropna(subset=[date_col])

    if train_start or train_end:
        lo = pd.to_datetime(train_start) if train_start else df[date_col].min()
        hi = pd.to_datetime(train_end) if train_end else df[date_col].max()
        train_base = df[(df[date_col] >= lo) & (df[date_col] <= hi)].copy()
    else:
        train_base = df.copy()

    # 2) Build neighbor map if requested
    if k_neighbors is not None and neighbor_map is None:
        neighbor_map = build_station_kneighbors(
            train_base,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=k_neighbors,
        )

    # 3) Add time features and resolve covariates in one pass
    train_base["year"] = train_base[date_col].dt.year
    train_base["month"] = train_base[date_col].dt.month
    train_base["doy"] = train_base[date_col].dt.dayofyear
    if add_cyclic:
        train_base["doy_sin"] = np.sin(2 * np.pi * train_base["doy"] / 365.25)
        train_base["doy_cos"] = np.cos(2 * np.pi * train_base["doy"] / 365.25)

    resolved_vars = _resolve_columns(train_base, var_cols)

    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
        feats += resolved_vars
    else:
        feats = list(feature_cols)

    # 4) Training pool: all except target (or only neighbors)
    is_target = train_base[id_col] == station_id
    if k_neighbors is not None:
        neigh_ids = (neighbor_map or {}).get(station_id, [])
        train_pool_mask = train_base[id_col].isin(neigh_ids) & (~is_target)
    else:
        train_pool_mask = ~is_target

    train_pool = train_base.loc[train_pool_mask]

    # 5) Optional inclusion of target rows in training (from train window)
    pct = max(0.0, min(float(include_target_pct), 100.0))
    if pct > 0.0:
        tgt_rows = train_base.loc[is_target]
        tgt_rows = tgt_rows.dropna(subset=feats + [target_col])
        if not tgt_rows.empty:
            n_take = int(np.ceil(len(tgt_rows) * (pct / 100.0)))
            sample_tgt = tgt_rows.sample(
                n=n_take,
                random_state=include_target_seed,
            )
            train_df = pd.concat([train_pool, sample_tgt], axis=0, copy=False)
        else:
            train_df = train_pool
    else:
        train_df = train_pool

    train_df = train_df.dropna(subset=feats + [target_col])
    if train_df.empty:
        raise ValueError(
            "Training set is empty after filtering (check features, neighbors, or periods)."
        )

    # 6) Station coordinates (median)
    st = df[df[id_col] == station_id]
    if st.empty:
        raise ValueError(f"No rows for station {station_id} in the input data.")
    lat0 = float(st[lat_col].median())
    lon0 = float(st[lon_col].median())
    alt0 = float(st[alt_col].median())

    # 7) Fit model
    model = RandomForestRegressor(**rf_params)
    X_train = train_df[feats].to_numpy(copy=False)
    y_train = train_df[target_col].to_numpy(copy=False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*is_sparse is deprecated and will be removed in a future version.*",
        )
        model.fit(X_train, y_train)

    # 8) Predict full continuous daily series on [start, end]
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
    synth["year"] = synth[date_col].dt.year
    synth["month"] = synth[date_col].dt.month
    synth["doy"] = synth[date_col].dt.dayofyear
    if add_cyclic:
        synth["doy_sin"] = np.sin(2 * np.pi * synth["doy"] / 365.25)
        synth["doy_cos"] = np.cos(2 * np.pi * synth["doy"] / 365.25)

    #y_pred_full = model.predict(synth[feats])
    y_pred_full = model.predict(synth[feats].to_numpy(copy=False))
    full_df = synth[[date_col, "station"]].copy()
    full_df["y_pred_full"] = y_pred_full

    # 9) Merge observed values (for scoring wherever available)
    obs = df[df[id_col] == station_id][[date_col, target_col]].rename(
        columns={target_col: "y_true"}
    )
    full_df = full_df.merge(obs, on=date_col, how="left")

    # 10) Metrics (only on overlapping observed days)
    comp = full_df.dropna(subset=["y_true"]).copy()
    if not comp.empty:
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
    else:
        daily = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
        monthly = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
        annual = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    metrics = {"daily": daily, "monthly": monthly, "annual": annual}

    # 11) Optional saves
    _save_df(full_df, save_series_path, parquet_compression=parquet_compression)
    _save_json(metrics, save_metrics_path)

    return full_df, metrics, model, feats


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------


def plot_compare_obs_rf_nasa(
    data: pd.DataFrame,
    *,
    station_id: int,
    # base columns in `data`
    id_col: str = "station",
    date_col: str = "date",
    # which columns to use (can be None to skip a series)
    obs_col: Optional[str] = "prec",
    nasa_col: Optional[str] = None,
    rf_df: Optional[pd.DataFrame] = None,
    rf_date_col: str = "date",
    rf_value_col: Optional[str] = "y_pred_full",
    rf_label: Optional[str] = "RF",
    # extra external source
    extra: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, list]] = None,
    extra_date_col: str = "date",
    extra_value_col: Optional[str] = None,
    extra_label: str = "External",
    # time window and transformations
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample: Optional[str] = "D",  # "D", "M", "YE", or None
    agg: Optional[str] = "mean",
    smooth: Optional[int] = None,
    # plot aesthetics / IO
    figsize: Optional[Tuple[int, int]] = (12, 5),
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_loc: Optional[str] = "best",
    grid: Optional[bool] = True,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    date_fmt: Optional[str] = None,
    save_to: Optional[str] = None,
    # style dictionaries
    obs_style: Optional[Dict] = None,
    nasa_style: Optional[Dict] = None,
    rf_style: Optional[Dict] = None,
    extra_style: Optional[Dict] = None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, Dict[str, Dict[str, float]]]:
    """
    Plot Observed vs RF vs NASA (plus an optional fourth external series) and
    report MAE/RMSE/R² against the observed values.

    All series are aligned in time; metrics are computed on the temporal
    intersection where observations exist.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    metrics : dict
        Nested mapping ``{"nasa": {...}, "rf": {...}, "extra": {...}}``.
    """

    # ------------------------ helpers ------------------------
    def _ensure_dt(s: pd.Series) -> pd.Series:
        s = pd.to_datetime(s, errors="coerce")
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_localize(None)
        return s

    def _clip(df_: pd.DataFrame, dcol: str) -> pd.DataFrame:
        out_ = df_.copy()
        out_[dcol] = _ensure_dt(out_[dcol])
        if start or end:
            lo_ = pd.to_datetime(start) if start else out_[dcol].min()
            hi_ = pd.to_datetime(end) if end else out_[dcol].max()
            out_ = out_[(out_[dcol] >= lo_) & (out_[dcol] <= hi_)]
        return out_

    def _prep(series: pd.Series) -> pd.Series:
        s_ = series.sort_index()
        if resample is not None:
            op = (agg or "mean").lower()
            if op == "sum":
                s_ = s_.resample(resample).sum()
            elif op == "median":
                s_ = s_.resample(resample).median()
            else:
                s_ = s_.resample(resample).mean()
        if smooth and isinstance(smooth, int) and smooth > 1:
            s_ = s_.rolling(smooth, min_periods=1, center=True).mean()
        return s_

    def _to_series(obj, dcol: str, vcol: Optional[str]) -> pd.Series:
        """Resolve a DataFrame/Series/array into a Series indexed by datetime."""
        if obj is None:
            return pd.Series(dtype=float)
        if isinstance(obj, pd.Series):
            s_ = obj.copy()
            if not isinstance(s_.index, pd.DatetimeIndex):
                raise ValueError("A plain Series must be datetime-indexed.")
            return s_.sort_index()
        if isinstance(obj, (list, np.ndarray)):
            return pd.Series(obj)
        if isinstance(obj, pd.DataFrame):
            if vcol is None or dcol not in obj.columns or vcol not in obj.columns:
                raise ValueError(
                    "For a DataFrame `extra`, provide valid extra_date_col and "
                    "extra_value_col."
                )
            tmp = obj[[dcol, vcol]].dropna().copy()
            tmp[dcol] = _ensure_dt(tmp[dcol])
            tmp = tmp.set_index(dcol).sort_index()[vcol]
            return tmp
        raise TypeError(
            "Unsupported type for `extra`. Use DataFrame/Series/ndarray/list."
        )

    # ------------------------ base selection ------------------------
    base = data.copy()
    if id_col in base.columns:
        base = base[base[id_col] == station_id]
    if base.empty:
        raise ValueError(f"No data for station {station_id}.")
    base = _clip(base, date_col)

    series: Dict[str, pd.Series] = {}
    labels: Dict[str, str] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    # Observed
    if obs_col is not None and obs_col in base.columns:
        obs = base[[date_col, obs_col]].dropna().rename(columns={obs_col: "obs"})
        obs[date_col] = _ensure_dt(obs[date_col])
        obs = obs.set_index(date_col).sort_index()["obs"]
        if not obs.empty:
            series["obs"] = obs
            labels["obs"] = "Observed"

    # NASA
    if nasa_col is not None and nasa_col in base.columns:
        nasa = base[[date_col, nasa_col]].dropna().rename(columns={nasa_col: "nasa"})
        nasa[date_col] = _ensure_dt(nasa[date_col])
        nasa = nasa.set_index(date_col).sort_index()["nasa"]
        if not nasa.empty:
            series["nasa"] = nasa
            labels["nasa"] = str(nasa_col)

    # RF
    if rf_df is not None and rf_value_col is not None and rf_value_col in rf_df.columns:
        rf = rf_df.copy()
        rf[rf_date_col] = _ensure_dt(rf[rf_date_col])
        rf = _clip(rf, rf_date_col)
        rf = (
            rf[[rf_date_col, rf_value_col]]
            .dropna()
            .set_index(rf_date_col)
            .sort_index()[rf_value_col]
        )
        if not rf.empty:
            series["rf"] = rf
            labels["rf"] = rf_label or "RF"

    # EXTRA
    extra_series = None
    if extra is not None:
        extra_series = _to_series(extra, extra_date_col, extra_value_col)
        if isinstance(extra_series.index, pd.DatetimeIndex):
            series["extra"] = extra_series.sort_index()
            labels["extra"] = extra_label

    def _pair_metrics(a: pd.Series, b: pd.Series) -> Dict[str, float]:
        pair = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(pair) < 2 or float(np.var(pair.iloc[:, 0])) == 0.0:
            return dict(MAE=np.nan, RMSE=np.nan, R2=np.nan)
        return {
            "MAE": mean_absolute_error(pair.iloc[:, 0], pair.iloc[:, 1]),
            "RMSE": mean_squared_error(
                pair.iloc[:, 0],
                pair.iloc[:, 1],
                squared=False,
            ),
            "R2": r2_score(pair.iloc[:, 0], pair.iloc[:, 1]),
        }

    # ------------------------ resample / smooth ------------------------
    for k in list(series.keys()):
        series[k] = _prep(series[k])

    # If extra was a plain vector, align with observed after resampling
    if extra is not None and not isinstance(extra_series.index, pd.DatetimeIndex):
        if "obs" not in series:
            raise ValueError(
                "Cannot align vector `extra` without an observed series."
            )
        obs_idx = series["obs"].index
        vec = pd.Series(extra_series, index=pd.RangeIndex(len(extra_series)))
        if len(vec) != len(series["obs"]):
            raise ValueError(
                f"Length mismatch: extra({len(vec)}) vs observed({len(series['obs'])})."
            )
        series["extra"] = pd.Series(
            np.asarray(vec.values, dtype=float),
            index=obs_idx,
        )
        labels["extra"] = extra_label

    # ------------------------ metrics ------------------------
    if "obs" in series:
        if "nasa" in series:
            metrics["nasa"] = _pair_metrics(series["obs"], series["nasa"])
        if "rf" in series:
            metrics["rf"] = _pair_metrics(series["obs"], series["rf"])
        if "extra" in series:
            metrics["extra"] = _pair_metrics(series["obs"], series["extra"])

    # ------------------------ plot ------------------------
    obs_style = {
        "marker": "o",
        "ms": 3,
        "alpha": 0.7,
        "color": "#37474F",
        "ls": "",
        **(obs_style or {}),
    }
    nasa_style = {
        "lw": 2.0,
        "alpha": 0.9,
        "color": "#B71C1C",
        **(nasa_style or {}),
    }
    rf_style = {
        "lw": 2.0,
        "alpha": 0.9,
        "color": "#1E88E5",
        **(rf_style or {}),
    }
    extra_style = {
        "lw": 2.0,
        "alpha": 0.9,
        "color": "#43A047",
        **(extra_style or {}),
    }

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
            lbl = f"{lbl} — MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R²={m['R2']:.2f}"
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
            lbl = f"{lbl} — MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R²={m['R2']:.2f}"
        ax.plot(
            series["rf"].index,
            series["rf"].values,
            label=lbl,
            **rf_style,
        )
    if "extra" in series:
        lbl = labels["extra"]
        if "extra" in metrics:
            m = metrics["extra"]
            lbl = f"{lbl} — MAE={m['MAE']:.2f} RMSE={m['RMSE']:.2f} R²={m['R2']:.2f}"
        ax.plot(
            series["extra"].index,
            series["extra"].values,
            label=lbl,
            **extra_style,
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


__all__ = [
    # core & full series
    "loso_train_predict_station",
    "loso_predict_full_series",
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
    # utility
    "ensure_datetime",
    "add_time_features",
    "aggregate_and_score",
    "set_warning_policy",
]
