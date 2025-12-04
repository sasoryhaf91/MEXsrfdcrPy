# SPDX-License-Identifier: MIT
"""
Global grid modelling utilities for MEXsrfdcrPy.

This module provides a high-level API to:

- Train a single **global Random Forest** using all stations
  (`train_global_rf_target`).
- Apply that model to:
    * all grid points of a domain
      (`predict_grid_daily_with_global_model`),
    * an arbitrary set of points
      (`predict_points_daily_with_global_model`),
    * or a single location
      (`predict_at_point_daily_with_global_model`).

The philosophy is the same que en el resto del paquete:

- Usar únicamente coordenadas espaciales (lat, lon, alt) +
  características de calendario (año, mes, día del año) como entrada.
- Entrenar un único modelo global con todas las estaciones que cumplen
  un umbral mínimo de datos.
- Luego, generalizar a cualquier punto / fecha del dominio, produciendo
  series completas diarias.

Compatibilidad hacia atrás
--------------------------

Versiones anteriores del paquete guardaban los metadatos del modelo
global con las claves ``"start"`` y ``"end"``. La clase
:class:`GlobalRFMeta` implementa un método de clase
:meth:`GlobalRFMeta.from_dict` que:

- Mapea ``start`` → ``train_start`` si no existe.
- Mapea ``end`` → ``train_end`` si no existe.
- Ignora silenciosamente claves desconocidas.

De esta forma, **los modelos viejos** (por ejemplo los que ya subiste a
Zenodo/Kaggle) siguen funcionando sin modificar nada del lado del
usuario.

Dependencias
------------

- numpy
- pandas
- scikit-learn
- joblib
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import json
import os

import pandas as pd
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.ensemble import RandomForestRegressor

from .metrics import regression_metrics
from .pipeline import ensure_datetime, add_time_features


# ---------------------------------------------------------------------
# Small I/O helpers (local to this module)
# ---------------------------------------------------------------------


def _ensure_parent_dir(path: Optional[str]) -> None:
    """Create parent directory if needed (no-op on None)."""
    if path is None:
        return
    d = os.path.dirname(str(path)) or "."
    os.makedirs(d, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: Optional[str]) -> Optional[str]:
    """Save a JSON-serialisable dict to *path* (pretty-printed)."""
    if path is None:
        return None
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a Python dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Metadata for global RF model
# ---------------------------------------------------------------------


@dataclass
class GlobalRFMeta:
    """
    Metadata for a global Random Forest trained with
    :func:`train_global_rf_target`.

    The class is intentionally simple and JSON-friendly. Use
    :meth:`from_dict` and :meth:`to_dict` to serialise/deserialise
    safely, including backwards compatibility with older metadata
    that used different key names (e.g. ``start`` / ``end``).
    """

    target_col: str
    id_col: str
    date_col: str
    lat_col: str
    lon_col: str
    alt_col: str
    feature_cols: List[str]
    n_stations: int
    n_rows: int
    train_start: str
    train_end: str
    rf_params: Dict[str, Any]
    metrics_daily: Optional[Dict[str, float]] = None
    metrics_monthly: Optional[Dict[str, float]] = None
    metrics_annual: Optional[Dict[str, float]] = None
    version: Optional[str] = None  # optional model version tag

    # ---- Backwards compatibility ------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalRFMeta":
        """
        Build a :class:`GlobalRFMeta` from a raw dictionary, handling
        older metadata layouts.

        Rules:
        - If ``train_start`` is missing but ``start`` is present,
          use ``start`` as ``train_start``.
        - If ``train_end`` is missing but ``end`` is present,
          use ``end`` as ``train_end``.
        - Unknown keys are silently dropped.
        """
        d = dict(d)  # shallow copy

        # Older versions used 'start' / 'end'
        if "train_start" not in d and "start" in d:
            d["train_start"] = d.pop("start")
        if "train_end" not in d and "end" in d:
            d["train_end"] = d.pop("end")

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in d.items() if k in allowed}
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------
# Training: single global RF model
# ---------------------------------------------------------------------


def train_global_rf_target(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "prec",
    start: Optional[str] = None,
    end: Optional[str] = None,
    min_rows_per_station: int = 365,
    rf_params: Optional[Dict[str, Any]] = None,
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    model_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    save_summary_path: Optional[str] = None,
) -> Tuple[RandomForestRegressor, Dict[str, Any], pd.DataFrame]:
    """
    Train a single global Random Forest model for a given target variable.

    Typical usage
    -------------

    .. code-block:: python

        from MEXsrfdcrPy.grid import train_global_rf_target

        model, meta, summary = train_global_rf_target(
            data,
            id_col="station",
            date_col="date",
            lat_col="latitude",
            lon_col="longitude",
            alt_col="altitude",
            target_col="tmin",
            start="1991-01-01",
            end="2020-12-31",
            min_rows_per_station=1825,
            rf_params=dict(
                n_estimators=200,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ),
            model_path="TemperatureMinModel-IMEX.joblib",
            meta_path="TemperatureMinModel-IMEX.meta.json",
        )

    Parameters
    ----------
    data:
        Long-format DataFrame containing all stations.
    id_col, date_col, lat_col, lon_col, alt_col, target_col:
        Column names in *data*.
    start, end:
        Optional period restriction for training. If any of them is
        ``None``, the corresponding bound is inferred from the data.
    min_rows_per_station:
        Minimum number of **non-missing** values required in ``target_col``
        for a station to be included in the training set.
    rf_params:
        Dictionary of parameters passed to
        :class:`sklearn.ensemble.RandomForestRegressor`. If ``None``,
        a conservative default is used.
    add_cyclic:
        Whether to include sinusoidal encodings ``doy_sin`` / ``doy_cos``
        of the day-of-year.
    feature_cols:
        Optional explicit feature list. If ``None``, the default
        ``[lat, lon, alt, year, month, doy]`` plus optional cyclic terms
        is used.
    model_path:
        Optional path where the fitted model is serialised via joblib.
    meta_path:
        Optional path where metadata are saved as a JSON file.
    save_summary_path:
        Optional CSV/Parquet/Feather path where the per-station training
        summary table is written.

    Returns
    -------
    model:
        Fitted :class:`RandomForestRegressor`.
    meta_dict:
        Metadata dictionary describing the training configuration and
        model context (this is exactly what is written to *meta_path*).
    summary:
        Per-station summary table with basic coverage information.
    """
    if rf_params is None:
        rf_params = dict(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

    # 1) Fecha → datetime + recorte de periodo
    df = ensure_datetime(data, date_col=date_col)
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # 2) Resumen por estación (cobertura del target)
    grp = df.groupby(id_col, dropna=True)
    coverage_rows = []
    for st, g in grp:
        n_total = int(len(g))
        n_valid = int(g[target_col].notna().sum())
        cov = float(n_valid) / n_total * 100.0 if n_total > 0 else 0.0
        coverage_rows.append(
            {
                "station": int(st),
                "n_total": n_total,
                "n_valid": n_valid,
                "coverage_pct": cov,
            }
        )
    summary = pd.DataFrame(coverage_rows).sort_values("station").reset_index(drop=True)

    # 3) Filtrar estaciones con umbral mínimo de datos válidos
    valid_stations = summary.loc[
        summary["n_valid"] >= int(min_rows_per_station), "station"
    ].tolist()
    if not valid_stations:
        raise ValueError(
            "No station satisfies min_rows_per_station="
            f"{min_rows_per_station}. Nothing to train."
        )

    df_train = df[df[id_col].isin(valid_stations)].copy()

    # 4) Añadir features temporales
    df_train = add_time_features(df_train, date_col=date_col, add_cyclic=add_cyclic)

    # 5) Definir lista de predictores
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # Asegurar que las columnas existen
    missing_feats = [c for c in feats if c not in df_train.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns in training data: {missing_feats}")

    # 6) Limpiar filas con NaN en features o target
    mask_valid = ~df_train[feats + [target_col]].isna().any(axis=1)
    df_train = df_train.loc[mask_valid]
    if df_train.empty:
        raise ValueError("Training set is empty after dropping NaNs in features/target.")

    # 7) Entrenar RF global
    X = df_train[feats].to_numpy(copy=False)
    y = df_train[target_col].to_numpy(copy=False)
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X, y)

    # 8) Métricas in-sample (solo informativas)
    y_hat = rf.predict(X)
    metrics_daily = regression_metrics(y, y_hat)
    metrics_monthly = None
    metrics_annual = None  # se podrían añadir, pero no son críticos aquí

    # 9) Construir metadatos
    train_start = str(df_train[date_col].min().date())
    train_end = str(df_train[date_col].max().date())

    meta_obj = GlobalRFMeta(
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        feature_cols=feats,
        n_stations=int(len(valid_stations)),
        n_rows=int(len(df_train)),
        train_start=train_start,
        train_end=train_end,
        rf_params=rf_params,
        metrics_daily=metrics_daily,
        metrics_monthly=metrics_monthly,
        metrics_annual=metrics_annual,
        version=None,
    )
    meta_dict = meta_obj.to_dict()

    # 10) Guardar modelo y metadatos si se pidió
    if model_path is not None:
        _ensure_parent_dir(model_path)
        joblib_dump(rf, model_path)
    _save_json(meta_dict, meta_path)

    # 11) Guardar summary si se pidió
    if save_summary_path is not None:
        ext = os.path.splitext(save_summary_path)[1].lower()
        _ensure_parent_dir(save_summary_path)
        if ext == ".csv":
            summary.to_csv(save_summary_path, index=False)
        elif ext == ".parquet":
            summary.to_parquet(save_summary_path, index=False, compression="snappy")
        elif ext == ".feather":
            summary.to_feather(save_summary_path)
        else:
            raise ValueError(
                "save_summary_path must have extension .csv, .parquet or .feather"
            )

    return rf, meta_dict, summary


# ---------------------------------------------------------------------
# Prediction on arbitrary points (grid or custom)
# ---------------------------------------------------------------------


def _prepare_points(
    points: pd.DataFrame,
    *,
    grid_id_col: str,
    grid_lat_col: str,
    grid_lon_col: str,
    grid_alt_col: str,
) -> pd.DataFrame:
    """
    Validate and normalise the *points* input for prediction.

    The returned DataFrame is a shallow copy with at least the required
    spatial columns and a unique identifier column.
    """
    if points is None or len(points) == 0:
        raise ValueError("`points` must be a non-empty DataFrame.")

    req = [grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]
    missing = [c for c in req if c not in points.columns]
    if missing:
        raise ValueError(f"Missing required point columns: {missing}")

    out = points.copy()
    out[grid_id_col] = out[grid_id_col].astype("int64")
    out[grid_lat_col] = out[grid_lat_col].astype(float)
    out[grid_lon_col] = out[grid_lon_col].astype(float)
    out[grid_alt_col] = out[grid_alt_col].astype(float)
    return out


def predict_points_daily_with_global_model(
    *,
    points: pd.DataFrame,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "grid_id",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Use a previously trained global RF model to reconstruct daily series
    for an arbitrary set of points.

    Parameters
    ----------
    points:
        DataFrame with at least the columns::

            grid_id_col | grid_lat_col | grid_lon_col | grid_alt_col

    model_path:
        Path to the joblib-serialised Random Forest model created by
        :func:`train_global_rf_target`.
    meta_path:
        Path to the JSON metadata file created alongside the model.
        Older metadata files that used ``"start"`` / ``"end"`` are
        supported transparently.
    start, end:
        Date range for the reconstructed series. If any of them is
        ``None``, the bound is taken from the training metadata.
    grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col:
        Column names in *points*.
    date_col:
        Name used for the date column in the output.

    Returns
    -------
    DataFrame
        Long-format table with columns::

            [date_col, grid_id_col, grid_lat_col, grid_lon_col,
             grid_alt_col, "y_pred_full"]
    """
    # 1) Cargar modelo y metadatos (compatibles hacia atrás)
    model: RandomForestRegressor = joblib_load(model_path)
    meta_dict = _load_json(meta_path)
    meta = GlobalRFMeta.from_dict(meta_dict)

    # 2) Rango de fechas
    start_eff = pd.to_datetime(start or meta.train_start)
    end_eff = pd.to_datetime(end or meta.train_end)
    if end_eff < start_eff:
        raise ValueError("`end` must be >= `start` (after resolving defaults).")

    dates = pd.date_range(start=start_eff, end=end_eff, freq="D")
    df_dates = pd.DataFrame({date_col: dates})

    # 3) Normalizar puntos
    pts = _prepare_points(
        points,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
    )

    # 4) Producto cartesiano fechas × puntos
    pts = pts.reset_index(drop=True)
    pts["_key"] = 1
    df_dates["_key"] = 1
    full = df_dates.merge(pts, on="_key").drop(columns="_key")

    # 5) Añadir features temporales
    full = add_time_features(full, date_col=date_col, add_cyclic=("doy_sin" in meta.feature_cols))

    # 6) Predicción
    missing_feats = [c for c in meta.feature_cols if c not in full.columns]
    if missing_feats:
        raise ValueError(
            f"Missing feature columns in prediction table: {missing_feats}. "
            "This usually indicates a mismatch between training and prediction inputs."
        )

    X_pred = full[meta.feature_cols].to_numpy(copy=False)
    y_pred = model.predict(X_pred)

    out = full[[date_col, grid_id_col, grid_lat_col, grid_lon_col, grid_alt_col]].copy()
    out["y_pred_full"] = y_pred.astype(float)

    return out.sort_values([grid_id_col, date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Prediction on a pre-defined grid
# ---------------------------------------------------------------------


def predict_grid_daily_with_global_model(
    grid: pd.DataFrame,
    *,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "grid_id",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Convenience wrapper around
    :func:`predict_points_daily_with_global_model` for a regular grid.

    Parameters
    ----------
    grid:
        DataFrame describing the spatial grid, with columns::

            grid_id_col | grid_lat_col | grid_lon_col | grid_alt_col

    model_path, meta_path, start, end, grid_*_col, date_col:
        See :func:`predict_points_daily_with_global_model`.

    Returns
    -------
    DataFrame
        Same format as :func:`predict_points_daily_with_global_model`.
    """
    return predict_points_daily_with_global_model(
        points=grid,
        model_path=model_path,
        meta_path=meta_path,
        start=start,
        end=end,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
        date_col=date_col,
    )


# ---------------------------------------------------------------------
# Prediction at a single point
# ---------------------------------------------------------------------


def predict_at_point_daily_with_global_model(
    *,
    latitude: float,
    longitude: float,
    altitude: float,
    station: Optional[int] = None,
    model_path: str,
    meta_path: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    grid_id_col: str = "grid_id",
    grid_lat_col: str = "latitude",
    grid_lon_col: str = "longitude",
    grid_alt_col: str = "altitude",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Predict the full daily series at a single location using a trained
    global RF model.

    This is a thin wrapper over
    :func:`predict_points_daily_with_global_model`.

    Parameters
    ----------
    latitude, longitude, altitude:
        Spatial coordinates of the point.
    station:
        Optional numeric identifier to be used as the grid ID. If
        ``None``, the ID is set to 0.
    model_path, meta_path, start, end, grid_*_col, date_col:
        See :func:`predict_points_daily_with_global_model`.

    Returns
    -------
    DataFrame
        Same format as :func:`predict_points_daily_with_global_model`,
        but restricted to a single grid ID.
    """
    gid = int(station) if station is not None else 0

    pts = pd.DataFrame(
        {
            grid_id_col: [gid],
            grid_lat_col: [float(latitude)],
            grid_lon_col: [float(longitude)],
            grid_alt_col: [float(altitude)],
        }
    )

    out = predict_points_daily_with_global_model(
        points=pts,
        model_path=model_path,
        meta_path=meta_path,
        start=start,
        end=end,
        grid_id_col=grid_id_col,
        grid_lat_col=grid_lat_col,
        grid_lon_col=grid_lon_col,
        grid_alt_col=grid_alt_col,
        date_col=date_col,
    )
    return out


# ---------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------

__all__ = [
    "GlobalRFMeta",
    "train_global_rf_target",
    "predict_points_daily_with_global_model",
    "predict_grid_daily_with_global_model",
    "predict_at_point_daily_with_global_model",
]
