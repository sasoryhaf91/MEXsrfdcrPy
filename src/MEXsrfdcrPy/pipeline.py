# SPDX-License-Identifier: MIT
"""
High-level pipeline utilities for LOSO-based climate reconstruction.

This module complements `loso.py` with end-to-end helpers:

- Dataset union with source tagging (SMN vs NASA).
- Global RandomForest training per variable (no LOSO) for grid inference.
- Daily predictions on a spatial grid (0.5°, 0.25°, 0.125°, 0.0625°, etc.).
- LOSO-FAST orchestration + metric maps (static & interactive) when available.
- Compact save/load helpers for CSV/Parquet.

The functions assume the *canonical column names* used across the codebase:
    station | date | latitude | longitude | altitude | prec | tmin | tmax | evap
You can pass alternative names via parameters if needed.

Dependencies:
- Core: numpy, pandas, scikit-learn, matplotlib
- Optional (maps): geopandas, folium
"""

from __future__ import annotations
import os
import json
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Local imports from the core LOSO module
from .loso import (
    ensure_datetime, add_time_features, resolve_columns, _save_df, _save_json,
    evaluate_all_stations_fast, build_station_kneighbors,
    aggregate_and_score, _safe_metrics
)

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_df(df: pd.DataFrame, path: Optional[str], *, parquet_compression: str = "snappy") -> Optional[str]:
    """Save a dataframe to CSV/Parquet/Feather/XLSX depending on file extension."""
    return _save_df(df, path, parquet_compression=parquet_compression)


def save_json(obj: dict, path: Optional[str]) -> Optional[str]:
    """Save a dict as JSON if path is provided."""
    return _save_json(obj, path)


# ---------------------------------------------------------------------
# Dataset union and normalization
# ---------------------------------------------------------------------
def union_with_source(
    df_smn: pd.DataFrame,
    df_nasa: pd.DataFrame,
    *,
    # incoming column names for each source
    id_col_smn: str = "station",
    date_col_smn: str = "date",
    lat_col_smn: str = "latitude",
    lon_col_smn: str = "longitude",
    alt_col_smn: str = "altitude",
    id_col_nasa: str = "station",
    date_col_nasa: str = "date",
    lat_col_nasa: str = "latitude",
    lon_col_nasa: str = "longitude",
    alt_col_nasa: str = "altitude",
    # variable mappings for NASA → canonical
    nasa_to_canonical: Optional[Dict[str, str]] = None,
    # canonical names to produce
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
) -> pd.DataFrame:
    """
    Return a unified long-format table containing both sources with a `source` tag.

    Notes
    -----
    - The function *does not* aggregate duplicates; it concatenates rows from both sources.
    - NASA variables are renamed using `nasa_to_canonical`, e.g.:
          {"T2M_MIN":"tmin", "T2M_MAX":"tmax", "PRECTOTCORR":"prec", "EVLAND":"evap"}
    - Missing canonical variables are left as-is if already present in the input dataframes.
    """
    nasa_to_canonical = nasa_to_canonical or {}

    # Normalize SMN columns
    smn = df_smn.copy()
    smn = smn.rename(columns={
        id_col_smn: id_col, date_col_smn: date_col,
        lat_col_smn: lat_col, lon_col_smn: lon_col, alt_col_smn: alt_col
    })
    smn["source"] = "smn"

    # Normalize NASA columns and variables
    nasa = df_nasa.copy()
    nasa = nasa.rename(columns={
        id_col_nasa: id_col, date_col_nasa: date_col,
        lat_col_nasa: lat_col, lon_col_nasa: lon_col, alt_col_nasa: alt_col,
        **nasa_to_canonical
    })
    nasa["source"] = "nasa"

    # Keep a consistent set of columns: metadata + all variables we find
    meta = [id_col, date_col, lat_col, lon_col, alt_col, "source"]
    vars_all = sorted(set([c for c in smn.columns if c not in meta] + [c for c in nasa.columns if c not in meta]))
    keep = meta + vars_all

    out = pd.concat([smn[keep].copy(), nasa[keep].copy()], axis=0, ignore_index=True)
    out = ensure_datetime(out, date_col)
    return out


# ---------------------------------------------------------------------
# Training (global RF per variable, no LOSO) and grid inference
# ---------------------------------------------------------------------
def fit_rf_global(
    data: pd.DataFrame,
    *,
    target_col: str,
    feature_cols: Optional[List[str]] = None,
    add_cyclic: bool = True,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    rf_params: Dict = dict(n_estimators=400, random_state=42, n_jobs=-1)
) -> Tuple[RandomForestRegressor, List[str], pd.DataFrame]:
    """
    Fit a *global* RandomForest on all available rows for `target_col` (no LOSO).

    Recommended when you need a fast model to infer values on a grid for many days.

    Returns
    -------
    model, feature_cols_used, training_dataframe
    """
    df = ensure_datetime(data, date_col)
    df = add_time_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feature_cols = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feature_cols += ["doy_sin", "doy_cos"]

    train_df = df.dropna(subset=feature_cols + [target_col])
    if train_df.empty:
        raise ValueError(f"No valid rows to train for target '{target_col}'.")

    X = train_df[feature_cols].to_numpy(copy=False)
    y = train_df[target_col].to_numpy(copy=False)

    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)
    return model, feature_cols, train_df


def predict_on_grid_daily(
    grid_df: pd.DataFrame,
    *,
    dates: Union[pd.DatetimeIndex, Iterable[pd.Timestamp], Tuple[str, str]],
    model: RandomForestRegressor,
    feature_cols: List[str],
    add_cyclic: bool = True,
    cell_id_col: str = "cell_id",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    out_value_col: str = "y_pred",
    chunk_size: int = 200_000
) -> pd.DataFrame:
    """
    Predict a *daily* time series on a spatial grid for the given `dates`.

    Parameters
    ----------
    grid_df : DataFrame
        Grid with at least [latitude, longitude, altitude] and optional cell_id.
    dates : sequence of timestamps *or* (start, end) tuple/strings.
    model : trained RandomForest (global) and its `feature_cols`.

    Notes
    -----
    - Generates a cartesian product of grid cells × days, **streamed** in chunks.
    - If `cell_id_col` does not exist, an integer id is assigned (0..N-1).
    """
    gd = grid_df.copy()
    if cell_id_col not in gd.columns:
        gd[cell_id_col] = np.arange(len(gd), dtype=int)

    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = pd.to_datetime(dates[0]), pd.to_datetime(dates[1])
        dt_index = pd.date_range(start=start, end=end, freq="D")
    else:
        dt_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))

    # Prepare static feature block from grid
    static_cols = [lat_col, lon_col, alt_col]
    missing = [c for c in static_cols if c not in gd.columns]
    if missing:
        raise ValueError(f"Grid is missing required columns: {missing}")

    # Create an iterator over chunks of days to keep memory bounded
    results = []
    for i in range(0, len(dt_index), max(1, chunk_size // max(1, len(gd)))):
        days = dt_index[i:i + max(1, chunk_size // max(1, len(gd)))]
        block = (
            gd[[cell_id_col] + static_cols]
            .assign(**{date_col: days.min()})
            .iloc[np.repeat(np.arange(len(gd)), len(days))]
            .reset_index(drop=True)
        )
        # expand dates across rows
        block[date_col] = np.tile(days.values, len(gd))
        block = add_time_features(block, date_col, add_cyclic=add_cyclic)

        X = block[feature_cols].to_numpy(copy=False)
        yhat = model.predict(X)
        block[out_value_col] = yhat
        results.append(block[[cell_id_col, date_col, out_value_col]])

    out = pd.concat(results, axis=0, ignore_index=True)
    return out


# ---------------------------------------------------------------------
# Orchestrating LOSO-FAST + maps
# ---------------------------------------------------------------------
def run_loso_fast_with_maps(
    data: pd.DataFrame,
    *,
    # core columns
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    # target and optional covariates
    target_col: str = "prec",
    var_cols: Optional[Iterable[str]] = None,
    # selection
    station_ids: Optional[Iterable[int]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter=None,
    # time
    start: str = "1961-01-01",
    end: str = "2023-12-31",
    # model + features
    rf_params: Dict = dict(n_estimators=200, random_state=42, n_jobs=-1),
    add_cyclic: bool = False,
    feature_cols: Optional[List[str]] = None,
    # neighbors
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    # leakage and station filter
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    # I/O
    out_dir: str = "/kaggle/working/loso",
    table_filename: str = "loso_metrics.parquet",
    make_static_map: bool = True,
    make_interactive_map: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Run LOSO-FAST over the dataset and (optionally) produce metric maps.

    Returns
    -------
    dict with keys: {"metrics_path", "static_map_path", "interactive_map_path"}
    """
    ensure_dir(out_dir)
    metrics_path = os.path.join(out_dir, table_filename)

    df_metrics = evaluate_all_stations_fast(
        data,
        id_col=id_col, date_col=date_col, lat_col=lat_col, lon_col=lon_col, alt_col=alt_col,
        var_cols=var_cols, target_col=target_col,
        prefix=prefix, station_ids=station_ids, regex=regex, custom_filter=custom_filter,
        start=start, end=end, rf_params=rf_params, agg_for_metrics="sum",
        add_cyclic=add_cyclic, feature_cols=feature_cols,
        k_neighbors=k_neighbors, neighbor_map=neighbor_map,
        show_progress=True, include_target_pct=include_target_pct, include_target_seed=include_target_seed,
        min_station_rows=min_station_rows,
        save_table_path=metrics_path
    )

    # Maps (optional dependencies)
    static_map_path = None
    interactive_map_path = None

    if make_static_map:
        try:
            from .maps import metric_bubble_map_static  # provided below, same file
            static_map_path = os.path.join(out_dir, "metric_map_static.png")
            metric_bubble_map_static(
                df_metrics, data,
                metric_col="R2_d",
                id_col_res="station", id_col_data=id_col,
                lat_col=lat_col, lon_col=lon_col,
                title="Daily R² (LOSO)",
                save_to=static_map_path
            )
        except Exception as e:
            print(f"[WARN] Static map generation skipped: {e}")

    if make_interactive_map:
        try:
            from .maps import metric_bubble_map_interactive  # provided below, same file
            interactive_map_path = os.path.join(out_dir, "metric_map_interactive.html")
            metric_bubble_map_interactive(
                df_metrics, data,
                metric_col="R2_d",
                id_col_res="station", id_col_data=id_col,
                lat_col=lat_col, lon_col=lon_col,
                save_to=interactive_map_path
            )
        except Exception as e:
            print(f"[WARN] Interactive map generation skipped: {e}")

    return {
        "metrics_path": metrics_path,
        "static_map_path": static_map_path,
        "interactive_map_path": interactive_map_path
    }


# ---------------------------------------------------------------------
# Lightweight maps (kept here to avoid hard dependency in `loso.py`)
# ---------------------------------------------------------------------
def _coords_unique(data: pd.DataFrame, id_col="station", lat_col="latitude", lon_col="longitude") -> pd.DataFrame:
    """Unique coordinates per station (median per id)."""
    return (
        data.dropna(subset=[lat_col, lon_col])
            .groupby(id_col)[[lat_col, lon_col]].median().reset_index()
            .rename(columns={id_col: "station"})
    )


def metric_bubble_map_static(
    df_result: pd.DataFrame,
    data: pd.DataFrame,
    *,
    metric_col: str = "R2_d",
    title: str = "Metric by Station",
    world_region: Optional[str] = "Mexico",
    use_diverging: Optional[bool] = None,
    cmap_div: str = "RdBu_r",
    cmap_seq: str = "Blues",
    size_by_n: bool = True,
    size_min: float = 30, size_max: float = 300,
    edgecolor: str = "white", edgewidth: float = 0.8,
    alpha: float = 0.9,
    figsize: Tuple[int, int] = (9, 9),
    save_to: Optional[str] = None,
    id_col_res: str = "station",
    id_col_data: str = "station",
    lat_col: str = "latitude", lon_col: str = "longitude",
):
    """
    Static bubble map (matplotlib + geopandas) for a metric column on station results.
    """
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import numpy as np

    if metric_col not in df_result.columns:
        raise ValueError(f"'{metric_col}' not found in df_result.")

    coords = _coords_unique(data, id_col=id_col_data, lat_col=lat_col, lon_col=lon_col)
    cols_to_merge = [id_col_res, metric_col] + (["n_rows"] if "n_rows" in df_result.columns else [])
    metrics = df_result[cols_to_merge].rename(columns={id_col_res: "station"})
    gdf = coords.merge(metrics, on="station", how="left").dropna(subset=[metric_col])
    if gdf.empty:
        raise ValueError("No stations with the requested metric.")

    gdf = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(gdf[lon_col], gdf[lat_col]), crs="EPSG:4326")
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    base = world if world_region is None else world[world.name == world_region]
    if base.empty: base = world

    v = gdf[metric_col].astype(float)
    if use_diverging is None:
        use_diverging = ("R2" in metric_col)

    if use_diverging:
        p5, p95 = np.nanpercentile(v, [5, 95])
        vmax_abs = max(abs(p5), abs(p95), 1e-9)
        vmin, vmax = -vmax_abs, vmax_abs
        cmap = cmap_div
    else:
        p5, p95 = np.nanpercentile(v, [5, 95])
        vmin = max(np.nanmin(v), p5); vmax = max(vmin + 1e-9, p95)
        cmap = cmap_seq

    if size_by_n and "n_rows" in gdf.columns:
        n = gdf["n_rows"].astype(float); n_min, n_max = n.min(), n.max()
        sizes = np.full(len(gdf), (size_min + size_max) / 2) if n_min == n_max else size_min + (n - n_min) * (size_max - size_min) / (n_max - n_min)
    else:
        sizes = np.full(len(gdf), (size_min + size_max) / 2)

    fig, ax = plt.subplots(figsize=figsize)
    base.plot(ax=ax, color="#F5F6F7", edgecolor="#D0D3D4", linewidth=0.8, zorder=0)
    gdf.plot(
        ax=ax, column=metric_col, cmap=cmap, markersize=sizes, alpha=alpha,
        edgecolor=edgecolor, linewidth=edgewidth, vmin=vmin, vmax=vmax,
        legend=True, legend_kwds={"label": metric_col, "shrink": 0.65}, zorder=2
    )
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    xmin, ymin, xmax, ymax = gdf.total_bounds
    pad_x, pad_y = (xmax - xmin) * 0.07, (ymax - ymin) * 0.07
    ax.set_xlim(xmin - pad_x, xmax + pad_x); ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect(1.0); ax.grid(ls=":", lw=0.6, color="#9E9E9E", alpha=0.35)

    if save_to:
        fig.savefig(save_to, dpi=300, bbox_inches="tight")
    return ax, gdf


def metric_bubble_map_interactive(
    df_result: pd.DataFrame,
    data: pd.DataFrame,
    *,
    metric_col: str = "R2_d",
    save_to: str = "/kaggle/working/metric_map.html",
    tiles: str = "CartoDB positron",
    scale_radius_by_n: bool = True,
    min_radius: float = 4, max_radius: float = 10,
    id_col_res: str = "station", id_col_data: str = "station",
    lat_col: str = "latitude", lon_col: str = "longitude",
):
    """
    Interactive bubble map (folium) for a metric column on station results.
    """
    import folium
    import branca.colormap as cm
    import matplotlib
    import numpy as np

    if metric_col not in df_result.columns:
        raise ValueError(f"'{metric_col}' not found in df_result.")

    coords = _coords_unique(data, id_col=id_col_data, lat_col=lat_col, lon_col=lon_col)
    cols_to_merge = [id_col_res, metric_col] + (["n_rows"] if "n_rows" in df_result.columns else [])
    metrics = df_result[cols_to_merge].rename(columns={id_col_res: "station"})
    dfm = coords.merge(metrics, on="station", how="left").dropna(subset=[metric_col])
    if dfm.empty:
        raise ValueError("No stations with the requested metric.")

    center = [dfm[lat_col].median(), dfm[lon_col].median()]
    m = folium.Map(location=center, zoom_start=6, tiles=tiles)

    v = dfm[metric_col].astype(float)
    if "R2" in metric_col:
        p5, p95 = v.quantile([0.05, 0.95]).tolist()
        vmax_abs = max(abs(p5), abs(p95), 1e-9)
        vmin, vmax = -vmax_abs, vmax_abs
        try:
            colormap = cm.linear.RdBu_11.scale(vmin, vmax)
        except AttributeError:
            rd = matplotlib.cm.get_cmap('RdBu_r')
            colors = [matplotlib.colors.rgb2hex(rd(x)) for x in np.linspace(0, 1, 256)]
            colormap = cm.LinearColormap(colors, vmin=vmin, vmax=vmax)
    else:
        p5, p95 = v.quantile([0.05, 0.95]).tolist()
        vmin = max(v.min(), p5); vmax = max(vmin + 1e-9, p95)
        try:
            colormap = cm.linear.Blues_09.scale(vmin, vmax)
        except Exception:
            blues = matplotlib.cm.get_cmap('Blues')
            colors = [matplotlib.colors.rgb2hex(blues(x)) for x in np.linspace(0, 1, 256)]
            colormap = cm.LinearColormap(colors, vmin=vmin, vmax=vmax)

    colormap.caption = metric_col
    colormap.add_to(m)

    if scale_radius_by_n and "n_rows" in dfm.columns:
        n = dfm["n_rows"].astype(float); n_min, n_max = n.min(), n.max()
        radii = np.full(len(dfm), (min_radius + max_radius) / 2) if n_min == n_max else min_radius + (n - n_min) * (max_radius - min_radius) / (n_max - n_min)
    else:
        radii = np.full(len(dfm), (min_radius + max_radius) / 2)

    for _, r in dfm.iterrows():
        val = float(r[metric_col])
        popup_html = f"<b>Station:</b> {int(r['station'])}<br><b>{metric_col}:</b> {val:.3f}"
        if "n_rows" in dfm.columns:
            popup_html += f"<br><b>n_rows:</b> {int(r['n_rows'])}"
        folium.CircleMarker(
            location=[r[lat_col], r[lon_col]],
            radius=float(radii[0] if np.isscalar(radii) else r["station"] * 0 + radii[dfm.index.get_loc(r.name)]),
            color=colormap(val), fill=True, fill_opacity=0.9, weight=0.8,
            popup=folium.Popup(html=popup_html, max_width=260)
        ).add_to(m)

    if save_to:
        m.save(save_to)
    return m, dfm

