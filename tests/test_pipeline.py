import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from MEXsrfdcrPy.pipeline import (
    ensure_datetime,
    add_time_features,
    select_stations,
    build_station_kneighbors,
    neighbor_correlation_table,
    sample_target_for_training,
    loso_train_predict_station,
    loso_predict_full_series,
    evaluate_all_stations,
    evaluate_all_stations_fast,
    export_full_series_station,
    export_full_series_batch,
    plot_compare_obs_rf_nasa,
)


# ---------------------------------------------------------------------
# Synthetic toy dataset
# ---------------------------------------------------------------------


@pytest.fixture
def toy_data() -> pd.DataFrame:
    """
    Small synthetic dataset with 3 stations and 20 days.

    Columns:
        station | date | latitude | longitude | altitude | prec | tmax

    The values are simple deterministic patterns so that:
    - All stations have data every day.
    - Correlations between stations are non-trivial but positive.
    """
    rng = pd.date_range("2000-01-01", periods=20, freq="D")
    stations = [101, 202, 303]
    rows = []
    for st in stations:
        lat = 19.0 + (st % 10) * 0.1
        lon = -99.0 - (st % 10) * 0.1
        alt = 2200 + (st % 10) * 10
        for i, d in enumerate(rng):
            # Simple pattern: base + station offset + seasonal-like trend
            prec = 5.0 + (st % 10) * 0.5 + np.sin(i / 3.0)
            tmax = 20.0 + (st % 10) * 0.3 + np.cos(i / 5.0)
            rows.append(
                {
                    "station": st,
                    "date": d.strftime("%Y-%m-%d"),  # as string, to test ensure_datetime
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "prec": prec,
                    "tmax": tmax,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------


def test_ensure_datetime_and_add_time_features(toy_data: pd.DataFrame) -> None:
    df = ensure_datetime(toy_data, date_col="date")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert len(df) == len(toy_data)

    df2 = add_time_features(df, date_col="date", add_cyclic=True)
    for col in ["year", "month", "doy", "doy_sin", "doy_cos"]:
        assert col in df2.columns
    # Check that day-of-year is within expected range
    assert df2["doy"].between(1, 366).all()


def test_select_stations_filters(toy_data: pd.DataFrame) -> None:
    # All stations
    all_ids = select_stations(toy_data, id_col="station")
    assert sorted(all_ids) == [101, 202, 303]

    # Prefix filter ("10" should select station 101)
    pref = select_stations(toy_data, id_col="station", prefix="10")
    assert pref == [101]

    # Regex filter (only station 303 ends with "03" here)
    re_ids = select_stations(toy_data, id_col="station", regex=r".03$")
    assert re_ids == [303]


def test_build_station_kneighbors_returns_neighbors(toy_data: pd.DataFrame) -> None:
    neigh_map = build_station_kneighbors(
        toy_data,
        id_col="station",
        lat_col="latitude",
        lon_col="longitude",
        k=2,
    )
    # We have 3 stations, so each station should see up to 2 neighbors
    assert set(neigh_map.keys()) == {101, 202, 303}
    for st, neighs in neigh_map.items():
        assert 1 <= len(neighs) <= 2
        assert st not in neighs  # self should not be in the neighbor list


def test_neighbor_correlation_table_basic(toy_data: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "corr_table.parquet"
    table = neighbor_correlation_table(
        toy_data,
        station_id=101,
        neighbor_ids=[202, 303],
        id_col="station",
        date_col="date",
        value_col="prec",
        start="2000-01-01",
        end="2000-01-20",
        min_overlap=5,
        save_table_path=str(out_path),
    )
    # Expected structure
    assert list(table.columns) == ["neighbor", "corr", "n_overlap"]
    assert len(table) == 2
    assert table["n_overlap"].min() >= 5
    # File was written
    assert out_path.exists()


def test_sample_target_for_training_percentage(toy_data: pd.DataFrame) -> None:
    df = add_time_features(ensure_datetime(toy_data, "date"), "date")
    feature_cols = ["latitude", "longitude", "altitude", "year", "month", "doy"]

    sampled = sample_target_for_training(
        df,
        id_col="station",
        date_col="date",
        target_col="prec",
        feature_cols=feature_cols,
        station_id=101,
        include_target_pct=50.0,
        random_state=0,
    )
    # Station 101 has 20 rows, all complete -> 50% -> ceil(10) rows expected
    assert 1 <= len(sampled) <= 20
    assert len(sampled) == int(np.ceil(20 * 0.5))
    assert (sampled["station"] == 101).all()
    assert not sampled["prec"].isna().any()
    assert not sampled[feature_cols].isna().any().any()


# ---------------------------------------------------------------------
# LOSO core (single station)
# ---------------------------------------------------------------------


def test_loso_train_predict_station_runs_and_returns_metrics(toy_data: pd.DataFrame) -> None:
    rf_params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=1)

    out, metrics, model, feats = loso_train_predict_station(
        toy_data,
        station_id=101,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        rf_params=rf_params,
        agg_for_metrics="sum",
        start="2000-01-01",
        end="2000-01-20",
        add_cyclic=True,
    )

    # Basic structure
    assert not out.empty
    assert set(out.columns) == {"date", "station", "y_true", "y_pred"}
    assert out["station"].nunique() == 1
    assert out["station"].iloc[0] == 101

    # Feature list is non-empty
    assert isinstance(feats, list)
    assert len(feats) > 0

    # Metrics structure (pipeline _safe_metrics -> MAE, RMSE, R2 only)
    for scale in ["daily", "monthly", "annual"]:
        assert scale in metrics
        m = metrics[scale]
        for key in ["MAE", "RMSE", "R2"]:
            assert key in m
            assert np.isfinite(m[key]) or np.isnan(m[key])


def test_loso_predict_full_series_generates_complete_range(toy_data: pd.DataFrame) -> None:
    rf_params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=1)
    start = "2000-01-01"
    end = "2000-01-10"

    full_df, metrics, model, feats = loso_predict_full_series(
        toy_data,
        station_id=101,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start=start,
        end=end,
        rf_params=rf_params,
        add_cyclic=False,
    )

    # 10 days inclusive
    expected_len = len(pd.date_range(start, end, freq="D"))
    assert len(full_df) == expected_len
    assert full_df["station"].nunique() == 1
    assert "y_pred_full" in full_df.columns
    # All predictions must be finite
    assert np.isfinite(full_df["y_pred_full"]).all()

    # Metrics structure (again only MAE, RMSE, R2)
    for scale in ["daily", "monthly", "annual"]:
        assert scale in metrics
        m = metrics[scale]
        for key in ["MAE", "RMSE", "R2"]:
            assert key in m
            assert np.isfinite(m[key]) or np.isnan(m[key])


# ---------------------------------------------------------------------
# Evaluation across stations
# ---------------------------------------------------------------------


def test_evaluate_all_stations_returns_row_per_station(toy_data: pd.DataFrame) -> None:
    rf_params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=1)

    summary = evaluate_all_stations(
        toy_data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        rf_params=rf_params,
        agg_for_metrics="sum",
        start="2000-01-01",
        end="2000-01-20",
        add_cyclic=False,
    )

    # One row per station
    assert len(summary) == 3
    assert set(summary["station"]) == {101, 202, 303}

    # Check some metric columns exist (pipeline returns only MAE/RMSE/R2 per scale)
    for col in [
        "MAE_d",
        "RMSE_d",
        "R2_d",
        "MAE_m",
        "RMSE_m",
        "R2_m",
        "MAE_y",
        "RMSE_y",
        "R2_y",
    ]:
        assert col in summary.columns


def test_evaluate_all_stations_fast_matches_station_count(toy_data: pd.DataFrame, tmp_path: Path) -> None:
    rf_params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=1)
    log_csv = tmp_path / "log_fast.csv"

    summary_fast = evaluate_all_stations_fast(
        toy_data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-20",
        rf_params=rf_params,
        agg_for_metrics="sum",
        add_cyclic=False,
        show_progress=False,
        log_csv=str(log_csv),
        flush_every=1,
        min_station_rows=5,
    )

    # It should evaluate all 3 stations
    assert len(summary_fast) == 3
    assert set(summary_fast["station"]) == {101, 202, 303}
    # Log file should exist
    assert log_csv.exists()


# ---------------------------------------------------------------------
# Export full series
# ---------------------------------------------------------------------


def test_export_full_series_station_writes_file_and_returns_metrics(
    toy_data: pd.DataFrame, tmp_path: Path
) -> None:
    rf_params = dict(n_estimators=10, max_depth=3, random_state=0, n_jobs=1)
    out_dir = tmp_path / "series"
    metrics_path = tmp_path / "metrics_101.json"

    full_df, metrics, out_path = export_full_series_station(
        toy_data,
        station_id=101,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-15",
        train_start=None,
        train_end=None,
        rf_params=rf_params,
        add_cyclic=False,
        feature_cols=None,
        k_neighbors=None,
        neighbor_map=None,
        out_dir=str(out_dir),
        file_format="parquet",
        parquet_compression="snappy",
        csv_index=False,
        include_target_pct=0.0,
        include_target_seed=42,
        save_metrics_path=str(metrics_path),
    )

    # Check series
    assert not full_df.empty
    assert out_path is not None
    assert os.path.exists(out_path)

    # Metrics file
    assert metrics_path.exists()
    for scale in ["daily", "monthly", "annual"]:
        assert scale in metrics
        assert "MAE" in metrics[scale]
        assert "RMSE" in metrics[scale]
        assert "R2" in metrics[scale]


def test_export_full_series_batch_writes_manifest_and_files(
    toy_data: pd.DataFrame, tmp_path: Path
) -> None:
    rf_params = dict(n_estimators=5, max_depth=3, random_state=0, n_jobs=1)
    out_dir = tmp_path / "batch_series"
    manifest_path = tmp_path / "manifest.csv"
    combined_path = tmp_path / "combined.csv"

    manifest = export_full_series_batch(
        toy_data,
        station_ids=[101, 202],  # subset for speed
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-10",
        train_start=None,
        train_end=None,
        rf_params=rf_params,
        add_cyclic=False,
        feature_cols=None,
        k_neighbors=None,
        neighbor_map=None,
        out_dir=str(out_dir),
        file_format="csv",
        parquet_compression="snappy",
        csv_index=False,
        manifest_path=str(manifest_path),
        show_progress=False,
        combine_output_path=str(combined_path),
        combine_format="csv",
        combine_schema="input_like",
        include_target_pct=0.0,
        include_target_seed=42,
    )

    # Two stations in manifest
    assert len(manifest) == 2
    assert set(manifest["station"]) == {101, 202}

    # Each row should have a valid path (file written)
    for p in manifest["path"]:
        assert isinstance(p, str)
        assert os.path.exists(p)

    # Manifest file exists
    assert manifest_path.exists()
    # Combined file exists
    assert combined_path.exists()


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------


def test_plot_compare_obs_rf_nasa_returns_figure(toy_data: pd.DataFrame, tmp_path: Path) -> None:
    # Build a small "NASA-like" product by adding noise to tmax
    df = toy_data.copy()
    df["nasa_tmax"] = df["tmax"] + 0.5

    # Build a small RF-like DF: here just copy tmax to mimic an almost-perfect model
    rf_df = (
        df[df["station"] == 101][["date", "tmax"]]
        .rename(columns={"tmax": "y_pred_full"})
        .copy()
    )

    from matplotlib import pyplot as plt

    fig, ax, metrics = plot_compare_obs_rf_nasa(
        df,
        station_id=101,
        id_col="station",
        date_col="date",
        obs_col="tmax",
        nasa_col="nasa_tmax",
        rf_df=rf_df,
        rf_date_col="date",
        rf_value_col="y_pred_full",
        rf_label="RF",
        start="2000-01-01",
        end="2000-01-20",
        resample="D",
        agg="mean",
        smooth=None,
        figsize=(6, 4),
        title="Test plot",
        ylabel="Tmax (°C)",
        legend_loc="best",
        grid=True,
        save_to=str(tmp_path / "test_plot.png"),
    )

    assert isinstance(fig, plt.Figure)
    assert metrics  # non-empty
    # If RF metrics are available, they should contain MAE/RMSE/R2
    if "rf" in metrics:
        for key in ["MAE", "RMSE", "R2"]:
            assert key in metrics["rf"]
