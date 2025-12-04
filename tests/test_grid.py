import json

import numpy as np
import pandas as pd
import pytest

from MEXsrfdcrPy.grid import (
    GlobalRFMeta,
    train_global_rf_target,
    predict_points_daily_with_global_model,
    predict_grid_daily_with_global_model,
    predict_at_point_daily_with_global_model,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_toy_data() -> pd.DataFrame:
    """
    Create a small synthetic climate-like dataset suitable for testing
    the global RF grid utilities.

    Columns:
        station | date | latitude | longitude | altitude | prec
    """
    rng = np.random.default_rng(42)

    dates = pd.date_range("2000-01-01", periods=30, freq="D")
    stations = [1001, 1002, 1003]

    rows = []
    for st in stations:
        lat = 20.0 + (st - 1000) * 0.1
        lon = -100.0 - (st - 1000) * 0.1
        alt = 2000.0 + (st - 1000) * 5.0
        # simple synthetic signal with noise
        values = 10.0 + 0.1 * np.arange(len(dates)) + rng.normal(0, 1, size=len(dates))
        for d, v in zip(dates, values):
            rows.append(
                {
                    "station": st,
                    "date": d,
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt,
                    "prec": float(v),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def toy_training_artifacts(tmp_path):
    """
    Train a small global RF model and return the artifacts needed for
    prediction tests.

    Returns
    -------
    dict with keys:
        data, model_path, meta_path, meta_dict
    """
    data = _make_toy_data()

    model_path = tmp_path / "global_prec_rf.joblib"
    meta_path = tmp_path / "global_prec_rf.meta.json"
    summary_path = tmp_path / "global_prec_rf.summary.csv"

    model, meta_dict, summary = train_global_rf_target(
        data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-01",
        end="2000-01-30",
        min_rows_per_station=10,
        rf_params=dict(
            n_estimators=20,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        ),
        add_cyclic=False,
        feature_cols=None,
        model_path=str(model_path),
        meta_path=str(meta_path),
        save_summary_path=str(summary_path),
    )

    # basic sanity checks on training output
    assert model_path.exists()
    assert meta_path.exists()
    assert summary_path.exists()

    assert isinstance(summary, pd.DataFrame)
    assert not summary.empty
    # all stations should be present
    assert set(summary["station"].tolist()) == {1001, 1002, 1003}

    return {
        "data": data,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "meta_dict": meta_dict,
    }


# ---------------------------------------------------------------------
# Tests for training and metadata
# ---------------------------------------------------------------------


def test_train_global_rf_target_basic(tmp_path):
    data = _make_toy_data()
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"

    model, meta_dict, summary = train_global_rf_target(
        data,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="2000-01-05",
        end="2000-01-25",
        min_rows_per_station=5,
        rf_params=dict(
            n_estimators=10,
            max_depth=3,
            random_state=0,
            n_jobs=-1,
        ),
        model_path=str(model_path),
        meta_path=str(meta_path),
    )

    # Files were written
    assert model_path.exists()
    assert meta_path.exists()

    # Summary has one row per station
    assert isinstance(summary, pd.DataFrame)
    assert set(summary["station"].tolist()) == {1001, 1002, 1003}

    # Metadata is a dict compatible with GlobalRFMeta
    meta_obj = GlobalRFMeta.from_dict(meta_dict)
    assert meta_obj.target_col == "prec"
    assert meta_obj.id_col == "station"
    assert "latitude" in meta_obj.feature_cols
    assert "longitude" in meta_obj.feature_cols
    assert "altitude" in meta_obj.feature_cols
    assert meta_obj.n_stations == 3
    assert meta_obj.n_rows > 0
    assert meta_obj.train_start <= meta_obj.train_end

    # Daily metrics are present
    assert isinstance(meta_obj.metrics_daily, dict)
    for k in ["MAE", "RMSE", "R2", "KGE", "NSE"]:
        assert k in meta_obj.metrics_daily


def test_globalrfmeta_from_dict_backwards_compatibility():
    """
    Older metadata may contain 'start'/'end' instead of 'train_start'/'train_end'
    and extra keys. This test ensures that from_dict handles them correctly.
    """
    old_meta = {
        "target_col": "prec",
        "id_col": "station",
        "date_col": "date",
        "lat_col": "latitude",
        "lon_col": "longitude",
        "alt_col": "altitude",
        "feature_cols": ["latitude", "longitude", "altitude", "year", "month", "doy"],
        "n_stations": 10,
        "n_rows": 1000,
        "start": "1991-01-01",
        "end": "2020-12-31",
        "rf_params": {"n_estimators": 100},
        "metrics_daily": {"MAE": 1.0, "RMSE": 2.0, "R2": 0.8, "KGE": 0.7, "NSE": 0.75},
        "metrics_monthly": None,
        "metrics_annual": None,
        "version": "legacy-1.0",
        "some_unused_key": "should_be_ignored",
    }

    meta_obj = GlobalRFMeta.from_dict(old_meta)
    assert meta_obj.train_start == "1991-01-01"
    assert meta_obj.train_end == "2020-12-31"
    assert meta_obj.target_col == "prec"
    assert meta_obj.version == "legacy-1.0"

    # Ensure unknown keys do not end up in the dataclass dictionary
    as_d = meta_obj.to_dict()
    assert "some_unused_key" not in as_d


# ---------------------------------------------------------------------
# Tests for prediction on points / grid / single point
# ---------------------------------------------------------------------


def test_predict_points_daily_with_global_model_shape(toy_training_artifacts):
    model_path = toy_training_artifacts["model_path"]
    meta_path = toy_training_artifacts["meta_path"]

    # Two arbitrary points
    points = pd.DataFrame(
        {
            "grid_id": [1, 2],
            "latitude": [19.5, 20.0],
            "longitude": [-99.5, -100.0],
            "altitude": [2200.0, 2300.0],
        }
    )

    out = predict_points_daily_with_global_model(
        points=points,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-05",
        end="2000-01-10",
        grid_id_col="grid_id",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    # 6 days × 2 puntos
    assert isinstance(out, pd.DataFrame)
    assert out["grid_id"].nunique() == 2
    assert out["date"].nunique() == 6
    assert len(out) == 2 * 6

    # Columnas esperadas
    for col in ["date", "grid_id", "latitude", "longitude", "altitude", "y_pred_full"]:
        assert col in out.columns

    # Valores finitos
    assert np.isfinite(out["y_pred_full"]).all()


def test_predict_grid_daily_with_global_model_wrapper(toy_training_artifacts):
    model_path = toy_training_artifacts["model_path"]
    meta_path = toy_training_artifacts["meta_path"]

    grid = pd.DataFrame(
        {
            "grid_id": [10, 11, 12],
            "latitude": [19.0, 19.1, 19.2],
            "longitude": [-99.0, -99.1, -99.2],
            "altitude": [2100.0, 2150.0, 2200.0],
        }
    )

    out_grid = predict_grid_daily_with_global_model(
        grid,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-01",
        end="2000-01-03",
        grid_id_col="grid_id",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    # 3 días × 3 puntos
    assert out_grid["grid_id"].nunique() == 3
    assert out_grid["date"].nunique() == 3
    assert len(out_grid) == 3 * 3
    assert np.isfinite(out_grid["y_pred_full"]).all()


def test_predict_at_point_daily_with_global_model_single_id(toy_training_artifacts):
    model_path = toy_training_artifacts["model_path"]
    meta_path = toy_training_artifacts["meta_path"]

    out = predict_at_point_daily_with_global_model(
        latitude=19.3,
        longitude=-99.3,
        altitude=2250.0,
        station=9999,
        model_path=model_path,
        meta_path=meta_path,
        start="2000-01-02",
        end="2000-01-06",
        grid_id_col="grid_id",
        grid_lat_col="latitude",
        grid_lon_col="longitude",
        grid_alt_col="altitude",
        date_col="date",
    )

    # Debe haber un solo grid_id (el station que se pasó)
    assert out["grid_id"].nunique() == 1
    assert out["grid_id"].iloc[0] == 9999

    # 5 días
    assert out["date"].nunique() == 5
    assert len(out) == 5

    # Columnas y valores válidos
    for col in ["date", "grid_id", "latitude", "longitude", "altitude", "y_pred_full"]:
        assert col in out.columns
    assert np.isfinite(out["y_pred_full"]).all()


def test_predict_points_raises_on_missing_feature_columns(toy_training_artifacts, tmp_path):
    """
    For robustness, if the prediction table lacks some of the feature columns
    expected by the metadata, the function should raise a clear ValueError.
    """
    # Cargamos el meta y lo modificamos artificialmente para añadir una
    # feature inexistente, simulando un desalineamiento entre entrenamiento
    # y predicción.
    meta_dict = toy_training_artifacts["meta_dict"].copy()
    meta_dict = dict(meta_dict)
    meta_dict["feature_cols"] = list(meta_dict["feature_cols"]) + ["non_existing_feature"]

    # Guardamos este meta "roto" en un archivo temporal
    bad_meta_path = tmp_path / "bad.meta.json"
    with open(bad_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, ensure_ascii=False, indent=2)

    model_path = toy_training_artifacts["model_path"]

    points = pd.DataFrame(
        {
            "grid_id": [1],
            "latitude": [19.5],
            "longitude": [-99.5],
            "altitude": [2200.0],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        predict_points_daily_with_global_model(
            points=points,
            model_path=model_path,
            meta_path=str(bad_meta_path),
            start="2000-01-01",
            end="2000-01-02",
            grid_id_col="grid_id",
            grid_lat_col="latitude",
            grid_lon_col="longitude",
            grid_alt_col="altitude",
            date_col="date",
        )

    msg = str(excinfo.value)
    assert "Missing feature columns in prediction table" in msg
    assert "non_existing_feature" in msg
