"""Leakage-safe fixed-lead India forecast calibration and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    brier_score_loss, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score,
)

LEADS = (1, 2, 3)
TARGETS = {
    "temperature_2m": "observed_temperature_2m",
    "relative_humidity_2m": "observed_relative_humidity_2m",
    "precipitation": "observed_precipitation",
    "wind_speed_10m": "observed_wind_speed_10m",
}
FORECAST_VARIABLES = tuple(TARGETS)
RAIN_THRESHOLD_MM = 0.1


def make_lead_frame(data: pd.DataFrame, lead: int):
    """Create one lead-specific frame; observed values are targets only."""
    if lead not in LEADS:
        raise ValueError(f"Unsupported lead: {lead}")
    required = {
        "location_id", "valid_time", "latitude", "longitude", "elevation_m",
        *TARGETS.values(),
        *(f"previous_run_{v}_previous_day{lead}" for v in FORECAST_VARIABLES),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required India dataset columns: {missing}")

    frame = data.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["valid_time", "location_id", "latitude", "longitude"])
    frame["hour"] = frame["valid_time"].dt.hour
    frame["day_of_year"] = frame["valid_time"].dt.dayofyear
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24.0)
    frame["doy_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)

    forecast_columns = [f"previous_run_{v}_previous_day{lead}" for v in FORECAST_VARIABLES]
    features = forecast_columns + [
        "latitude", "longitude", "elevation_m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"
    ]
    for column in forecast_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for target in TARGETS.values():
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=features + list(TARGETS.values()))
    return frame, features, forecast_columns


def chronological_split(frame: pd.DataFrame, test_fraction: float = 0.2):
    """Split on valid_time, never randomly across rows or locations."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    dates = np.sort(frame["valid_time"].dropna().unique())
    if len(dates) < 10:
        raise ValueError("At least 10 distinct valid_time values are required")
    cutoff_index = min(max(int(len(dates) * (1 - test_fraction)), 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[cutoff_index])
    train = frame[frame["valid_time"] < cutoff].copy()
    test = frame[frame["valid_time"] >= cutoff].copy()
    if train.empty or test.empty or train.valid_time.max() >= test.valid_time.min():
        raise ValueError("Chronological split is invalid or overlaps")
    return train, test, cutoff


def _regression_metrics(actual, prediction):
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
    }


def train_india_dataset(data_path: str, output_dir: str = "artifacts",
                        minimum_locations: int = 30, minimum_days: int = 300):
    data = pd.read_parquet(data_path)
    if data.empty:
        raise ValueError("India training dataset is empty")
    if "country_code" not in data or not data.country_code.eq("IN").all():
        raise ValueError("Training dataset must contain only India rows")
    locations = int(data.location_id.nunique())
    days = int(pd.to_datetime(data.valid_time, utc=True).dt.floor("D").nunique())
    if locations < minimum_locations:
        raise ValueError(f"Expected at least {minimum_locations} locations; found {locations}")
    if days < minimum_days:
        raise ValueError(f"Expected at least {minimum_days} UTC days; found {days}")
    if data.duplicated(["location_id", "valid_time"]).any():
        raise ValueError("Duplicate location_id + valid_time rows detected")

    model_bundle = {"version": "india-fixed-lead-bias-correction-v1", "leads": {}}
    report = {
        "dataset": {"rows": int(len(data)), "locations": locations, "utc_days": days,
                    "start_utc": str(pd.to_datetime(data.valid_time, utc=True).min()),
                    "end_utc": str(pd.to_datetime(data.valid_time, utc=True).max())},
        "method": "per-lead HistGradientBoosting calibration; chronological 80/20 valid_time holdout; no row shuffle",
        "feature_policy": "Only fixed-lead Previous Runs forecasts, location metadata and valid-time calendar features. Historical Weather reanalysis columns are targets only.",
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "leads": {},
    }

    for lead in LEADS:
        frame, features, forecast_columns = make_lead_frame(data, lead)
        train, test, cutoff = chronological_split(frame)
        lead_models = {"regressors": {}, "rain_probability": None}
        lead_metrics = {
            "split_date_utc": cutoff.isoformat(),
            "rows": {"total": int(len(frame)), "train": int(len(train)), "test": int(len(test))},
            "feature_columns": features,
            "targets": {},
        }
        for variable, target_column in TARGETS.items():
            raw_column = f"previous_run_{variable}_previous_day{lead}"
            actual = test[target_column].to_numpy()
            raw = test[raw_column].to_numpy()
            reg = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=0.06, max_leaf_nodes=23,
                l2_regularization=2.0, random_state=42,
            )
            reg.fit(train[features], train[target_column])
            corrected = reg.predict(test[features])
            if variable in ("precipitation", "relative_humidity_2m", "wind_speed_10m"):
                corrected = np.clip(corrected, 0, None)
            if variable == "relative_humidity_2m":
                corrected = np.clip(corrected, 0, 100)
            if variable == "precipitation":
                raw = np.clip(raw, 0, None)
            lead_models["regressors"][variable] = reg
            lead_metrics["targets"][variable] = {
                "raw_forecast": _regression_metrics(actual, raw),
                "bias_corrected": _regression_metrics(actual, corrected),
            }

        rain_actual = (test["observed_precipitation"].to_numpy() > RAIN_THRESHOLD_MM).astype(int)
        rain_train = (train["observed_precipitation"] > RAIN_THRESHOLD_MM).astype(int)
        classifier = (
            HistGradientBoostingClassifier(
                max_iter=180, learning_rate=0.06, max_leaf_nodes=23,
                l2_regularization=2.0, random_state=42,
            )
            if rain_train.nunique() > 1
            else DummyClassifier(strategy="constant", constant=int(rain_train.iloc[0]))
        )
        classifier.fit(train[features], rain_train)
        if 1 in classifier.classes_:
            rain_probability = classifier.predict_proba(test[features])[:, list(classifier.classes_).index(1)]
        else:
            rain_probability = np.zeros(len(test), dtype=float)
        rain_prediction = (rain_probability >= 0.5).astype(int)
        raw_rain_probability = (test[f"previous_run_precipitation_previous_day{lead}"].to_numpy() > RAIN_THRESHOLD_MM).astype(float)
        lead_models["rain_probability"] = classifier
        lead_metrics["rain_probability"] = {
            "positive_rate": float(rain_actual.mean()),
            "raw_binary_forecast_brier": float(brier_score_loss(rain_actual, raw_rain_probability)),
            "calibrated_probability_brier": float(brier_score_loss(rain_actual, rain_probability)),
            "precision_at_0_5": float(precision_score(rain_actual, rain_prediction, zero_division=0)),
            "recall_at_0_5": float(recall_score(rain_actual, rain_prediction, zero_division=0)),
            "f1_at_0_5": float(f1_score(rain_actual, rain_prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(rain_actual, rain_probability)) if np.unique(rain_actual).size > 1 else None,
        }
        model_bundle["leads"][str(lead)] = lead_models
        report["leads"][str(lead)] = lead_metrics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, out / "india_weather_calibration.joblib", compress=3)
    (out / "india_model_metrics.json").write_text(json.dumps(report, indent=2))
    return report
