"""Inference for the persisted India fixed-lead calibration bundle."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "india_weather_calibration.joblib"
METRICS_PATH = ARTIFACT_DIR / "india_model_metrics.json"
VARIABLES = ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m")
OUTPUT_NAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "precipitation": "rainfall_mm",
    "wind_speed_10m": "wind_speed_ms",
}


@lru_cache(maxsize=1)
def load_india_model(path: str | None = None):
    model_path = Path(path) if path else MODEL_PATH
    return joblib.load(model_path) if model_path.is_file() else None


def load_india_metrics():
    if not METRICS_PATH.is_file():
        return {"status": "india_model_not_installed"}
    import json
    return json.loads(METRICS_PATH.read_text())


def predict_india_forecasts(hourly_payload: dict, latitude: float, longitude: float,
                            elevation_m: float = 0.0, model=None,
                            now: datetime | None = None):
    """Apply 1/2/3-day models to matching lead-time hourly forecast values."""
    bundle = model if model is not None else load_india_model()
    if bundle is None:
        return None
    hourly = hourly_payload.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True, errors="coerce")
    if len(times) == 0:
        raise ValueError("Open-Meteo response contains no hourly forecast timestamps.")

    current = pd.Timestamp(now or datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")

    predictions = []
    for lead in (1, 2, 3):
        target_time = current + pd.Timedelta(days=lead)
        index = int(np.argmin(np.abs((times - target_time).total_seconds())))
        if pd.isna(times[index]) or abs((times[index] - target_time).total_seconds()) > 90 * 60:
            raise ValueError(f"No hourly forecast close to the {lead}-day target time.")
        features = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "elevation_m": float(elevation_m),
            "hour_sin": float(np.sin(2 * np.pi * times[index].hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * times[index].hour / 24.0)),
            "doy_sin": float(np.sin(2 * np.pi * times[index].dayofyear / 365.25)),
            "doy_cos": float(np.cos(2 * np.pi * times[index].dayofyear / 365.25)),
        }
        raw = {}
        for variable in VARIABLES:
            column = f"previous_run_{variable}_previous_day{lead}"
            values = hourly.get(variable)
            if values is None or index >= len(values) or values[index] is None:
                raise ValueError(f"Open-Meteo hourly forecast is missing {variable}.")
            raw[variable] = float(values[index])
            features[column] = raw[variable]

        model_features = bundle["leads"][str(lead)]["regressors"]
        ordered = [
            f"previous_run_{variable}_previous_day{lead}" for variable in VARIABLES
        ] + ["latitude", "longitude", "elevation_m", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        x = pd.DataFrame([[features[name] for name in ordered]], columns=ordered)
        calibrated = {}
        for variable in VARIABLES:
            value = float(model_features[variable].predict(x)[0])
            if variable == "relative_humidity_2m":
                value = float(np.clip(value, 0, 100))
            elif variable in ("precipitation", "wind_speed_10m"):
                value = max(0.0, value)
            calibrated[OUTPUT_NAMES[variable]] = value

        rain_model = bundle["leads"][str(lead)]["rain_probability"]
        classes = list(rain_model.classes_)
        probability = (
            float(rain_model.predict_proba(x)[0][classes.index(1)])
            if 1 in classes else 0.0
        )
        calibrated.update({
            "rainfall_probability": probability,
            "raw_forecast": {
                OUTPUT_NAMES[key]: raw[key] for key in VARIABLES
            },
            "forecast_valid_time_utc": times[index].isoformat(),
            "date": times[index].date().isoformat(),
            "horizon_days": lead,
        })
        predictions.append(calibrated)

    return {
        "horizon_days": 3,
        "model_version": bundle.get("version", "india-fixed-lead-unknown"),
        "model_mode": "India fixed-lead calibrated model; inference only",
        "predictions": predictions,
    }
