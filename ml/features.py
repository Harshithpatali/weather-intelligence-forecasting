import pandas as pd

BASE_FEATURES = ["temperature_c", "feels_like_c", "humidity_pct", "pressure_hpa",
                 "wind_speed_ms", "cloud_pct", "rain_1h_mm"]

def build_training_frame(frame: pd.DataFrame, city: str | None = None):
    required = {"city", "observed_at", *BASE_FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["observed_at"] = pd.to_datetime(data["observed_at"], utc=True, errors="coerce")
    data = data.dropna(subset=["observed_at", *BASE_FEATURES])
    if city:
        data = data[data["city"].str.casefold() == city.casefold()]
    data = data.sort_values(["city", "observed_at"])
    grouped = data.groupby("city", group_keys=False)
    for col in BASE_FEATURES:
        data[f"{col}_lag1"] = grouped[col].shift(1)
        data[f"{col}_lag2"] = grouped[col].shift(2)
    data["target_temperature_c"] = grouped["temperature_c"].shift(-1)
    data["target_rain"] = (grouped["rain_1h_mm"].shift(-1).fillna(0) > 0).astype(int)
    data = data.dropna()
    features = [f"{col}_lag{lag}" for lag in (1, 2) for col in BASE_FEATURES]
    return data, features
