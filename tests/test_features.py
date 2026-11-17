import pandas as pd
from ml.features import build_training_frame

def test_feature_builder_creates_lags_and_targets():
    n = 8
    df = pd.DataFrame({
        "city": ["Mysuru"] * n,
        "observed_at": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
        "temperature_c": range(n), "feels_like_c": range(n),
        "humidity_pct": [60] * n, "pressure_hpa": [1010] * n,
        "wind_speed_ms": [2] * n, "cloud_pct": [20] * n,
        "rain_1h_mm": [0, 0, 1, 0, 0, 2, 0, 0],
    })
    out, features = build_training_frame(df, "Mysuru")
    assert len(features) == 14
    assert len(out) == 5
    assert "target_temperature_c" in out
    assert set(out["target_rain"].unique()).issubset({0, 1})
