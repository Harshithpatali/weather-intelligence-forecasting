import numpy as np
import pandas as pd

from ml.india_predict import predict_india_forecasts


class FakeRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        assert len(frame) == 1
        return np.array([self.value])


class FakeClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.array([[0.25, 0.75]])


def test_india_predictor_returns_three_fixed_leads():
    now = pd.Timestamp("2026-11-21T00:00:00Z")
    times = pd.date_range(now.floor("h"), periods=97, freq="h", tz="UTC")
    hourly = {"time": [value.isoformat() for value in times]}
    for variable, value in {
        "temperature_2m": 28.0,
        "relative_humidity_2m": 65.0,
        "precipitation": 1.2,
        "wind_speed_10m": 3.5,
    }.items():
        hourly[variable] = [value] * len(times)

    bundle = {"version": "test-india-v1", "leads": {}}
    for lead in (1, 2, 3):
        bundle["leads"][str(lead)] = {
            "regressors": {
                "temperature_2m": FakeRegressor(27.0 + lead),
                "relative_humidity_2m": FakeRegressor(70.0),
                "precipitation": FakeRegressor(2.0),
                "wind_speed_10m": FakeRegressor(4.0),
            },
            "rain_probability": FakeClassifier(),
        }

    result = predict_india_forecasts(
        {"hourly": hourly}, 12.9, 77.6, model=bundle, now=now.to_pydatetime()
    )
    assert [row["horizon_days"] for row in result["predictions"]] == [1, 2, 3]
    assert result["predictions"][0]["temperature_c"] == 28.0
    assert result["predictions"][2]["rainfall_probability"] == 0.75
    assert result["predictions"][0]["raw_forecast"]["temperature_c"] == 28.0
