import numpy as np
import pandas as pd
import pytest

from ml.india_training import chronological_split, make_lead_frame


def sample_data(days=40, locations=2):
    rows = []
    times = pd.date_range("2026-01-01", periods=days * 24, freq="h", tz="UTC")
    for location in range(locations):
        for i, timestamp in enumerate(times):
            row = {
                "location_id": str(location), "country_code": "IN",
                "valid_time": timestamp, "latitude": 12.0 + location,
                "longitude": 76.0, "elevation_m": 800.0,
                "observed_temperature_2m": 25 + np.sin(i / 24),
                "observed_relative_humidity_2m": 60.0,
                "observed_precipitation": 0.0 if i % 20 else 2.0,
                "observed_wind_speed_10m": 3.0,
            }
            for lead in (1, 2, 3):
                for variable, value in {
                    "temperature_2m": 24.0, "relative_humidity_2m": 58.0,
                    "precipitation": 0.0, "wind_speed_10m": 2.5,
                }.items():
                    row[f"previous_run_{variable}_previous_day{lead}"] = value
            rows.append(row)
    return pd.DataFrame(rows)


def test_lead_frame_uses_forecast_inputs_and_observation_targets_separately():
    frame, features, raw_columns = make_lead_frame(sample_data(), 2)
    assert len(frame) > 0
    assert "previous_run_temperature_2m_previous_day2" in features
    assert "observed_temperature_2m" not in features
    assert "observed_precipitation" not in features
    assert "previous_run_precipitation_previous_day2" in raw_columns


def test_chronological_split_has_no_timestamp_overlap():
    frame, _, _ = make_lead_frame(sample_data(), 1)
    train, test, cutoff = chronological_split(frame)
    assert train.valid_time.max() < cutoff
    assert test.valid_time.min() == cutoff
    assert set(train.valid_time).isdisjoint(set(test.valid_time))


def test_missing_forecast_column_is_reported():
    data = sample_data().drop(columns=["previous_run_temperature_2m_previous_day3"])
    with pytest.raises(ValueError, match="Missing required"):
        make_lead_frame(data, 3)
