import pandas as pd
from ml.features import prepare_daily_weather

def test_global_feature_builder_creates_lags_and_three_day_targets():
    rows=[]
    for city_id in range(2):
        for day in pd.date_range("2025-01-01",periods=30,freq="D"):
            rows.append({"location_id":str(city_id),"country_code":"IN","date":day,
             "latitude":12.9,"longitude":77.6,"elevation_m":900,
             "temperature_c":25.,"humidity_pct":60.,"rainfall_mm":0.,"wind_speed_ms":3.})
    frame,features=prepare_daily_weather(pd.DataFrame(rows))
    assert len(frame)==42
    assert "temperature_c_lag6" in features
    assert "target_temperature_c_d3" in frame
    assert "target_rainfall_mm_d3" in frame
    assert "latitude" in features and "country_code" not in features
