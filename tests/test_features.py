import pandas as pd
from ml.features import build_daily_frame

def test_daily_frame_has_three_day_targets():
    rows=[]
    for city in ["Mysuru","Bengaluru"]:
        for day in pd.date_range("2024-01-01",periods=12,freq="D",tz="UTC"):
            rows.append({"city":city,"observed_at":day,"temperature_c":25.,"feels_like_c":26.,
            "humidity_pct":60.,"pressure_hpa":1010.,"wind_speed_ms":2.,"cloud_pct":20.,"rain_1h_mm":0.})
    result=build_daily_frame(pd.DataFrame(rows))
    assert len(result)==14
    assert "target_temperature_c_d3" in result
    assert "target_humidity_pct_d2" in result
    assert "target_wind_speed_ms_d1" in result
    assert "target_rain_1h_mm_d3" in result
