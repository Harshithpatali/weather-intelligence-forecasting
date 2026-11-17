import numpy as np
import pandas as pd

WEATHER=["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"]
BASE_FEATURES=["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms","latitude","longitude","elevation_m","doy_sin","doy_cos"]

def prepare_daily_weather(frame):
    required={"location_id","country_code","date","latitude","longitude",*WEATHER}
    missing=required-set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    d=frame.copy()
    d["date"]=pd.to_datetime(d["date"],errors="coerce")
    d=d.dropna(subset=["date",*WEATHER,"latitude","longitude"])
    d=d.sort_values(["location_id","date"])
    d["doy_sin"]=np.sin(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["doy_cos"]=np.cos(2*np.pi*d["date"].dt.dayofyear/365.25)
    g=d.groupby("location_id",group_keys=False)
    for col in WEATHER:
        for lag in (0,1,2,3,4,5,6):
            d[f"{col}_lag{lag}"]=g[col].shift(lag)
        d[f"{col}_mean7"]=g[col].transform(lambda s:s.rolling(7,min_periods=7).mean())
    inputs=BASE_FEATURES+[f"{c}_lag{lag}" for c in WEATHER for lag in range(7)]+[f"{c}_mean7" for c in WEATHER]
    for target in WEATHER:
        for horizon in (1,2,3):
            d[f"target_{target}_d{horizon}"]=g[target].shift(-horizon)
    return d.dropna(subset=inputs+[f"target_{t}_d{h}" for t in WEATHER for h in (1,2,3)]).reset_index(drop=True),inputs
