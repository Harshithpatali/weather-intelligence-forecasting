import numpy as np
import pandas as pd
FEATURES=["temperature_c","feels_like_c","humidity_pct","pressure_hpa","wind_speed_ms","cloud_pct","rain_1h_mm"]
TARGETS=["temperature_c","rain_1h_mm","humidity_pct","wind_speed_ms"]

def build_daily_frame(frame):
    needed={"city","observed_at",*FEATURES}
    missing=needed-set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    d=frame.copy()
    d["observed_at"]=pd.to_datetime(d["observed_at"],utc=True,errors="coerce")
    d=d.dropna(subset=["observed_at",*FEATURES])
    d["day"]=d["observed_at"].dt.floor("D")
    d=d.groupby(["city","day"],as_index=False)[FEATURES].mean().sort_values(["city","day"])
    g=d.groupby("city",group_keys=False)
    for f in FEATURES:
        for lag in (0,1,2): d[f"{f}_lag{lag+1}"]=g[f].shift(lag)
    for t in TARGETS:
        for h in (1,2,3): d[f"target_{t}_d{h}"]=g[t].shift(-h)
    d["doy_sin"]=np.sin(2*np.pi*d["day"].dt.dayofyear/365.25)
    d["doy_cos"]=np.cos(2*np.pi*d["day"].dt.dayofyear/365.25)
    return d.dropna().reset_index(drop=True)
