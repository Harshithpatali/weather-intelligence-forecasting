import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
ART=Path("artifacts")
def load_model(path=None):
    p=Path(path) if path else ART/"global_weather_model.joblib"
    return joblib.load(p) if p.exists() else None
def load_metrics():
    p=ART/"metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"status":"model_not_trained"}
def predict_from_recent(recent,latitude,longitude,elevation_m=0.0,model=None):
    bundle=model or load_model()
    if bundle is None: return None
    d=pd.DataFrame(recent).copy()
    required=["date","temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"]
    if any(c not in d for c in required): raise ValueError("Recent weather is missing required variables.")
    d["date"]=pd.to_datetime(d["date"])
    d=d.sort_values("date").tail(7)
    if len(d)<7: raise ValueError("Seven daily observations are required for global model inference.")
    for c in ["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.dropna(subset=["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"])
    if len(d)<7: raise ValueError("Seven valid daily weather records are required.")
    vals={"temperature_c":float(d.iloc[-1].temperature_c),"humidity_pct":float(d.iloc[-1].humidity_pct),
      "rainfall_mm":float(d.iloc[-1].rainfall_mm),"wind_speed_ms":float(d.iloc[-1].wind_speed_ms),
      "latitude":float(latitude),"longitude":float(longitude),"elevation_m":float(elevation_m)}
    now=pd.Timestamp.now()
    vals["doy_sin"]=float(np.sin(2*np.pi*now.dayofyear/365.25));vals["doy_cos"]=float(np.cos(2*np.pi*now.dayofyear/365.25))
    for c in ["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"]:
        series=d[c].astype(float).tolist()
        for lag in range(7): vals[f"{c}_lag{lag}"]=series[-1-lag]
        vals[f"{c}_mean7"]=float(np.mean(series[-7:]))
    x=pd.DataFrame([[vals[k] for k in bundle["features"]]],columns=bundle["features"])
    result=[]
    for h in (1,2,3):
        models=bundle["models"][str(h)]
        row={t:float(models[t].predict(x)[0]) for t in ["temperature_c","humidity_pct","rainfall_mm","wind_speed_ms"]}
        row["humidity_pct"]=float(np.clip(row["humidity_pct"],0,100))
        row["wind_speed_ms"]=max(0,row["wind_speed_ms"]);row["rainfall_mm"]=max(0,row["rainfall_mm"])
        clf=models["rain_probability"]
        row["rainfall_probability"]=float(clf.predict_proba(x)[0][list(clf.classes_).index(1)]) if 1 in clf.classes_ else 0.0
        row["date"]=(now+pd.Timedelta(days=h)).date().isoformat()
        result.append(row)
    return {"horizon_days":3,"model_version":bundle.get("version","unknown"),"predictions":result,
      "input_observations":len(d),"model_mode":"pretrained global model; inference only"}
