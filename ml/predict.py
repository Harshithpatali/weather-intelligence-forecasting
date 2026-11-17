import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sqlalchemy import select
from app.database import SessionLocal,WeatherObservation
from ml.features import FEATURES,TARGETS
ART=Path("artifacts")
def load_metrics():
    p=ART/"metrics.json"
    return json.loads(p.read_text()) if p.exists() else {"status":"not_trained"}

def predict_three_days(place,history_days=10):
    p=ART/"three_day_models.joblib"
    if not p.exists() or not SessionLocal:return None
    with SessionLocal() as s:
        rows=s.scalars(select(WeatherObservation).where(WeatherObservation.city.ilike(place))
          .order_by(WeatherObservation.observed_at.desc()).limit(300)).all()
    if len(rows)<3:return None
    d=pd.DataFrame([{k:getattr(r,k) for k in ["city","observed_at",*FEATURES]} for r in rows])
    d["observed_at"]=pd.to_datetime(d["observed_at"],utc=True)
    d=d.sort_values("observed_at").groupby("city").tail(history_days)
    daily=d.groupby("city",as_index=False)[FEATURES].mean()
    if len(daily)<1:return None
    # This initial inference uses the latest three observed records as lag inputs.
    latest=d.sort_values("observed_at").tail(3)
    if len(latest)<3:return None
    vals={}
    for lag,(_,row) in zip((3,2,1),latest.iterrows()):
        for f in FEATURES:vals[f"{f}_lag{lag}"]=row[f]
    now=pd.Timestamp.now(tz="UTC")
    vals["doy_sin"]=float(np.sin(2*np.pi*now.dayofyear/365.25));vals["doy_cos"]=float(np.cos(2*np.pi*now.dayofyear/365.25))
    b=joblib.load(p);x=pd.DataFrame([[vals[f] for f in b["features"]]],columns=b["features"])
    out=[]
    for h in (1,2,3):
        row={t:float(b["models"][str(h)][t].predict(x)[0]) for t in TARGETS}
        row["temperature_c"]=row["temperature_c"]
        row["humidity_pct"]=min(100,max(0,row["humidity_pct"]))
        row["wind_speed_ms"]=max(0,row["wind_speed_ms"])
        row["rainfall_estimate_mm"]=max(0,row.pop("rain_1h_mm"))
        row["date"]=(now+pd.Timedelta(days=h)).date().isoformat()
        out.append(row)
    return {"location":place,"history_records_used":len(latest),"horizon_days":3,"forecast":out,
      "model":"Random Forest multi-horizon regression","metrics":load_metrics(),
      "rainfall_probability":"Not estimated; rainfall amount is a different target."}
