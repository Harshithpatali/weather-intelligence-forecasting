import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from ml.features import build_daily_frame,FEATURES,TARGETS
ART=Path("artifacts")
INPUTS=[f"{f}_lag{n}" for n in (1,2,3) for f in FEATURES]+["doy_sin","doy_cos"]

def train(path):
    d=build_daily_frame(pd.read_csv(path))
    if len(d)<100: raise ValueError(f"Only {len(d)} usable city-day rows; collect at least 100 before training.")
    cut=int(len(d)*.8); tr,te=d.iloc[:cut],d.iloc[cut:]
    models={}; metrics={"rows":len(d),"train_rows":len(tr),"test_rows":len(te),"validation":"chronological 80/20, no shuffle","horizons":{}}
    for h in (1,2,3):
        models[str(h)]={}; metrics["horizons"][str(h)]={}
        for target in TARGETS:
            y=f"target_{target}_d{h}"
            m=RandomForestRegressor(n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1)
            m.fit(tr[INPUTS],tr[y]); p=m.predict(te[INPUTS])
            models[str(h)][target]=m
            metrics["horizons"][str(h)][target]={"mae":float(mean_absolute_error(te[y],p)),
              "rmse":float(mean_squared_error(te[y],p)**.5),"r2":float(r2_score(te[y],p))}
    ART.mkdir(exist_ok=True)
    joblib.dump({"models":models,"features":INPUTS,"targets":TARGETS},ART/"three_day_models.joblib")
    (ART/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);a=p.parse_args()
    print(json.dumps(train(a.data),indent=2))
