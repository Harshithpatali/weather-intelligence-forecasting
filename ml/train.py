import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score
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
        rain_y=(tr[f"target_rain_1h_mm_d{h}"]>0).astype(int)
        rain_test=(te[f"target_rain_1h_mm_d{h}"]>0).astype(int)
        clf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1)
        clf.fit(tr[INPUTS],rain_y)
        prob=clf.predict_proba(te[INPUTS])[:,list(clf.classes_).index(1)] if 1 in clf.classes_ else [0.0]*len(te)
        pred=(prob>=0.5).astype(int)
        models[str(h)]["rain_probability"]=clf
        metrics["horizons"][str(h)]["rain_probability"]={
          "precision":float(precision_score(rain_test,pred,zero_division=0)),
          "recall":float(recall_score(rain_test,pred,zero_division=0)),
          "f1":float(f1_score(rain_test,pred,zero_division=0)),
          "roc_auc":float(roc_auc_score(rain_test,prob)) if len(set(rain_test))>1 else None}
    ART.mkdir(exist_ok=True)
    joblib.dump({"models":models,"features":INPUTS,"targets":TARGETS},ART/"three_day_models.joblib")
    (ART/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);a=p.parse_args()
    print(json.dumps(train(a.data),indent=2))
