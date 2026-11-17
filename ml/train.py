import argparse,json
from pathlib import Path
import joblib,pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score,roc_auc_score,brier_score_loss
from ml.features import prepare_daily_weather,WEATHER

ART=Path("artifacts")
def train(data_path,output_dir="artifacts"):
    d=pd.read_csv(data_path)
    frame,features=prepare_daily_weather(d)
    if frame["country_code"].nunique()<40: raise ValueError(f"Need at least 40 countries; found {frame['country_code'].nunique()}.")
    if frame["location_id"].nunique()<1200: raise ValueError(f"Need at least 1,200 locations (30 per country x 40); found {frame['location_id'].nunique()}.")
    per_country=frame.groupby("country_code")["location_id"].nunique()
    if len(per_country)<40 or (per_country<30).any():
        raise ValueError("Every one of the 40 countries must contribute at least 30 unique locations.")
    if (frame["date"].max()-frame["date"].min()).days<330:
        raise ValueError("Training data must cover approximately one full year (at least 330 days).")
    dates=sorted(frame["date"].unique())
    if len(dates)<250: raise ValueError("Need at least 250 distinct dates after feature engineering.")
    cutoff=dates[int(len(dates)*.8)]
    train_df=frame[frame.date<cutoff]; test_df=frame[frame.date>=cutoff]
    if len(train_df)<10000 or len(test_df)<1000: raise ValueError("Insufficient global train/test rows.")
    models={};metrics={"dataset_rows":len(frame),"locations":int(frame.location_id.nunique()),
      "countries":int(frame.country_code.nunique()),"date_start":str(frame.date.min().date()),
      "date_end":str(frame.date.max().date()),"split_date":str(pd.Timestamp(cutoff).date()),
      "train_rows":len(train_df),"test_rows":len(test_df),"validation":"global chronological 80/20 date holdout; no shuffle","horizons":{}}
    for h in (1,2,3):
        models[str(h)]={};metrics["horizons"][str(h)]={}
        for target in WEATHER:
            y=f"target_{target}_d{h}"
            reg=HistGradientBoostingRegressor(max_iter=250,learning_rate=.07,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)
            reg.fit(train_df[features],train_df[y]);pred=reg.predict(test_df[features])
            models[str(h)][target]=reg
            metrics["horizons"][str(h)][target]={"mae":float(mean_absolute_error(test_df[y],pred)),
              "rmse":float(mean_squared_error(test_df[y],pred)**.5),"r2":float(r2_score(test_df[y],pred))}
        ytr=(train_df[f"target_rainfall_mm_d{h}"]>0.1).astype(int)
        yte=(test_df[f"target_rainfall_mm_d{h}"]>0.1).astype(int)
        clf=(HistGradientBoostingClassifier(max_iter=200,learning_rate=.07,max_leaf_nodes=31,l2_regularization=1.0,random_state=42) if ytr.nunique()>1 else DummyClassifier(strategy="constant",constant=int(ytr.iloc[0])))
        clf.fit(train_df[features],ytr);prob=clf.predict_proba(test_df[features])[:,list(clf.classes_).index(1)] if 1 in clf.classes_ else [0.0]*len(test_df)
        pred=(prob>=.5).astype(int);models[str(h)]["rain_probability"]=clf
        metrics["horizons"][str(h)]["rain_probability"]={"precision":float(precision_score(yte,pred,zero_division=0)),
          "recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),
          "roc_auc":float(roc_auc_score(yte,prob)) if yte.nunique()>1 else None,
          "brier":float(brier_score_loss(yte,prob))}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    joblib.dump({"models":models,"features":features,"weather_targets":WEATHER,"version":"global-weather-v1"},out/"global_weather_model.joblib",compress=3)
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return metrics

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output-dir",default="artifacts");a=p.parse_args()
    print(json.dumps(train(a.data,a.output_dir),indent=2))
