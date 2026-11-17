import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score)
from ml.features import build_training_frame

ARTIFACTS = Path("artifacts")

def train(data_path: str, city: str | None = None):
    raw = pd.read_csv(data_path)
    frame, features = build_training_frame(raw, city)
    if len(frame) < 50:
        raise ValueError(f"Only {len(frame)} usable rows. Need at least 50; collect more observations.")
    split = int(len(frame) * 0.8)
    train_df, test_df = frame.iloc[:split], frame.iloc[split:]
    if len(test_df) < 5:
        raise ValueError("Not enough chronological test observations.")
    x_train, x_test = train_df[features], test_df[features]
    ARTIFACTS.mkdir(exist_ok=True)
    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
    reg.fit(x_train, train_df["target_temperature_c"])
    temp_pred = reg.predict(x_test)
    rain = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    rain.fit(x_train, train_df["target_rain"])
    rain_pred = rain.predict(x_test)
    rain_prob = (rain.predict_proba(x_test)[:, list(rain.classes_).index(1)]
                 if 1 in rain.classes_ else [0.0] * len(x_test))
    metrics = {
        "city": city or "all", "rows": int(len(frame)), "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)), "validation": "chronological 80/20 split; no shuffle",
        "temperature": {
            "mae_c": float(mean_absolute_error(test_df["target_temperature_c"], temp_pred)),
            "rmse_c": float(mean_squared_error(test_df["target_temperature_c"], temp_pred) ** 0.5),
            "r2": float(r2_score(test_df["target_temperature_c"], temp_pred))},
        "rain": {
            "precision": float(precision_score(test_df["target_rain"], rain_pred, zero_division=0)),
            "recall": float(recall_score(test_df["target_rain"], rain_pred, zero_division=0)),
            "f1": float(f1_score(test_df["target_rain"], rain_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_df["target_rain"], rain_prob))
                if len(set(test_df["target_rain"])) > 1 else None},
        "warning": "Metrics apply only to this dataset, city filter and chronological holdout."
    }
    joblib.dump({"model": reg, "features": features}, ARTIFACTS / "temperature_model.joblib")
    joblib.dump({"model": rain, "features": features}, ARTIFACTS / "rain_model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--city")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.city), indent=2))
