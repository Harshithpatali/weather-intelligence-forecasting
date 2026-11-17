import json
from pathlib import Path
import joblib
import pandas as pd
from sqlalchemy import select
from app.database import SessionLocal, WeatherObservation
from ml.features import BASE_FEATURES

ARTIFACTS = Path("artifacts")

def load_metrics():
    path = ARTIFACTS / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {"status": "not_trained"}

def predict_for_city(city: str):
    temp_path, rain_path = ARTIFACTS / "temperature_model.joblib", ARTIFACTS / "rain_model.joblib"
    if not temp_path.exists() or not rain_path.exists() or not SessionLocal:
        return None
    with SessionLocal() as session:
        rows = session.scalars(select(WeatherObservation)
            .where(WeatherObservation.city.ilike(city))
            .order_by(WeatherObservation.observed_at.desc()).limit(2)).all()
    if len(rows) < 2:
        return None
    rows = list(reversed(rows))
    latest = {}
    for lag, row in zip((2, 1), rows):
        for col in BASE_FEATURES:
            latest[f"{col}_lag{lag}"] = getattr(row, col)
    temp_bundle, rain_bundle = joblib.load(temp_path), joblib.load(rain_path)
    x_temp = pd.DataFrame([[latest[f] for f in temp_bundle["features"]]], columns=temp_bundle["features"])
    x_rain = pd.DataFrame([[latest[f] for f in rain_bundle["features"]]], columns=rain_bundle["features"])
    rain_model = rain_bundle["model"]
    probability = (float(rain_model.predict_proba(x_rain)[0][list(rain_model.classes_).index(1)])
                   if 1 in rain_model.classes_ else 0.0)
    return {
        "city": city,
        "predicted_next_observation_temperature_c": float(temp_bundle["model"].predict(x_temp)[0]),
        "rain_probability_next_observation": probability,
        "prediction_horizon": "next collected observation; interval depends on collection cadence",
        "model_source": "locally trained Random Forest",
        "metrics": load_metrics()
    }
