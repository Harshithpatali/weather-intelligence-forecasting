from fastapi import FastAPI, HTTPException, Query
from app.weather_client import OpenWeatherClient, WeatherProviderError
from app.database import init_db, save_observation
from ml.predict import predict_for_city, load_metrics

app = FastAPI(title="Weather Intelligence API", version="1.0.0",
              description="Live weather, provider forecast and separately trained ML predictions.")

@app.on_event("startup")
def startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok", "service": "weather-intelligence-api"}

@app.get("/api/weather/current")
async def current_weather(city: str = Query(min_length=2, max_length=80)):
    try:
        result = await OpenWeatherClient().current(city)
        save_observation(result)
        return result
    except WeatherProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get("/api/weather/forecast")
async def weather_forecast(city: str = Query(min_length=2, max_length=80)):
    try:
        return await OpenWeatherClient().forecast(city)
    except WeatherProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

@app.get("/api/predictions/ml")
def ml_prediction(city: str = Query(min_length=2, max_length=80)):
    result = predict_for_city(city)
    if result is None:
        raise HTTPException(status_code=409, detail="ML model not trained or insufficient recent observations. Collect data and train first.")
    return result

@app.get("/api/models/metrics")
def model_metrics():
    return load_metrics()
