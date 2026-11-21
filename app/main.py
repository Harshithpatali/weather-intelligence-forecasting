from fastapi import FastAPI, HTTPException, Query
from app.weather_client import OpenWeatherClient, WeatherProviderError
from app.database import init_db, save_observation
from app.recent_weather import fetch_recent_daily, fetch_hourly_forecast
from ml.india_predict import (
    load_india_metrics, load_india_model, predict_india_forecasts,
)

app = FastAPI(title="Weather Intelligence API", version="4.0.0")


@app.on_event("startup")
def startup():
    init_db()
    load_india_model()  # Load the persisted India model once per API worker.


@app.get("/health")
def health():
    loaded = load_india_model() is not None
    return {
        "status": "ok",
        "model_loaded": loaded,
        "model_mode": "India fixed-lead calibrated model; inference only" if loaded
                      else "India model artifact not installed",
    }


@app.get("/api/weather/location")
async def location_weather(
    place: str = Query(min_length=2, max_length=120),
    history_days: int = Query(10, ge=5, le=15),
):
    client = OpenWeatherClient()
    try:
        location = await client.geocode(place)
        current = await client.current(location["lat"], location["lon"])
        current.update({"city": location["name"], "country": location["country"]})
        save_observation(current)
        history = await fetch_recent_daily(
            location["lat"], location["lon"], max(7, history_days)
        )
        return {
            "location": location,
            "current": current,
            "history": history,
            "history_days_requested": history_days,
            "history_status": "Open-Meteo recent daily context",
        }
    except WeatherProviderError as exc:
        raise HTTPException(502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, detail=f"Recent weather retrieval failed: {exc}") from exc


@app.get("/api/predictions/three-day")
async def three_day(place: str = Query(min_length=2, max_length=120)):
    """Return calibrated forecasts for fixed 24h, 48h and 72h lead times."""
    model = load_india_model()
    if model is None:
        raise HTTPException(
            503,
            detail="India model artifact is not installed. Add artifacts/india_weather_calibration.joblib.",
        )
    client = OpenWeatherClient()
    try:
        location = await client.geocode(place)
        current = await client.current(location["lat"], location["lon"])
        hourly = await fetch_hourly_forecast(location["lat"], location["lon"], days=4)
        result = predict_india_forecasts(
            hourly, location["lat"], location["lon"],
            elevation_m=0.0, model=model,
        )
        if result is None:
            raise HTTPException(503, detail="India model could not be loaded.")
        result["location"] = {
            key: location.get(key) for key in ("name", "state", "country")
        }
        result["current_weather"] = current
        return result
    except WeatherProviderError as exc:
        raise HTTPException(502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, detail=f"Forecast inference failed: {exc}") from exc


@app.get("/api/predictions")
async def india_predictions(place: str = Query(min_length=2, max_length=120)):
    """Alias for clients that prefer a resource-oriented predictions route."""
    return await three_day(place)


@app.get("/api/models/metrics")
def metrics():
    return load_india_metrics()
