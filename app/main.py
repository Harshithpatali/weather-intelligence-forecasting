from fastapi import FastAPI,HTTPException,Query
from app.weather_client import OpenWeatherClient,WeatherProviderError
from app.database import init_db,save_observation
from app.recent_weather import fetch_recent_daily
from ml.predict import predict_from_recent,load_metrics,load_model

app=FastAPI(title="Weather Intelligence API",version="3.0.0")
@app.on_event("startup")
def startup():
    init_db()
    load_model()  # load the persisted global model once per API worker

@app.get("/health")
def health():
    return {"status":"ok","model_loaded":load_model() is not None,"model_mode":"pretrained global model; no request-time training"}

@app.get("/api/weather/location")
async def location_weather(place:str=Query(min_length=2,max_length=120),history_days:int=Query(10,ge=5,le=15)):
    c=OpenWeatherClient()
    try:
        loc=await c.geocode(place);current=await c.current(loc["lat"],loc["lon"])
        current.update({"city":loc["name"],"country":loc["country"]})
        save_observation(current)
        history=await fetch_recent_daily(loc["lat"],loc["lon"],max(7,history_days))
        return {"location":loc,"current":current,"history":history,"history_days_requested":history_days,
          "history_status":"Open-Meteo recent daily context"}
    except WeatherProviderError as e:raise HTTPException(502,detail=str(e)) from e
    except Exception as e:raise HTTPException(502,detail=f"Recent weather retrieval failed: {e}") from e

@app.get("/api/predictions/three-day")
async def three_day(place:str=Query(min_length=2,max_length=120)):
    c=OpenWeatherClient()
    try:
        loc=await c.geocode(place)
        current=await c.current(loc["lat"],loc["lon"])
        recent=await fetch_recent_daily(loc["lat"],loc["lon"],7)
        if recent:
            recent[-1].update({"temperature_c":current["temperature_c"],"humidity_pct":current["humidity_pct"],
              "rainfall_mm":current["rain_1h_mm"],"wind_speed_ms":current["wind_speed_ms"]})
        result=predict_from_recent(recent,loc["lat"],loc["lon"])
        if result is None:raise HTTPException(503,detail="Global model artifact is not installed. Train once and mount artifacts/global_weather_model.joblib.")
        result["location"]={k:loc[k] for k in ("name","state","country")}
        result["current_weather"]=current
        return result
    except WeatherProviderError as e:raise HTTPException(502,detail=str(e)) from e

@app.get("/api/models/metrics")
def metrics():return load_metrics()
