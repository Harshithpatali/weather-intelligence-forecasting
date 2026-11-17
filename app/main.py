from fastapi import FastAPI, HTTPException, Query
from app.weather_client import OpenWeatherClient, WeatherProviderError
from app.database import init_db, save_observation
from ml.predict import predict_three_days, load_metrics

app=FastAPI(title="Weather Intelligence API",version="2.0.0")
@app.on_event("startup")
def startup(): init_db()

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/api/weather/location")
async def location_weather(place:str=Query(min_length=2,max_length=120),history_days:int=Query(10,ge=5,le=15)):
    c=OpenWeatherClient()
    try:
        loc=await c.geocode(place)
        current=await c.current(loc["lat"],loc["lon"])
        current.update({"city":loc["name"],"country":loc["country"]})
        save_observation(current)
        status="available"
        try:
            hist=await c.history(loc["lat"],loc["lon"],history_days)
            for row in hist:
                row["city"]=loc["name"]
                save_observation(row)
        except WeatherProviderError as e:
            hist=[]; status=str(e)
        return {"location":loc,"current":current,"history":hist,"history_days_requested":history_days,"history_status":status}
    except WeatherProviderError as e: raise HTTPException(502,detail=str(e)) from e

@app.get("/api/predictions/three-day")
def three_day(place:str=Query(min_length=2),history_days:int=Query(10,ge=5,le=15)):
    result=predict_three_days(place,history_days)
    if result is None: raise HTTPException(409,detail="Three-day model is not trained or this location lacks sufficient stored history.")
    return result

@app.get("/api/models/metrics")
def metrics(): return load_metrics()
