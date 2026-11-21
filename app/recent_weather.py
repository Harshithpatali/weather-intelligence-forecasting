"""Recent daily context and live hourly forecasts from Open-Meteo."""
import httpx
from datetime import datetime, timezone
from app.config import settings

BASE_URL = "https://api.open-meteo.com/v1/forecast"


async def fetch_recent_daily(latitude, longitude, days=7):
    params = {"latitude": latitude, "longitude": longitude, "past_days": days,
              "forecast_days": 1,
              "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_mean",
              "hourly": "relative_humidity_2m", "timezone": "UTC",
              "wind_speed_unit": "ms"}
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds,
                                 follow_redirects=True) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

    daily, hourly = data.get("daily", {}), data.get("hourly", {})
    import pandas as pd
    humidity_frame = pd.DataFrame({
        "time": hourly.get("time", []),
        "humidity": hourly.get("relative_humidity_2m", []),
    })
    if not humidity_frame.empty:
        humidity_frame["date"] = pd.to_datetime(humidity_frame.time).dt.date.astype(str)
        humidity = humidity_frame.groupby("date").humidity.mean().to_dict()
    else:
        humidity = {}

    rows = []
    today = datetime.now(timezone.utc).date().isoformat()
    for i, day in enumerate(daily.get("time", [])):
        if day > today:
            continue
        rows.append({
            "date": day,
            "temperature_c": daily["temperature_2m_mean"][i],
            "humidity_pct": humidity.get(day),
            "rainfall_mm": daily["precipitation_sum"][i],
            "wind_speed_ms": daily["wind_speed_10m_mean"][i],
        })
    return rows


async def fetch_hourly_forecast(latitude, longitude, days=4):
    """Fetch the live hourly forecast variables used by the India model."""
    params = {
        "latitude": latitude, "longitude": longitude,
        "forecast_days": days, "timezone": "UTC",
        "wind_speed_unit": "ms",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds,
                                 follow_redirects=True) as client:
        response = await client.get(BASE_URL, params=params)
        response.raise_for_status()
        payload = response.json()
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo forecast request failed"))
    return payload
