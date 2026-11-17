"""Recent daily context for inference; separate from the offline training archive."""
import httpx
from app.config import settings

async def fetch_recent_daily(latitude,longitude,days=7):
    params={"latitude":latitude,"longitude":longitude,"past_days":days,"forecast_days":1,
      "daily":"temperature_2m_mean,precipitation_sum,wind_speed_10m_mean",
      "hourly":"relative_humidity_2m","timezone":"UTC","wind_speed_unit":"ms"}
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds,follow_redirects=True) as client:
        r=await client.get("https://api.open-meteo.com/v1/forecast",params=params)
        r.raise_for_status();data=r.json()
    daily=data.get("daily",{});hourly=data.get("hourly",{})
    import pandas as pd
    h=pd.DataFrame({"time":hourly.get("time",[]),"humidity":hourly.get("relative_humidity_2m",[])})
    if not h.empty:
        h["date"]=pd.to_datetime(h.time).dt.date.astype(str)
        humidity=h.groupby("date").humidity.mean().to_dict()
    else: humidity={}
    rows=[]
    for i,day in enumerate(daily.get("time",[])):
        rows.append({"date":day,"temperature_c":daily["temperature_2m_mean"][i],
          "humidity_pct":humidity.get(day),"rainfall_mm":daily["precipitation_sum"][i],
          "wind_speed_ms":daily["wind_speed_10m_mean"][i]})
    return rows
