import httpx
from datetime import datetime, timedelta, timezone
from app.config import settings

class WeatherProviderError(RuntimeError): pass

class OpenWeatherClient:
    def __init__(self):
        self.key=settings.openweather_api_key
        self.base=settings.openweather_base_url.rstrip("/")
        self.timeout=settings.request_timeout_seconds

    async def get(self,url,params):
        if not self.key: raise WeatherProviderError("OPENWEATHER_API_KEY is not configured.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r=await client.get(url,params={**params,"appid":self.key})
            if r.status_code in (401,403): raise WeatherProviderError("OpenWeather rejected this request. Historical data requires the appropriate One Call subscription.")
            if r.status_code==404: raise WeatherProviderError("Location not found.")
            if r.status_code==429: raise WeatherProviderError("OpenWeather rate limit reached.")
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e: raise WeatherProviderError(f"OpenWeather request failed: {e}") from e

    async def geocode(self,place):
        url=self.base.replace("/data/2.5","/geo/1.0")+"/direct"
        data=await self.get(url,{"q":place,"limit":1})
        if not data: raise WeatherProviderError(f"No location found for {place}.")
        x=data[0]
        return {"name":x["name"],"state":x.get("state"),"country":x.get("country"),"lat":x["lat"],"lon":x["lon"]}

    async def current(self,lat,lon):
        x=await self.get(self.base+"/weather",{"lat":lat,"lon":lon,"units":"metric"})
        m,w=x.get("main",{}),x.get("wind",{})
        return {"observed_at":x.get("dt"),"temperature_c":m.get("temp"),"feels_like_c":m.get("feels_like"),
        "humidity_pct":m.get("humidity"),"pressure_hpa":m.get("pressure"),"wind_speed_ms":w.get("speed",0),
        "cloud_pct":x.get("clouds",{}).get("all",0),"rain_1h_mm":x.get("rain",{}).get("1h",0),
        "condition":(x.get("weather") or [{}])[0].get("description","unknown")}

    async def history(self,lat,lon,days):
        rows=[]
        for n in range(days,0,-1):
            dt=datetime.now(timezone.utc)-timedelta(days=n)
            dt=dt.replace(hour=12,minute=0,second=0,microsecond=0)
            x=await self.get("https://api.openweathermap.org/data/3.0/onecall/timemachine",
                {"lat":lat,"lon":lon,"dt":int(dt.timestamp()),"units":"metric"})
            d=(x.get("data") or [{}])[0]
            if d:
                rows.append({"observed_at":d.get("dt",int(dt.timestamp())),"temperature_c":d.get("temp"),
                "feels_like_c":d.get("feels_like",d.get("temp")),"humidity_pct":d.get("humidity"),
                "pressure_hpa":d.get("pressure"),"wind_speed_ms":d.get("wind_speed",0),
                "cloud_pct":d.get("clouds",0),"rain_1h_mm":d.get("rain",{}).get("1h",0)})
        return rows
