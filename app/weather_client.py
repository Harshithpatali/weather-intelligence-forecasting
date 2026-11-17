import httpx
from app.config import settings
class WeatherProviderError(RuntimeError): pass
class OpenWeatherClient:
    def __init__(self, api_key=None):
        self.key=settings.openweather_api_key if api_key is None else api_key
        self.base=settings.openweather_base_url.rstrip("/")
        self.timeout=settings.request_timeout_seconds
    async def get(self,url,params):
        if not self.key: raise WeatherProviderError("OPENWEATHER_API_KEY is not configured.")
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r=await client.get(url,params={**params,"appid":self.key})
            if r.status_code in (401,403): raise WeatherProviderError("OpenWeather rejected the API key or API access.")
            if r.status_code==404: raise WeatherProviderError("Location not found.")
            if r.status_code==429: raise WeatherProviderError("OpenWeather rate limit reached.")
            r.raise_for_status();return r.json()
        except httpx.HTTPError as e: raise WeatherProviderError(f"OpenWeather request failed: {e}") from e
    async def geocode(self,place):
        data=await self.get("https://api.openweathermap.org/geo/1.0/direct",{"q":place,"limit":1})
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
