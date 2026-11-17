import httpx
from app.config import settings

class WeatherProviderError(RuntimeError):
    pass

class OpenWeatherClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else settings.openweather_api_key
        self.base_url = settings.openweather_base_url.rstrip("/")

    async def _get(self, endpoint: str, city: str) -> dict:
        if not self.api_key:
            raise WeatherProviderError("OPENWEATHER_API_KEY is not configured.")
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
                response = await client.get(
                    f"{self.base_url}/{endpoint}",
                    params={"q": city, "appid": self.api_key, "units": "metric"},
                )
            if response.status_code == 401:
                raise WeatherProviderError("OpenWeather rejected the API key. Check key and API access.")
            if response.status_code == 404:
                raise WeatherProviderError(f"City '{city}' was not found by OpenWeather.")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise WeatherProviderError(f"Weather provider request failed: {exc}") from exc

    async def current(self, city: str) -> dict:
        raw = await self._get("weather", city)
        main, wind = raw.get("main", {}), raw.get("wind", {})
        weather = (raw.get("weather") or [{}])[0]
        return {
            "city": raw.get("name", city), "country": raw.get("sys", {}).get("country"),
            "observed_at": raw.get("dt"), "temperature_c": main.get("temp"),
            "feels_like_c": main.get("feels_like"), "humidity_pct": main.get("humidity"),
            "pressure_hpa": main.get("pressure"), "wind_speed_ms": wind.get("speed", 0.0),
            "cloud_pct": raw.get("clouds", {}).get("all", 0),
            "rain_1h_mm": raw.get("rain", {}).get("1h", 0.0),
            "condition": weather.get("description", "unknown"), "icon": weather.get("icon"),
            "source": "OpenWeather current conditions",
        }

    async def forecast(self, city: str) -> dict:
        raw = await self._get("forecast", city)
        items = []
        for entry in raw.get("list", []):
            weather = (entry.get("weather") or [{}])[0]
            items.append({
                "timestamp": entry.get("dt"), "temperature_c": entry.get("main", {}).get("temp"),
                "feels_like_c": entry.get("main", {}).get("feels_like"),
                "humidity_pct": entry.get("main", {}).get("humidity"),
                "precipitation_probability": entry.get("pop"),
                "rain_3h_mm": entry.get("rain", {}).get("3h", 0.0),
                "wind_speed_ms": entry.get("wind", {}).get("speed"),
                "condition": weather.get("description", "unknown"),
            })
        return {"city": raw.get("city", {}).get("name", city), "forecast": items,
                "source": "OpenWeather 5-day / 3-hour forecast"}
