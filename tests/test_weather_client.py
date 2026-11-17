import pytest
from app.weather_client import OpenWeatherClient, WeatherProviderError

@pytest.mark.asyncio
async def test_missing_key_fails_before_network_call():
    client = OpenWeatherClient(api_key="")
    with pytest.raises(WeatherProviderError, match="not configured"):
        await client.current("Mysuru")
