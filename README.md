# Weather Intelligence & Forecasting

Global pretrained ML weather forecasting. Train offline once using one year of reanalysis weather for 1,200 populated places across 40 countries (30 per country). Runtime requests never retrain the model.

## Global training data
- Source: Open-Meteo Historical Weather API, ERA5 reanalysis.
- Location catalogue: GeoNames populated places (cities500); top 30 population-ranked cities in each of 40 configured countries.
- Variables: daily mean temperature, daily precipitation sum, daily maximum wind speed, and daily mean relative humidity derived from hourly values.
- One year of daily data, with a 6-day publication buffer to avoid incomplete recent archive days.
- The source is reanalysis (model-assimilated historical estimate), not direct station observations. See https://open-meteo.com/en/docs/historical-weather-api and GeoNames attribution.

## Build and train once
```bash
pip install -r requirements.txt
python -m scripts.build_global_dataset --days 365
python -m scripts.train_global_model
```
The dataset builder makes batched multi-coordinate archive requests and writes `data/global_weather_1y.csv`. It validates 30 places for each of 40 configured countries. If the GeoNames top-populated-places catalogue does not provide 30 for a country, supply a complete `data/locations.csv` containing at least 30 records for each configured country with columns: geonameid, name, latitude, longitude, country_code, population, dem.
Training writes `artifacts/global_weather_model.joblib` and `artifacts/metrics.json`. Model artifact is not committed by default; persist it in durable object storage or attach it to a release/deployment image. Training requires 1,200 locations, 40 countries and sufficient valid daily rows. It evaluates a global chronological 80/20 date holdout without shuffling.

## Runtime inference
1. Geocode the requested location using the configured OpenWeather API key.
2. Fetch current weather through OpenWeather.
3. Fetch the latest seven complete daily context records from Open-Meteo Forecast API.
4. Load the saved global artifact and run inference only. No runtime training.
5. Return daily temperature, rainfall amount and probability, humidity and wind for days 1–3.

The archive and runtime context are sourced from Open-Meteo reanalysis/forecast products, while current conditions are from OpenWeather. Reanalysis/forecast data are not guaranteed to equal station observations. Rainfall probability is the classifier's estimated probability of a day with measurable precipitation (>0.1 mm), not an official warning.

## Environment
Set `OPENWEATHER_API_KEY` for geocoding and current conditions. Set `DATABASE_URL` only if storing live observations. Keep API credentials in host secrets, never in source control.

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run dashboard/streamlit_app.py
```
