# Weather Intelligence & Forecasting

A location-based weather intelligence platform. Users enter a place, choose a 5/10/15-day historical lookback, retrieve current and historical observations, and view a three-day ML forecast.

## Data flow
1. OpenWeather Geocoding resolves the place to coordinates.
2. Current conditions are retrieved and stored in PostgreSQL.
3. Historical daily snapshots are requested from One Call API 3.0 Time Machine and stored.
4. A shared multi-location daily dataset is exported and used to train multi-horizon regressors.
5. The UI presents historical trends, current conditions and day 1/2/3 ML predictions.

## Historical API subscription
Historical snapshots use https://api.openweathermap.org/data/3.0/onecall/timemachine. This requires the relevant One Call API subscription. A current-weather API key alone may not grant access. A 15-day backfill issues 15 API calls. If access is denied, the app shows the provider error and does not fabricate history.

## Train
Collect observations from multiple locations and dates. The user-selected lookback is an inference context window, not enough data to train a reliable forecasting model.
```bash
python -m scripts.export_observations
python -m ml.train --data data/observations.csv
```
At least 100 complete city-day rows are required by the initial pipeline. This is only a minimum code guard, not a quality guarantee. Use many locations and seasons, and evaluate chronologically.

## Model targets
- Temperature, humidity and wind speed: regression for day 1, 2 and 3.
- Rainfall: initial version predicts rainfall amount, not probability. A separately calibrated occurrence classifier is needed before displaying rainfall probability.
- The model uses Random Forest regressors. Metrics include MAE, RMSE and R² per target and horizon.

## Run
Configure OPENWEATHER_API_KEY and DATABASE_URL. Install requirements, run `uvicorn app.main:app --reload`, then `streamlit run dashboard/streamlit_app.py`.

API:
- GET /health
- GET /api/weather/location?place=Mysuru&history_days=10
- GET /api/predictions/three-day?place=Mysuru&history_days=10
- GET /api/models/metrics
