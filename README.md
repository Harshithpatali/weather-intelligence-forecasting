# Weather Intelligence & Forecasting

A production-oriented weather analytics platform combining OpenWeather live conditions and 5-day forecasts with independently trained machine-learning models.

## Features
- FastAPI backend for current conditions, provider forecast, ML predictions and health.
- Streamlit dashboard with live weather cards, forecast charts and model status.
- OpenWeather API key is read only from server-side environment variables.
- Optional PostgreSQL persistence for normalized observations.
- Random Forest regression for next-observation temperature and classification for rain.
- Chronological 80/20 holdout validation; MAE, RMSE, R², precision, recall, F1 and ROC-AUC.
- Docker Compose, tests and GitHub Actions CI.

**Scientific distinction:** OpenWeather's forecast is displayed separately from local ML predictions. ML predictions remain unavailable until sufficient historical observations are collected and models are trained; no synthetic prediction is fabricated.

## Quick start
1. Create an API key at https://openweathermap.org/api.
2. Copy `.env.example` to `.env` and set `OPENWEATHER_API_KEY`. Never commit secrets.
3. Install and run:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```
4. In another terminal: `streamlit run dashboard/streamlit_app.py`
5. API docs: http://localhost:8000/docs

## Docker
```bash
cp .env.example .env
# Add OPENWEATHER_API_KEY to .env
docker compose up --build
```
Dashboard: http://localhost:8501 · API: http://localhost:8000/docs

## Data and model training
Each successful `/api/weather/current` request stores an observation when PostgreSQL is configured. Collect observations across different days and weather conditions. Export the observation table to CSV with columns:
`city, observed_at, temperature_c, feels_like_c, humidity_pct, pressure_hpa, wind_speed_ms, cloud_pct, rain_1h_mm`

Train:
```bash
python -m ml.train --data data/observations.csv --city Mysuru
```
Artifacts are written to `artifacts/`. The pipeline creates two lagged observations per city and predicts the next collected observation. The prediction horizon depends on the collection cadence; it is not a fixed 24-hour forecast.

At least 50 usable rows are required by the starter training pipeline. More data spanning seasons and weather regimes is strongly recommended. The last 20% in time is held out; data is never shuffled.

## API
- `GET /health`
- `GET /api/weather/current?city=Mysuru`
- `GET /api/weather/forecast?city=Mysuru`
- `GET /api/predictions/ml?city=Mysuru`
- `GET /api/models/metrics`

## Environment
See `.env.example`. For deployment, configure `OPENWEATHER_API_KEY` as a secret/environment variable in the hosting platform. GitHub Actions secrets are only available to workflows that explicitly map them; they do not automatically become runtime secrets for a deployed service.

## Responsible use
Rain probability is a model estimate, not an official warning. Do not use this educational project as an emergency alert system.
