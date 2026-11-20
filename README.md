# Weather Intelligence & Forecasting — India Pilot

An India-wide hourly weather forecasting and forecast-calibration project. The initial dataset covers 30 populated Indian locations selected from GeoNames by population.

## Data design

- **Previous Runs API:** hourly forecasts at fixed 1-, 2- and 3-day lead times; these are the forecast inputs evaluated by lead.
- **Historical Forecast API:** archived forecast context, including precipitation probability. It is retained in the source dataset but is not mixed into fixed-lead model features.
- **Historical Weather API:** hourly reanalysis values used only as verification targets, never as forecast features.

Rows are keyed by location and valid UTC hour. Reanalysis is a model-based estimate, not a direct station measurement. GeoNames supplies the Indian populated-place catalogue and location metadata.

Documentation: https://open-meteo.com/en/docs/previous-runs-api, https://open-meteo.com/en/docs/historical-forecast-api, https://open-meteo.com/en/docs/historical-weather-api

## Build the India dataset

```bash
pip install -r requirements.txt
python -m scripts.build_india_dataset --days 365 --locations 30
```

Outputs: `data/india_forecast_training.parquet` and `artifacts/india_dataset_report.json`. The **Build India forecast dataset** GitHub Actions workflow uploads both as a 90-day artifact. The full-year dataset should be built before training; the 30-day pilot is for pipeline checks only.

## Leakage-safe model evaluation

The India-specific pipeline is `ml/india_training.py`, invoked with:

```bash
python -m scripts.train_india_model --data data/india_forecast_training.parquet
```

For each fixed lead (1, 2 and 3 days), it compares the raw Previous Runs forecast against a separately fitted HistGradientBoosting bias-correction model. Temperature, relative humidity, rainfall amount and wind speed are evaluated with MAE and RMSE. Rain occurrence (observed precipitation greater than 0.1 mm) is evaluated with Brier score, precision, recall, F1 and ROC-AUC where defined.

The holdout is chronological by UTC valid time (first 80% train, final 20% test); timestamps are not randomly shuffled. Only fixed-lead forecast variables, location metadata and valid-time calendar features enter the models. Historical Weather reanalysis columns are targets only. Duplicate location/time rows are rejected.

Outputs:
- `artifacts/india_weather_calibration.joblib`
- `artifacts/india_model_metrics.json`

### Run training from GitHub Actions

1. Open a successful **Build India forecast dataset** run.
2. Note its run ID and artifact name (for example, `india-forecast-dataset-2`).
3. Run **Train India weather model**, supplying that artifact name and run ID.
4. Download the resulting `india-weather-model-*` artifact to inspect the model and metrics.

A successful training run produces evaluation evidence; it does not by itself mean the model has been integrated into the API or Streamlit app.

## Runtime target

The intended runtime accepts an Indian location, fetches current conditions and recent forecast context, loads a saved offline-trained model, and returns a three-day forecast for temperature, rainfall probability/amount, humidity and wind. Runtime requests will not retrain the model.
