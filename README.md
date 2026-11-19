# Weather Intelligence & Forecasting — Karnataka Pilot

A Karnataka-only weather forecasting and model-evaluation pilot. The first data build covers 30 populated locations across Karnataka, India. Global expansion is out of scope until this pipeline is validated.

## Data design

- **Previous Runs API:** hourly forecasts at fixed 1-, 2- and 3-day lead times; primary inputs for lead-time-specific evaluation.
- **Historical Forecast API:** archived continuous forecast context, including precipitation probability.
- **Historical Weather API:** hourly reanalysis observations used as verification targets, not as forecast inputs.

Each record is keyed by location and valid UTC hour. Forecast lead-time columns and verifying observations remain separate. Reanalysis is a model-based estimate, not a direct station measurement.

Documentation: https://open-meteo.com/en/docs/previous-runs-api, https://open-meteo.com/en/docs/historical-forecast-api, https://open-meteo.com/en/docs/historical-weather-api

## Build

```bash
pip install -r requirements.txt
python -m scripts.build_karnataka_dataset --days 365 --locations 30
```

Outputs: `data/karnataka_forecast_training.parquet` and `artifacts/karnataka_dataset_report.json`. Run **Actions → Build Karnataka forecast dataset** to upload them as a 90-day artifact. Generated data is not committed into Git history. The default end date leaves a six-day archive-completeness buffer; use `days=30` for a pilot before the full year.

## Model plan

The data build is not a trained model. Next, create a leakage-safe training/evaluation pipeline pairing each fixed-lead forecast with the observation at the same valid time. Compare lead-time-specific bias-correction models. Report MAE/RMSE for temperature, humidity and wind, plus rainfall amount and probability metrics. Do not claim a trained or deployed model until evaluation artifacts exist.

## Runtime target

The intended runtime accepts a Karnataka location, fetches current conditions and recent forecast context, loads a saved offline-trained model, and returns a three-day forecast for temperature, rainfall probability/amount, humidity and wind. Runtime requests will not retrain the model.
