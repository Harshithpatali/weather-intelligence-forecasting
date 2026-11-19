# Weather Intelligence & Forecasting — India Pilot

An India-wide weather forecasting and model-evaluation pilot. The initial dataset uses 30 populated locations selected from GeoNames across India. The scope is national rather than state-specific so the pipeline does not depend on resolving a single state's administrative catalogue.

## Data design

- **Previous Runs API:** hourly forecasts at fixed 1-, 2- and 3-day lead times; primary inputs for lead-time-specific evaluation.
- **Historical Forecast API:** archived continuous forecast context, including precipitation probability.
- **Historical Weather API:** hourly reanalysis observations used as verification targets, not as forecast inputs.

Each record is keyed by location and valid UTC hour. Forecast lead-time columns and verifying observations remain separate. Reanalysis is a model-based estimate, not a direct station measurement. GeoNames supplies the Indian populated-place catalogue and location metadata.

Documentation: https://open-meteo.com/en/docs/previous-runs-api, https://open-meteo.com/en/docs/historical-forecast-api, https://open-meteo.com/en/docs/historical-weather-api

## Build

```bash
pip install -r requirements.txt
python -m scripts.build_india_dataset --days 30 --locations 30
```

Outputs: `data/india_forecast_training.parquet` and `artifacts/india_dataset_report.json`. Run **Actions → Build India forecast dataset** to upload them as a 90-day artifact. Generated data is not committed into Git history. Validate the 30-day pilot before requesting the full 365-day history.

## Model plan

The data build is not a trained model. Next, create a leakage-safe training/evaluation pipeline pairing each fixed-lead forecast with the observation at the same valid time. Compare lead-time-specific bias-correction models. Report MAE/RMSE for temperature, humidity and wind, plus rainfall amount and probability metrics. Do not claim a trained or deployed model until evaluation artifacts exist.

## Runtime target

The intended runtime accepts an Indian location, fetches current conditions and recent forecast context, loads a saved offline-trained model, and returns a three-day forecast for temperature, rainfall probability/amount, humidity and wind. Runtime requests will not retrain the model.
