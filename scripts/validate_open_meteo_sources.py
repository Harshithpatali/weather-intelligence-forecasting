"""Smoke-test Open-Meteo forecast archives before global dataset collection."""
import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

LAT, LON = 52.52, 13.41
END = date.today() - timedelta(days=10)
START = END - timedelta(days=6)

HISTORICAL_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
OBSERVATIONS = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "precipitation_probability", "wind_speed_10m",
]
PREVIOUS_HOURLY = [
    f"{name}_previous_day{lead}"
    for lead in (1, 2, 3)
    for name in ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m")
]
OBS_DAILY = [
    "temperature_2m_mean", "relative_humidity_2m_mean",
    "precipitation_sum", "wind_speed_10m_max",
]

def get_json(client, url, params):
    response = client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo returned an API error"))
    return payload

def inspect_series(payload, key, expected):
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"No hourly timestamps returned for {key}")
    available = [name for name in expected if name in hourly]
    missing = [name for name in expected if name not in hourly]
    return {
        "hourly_rows": len(hourly["time"]),
        "start": hourly["time"][0],
        "end": hourly["time"][-1],
        "available_variables": available,
        "missing_variables": missing,
    }

def main():
    start, end = START.isoformat(), END.isoformat()
    common = {"latitude": LAT, "longitude": LON, "start_date": start,
              "end_date": end, "timezone": "UTC"}
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        historical = get_json(client, HISTORICAL_FORECAST, {
            **common, "hourly": ",".join(FORECAST_HOURLY)
        })
        previous = get_json(client, PREVIOUS_RUNS, {
            **common, "hourly": ",".join(PREVIOUS_HOURLY)
        })
        observations = get_json(client, OBSERVATIONS, {
            **common, "daily": ",".join(OBS_DAILY)
        })

    report = {
        "sample_location": {"latitude": LAT, "longitude": LON},
        "sample_period": {"start": start, "end": end},
        "historical_forecast": inspect_series(historical, "historical_forecast", FORECAST_HOURLY),
        "previous_runs": inspect_series(previous, "previous_runs", PREVIOUS_HOURLY),
        "observations": {
            "daily_rows": len(observations.get("daily", {}).get("time", [])),
            "available_variables": [x for x in OBS_DAILY if x in observations.get("daily", {})],
            "missing_variables": [x for x in OBS_DAILY if x not in observations.get("daily", {})],
        },
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/open_meteo_source_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if report["historical_forecast"]["missing_variables"]:
        raise SystemExit("Historical Forecast API did not return all requested variables.")
    if report["previous_runs"]["missing_variables"]:
        raise SystemExit("Previous Runs API did not return all requested lead-time variables.")
    if report["observations"]["missing_variables"]:
        raise SystemExit("Historical Weather API did not return all verification variables.")
    if report["observations"]["daily_rows"] < 5:
        raise SystemExit("Too few observation rows returned for the sample period.")

if __name__ == "__main__":
    main()
