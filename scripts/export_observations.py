from pathlib import Path
import pandas as pd
from app.database import SessionLocal,WeatherObservation
if not SessionLocal: raise SystemExit("DATABASE_URL is required.")
with SessionLocal() as session:
    rows=session.query(WeatherObservation).order_by(WeatherObservation.city,WeatherObservation.observed_at).all()
data=[{k:getattr(r,k) for k in ["city","observed_at","temperature_c","feels_like_c","humidity_pct","pressure_hpa","wind_speed_ms","cloud_pct","rain_1h_mm"]} for r in rows]
Path("data").mkdir(exist_ok=True)
pd.DataFrame(data).to_csv("data/observations.csv",index=False)
print(f"Exported {len(data)} weather observations to data/observations.csv")
