from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Float, DateTime, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from app.config import settings

class Base(DeclarativeBase):
    pass

class WeatherObservation(Base):
    __tablename__ = "weather_observations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    city: Mapped[str] = mapped_column(String(120), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    temperature_c: Mapped[float] = mapped_column(Float)
    feels_like_c: Mapped[float] = mapped_column(Float)
    humidity_pct: Mapped[float] = mapped_column(Float)
    pressure_hpa: Mapped[float] = mapped_column(Float)
    wind_speed_ms: Mapped[float] = mapped_column(Float)
    cloud_pct: Mapped[float] = mapped_column(Float)
    rain_1h_mm: Mapped[float] = mapped_column(Float)

engine = create_engine(settings.database_url, pool_pre_ping=True) if settings.database_url else None
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False) if engine else None

def init_db():
    if engine:
        Base.metadata.create_all(engine)

def save_observation(payload: dict):
    if not SessionLocal or not payload.get("observed_at"):
        return
    fields = ("temperature_c", "feels_like_c", "humidity_pct", "pressure_hpa",
              "wind_speed_ms", "cloud_pct", "rain_1h_mm")
    values = {key: payload.get(key) for key in fields}
    if any(value is None for value in values.values()):
        return
    with SessionLocal() as session:
        session.add(WeatherObservation(
            city=payload["city"],
            observed_at=datetime.fromtimestamp(payload["observed_at"], tz=timezone.utc),
            **values
        ))
        session.commit()
