import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Weather Intelligence", page_icon="🌦️", layout="wide")
API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

# Keep the existing dashboard layout and visual hierarchy.
st.title("🌦️ Weather Intelligence & Forecasting")
st.caption("India-wide calibrated weather forecasts · 1-, 2- and 3-day horizons")
with st.sidebar:
    place = st.text_input("Enter a city or location", "Mysuru, Karnataka, India")
    run = st.button("Analyse location", type="primary", use_container_width=True)
if run:
    st.session_state["place"] = place
if "place" not in st.session_state:
    st.info("Enter a location to retrieve current weather and run the saved India model.")
    st.stop()

place = st.session_state["place"]
try:
    with st.spinner("Loading current weather and applying the saved India forecast model..."):
        current_response = requests.get(
            f"{API}/api/weather/location",
            params={"place": place, "history_days": 7}, timeout=60,
        )
        current_response.raise_for_status()
        weather = current_response.json()
        prediction_response = requests.get(
            f"{API}/api/predictions/three-day", params={"place": place}, timeout=60,
        )
        prediction_response.raise_for_status()
        result = prediction_response.json()

    location = weather["location"]
    current = weather["current"]
    st.subheader(
        f"{location['name']}, {location.get('state') or ''} · {location['country']}"
    )
    a, b, c, d = st.columns(4)
    a.metric("Current temperature", f"{current['temperature_c']:.1f} °C")
    b.metric("Humidity", f"{current['humidity_pct']}%")
    c.metric("Wind", f"{current['wind_speed_ms']:.1f} m/s")
    d.metric("Pressure", f"{current['pressure_hpa']} hPa")

    history = pd.DataFrame(weather["history"])
    if not history.empty:
        st.subheader("Recent weather context")
        st.plotly_chart(
            px.line(history, x="date", y="temperature_c", markers=True,
                    title="Recent daily temperature"),
            use_container_width=True,
        )
        st.dataframe(history, use_container_width=True)

    st.divider()
    st.header("India ML forecast · Next 3 days")
    st.caption(f"Model: {result['model_version']} · {result['model_mode']}")
    forecast = pd.DataFrame(result["predictions"])
    st.dataframe(forecast, use_container_width=True)
    for column, label in [
        ("temperature_c", "Temperature (°C)"),
        ("rainfall_probability", "Rainfall probability"),
        ("rainfall_mm", "Rainfall amount (mm)"),
        ("humidity_pct", "Humidity (%)"),
        ("wind_speed_ms", "Wind speed (m/s)"),
    ]:
        st.plotly_chart(
            px.line(forecast, x="date", y=column, markers=True, title=label),
            use_container_width=True,
        )
    with st.expander("India model validation metrics"):
        metrics_response = requests.get(f"{API}/api/models/metrics", timeout=20)
        metrics_response.raise_for_status()
        st.json(metrics_response.json())
except requests.RequestException as exc:
    st.error(f"Could not complete weather inference: {exc}")
    if getattr(exc, "response", None) is not None:
        st.code(exc.response.text)
