import os
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Weather Intelligence", page_icon="🌦️", layout="wide")
API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
st.title("🌦️ Weather Intelligence")
st.caption("Live observations · provider forecast · independently trained ML")
city = st.sidebar.text_input("City", value=os.getenv("DEFAULT_CITY", "Mysuru"))
if st.sidebar.button("Refresh weather", type="primary"):
    st.cache_data.clear()

@st.cache_data(ttl=300)
def get_json(path, city_name):
    response = requests.get(f"{API}{path}", params={"city": city_name}, timeout=20)
    response.raise_for_status()
    return response.json()

try:
    current = get_json("/api/weather/current", city)
    forecast = get_json("/api/weather/forecast", city)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{current['temperature_c']:.1f} °C")
    c2.metric("Feels like", f"{current['feels_like_c']:.1f} °C")
    c3.metric("Humidity", f"{current['humidity_pct']}%")
    c4.metric("Wind", f"{current['wind_speed_ms']:.1f} m/s")
    st.subheader(f"{current['city']}, {current.get('country') or ''}")
    st.write(f"**Conditions:** {current['condition'].title()} · **Pressure:** {current['pressure_hpa']} hPa · **Cloud:** {current['cloud_pct']}%")
    df = pd.DataFrame(forecast["forecast"])
    if not df.empty:
        df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        left, right = st.columns(2)
        with left:
            st.subheader("OpenWeather temperature forecast")
            st.plotly_chart(px.line(df, x="time", y="temperature_c", markers=True,
                labels={"temperature_c": "Temperature (°C)", "time": "Time"}), use_container_width=True)
        with right:
            st.subheader("Precipitation probability")
            st.plotly_chart(px.area(df, x="time", y="precipitation_probability",
                labels={"precipitation_probability": "Probability", "time": "Time"}), use_container_width=True)
        st.subheader("Forecast detail")
        st.dataframe(df[["time", "temperature_c", "humidity_pct", "precipitation_probability",
                         "rain_3h_mm", "condition"]], use_container_width=True)
except requests.RequestException as exc:
    st.error(f"Could not load live weather: {exc}")
    st.info("Check that the API is running and OPENWEATHER_API_KEY is configured on the backend.")

st.divider()
st.header("Machine-learning predictions")
st.caption("Independent local model outputs; these are not the OpenWeather forecast.")
try:
    response = requests.get(f"{API}/api/predictions/ml", params={"city": city}, timeout=15)
    if response.ok:
        result = response.json()
        p1, p2 = st.columns(2)
        p1.metric("Next-observation temperature", f"{result['predicted_next_observation_temperature_c']:.1f} °C")
        p2.metric("Rain probability", f"{result['rain_probability_next_observation']:.1%}")
        st.caption(result["prediction_horizon"])
    else:
        st.warning("ML model is not ready. Collect historical observations and train the model first.")
except requests.RequestException:
    st.info("ML prediction endpoint is unavailable.")

with st.expander("Model evaluation"):
    try:
        st.json(requests.get(f"{API}/api/models/metrics", timeout=10).json())
    except requests.RequestException:
        st.write("Metrics endpoint unavailable.")
