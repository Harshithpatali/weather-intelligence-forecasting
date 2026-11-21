import os
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Weather Intelligence", page_icon="🌦️", layout="wide", initial_sidebar_state="collapsed")
API = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:1.5rem;padding-bottom:3rem}
[data-testid="stMetric"]{background:var(--secondary-background-color);padding:14px 16px;border-radius:14px;border:1px solid rgba(128,128,128,.16)}
[data-testid="stMetricLabel"] p{font-size:.86rem}
[data-testid="stTabs"] button{font-weight:600}
.muted{color:var(--secondary-text-color);font-size:.93rem}
.eyebrow{color:var(--secondary-text-color);font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}
.forecast-temp{font-size:1.8rem;font-weight:700;line-height:1.35}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather Intelligence")
st.markdown('<p class="muted">Your local weather, a clear three-day outlook, and useful forecast insights.</p>', unsafe_allow_html=True)

with st.form("location_form"):
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    with c1:
        place = st.text_input("Search for a city or town", value=st.session_state.get("place", "Mysuru, Karnataka, India"), placeholder="e.g. Mysuru, Karnataka", help="Enter a recognised location in India. Including the state helps distinguish places with similar names.")
    with c2:
        submitted = st.form_submit_button("Get forecast", type="primary", use_container_width=True)
st.caption("Try Mysuru, Bengaluru, Mumbai, Delhi or another Indian location.")

if submitted:
    if not place.strip():
        st.warning("Please enter a location.")
    else:
        try:
            with st.spinner("Loading weather and preparing your forecast… The first request may take longer while the free backend wakes up."):
                wr = requests.get(f"{API}/api/weather/location", params={"place": place.strip(), "history_days": 7}, timeout=90)
                wr.raise_for_status()
                weather_data = wr.json()
                fr = requests.get(f"{API}/api/predictions/three-day", params={"place": place.strip()}, timeout=90)
                fr.raise_for_status()
                forecast_data = fr.json()
            st.session_state["place"] = place.strip()
            st.session_state["weather_data"] = weather_data
            st.session_state["forecast_data"] = forecast_data
            st.session_state.pop("load_error", None)
        except requests.RequestException as exc:
            detail = str(exc)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text or detail
            st.session_state["load_error"] = detail
        except (KeyError, TypeError, ValueError) as exc:
            st.session_state["load_error"] = f"Unexpected response from the weather service: {exc}"

if st.session_state.get("load_error"):
    st.error("We couldn't load the forecast.")
    st.write(st.session_state["load_error"])
    st.info("Check the spelling and try again. If the backend has just woken up, wait a moment and submit again.")

weather = st.session_state.get("weather_data")
result = st.session_state.get("forecast_data")
if not weather or not result:
    st.divider()
    st.subheader("Your forecast starts here")
    st.write("Search for an Indian city or town to see current conditions, the next three days and recent weather trends.")
    a, b, c = st.columns(3)
    a.markdown("**🌡️ Current weather**\n\nTemperature, humidity and wind at a glance.")
    b.markdown("**📅 Three-day outlook**\n\nRain probability and model-adjusted estimates.")
    c.markdown("**📈 Weather trends**\n\nExplore recent conditions and model validation.")
    st.caption("Forecasts use a saved model. The app does not train a model when you open it.")
    st.stop()

location = weather.get("location", {})
current = weather.get("current", {})
predictions = result.get("predictions", [])
history_rows = weather.get("history", [])
location_name = ", ".join(str(v) for v in [location.get("name"), location.get("state"), location.get("country")] if v) or st.session_state.get("place", "Selected location")

def fmt(value, digits=1):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"

def day_label(value, horizon):
    try:
        return pd.to_datetime(value).strftime("%a, %d %b") if value else f"Day {horizon}"
    except (TypeError, ValueError):
        return f"Day {horizon}"

def icon_for_rain(value):
    try:
        p = float(value)
    except (TypeError, ValueError):
        p = 0
    return "🌧️" if p >= .6 else "🌦️" if p >= .3 else "☀️"

st.divider()
h1, h2 = st.columns([4, 1])
with h1:
    st.markdown('<div class="eyebrow">YOUR LOCATION</div>', unsafe_allow_html=True)
    st.subheader(f"📍 {location_name}")
    st.caption(f"Current conditions · {str(current.get('condition') or 'Latest available observation').title()}")
with h2:
    st.caption("Forecast status")
    st.success("Ready")

st.markdown("#### Current conditions")
m1, m2, m3, m4 = st.columns(4)
m1.metric("🌡️ Temperature", f"{fmt(current.get('temperature_c'))} °C")
m2.metric("Feels like", f"{fmt(current.get('feels_like_c'))} °C")
m3.metric("💧 Humidity", f"{fmt(current.get('humidity_pct'), 0)}%")
m4.metric("💨 Wind speed", f"{fmt(current.get('wind_speed_ms'))} m/s")

st.markdown("#### Your 3-day outlook")
st.caption("Model-adjusted estimates for approximately 24, 48 and 72 hours ahead.")
if not predictions:
    st.warning("The API did not return forecast values for this location.")
else:
    cols = st.columns(min(3, len(predictions)))
    for i, item in enumerate(predictions[:3]):
        with cols[i]:
            horizon = item.get("horizon_days", i + 1)
            probability = item.get("rainfall_probability", 0)
            try:
                rain_pct = max(0.0, min(100.0, float(probability) * 100))
            except (TypeError, ValueError):
                rain_pct = 0.0
            with st.container(border=True):
                st.caption(day_label(item.get("date"), horizon))
                st.markdown(f'<div class="forecast-temp">{icon_for_rain(probability)} {fmt(item.get("temperature_c"))} °C</div>', unsafe_allow_html=True)
                st.metric("Chance of rain", f"{rain_pct:.0f}%")
                st.progress(rain_pct / 100)
                st.caption(f"💧 {fmt(item.get('rainfall_mm'))} mm estimated rain")
                st.caption(f"Humidity {fmt(item.get('humidity_pct'), 0)}% · Wind {fmt(item.get('wind_speed_ms'))} m/s")

st.info("Rain chance is the estimated probability of measurable rainfall. Rainfall amount is a separate estimate in millimetres; a high chance does not guarantee rain at your exact address.")

tab_forecast, tab_history, tab_model = st.tabs(["📊 Forecast details", "🕘 Recent weather", "🧪 About this model"])

with tab_forecast:
    st.subheader("Compare the next three days")
    if predictions:
        forecast = pd.DataFrame(predictions)
        labels = {"horizon_days":"Days ahead","date":"Date","temperature_c":"Temperature (°C)","rainfall_probability":"Rain chance","rainfall_mm":"Rainfall (mm)","humidity_pct":"Humidity (%)","wind_speed_ms":"Wind (m/s)"}
        keys = [k for k in labels if k in forecast.columns]
        display = forecast[keys].rename(columns=labels).copy()
        if "Rain chance" in display:
            display["Rain chance"] = (pd.to_numeric(display["Rain chance"], errors="coerce") * 100).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True, hide_index=True)
        left, right = st.columns(2)
        with left:
            if {"date", "temperature_c"}.issubset(forecast.columns):
                fig = px.line(forecast, x="date", y="temperature_c", markers=True, labels={"date":"Date","temperature_c":"Temperature (°C)"}, title="🌡️ Temperature trend")
                fig.update_layout(margin=dict(l=10,r=10,t=55,b=10),height=300)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            if {"date", "rainfall_probability"}.issubset(forecast.columns):
                chart = forecast.copy()
                chart["rain_chance_pct"] = pd.to_numeric(chart["rainfall_probability"], errors="coerce") * 100
                fig = px.bar(chart, x="date", y="rain_chance_pct", labels={"date":"Date","rain_chance_pct":"Rain chance (%)"}, title="🌧️ Rain probability", range_y=[0,100])
                fig.update_layout(margin=dict(l=10,r=10,t=55,b=10),height=300)
                st.plotly_chart(fig, use_container_width=True)
        with st.expander("What do these numbers mean?"):
            st.markdown("- **Temperature:** predicted air temperature in °C.\n- **Rain chance:** probability of measurable rainfall.\n- **Rainfall:** estimated amount of rain in millimetres.\n- **Humidity:** relative humidity as a percentage.\n- **Wind:** wind speed in metres per second.")
    else:
        st.info("No forecast details are available yet.")

with tab_history:
    st.subheader("Recent weather")
    history = pd.DataFrame(history_rows)
    if not history.empty:
        if {"date", "temperature_c"}.issubset(history.columns):
            fig = px.line(history, x="date", y="temperature_c", markers=True, labels={"date":"Date","temperature_c":"Temperature (°C)"}, title="Temperature over the last 7 days")
            fig.update_layout(margin=dict(l=10,r=10,t=55,b=10),height=320)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(history, use_container_width=True, hide_index=True)
    else:
        st.info("Recent weather history is not available for this location.")

with tab_model:
    st.subheader("How this forecast works")
    st.write("A saved India-wide model calibrates the underlying forecast for 1-, 2- and 3-day lead times. Training happens offline, not when you request a forecast.")
    st.caption(f"Model version: {result.get('model_version', 'Not reported')}")
    st.caption(result.get("model_mode", "India fixed-lead calibrated model"))
    st.warning("Validation targets were Open-Meteo historical reanalysis, not independent weather-station observations. These results are reanalysis-based validation, not proof of station-level accuracy.")
    with st.expander("View model validation metrics"):
        try:
            response = requests.get(f"{API}/api/models/metrics", timeout=30)
            response.raise_for_status()
            st.json(response.json())
        except requests.RequestException as exc:
            st.error(f"Could not load model metrics: {exc}")

st.divider()
st.caption("Weather Intelligence · Forecasts are estimates and should not be used as the sole basis for safety-critical decisions.")
