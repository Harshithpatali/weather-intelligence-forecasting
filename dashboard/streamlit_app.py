import os,requests,pandas as pd,plotly.express as px,streamlit as st
st.set_page_config(page_title="Weather Intelligence",page_icon="🌦️",layout="wide")
API=os.getenv("API_BASE_URL","http://localhost:8000").rstrip("/")
st.title("🌦️ Weather Intelligence & Forecasting")
st.caption("A globally pretrained model · location-aware three-day inference")
with st.sidebar:
    place=st.text_input("Enter a city or location","Mysuru, Karnataka, India")
    run=st.button("Analyse location",type="primary",use_container_width=True)
if run:st.session_state["place"]=place
if "place" not in st.session_state:
    st.info("Enter a location to retrieve current weather and run the saved global model.")
    st.stop()
place=st.session_state["place"]
try:
    with st.spinner("Loading current and recent weather, then applying the saved global model..."):
        current=requests.get(f"{API}/api/weather/location",params={"place":place,"history_days":7},timeout=60)
        current.raise_for_status();weather=current.json()
        response=requests.get(f"{API}/api/predictions/three-day",params={"place":place},timeout=60)
        response.raise_for_status();result=response.json()
    loc=weather["location"];cur=weather["current"]
    st.subheader(f"{loc['name']}, {loc.get('state') or ''} · {loc['country']}")
    a,b,c,d=st.columns(4)
    a.metric("Current temperature",f"{cur['temperature_c']:.1f} °C")
    b.metric("Humidity",f"{cur['humidity_pct']}%")
    c.metric("Wind",f"{cur['wind_speed_ms']:.1f} m/s")
    d.metric("Pressure",f"{cur['pressure_hpa']} hPa")
    hist=pd.DataFrame(weather["history"])
    if not hist.empty:
        st.subheader("Recent weather context")
        st.plotly_chart(px.line(hist,x="date",y="temperature_c",markers=True,title="Recent daily temperature"),use_container_width=True)
        st.dataframe(hist,use_container_width=True)
    st.divider();st.header("Global ML forecast · Next 3 days")
    st.caption(f"Model: {result['model_version']} · {result['model_mode']}")
    forecast=pd.DataFrame(result["predictions"])
    st.dataframe(forecast,use_container_width=True)
    for col,label in [("temperature_c","Temperature (°C)"),("rainfall_probability","Rainfall probability"),
      ("rainfall_mm","Rainfall amount (mm)"),("humidity_pct","Humidity (%)"),("wind_speed_ms","Wind speed (m/s)")]:
        st.plotly_chart(px.line(forecast,x="date",y=col,markers=True,title=label),use_container_width=True)
    with st.expander("Global model validation metrics"):
        st.json(requests.get(f"{API}/api/models/metrics",timeout=20).json())
except requests.RequestException as e:
    st.error(f"Could not complete weather inference: {e}")
    if getattr(e,"response",None) is not None:st.code(e.response.text)
