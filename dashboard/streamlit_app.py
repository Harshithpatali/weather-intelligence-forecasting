import os,requests,pandas as pd,plotly.express as px,streamlit as st
st.set_page_config(page_title="Weather Intelligence",page_icon="🌦️",layout="wide")
API=os.getenv("API_BASE_URL","http://localhost:8000").rstrip("/")
st.title("🌦️ Weather Intelligence & Forecasting")
st.caption("Location-aware historical analysis and three-day machine-learning predictions")
with st.sidebar:
    place=st.text_input("Location","Mysuru, Karnataka, India")
    days=st.select_slider("Historical lookback",options=[5,10,15],value=10)
    run=st.button("Analyse location",type="primary",use_container_width=True)
if run: st.session_state["selection"]=(place,days)
if "selection" not in st.session_state:
    st.info("Enter a location, choose 5, 10 or 15 historical days, then analyse.")
    st.stop()
place,days=st.session_state["selection"]
try:
    with st.spinner("Resolving location and retrieving current and historical weather..."):
        r=requests.get(f"{API}/api/weather/location",params={"place":place,"history_days":days},timeout=180)
        r.raise_for_status(); payload=r.json()
    loc,cur=payload["location"],payload["current"]
    st.subheader(f"{loc['name']}, {loc.get('state') or ''} {loc['country']}")
    a,b,c,d=st.columns(4)
    a.metric("Temperature",f"{cur['temperature_c']:.1f} °C")
    b.metric("Humidity",f"{cur['humidity_pct']}%")
    c.metric("Wind",f"{cur['wind_speed_ms']} m/s")
    d.metric("Pressure",f"{cur['pressure_hpa']} hPa")
    st.caption(f"Historical data status: {payload['history_status']}")
    hist=pd.DataFrame(payload["history"])
    if not hist.empty:
        hist["date"]=pd.to_datetime(hist["observed_at"],unit="s",utc=True)
        st.subheader(f"Historical weather · {days} days")
        st.plotly_chart(px.line(hist,x="date",y="temperature_c",markers=True),use_container_width=True)
        st.dataframe(hist,use_container_width=True)
    else: st.warning("No historical records returned. Check your One Call historical API subscription.")
    st.divider();st.header("Three-day ML prediction")
    pr=requests.get(f"{API}/api/predictions/three-day",params={"place":loc["name"],"history_days":days},timeout=30)
    if pr.ok:
        result=pr.json();fc=pd.DataFrame(result["forecast"])
        st.caption(f"{result['model']} · stored records used: {result['history_records_used']}")
        st.dataframe(fc,use_container_width=True)
        for col,label in [("temperature_c","Temperature (°C)"),("humidity_pct","Humidity (%)"),
          ("wind_speed_ms","Wind speed (m/s)"),("rainfall_estimate_mm","Rainfall amount (mm)")]:
            st.plotly_chart(px.line(fc,x="date",y=col,markers=True,title=label),use_container_width=True)
        st.info("Rainfall amount is estimated; rainfall probability is not yet modelled.")
    else: st.warning("The three-day model is not trained or there is insufficient stored history. Train using a diverse multi-city historical dataset.")
except requests.RequestException as e:
    st.error(f"Weather request failed: {e}")
    if getattr(e,"response",None) is not None:st.code(e.response.text)
