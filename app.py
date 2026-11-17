from pathlib import Path
import json,joblib,pandas as pd,streamlit as st
R=Path(__file__).resolve().parent; st.set_page_config(page_title="Smart Traffic AI",page_icon="🚦")
st.title("🚦 Smart Traffic Congestion Predictor")
st.write("Estimate congestion and average speed from road and weather conditions.")
if not (R/"models"/"congestion_model.joblib").exists(): st.warning("Run `python src/train_model.py` first."); st.stop()
clf=joblib.load(R/"models"/"congestion_model.joblib"); reg=joblib.load(R/"models"/"speed_model.joblib")
hour=st.slider("Hour",0,23,8); day=st.selectbox("Day (0=Mon ... 6=Sun)",range(7)); vehicles=st.slider("Vehicle count",20,250,150)
rain=st.selectbox("Rain",["No","Yes"]); vis=st.slider("Visibility (km)",1.0,10.0,7.0); road=st.selectbox("Road type",["Residential","Arterial","Highway"])
if st.button("Predict traffic",type="primary"):
 x=pd.DataFrame([dict(hour=hour,day_of_week=day,vehicle_count=vehicles,rain=int(rain=="Yes"),visibility_km=vis,road_type=road)])
 st.subheader("Congestion: "+clf.predict(x)[0]); st.metric("Estimated speed",f"{reg.predict(x)[0]:.1f} km/h")
 st.info("Educational prediction using synthetic data; not a live traffic or safety alert.")
m=R/"models"/"metrics.json"
if m.exists(): st.caption("Test metrics"); st.json(json.loads(m.read_text()))
