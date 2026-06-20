import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved models - these were trained in the notebook, not retrained here
model = joblib.load('model_a_lite.pkl')
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Hyrox Performance Predictor")
st.caption("Trained on 90,000+ real Hyrox race results")

st.header("Tell us about your race (or a hypothetical one)")

col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox("Age group", 
        ['16-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74'])
    gender = st.radio("Gender", ['male', 'female'], horizontal=True)

with col2:
    total_run_min = st.number_input("Total running time (minutes)", min_value=10, max_value=120, value=45)
    total_work_min = st.number_input("Total station time (minutes)", min_value=10, max_value=120, value=38)

fastest_station = st.number_input("Fastest station time (seconds)", min_value=30, max_value=900, value=180)
slowest_station = st.number_input("Slowest station time (seconds)", min_value=30, max_value=900, value=350)

if st.button("Predict my Hyrox performance"):

    # Convert inputs to match training format
    total_run_secs = total_run_min * 60
    total_work_secs = total_work_min * 60

    run_to_work_ratio = total_run_secs / total_work_secs
    proxy_consistency = slowest_station - fastest_station

    age_order = ['16-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74']
    age_group_encoded = age_order.index(age_group)

    gender_encoded = 0 if gender == 'male' else 1

    input_data = pd.DataFrame(
        [[age_group_encoded, gender_encoded, run_to_work_ratio, proxy_consistency]],
        columns=['age_group_encoded', 'gender_encoded', 'run_to_work_ratio', 'proxy_consistency']
    )

    predicted_secs = model.predict(input_data)[0]
    pred_min, pred_sec = divmod(int(predicted_secs), 60)

    # --- cluster code starts here ---
    cluster_input = pd.DataFrame(
        [[run_to_work_ratio, proxy_consistency]],
        columns=['run_to_work_ratio', 'station_consistency']
    )
    cluster_input_scaled = scaler.transform(cluster_input)

    archetype = kmeans.predict(cluster_input_scaled)[0]

    archetype_names = {
        0: 'Balanced & Consistent',
        1: 'Inconsistent (has weak points)',
        2: 'Runner-Leaning'
    }
    archetype_descriptions = {
        0: "You perform evenly across all stations — no major weak points, which tends to mean faster overall times.",
        1: "Your station performance varies a lot between exercises — working on your weakest stations could meaningfully improve your time.",
        2: "You lean toward being stronger on the run than on stations, relative to other athletes."
    }

    # --- both results displayed at the end, once, in order ---
    st.success(f"Predicted finish time: {pred_min} min {pred_sec} sec")
    st.info(f"**Your archetype: {archetype_names[archetype]}**\n\n{archetype_descriptions[archetype]}")