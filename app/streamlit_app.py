import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved models - these were trained in the notebook, not retrained here
import os

# Get the folder this script itself lives in, regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model_a_lite.pkl'))
kmeans = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
all_times = pd.read_csv(os.path.join(BASE_DIR, 'all_finish_times.csv'))['total_time_secs']

st.title("🏋️ Hyrox Performance Predictor")
st.caption("Trained on 90,000+ real Hyrox race results")

st.markdown("""
This tool estimates your Hyrox finish time and athlete archetype based on a 
small set of inputs. It uses a simplified version of a larger model — built 
this way so you don't need detailed per-station splits to try it.

**Note:** predictions are estimates based on patterns in race data, not a 
guarantee — actual performance depends on many factors not captured here.
""")

st.header("Tell us about your race (or a hypothetical one)")

col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox("Age group", 
        ['16-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74'])
    gender = st.radio("Gender", ['male', 'female'], horizontal=True)

with col2:
    total_run_min = st.number_input("Total running time (minutes)", min_value=35, max_value=120, value=45)
    total_work_min = st.number_input("Total station time (minutes)", min_value=10, max_value=120, value=38)

fastest_station = st.number_input("Fastest station time (seconds)", min_value=30, max_value=900, value=180)
slowest_station = st.number_input("Slowest station time (seconds)", min_value=30, max_value=900, value=350)

if st.button("Predict my Hyrox performance"):

    # Convert inputs to match training
    total_run_secs = total_run_min * 60
    total_work_secs = total_work_min * 60

    run_to_work_ratio = total_run_secs / total_work_secs
    proxy_consistency = slowest_station - fastest_station

    age_order = ['16-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74']
    age_group_encoded = age_order.index(age_group)

    gender_encoded = 0 if gender == 'male' else 1

    input_data = pd.DataFrame(
        [[age_group_encoded, gender_encoded, total_run_secs, run_to_work_ratio, proxy_consistency]],
        columns=['age_group_encoded', 'gender_encoded', 'total_run_secs', 'run_to_work_ratio', 'proxy_consistency']
    )

    predicted_secs = model.predict(input_data)[0]
    percentile = (all_times < predicted_secs).mean() * 100
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
    st.write(f"This would put you faster than approximately **{percentile:.0f}%** of athletes in our dataset.")
    st.info(f"**Your archetype: {archetype_names[archetype]}**\n\n{archetype_descriptions[archetype]}")