import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Diamond Dynamics", layout="wide")

# ---------------- LOAD MODELS ----------------
price_model = joblib.load("models/price_model.pkl")
cluster_model = joblib.load("models/cluster_model.pkl")
scaler_cluster = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

with open("models/cluster_names.json", "r") as f:
    cluster_names = json.load(f)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>💎 Diamond Dynamics</h1>
    <h4 style='text-align: center;'>Price Prediction & Market Segmentation</h4>
    """,
    unsafe_allow_html=True
)

st.write("Enter diamond details in the sidebar to get predictions.")

# ---------------- PRESET DIAMOND PROFILES ----------------
st.sidebar.subheader("💡 Try Sample Diamonds")

preset_options = {
    "💍 Budget Small Diamond": {"carat": 0.3, "x": 4.2, "y": 4.1, "z": 2.5, "cut": "Good", "color": "H", "clarity": "SI2"},
    "✨ Everyday Wear Diamond": {"carat": 0.6, "x": 5.5, "y": 5.4, "z": 3.2, "cut": "Very Good", "color": "G", "clarity": "SI1"},
    "🎁 Mid-Range Gift Diamond": {"carat": 1.0, "x": 6.5, "y": 6.4, "z": 4.0, "cut": "Premium", "color": "F", "clarity": "VS2"},
    "💎 Premium Engagement Diamond": {"carat": 1.8, "x": 7.8, "y": 7.7, "z": 4.8, "cut": "Ideal", "color": "E", "clarity": "VVS2"},
    "👑 Luxury Showcase Diamond": {"carat": 2.5, "x": 8.8, "y": 8.7, "z": 5.5, "cut": "Ideal", "color": "D", "clarity": "IF"}
}

selected_preset = st.sidebar.selectbox("Choose a preset", ["Custom Input"] + list(preset_options.keys()))


# ---------------- SIDEBAR INPUTS ----------------
if selected_preset != "Custom Input":
    preset = preset_options[selected_preset]
    carat = st.sidebar.number_input("Carat", value=preset["carat"])
    x = st.sidebar.number_input("Length (mm)", value=preset["x"])
    y = st.sidebar.number_input("Width (mm)", value=preset["y"])
    z = st.sidebar.number_input("Depth (mm)", value=preset["z"])
    cut = st.sidebar.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'], index=['Fair','Good','Very Good','Premium','Ideal'].index(preset["cut"]))
    color = st.sidebar.selectbox("Color", ['J','I','H','G','F','E','D'], index=['J','I','H','G','F','E','D'].index(preset["color"]))
    clarity = st.sidebar.selectbox("Clarity", ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'], index=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'].index(preset["clarity"]))
else:
    carat = st.sidebar.number_input("Carat", 0.1, 5.0, 1.0)
    x = st.sidebar.number_input("Length (mm)", 0.1, 10.0, 5.0)
    y = st.sidebar.number_input("Width (mm)", 0.1, 10.0, 5.0)
    z = st.sidebar.number_input("Depth (mm)", 0.1, 10.0, 3.0)
    cut = st.sidebar.selectbox("Cut", ['Fair','Good','Very Good','Premium','Ideal'])
    color = st.sidebar.selectbox("Color", ['J','I','H','G','F','E','D'])
    clarity = st.sidebar.selectbox("Clarity", ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
predict_btn = st.sidebar.button("Predict")

# ---------------- FEATURE ENGINEERING ----------------
def engineer_features(carat, x, y, z):
    volume = x * y * z
    price_per_carat = 0
    dimension_ratio = (x + y) / (2 * z)

    if carat < 0.5:
        carat_category = 0
    elif carat < 1.5:
        carat_category = 1
    else:
        carat_category = 2

    return volume, price_per_carat, dimension_ratio, carat_category

# ---------------- PREDICTION ----------------
if predict_btn:

    cat_input = pd.DataFrame([[cut, color, clarity]], columns=['cut','color','clarity'])
    encoded = encoder.transform(cat_input)

    volume, price_per_carat, dimension_ratio, carat_category = engineer_features(carat, x, y, z)

    features = np.array([[carat,x,y,z,encoded[0][0],encoded[0][1],encoded[0][2],
                          volume,price_per_carat,dimension_ratio,carat_category]])

    # Predictions
    price = price_model.predict(features)[0]
    cluster = cluster_model.predict(scaler_cluster.transform(features))[0]
    cluster_label = cluster_names[str(cluster)]

    # ---------------- DISPLAY RESULTS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background-color:#E8F6F3;">
                <h3 style="color:#117864;">💰 Estimated Price</h3>
                <h2 style="color:#0B5345;">₹ {int(price):,}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="padding:20px; border-radius:10px; background-color:#FEF5E7;">
                <h3 style="color:#B9770E;">📦 Market Segment</h3>
                <h2 style="color:#7D6608;">{cluster_label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

else:
    st.info("👈 Enter values in the sidebar and click **Predict**")
