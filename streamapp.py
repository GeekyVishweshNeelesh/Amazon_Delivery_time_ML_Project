# -------------------------------------------------------
# 🚚 Streamlit App: Delivery Time Prediction + MLflow Viewer
# -------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# -----------------------------
# Load Pre-trained Models
# -----------------------------
model_lr = joblib.load("/home/vishwesh/Documents/Labmentix_Internship/Project_10/linear_model.joblib")
model_rf = joblib.load("/home/vishwesh/Documents/Labmentix_Internship/Project_10/rf_model.joblib")
model_gb = joblib.load("/home/vishwesh/Documents/Labmentix_Internship/Project_10/gb_model.joblib")

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Delivery Time Prediction + MLflow", layout="wide")
st.title("🚚 Delivery Time Prediction App with MLflow Tracking")
st.markdown("Predict delivery time using multiple ML models and track them using MLflow.")

# -----------------------------
# 1️⃣ User Input Section
# -----------------------------
st.subheader("Order Details Input")
with st.form("order_form"):
    col1, col2 = st.columns(2)
    with col1:
        Agent_Age = st.number_input("Agent Age", 18, 70, 30)
        Agent_Rating = st.number_input("Agent Rating", 1.0, 5.0, 4.5, 0.1)
        Store_Latitude = st.number_input("Store Latitude", 12.9716)
        Store_Longitude = st.number_input("Store Longitude", 77.5946)
        Drop_Latitude = st.number_input("Drop Latitude", 12.2958)
        Drop_Longitude = st.number_input("Drop Longitude", 76.6394)
    with col2:
        Order_Day = st.number_input("Order Day", 1, 31, 1)
        Order_Month = st.number_input("Order Month", 1, 12, 1)
        Order_Weekday = st.number_input("Order Weekday (0=Mon)", 0, 6, 0)
        Order_Time = st.number_input("Order Hour (0-23)", 0, 23, 12)
        Pickup_Time = st.number_input("Pickup Hour (0-23)", 0, 23, 13)

    Traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    Weather = st.selectbox("Weather", ["Sunny", "Rainy", "Cloudy", "Foggy"])
    Vehicle = st.selectbox("Vehicle", ["Bike", "Car", "Van"])
    Area = st.selectbox("Area", ["Urban", "Suburban"])
    Category = st.selectbox("Category", ["Food", "Grocery", "Electronics", "Clothing"])

    submitted = st.form_submit_button("Predict Delivery Time")

# -----------------------------
# 2️⃣ Prediction & MLflow Logging
# -----------------------------
if submitted:
    # Prepare input dataframe
    input_df = pd.DataFrame({
        "Agent_Age": [Agent_Age],
        "Agent_Rating": [Agent_Rating],
        "Store_Latitude": [Store_Latitude],
        "Store_Longitude": [Store_Longitude],
        "Drop_Latitude": [Drop_Latitude],
        "Drop_Longitude": [Drop_Longitude],
        "Order_Day": [Order_Day],
        "Order_Month": [Order_Month],
        "Order_Weekday": [Order_Weekday],
        "Order_Time": [Order_Time],
        "Pickup_Time": [Pickup_Time],
        "Traffic_High": [int(Traffic=="High")],
        "Traffic_Medium": [int(Traffic=="Medium")],
        "Weather_Cloudy": [int(Weather=="Cloudy")],
        "Weather_Foggy": [int(Weather=="Foggy")],
        "Weather_Rainy": [int(Weather=="Rainy")],
        "Weather_Sunny": [int(Weather=="Sunny")],
        "Vehicle_Car": [int(Vehicle=="Car")],
        "Vehicle_Van": [int(Vehicle=="Van")],
        "Area_Urban": [int(Area=="Urban")],
        "Category_Grocery": [int(Category=="Grocery")],
        "Category_Electronics": [int(Category=="Electronics")],
        "Category_Clothing": [int(Category=="Clothing")],
        "Category_Food": [int(Category=="Food")]
    })
    input_df = input_df.reindex(columns=model_rf.feature_names_in_, fill_value=0)

    # Predictions
    pred_lr = model_lr.predict(input_df)[0]
    pred_rf = model_rf.predict(input_df)[0]
    pred_gb = model_gb.predict(input_df)[0]

    # Display Predictions
    st.subheader("📊 Predicted Delivery Time (minutes)")
    st.success(f"Linear Regression: {pred_lr:.2f}")
    st.success(f"Random Forest: {pred_rf:.2f}")
    st.success(f"Gradient Boosting: {pred_gb:.2f}")

    # Model Comparison Chart (medium size)
    comp_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
        "Prediction": [pred_lr, pred_rf, pred_gb]
    })
    plt.figure(figsize=(5,3))
    sns.barplot(x="Model", y="Prediction", data=comp_df, palette=["#3498db","#2ecc71","#e74c3c"])
    plt.ylabel("Delivery Time (minutes)", fontsize=10)
    plt.title("🚀 Model Prediction Comparison", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    for index, row in comp_df.iterrows():
        plt.text(index, row.Prediction + 0.3, f"{row.Prediction:.2f}", ha='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    st.pyplot(plt)

    # MLflow Logging
    mlflow.set_experiment("Delivery_Time_Models")
    with mlflow.start_run():
        mlflow.log_params({
            "Agent_Age": Agent_Age,
            "Agent_Rating": Agent_Rating,
            "Traffic": Traffic,
            "Weather": Weather,
            "Vehicle": Vehicle,
            "Area": Area,
            "Category": Category
        })
        mlflow.log_metrics({
            "Prediction_LR": float(pred_lr),
            "Prediction_RF": float(pred_rf),
            "Prediction_GB": float(pred_gb)
        })
        mlflow.sklearn.log_model(model_lr, "Linear_Regression_Model")
        mlflow.sklearn.log_model(model_rf, "Random_Forest_Model")
        mlflow.sklearn.log_model(model_gb, "Gradient_Boosting_Model")
    st.info("✅ Prediction run logged successfully in MLflow!")

# -----------------------------
# 3️⃣ MLflow Metrics Viewer
# -----------------------------
st.subheader("📈 MLflow Run Metrics Viewer")
client = MlflowClient()
experiment = client.get_experiment_by_name("Delivery_Time_Models")
if experiment:
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=10)
    if runs:
        mlflow_data = []
        for run in runs:
            mlflow_data.append({
                "Run ID": run.info.run_id,
                "Start Time": pd.to_datetime(run.info.start_time, unit='ms'),
                "Prediction_LR": run.data.metrics.get("Prediction_LR", None),
                "Prediction_RF": run.data.metrics.get("Prediction_RF", None),
                "Prediction_GB": run.data.metrics.get("Prediction_GB", None),
            })
        df_mlflow = pd.DataFrame(mlflow_data)
        st.dataframe(df_mlflow)

        # Line chart for comparison
        st.line_chart(df_mlflow[["Prediction_LR", "Prediction_RF", "Prediction_GB"]])
    else:
        st.info("No MLflow runs found.")
else:
    st.warning("MLflow experiment not found.")
