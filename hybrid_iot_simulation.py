"""
Enhanced Hybrid IoT Architecture Simulation — Smart Energy Monitoring
Multi-node + RandomForest forecasting version for ICASET demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import os

# --- CONFIG ---
DB_PATH = "smartcity_cloud.db"
CLOUD_BATCH_SIZE = 10
FOG_FILTER_WINDOW = 4
LATENCY = {"edge": 10, "fog": 40, "cloud": 200}  # ms
ENERGY = {"edge_tx": 0.15, "fog_proc": 0.1, "cloud_tx": 0.4}

NODES = ["Building-A", "Building-B", "Smart-Streetlights"]

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cloud_data(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            node_id TEXT,
            raw REAL,
            fog REAL
        )
    """)
    conn.commit()
    return conn

# --- SIMULATION FUNCTIONS ---
def generate_reading(base_load=1.0, hour=None):
    hour = hour or datetime.utcnow().hour
    pattern = 1.0 + 0.6 * np.exp(-((hour - 15) ** 2) / (2 * 4**2))
    noise = np.random.normal(0, 0.05)
    return round(base_load * pattern * (1 + noise), 3)

def fog_filter(data):
    ser = pd.Series(data)
    return ser.rolling(window=FOG_FILTER_WINDOW, min_periods=1).mean().iloc[-1]

def randomforest_forecast(df):
    if len(df) < 15:
        return None, None
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")
    df["y"] = df["fog"]
    for lag in range(1, 5):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, 5)]].values
    y = df["y"].values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    recent = list(df["y"].tail(4))
    preds = []
    for _ in range(6):
        p = float(model.predict(np.array(recent[-4:]).reshape(1, -1))[0])
        preds.append(p)
        recent.append(p)
    return preds, model

def compute_metrics(samples, fog_batches):
    latency_hybrid = samples * (LATENCY["edge"] + LATENCY["fog"]) + fog_batches * LATENCY["cloud"]
    latency_direct = samples * (LATENCY["edge"] + LATENCY["cloud"])
    energy_hybrid = samples * (ENERGY["edge_tx"] + ENERGY["fog_proc"]) + fog_batches * ENERGY["cloud_tx"]
    energy_direct = samples * ENERGY["cloud_tx"]
    volume_hybrid = fog_batches
    volume_direct = samples
    return latency_hybrid, latency_direct, energy_hybrid, energy_direct, volume_hybrid, volume_direct

# --- STREAMLIT UI ---
st.set_page_config(page_title="Hybrid IoT Smart Energy Simulation", layout="wide")
st.title("⚡ Hybrid IoT Architecture Simulation — Smart Energy Monitoring (ICASET 2025)")

st.markdown("""
### 🎯 Objective
Simulate a **Hybrid IoT Framework (Edge → Fog → Cloud)** for smart energy management across multiple nodes,  
and visualize how it improves **latency**, **energy efficiency**, and **data transfer volume** compared to a traditional system.
""")

conn = init_db()
cur = conn.cursor()

# Sidebar
st.sidebar.header("⚙️ Simulation Controls")
steps = st.sidebar.slider("Simulation Steps", 20, 300, 100, 10)
delay = st.sidebar.slider("Delay (seconds per step)", 0.1, 1.0, 0.3, 0.1)
clear = st.sidebar.button("Clear Cloud Database")
if clear:
    cur.execute("DELETE FROM cloud_data")
    conn.commit()
    st.sidebar.success("Database cleared.")

start = st.sidebar.button("Run Simulation")

# Layout
col1, col2 = st.columns(2)

if start:
    st.info("Running simulation...")
    fog_buffer = {n: [] for n in NODES}
    samples = 0
    fog_batches = 0
    progress = st.progress(0)

    for step in range(steps):
        now = datetime.utcnow()
        for node in NODES:
            val = generate_reading(base_load=np.random.uniform(0.8, 1.5), hour=now.hour)
            fog_buffer[node].append(val)
            samples += 1
            # When enough samples gathered -> fog filter + upload to cloud
            if len(fog_buffer[node]) >= CLOUD_BATCH_SIZE:
                fog_val = fog_filter(fog_buffer[node])
                for v in fog_buffer[node]:
                    cur.execute("INSERT INTO cloud_data(ts,node_id,raw,fog) VALUES(?,?,?,?)",
                                (now.isoformat(), node, float(v), float(fog_val)))
                conn.commit()
                fog_batches += 1
                fog_buffer[node] = []
        progress.progress((step + 1) / steps)
        time.sleep(delay)

    # Fetch cloud data
    df = pd.read_sql_query("SELECT * FROM cloud_data", conn)
    df["ts"] = pd.to_datetime(df["ts"])

    st.success(f"Simulation complete — {len(df)} total records stored in cloud database.")

    # Metrics
    lat_h, lat_d, en_h, en_d, vol_h, vol_d = compute_metrics(samples, fog_batches)
    metrics = pd.DataFrame({
        "Metric": ["Latency (ms)", "Energy (J)", "Cloud Data Volume"],
        "Without Hybrid": [lat_d, en_d, vol_d],
        "With Hybrid": [lat_h, en_h, vol_h]
    })
    st.subheader("📊 Comparative Metrics")
    st.dataframe(metrics)

    # Visualization
    with col1:
        fig = go.Figure()
        for node in NODES:
            node_df = df[df["node_id"] == node]
            fig.add_trace(go.Scatter(x=node_df["ts"], y=node_df["fog"], mode="lines+markers", name=node))
        fig.update_layout(title="Fog-Processed Energy Data (Cloud View)", xaxis_title="Time", yaxis_title="kW")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        preds, model = randomforest_forecast(df)
        if preds:
            last_ts = df["ts"].max()
            future_ts = [last_ts + pd.Timedelta(minutes=i) for i in range(len(preds))]
            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(x=df["ts"].tail(30), y=df["fog"].tail(30), name="Historical (Fog)"))
            pred_fig.add_trace(go.Scatter(x=future_ts, y=preds, mode="lines+markers", name="Forecast"))
            pred_fig.update_layout(title="AI-Based Energy Demand Forecast", xaxis_title="Time", yaxis_title="Predicted kW")
            st.plotly_chart(pred_fig, use_container_width=True)
        else:
            st.warning("Not enough data for forecast yet.")

    st.caption("Simulation finished. Database: smartcity_cloud.db")
else:
    st.info("Press **Run Simulation** to start the hybrid IoT demonstration.")
