# --- DLL FIX FOR WINDOWS [WinError 1114] ---
# This block must be at the very top, before any other imports
import os
import torch 
# Set this environment variable to prevent Intel MKL conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple

# Darts Imports
from darts import TimeSeries
from darts.utils.timeseries_generation import (
    sine_timeseries, linear_timeseries, random_walk_timeseries, constant_timeseries
)
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers.reconciliation import MinTReconciliator
from darts.metrics import mae, mape
from darts import concatenate

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Deep Learning Hierarchical Forecasting",
    layout="wide",
    page_icon="📈"
)

st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .stAlert {margin-top: 1rem;}
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def ts_to_df(ts: TimeSeries) -> pd.DataFrame:
    """Safe conversion of Darts TimeSeries to Pandas DataFrame."""
    return pd.DataFrame(ts.values(), index=ts.time_index, columns=ts.components)

# --- BACKEND LOGIC (Cached) ---

class HierarchicalDataGenerator:
    """
    Generates synthetic hierarchical sales data consistent with a retail tree:
    Global -> Region -> Product
    Includes 'Cold Start' simulation.
    """
    def __init__(self, length: int = 100):
        self.length = length
        
    def generate(self) -> Tuple[Dict[str, TimeSeries], Dict[str, List[str]]]:
        # Define the list of countries/regions
        countries = ["US", "EU", "India", "UAE", "China", "SouthKorea", "Japan", "Singapore"]
        
        all_series = {}
        hierarchy = {}
        global_total = None
        
        print(f"Generating data for: {countries}")

        for country in countries:
            # --- 1. Generate Parameters (Randomized for variety) ---
            # Random trends and seasonality per country
            trend_start = np.random.randint(10, 50)
            trend_end = np.random.randint(50, 150)
            freq = np.random.uniform(0.05, 0.15)
            noise_scale = np.random.uniform(1, 3)

            # --- 2. Electronics (Trend + Seasonality) ---
            elec = (linear_timeseries(start_value=trend_start, end_value=trend_end, length=self.length) + 
                    sine_timeseries(value_frequency=freq, value_amplitude=10, length=self.length) + 
                    random_walk_timeseries(length=self.length) * noise_scale)
            
            # --- 3. Clothing (Volatile or Cold Start) ---
            # 25% chance of being a "Cold Start" product (new launch)
            is_cold_start = np.random.random() < 0.25
            
            if is_cold_start:
                # Create a series that is 0 for the first 30% of the timeline
                # FIX: We must generate the full TimeSeries first to preserve the DatetimeIndex
                base_ts = linear_timeseries(start_value=trend_end, end_value=trend_start, length=self.length)
                base_vals = base_ts.values()
                start_idx = int(self.length * 0.3)
                base_vals[:start_idx] = 0
                
                # Reconstruct using the original time index (Fixes ValueError in concatenation)
                cloth = TimeSeries.from_times_and_values(base_ts.time_index, base_vals)
            else:
                cloth = (linear_timeseries(start_value=trend_end, end_value=trend_start, length=self.length) + 
                         random_walk_timeseries(length=self.length) * (noise_scale * 1.5))

            # --- 4. Country Aggregation ---
            country_total = elec + cloth
            
            # --- 5. Store in Dictionary ---
            all_series[country] = country_total
            all_series[f"{country}_Elec"] = elec
            all_series[f"{country}_Cloth"] = cloth
            
            # --- 6. Build Hierarchy Tree (Child -> [Parents]) ---
            hierarchy[f"{country}_Elec"] = [country]
            hierarchy[f"{country}_Cloth"] = [country]
            hierarchy[country] = ["Total"]
            
            # --- 7. Accumulate Global Total ---
            if global_total is None:
                global_total = country_total
            else:
                global_total = global_total + country_total

        # Add Global Total to the series list
        all_series["Total"] = global_total
        
        return all_series, hierarchy

@st.cache_resource
def train_model(_train_series_list, input_chunk, output_chunk, epochs):
    """
    Trains the N-BEATS model. Cached to avoid retraining on simple UI refreshes.
    Prefixing arguments with '_' tells Streamlit NOT to hash them.
    """
    model = NBEATSModel(
        input_chunk_length=input_chunk,
        output_chunk_length=output_chunk,
        n_epochs=epochs,
        random_state=42,
        force_reset=True,
        pl_trainer_kwargs={"accelerator": "cpu"},
        save_checkpoints=False
    )
    model.fit(_train_series_list, verbose=False)
    return model

def reconcile_forecasts(base_forecasts, train_series, hierarchy):
    """
    Applies MinT (Minimum Trace) Reconciliation.
    """
    # 1. Prepare Multivariate Series for Darts
    # Dynamically fetch all keys from the training data to ensure we capture all countries
    cols = list(train_series.keys())
    
    # Helper to stack list of TimeSeries into one Multivariate TimeSeries
    def to_multivariate(series_dict):
        # Sort by predefined col list to ensure alignment
        s_list = [series_dict[c] for c in cols]
        joined = concatenate(s_list, axis="component")
        return joined.with_columns_renamed(joined.components, cols)

    train_multi = to_multivariate(train_series)
    pred_multi = to_multivariate(base_forecasts)

    # 2. Add Hierarchy
    train_multi = train_multi.with_hierarchy(hierarchy)
    pred_multi = pred_multi.with_hierarchy(hierarchy)

    # 3. Apply MinT (WLS with Variance scaling)
    reconciler = MinTReconciliator(method="wls_val")
    reconciler.fit(train_multi)
    reconciled_pred = reconciler.transform(pred_multi)
    
    return pred_multi, reconciled_pred

# --- FRONTEND UI ---

# Sidebar
st.sidebar.title("🛠 Control Panel")
data_len = st.sidebar.slider("Dataset Length (Days)", 60, 365, 150)
epochs = st.sidebar.slider("Training Epochs (N-BEATS)", 1, 50, 5)
input_chunk = st.sidebar.slider("Lookback Window", 12, 60, 24)
output_chunk = st.sidebar.slider("Forecast Horizon", 1, 30, 12)

st.sidebar.markdown("---")

# Title Area
st.title("📦 Hierarchical Demand Forecasting")
st.markdown("Global $\\to$ Region (India, China, UAE, etc) $\\to$ Product Level forecasting with **N-BEATS** and **MinT Reconciliation**.")

# 1. Data Generation
gen = HierarchicalDataGenerator(length=data_len)
all_series, hierarchy = gen.generate()

# Split
val_len = output_chunk
train_dict = {k: v[:-val_len] for k, v in all_series.items()}
val_dict = {k: v[-val_len:] for k, v in all_series.items()}

# Preprocessing (Scaling)
scaler = Scaler()
train_scaled_list = scaler.fit_transform(list(train_dict.values()))

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Explorer", "🧠 Model Training", "🎯 Forecast & Reconciliation"])

with tab1:
    st.subheader("Hierarchy Visualization")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Tree Structure:**")
        st.code("""
Total
├── India
│   ├── India_Elec
│   └── India_Cloth
├── UAE
│   ├── UAE_Elec
│   └── ...
├── China
├── SouthKorea
├── Japan
├── Singapore
├── US
└── EU
        """)
        
        selected_series = st.selectbox("Select Series to Inspect", list(all_series.keys()), index=0)
    
    with col2:
        # Simple plotly chart
        df_plot = ts_to_df(all_series[selected_series])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot.iloc[:,0], mode='lines', name='Sales'))
        fig.update_layout(title=f"Raw Data: {selected_series}", height=350)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Deep Learning Engine (N-BEATS)")
    
    if st.button("Train Global Model"):
        with st.spinner("Training N-BEATS on all levels simultaneously..."):
            # Train
            model = train_model(train_scaled_list, input_chunk, output_chunk, epochs)
            
            # Predict
            pred_scaled_list = model.predict(n=val_len, series=train_scaled_list)
            pred_list = scaler.inverse_transform(pred_scaled_list)
            
            # Repack into dict for easier handling
            base_forecasts = {k: v for k, v in zip(train_dict.keys(), pred_list)}
            
            # Store in session state to pass to next tab
            st.session_state['base_forecasts'] = base_forecasts
            st.session_state['model_trained'] = True
            
        st.success("Training Complete! Model has learned patterns across all hierarchy levels.")
    else:
        st.warning("Click the button above to train the model.")

with tab3:
    if 'model_trained' in st.session_state:
        st.subheader("Reconciliation Analysis")
        
        base_forecasts = st.session_state['base_forecasts']
        
        # Run Reconciliation
        with st.spinner("Solving Linear Algebra for MinT Reconciliation..."):
            pred_multi, recon_multi = reconcile_forecasts(base_forecasts, train_dict, hierarchy)
        
        # Metrics Calculation
        st.markdown("### Performance Comparison (MAPE)")
        
        # Prepare metrics table with a sample of interesting components
        results = []
        # Dynamic sampling of components to display
        components_to_show = ["Total", "India", "China_Elec", "UAE_Cloth", "Singapore"]
        # Ensure they exist (in case names change)
        components_to_show = [c for c in components_to_show if c in all_series]
        
        for comp in components_to_show:
            actual = val_dict[comp]
            base = pred_multi[comp]
            recon = recon_multi[comp]
            
            mape_base = mape(actual, base)
            mape_recon = mape(actual, recon)
            improvement = mape_base - mape_recon
            
            results.append({
                "Level": comp,
                "Base MAPE (%)": round(mape_base, 2),
                "Reconciled MAPE (%)": round(mape_recon, 2),
                "Improvement": round(improvement, 2)
            })
            
        st.dataframe(pd.DataFrame(results).style.applymap(
            lambda x: 'color: green' if x > 0 else 'color: red', subset=['Improvement']
        ))

        # Visual Comparison
        st.markdown("### Visual Validation")
        comp_select = st.selectbox("Select Component to visualize", list(all_series.keys()), index=4)
        
        # Plotly comparison
        actual_df = ts_to_df(all_series[comp_select])
        base_df = ts_to_df(pred_multi[comp_select])
        recon_df = ts_to_df(recon_multi[comp_select])
        
        fig = go.Figure()
        
        # Plot History
        fig.add_trace(go.Scatter(
            x=actual_df.index[:-val_len], 
            y=actual_df.iloc[:-val_len, 0], 
            mode='lines', 
            name='Training History',
            line=dict(color='gray')
        ))
        
        # Plot Actual Validation
        fig.add_trace(go.Scatter(
            x=actual_df.index[-val_len:], 
            y=actual_df.iloc[-val_len:, 0], 
            mode='lines', 
            name='Actual (Ground Truth)',
            line=dict(color='black', dash='dot')
        ))
        
        # Plot Base
        fig.add_trace(go.Scatter(
            x=base_df.index, 
            y=base_df.iloc[:, 0], 
            mode='lines', 
            name='Base (N-BEATS)',
            line=dict(color='red')
        ))
        
        # Plot Reconciled
        fig.add_trace(go.Scatter(
            x=recon_df.index, 
            y=recon_df.iloc[:, 0], 
            mode='lines', 
            name='Reconciled (MinT)',
            line=dict(color='green', width=4)
        ))
        
        fig.update_layout(title=f"Forecast Analysis: {comp_select}", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        if "_Cloth" in comp_select:
            st.info("Note: Some Clothing lines have simulated 'Cold Starts' (zeros at beginning). Watch how reconciliation handles this!")
            
    else:
        st.info("Please train the model in the previous tab first.")