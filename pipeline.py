import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Darts imports for Time Series
from darts import TimeSeries
from darts.utils.timeseries_generation import sine_timeseries, linear_timeseries, random_walk_timeseries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.dataprocessing.transformers.reconciliation import MinTReconciliator
from darts.metrics import mae, mape

# Visualization settings
plt.style.use('fivethirtyeight')
pd.options.mode.chained_assignment = None

class HierarchicalDataGenerator:
    """
    Generates synthetic hierarchical sales data:
    Level 0: Total (Global)
    Level 1: Regions (US, EU)
    Level 2: Products (Electronics, Clothing within regions)
    
    Includes 'Cold Start' simulation by introducing zero-padding at the start of some series.
    """
    def __init__(self, length: int = 100):
        self.length = length
        self.hierarchy = {}
        
    def generate(self) -> Tuple[Dict[str, TimeSeries], Dict[str, List[str]]]:
        print("Generating synthetic hierarchical data...")
        
        # 1. Generate Bottom-Level Series (Products) with different patterns
        # We simulate 4 bottom-level series
        # Pattern: Trend + Seasonality + Noise
        
        # Product 1: US - Electronics (Stable history)
        us_elec = (linear_timeseries(start_value=10, end_value=50, length=self.length) + 
                   sine_timeseries(value_frequency=0.1, value_amplitude=10, length=self.length) + 
                   random_walk_timeseries(length=self.length) * 2)
        
        # Product 2: US - Clothing (Cold Start - simulated by zeroing first 50 steps)
        # In real life, this would be NaN or missing, but N-BEATS handles zeros well as "no sales"
        us_cloth_base = (linear_timeseries(start_value=50, end_value=20, length=self.length) + 
                         sine_timeseries(value_frequency=0.2, value_amplitude=5, length=self.length))
        # Masking first half to simulate product launch mid-year
        values = us_cloth_base.values()
        values[:50] = 0 
        us_cloth = TimeSeries.from_values(values)

        # Product 3: EU - Electronics
        eu_elec = (linear_timeseries(start_value=20, end_value=80, length=self.length) + 
                   sine_timeseries(value_frequency=0.05, value_amplitude=20, length=self.length))

        # Product 4: EU - Clothing
        eu_cloth = (random_walk_timeseries(length=self.length) * 5 + 100)

        # 2. Aggregate to create Higher Levels (Ground Truth)
        # Region Level
        us_total = us_elec + us_cloth
        eu_total = eu_elec + eu_cloth
        
        # Global Level
        global_total = us_total + eu_total

        # 3. Define Hierarchy Dictionary (Child -> [Parents])
        # This tells the Reconciler how the tree is structured
        hierarchy = {
            "US_Elec": ["US"],
            "US_Cloth": ["US"],
            "EU_Elec": ["EU"],
            "EU_Cloth": ["EU"],
            "US": ["Total"],
            "EU": ["Total"],
            "Total": []
        }

        # Collect all series into a dictionary
        all_series = {
            "Total": global_total,
            "US": us_total,
            "EU": eu_total,
            "US_Elec": us_elec,
            "US_Cloth": us_cloth,
            "EU_Elec": eu_elec,
            "EU_Cloth": eu_cloth
        }

        # Convert to Darts multivariate list for easier processing, 
        # but keep keys for identification
        return all_series, hierarchy

class ForecastingEngine:
    def __init__(self):
        # N-BEATS Configuration
        # input_chunk_length: Lookback window (e.g., look at past 24 days)
        # output_chunk_length: Forecast horizon (e.g., predict next 12 days)
        self.model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=12,
            n_epochs=20,          # Low for demo speed. Increase to 100+ for production
            random_state=42,
            force_reset=True,
            pl_trainer_kwargs={"accelerator": "cpu"} # Use 'gpu' if available
        )
        self.scaler = Scaler()
        self.reconciler = MinTReconciliator(method="wls_val") 

    def preprocess(self, series_dict: Dict[str, TimeSeries]) -> List[TimeSeries]:
        """
        Scales data to 0-1 range (critical for Deep Learning convergence).
        """
        series_list = list(series_dict.values())
        # Embed hierarchy info into the TimeSeries objects
        # Note: In production, you might handle hierarchy externally, 
        # but Darts likes it embedded for the Reconciler.
        return self.scaler.fit_transform(series_list)

    def train_and_predict(self, 
                         train_series: List[TimeSeries], 
                         val_series: List[TimeSeries], 
                         hierarchy: Dict[str, List[str]]) -> Tuple[List[TimeSeries], List[TimeSeries]]:
        
        print("Training Global N-BEATS model on all hierarchy levels...")
        # We train on ALL levels (Total, Region, Product).
        # This allows the model to learn "macro" trends from Total and "micro" from Products.
        self.model.fit(train_series, verbose=True)

        print("Generating Base Forecasts...")
        pred_horizon = len(val_series[0])
        base_forecasts = self.model.predict(n=pred_horizon, series=train_series)
        
        # Inverse transform to get original scale back
        base_forecasts_unscaled = self.scaler.inverse_transform(base_forecasts)
        
        # Add hierarchy information to the TimeSeries objects for Reconciliation
        # We map the list back to a dictionary temporarily to assign component names if needed,
        # but Darts handles multi-series reconciliation if passed as a list with correct components.
        
        # For MinT, we need to ensure the forecasts are grouped correctly.
        # We will restructure the forecasts into a single Multivariate TimeSeries or 
        # a list that the Reconciler accepts.
        
        # Simplest approach for MinT in Darts:
        # 1. Convert list of univariate series to one multivariate series
        # 2. Define the grouping structure
        
        return base_forecasts_unscaled

    def reconcile(self, base_forecasts: List[TimeSeries], hierarchy: Dict[str, List[str]], train_data: List[TimeSeries]):
        """
        Applies MinT (Minimum Trace) Reconciliation.
        MinT adjusts forecasts based on the correlation of errors in the training set.
        """
        print("Applying MinT Reconciliation...")
        
        # Convert list of series to a single Multivariate Series for Reconciliation
        # We assume the order matches the keys in our hierarchy logic
        # Ideally, we stack them.
        
        # For this demo, we will manually implementation a simple Bottom-Up check 
        # or use Darts' built-in if the version supports the hierarchy dict directly.
        # Darts MinT expects a specific format. Let's use the Post-Hoc Reconciler.
        
        # We need to reshape our list of TimeSeries into one Multivariate TimeSeries
        # to apply the hierarchy constraint easily.
        
        # Helper to stack
        from darts import concatenate
        
        # Combine train and forecast into multivariate
        multivariate_train = concatenate(train_data, axis="component")
        multivariate_pred = concatenate(base_forecasts, axis="component")
        
        # Assign static component names matching our hierarchy keys
        # The generator created them in specific order: Total, US, EU, US_Elec...
        # We need to ensure columns match hierarchy keys.
        col_names = ["Total", "US", "EU", "US_Elec", "US_Cloth", "EU_Elec", "EU_Cloth"]
        multivariate_train = multivariate_train.with_columns_renamed(multivariate_train.components, col_names)
        multivariate_pred = multivariate_pred.with_columns_renamed(multivariate_pred.components, col_names)
        
        # Define hierarchy for Darts (Dictionary mapping Component -> Parent)
        # Note: Darts format is Child: [Parent]
        darts_hierarchy = {
            "US_Elec": ["US"], "US_Cloth": ["US"],
            "EU_Elec": ["EU"], "EU_Cloth": ["EU"],
            "US": ["Total"], "EU": ["Total"]
        }
        
        # Add hierarchy to the series
        multivariate_train = multivariate_train.with_hierarchy(darts_hierarchy)
        multivariate_pred = multivariate_pred.with_hierarchy(darts_hierarchy)

        # Fit Reconciler on Training errors (residuals)
        self.reconciler.fit(multivariate_train)
        reconciled_pred = self.reconciler.transform(multivariate_pred)
        
        return multivariate_pred, reconciled_pred

def evaluate_and_plot(actual_dict: Dict[str, TimeSeries], 
                      base_pred: TimeSeries, 
                      recon_pred: TimeSeries):
    
    components = base_pred.components
    
    # Calculate simple accuracy (MAPE)
    print("\n--- Performance Report (MAPE) ---")
    print(f"{'Component':<15} | {'Base Model':<12} | {'Reconciled':<12} | {'Improvement'}")
    print("-" * 60)
    
    for comp in components:
        # Extract individual series
        act = actual_dict[comp][-len(base_pred):]
        base = base_pred[comp]
        recon = recon_pred[comp]
        
        err_base = mape(act, base)
        err_recon = mape(act, recon)
        
        imp = "YES" if err_recon < err_base else "NO"
        print(f"{comp:<15} | {err_base:.2f}%       | {err_recon:.2f}%       | {imp}")

    # Plot Total (Level 0) and one Leaf (Level 2)
    plt.figure(figsize=(12, 6))
    
    # Plot Total
    plt.subplot(1, 2, 1)
    actual_dict['Total'].plot(label='Actual', color='black', alpha=0.5)
    base_pred['Total'].plot(label='Base Forecast (N-BEATS)', linestyle='--')
    recon_pred['Total'].plot(label='Reconciled (MinT)', linewidth=2)
    plt.title("Level 0: Global Demand")
    
    # Plot a Product
    plt.subplot(1, 2, 2)
    actual_dict['US_Cloth'].plot(label='Actual', color='black', alpha=0.5)
    base_pred['US_Cloth'].plot(label='Base Forecast', linestyle='--')
    recon_pred['US_Cloth'].plot(label='Reconciled', linewidth=2)
    plt.title("Level 2: US Clothing (Cold Start?)")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Generate Data
    gen = HierarchicalDataGenerator(length=120)
    all_series, hierarchy = gen.generate()
    
    # Split Train/Test
    train_dict = {k: v[:-24] for k, v in all_series.items()}
    val_dict = {k: v[-24:] for k, v in all_series.items()}
    
    # 2. Setup Engine
    engine = ForecastingEngine()
    
    # 3. Preprocess
    # Train scaler on training data only to avoid leakage
    train_scaled = engine.preprocess(train_dict)
    
    # 4. Train & Predict
    # We create a dummy validation list just to define horizon, 
    # real validation happens in evaluation
    raw_forecasts = engine.train_and_predict(train_scaled, list(val_dict.values()), hierarchy)
    
    # 5. Reconcile
    base_multi, recon_multi = engine.reconcile(
        raw_forecasts, 
        hierarchy, 
        engine.scaler.transform(list(train_dict.values()))
    )
    
    # 6. Evaluate
    evaluate_and_plot(all_series, base_multi, recon_multi)
    
    print("\nDone! Use the plot to verify that Reconciled forecasts (MinT) are smoother and coherent.")