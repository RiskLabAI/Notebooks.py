# preprocessing.py

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from data_loader import get_fred_data, load_hfrx_data
from config import Config
import numpy as np

def get_daily_log_returns(data_series):
    """Calculates daily logarithmic returns for a given price series."""
    # Using .pct_change() then log1p for log returns, as commonly done for financial series
    return np.log1p(data_series.pct_change()).dropna()

def prepare_hedgefund_and_sp500(hf_name, start_date, end_date, hfrx_data_path):
    """
    Prepares the dataset by merging hedge fund returns, S&P 500 returns,
    and VIX levels, ensuring synchronized trading dates.
    """
    # Load HFRX data
    hfrx_df = load_hfrx_data(hf_name, hfrx_data_path)
    if hfrx_df is None:
        return None

    # Get S&P 500 data from FRED and calculate log returns
    sp500_levels = get_fred_data(Config.FRED_SP500_SERIES_ID, start_date, end_date, Config.FRED_API_KEY)
    if sp500_levels is None:
        return None
    sp500_returns = get_daily_log_returns(sp500_levels)
    sp500_returns.name = 'S&P500_returns'
    sp500_returns.index = pd.to_datetime(sp500_returns.index)

    # Get VIX data (levels) from FRED
    vix_levels = get_fred_data(Config.FRED_VIX_SERIES_ID, start_date, end_date, Config.FRED_API_KEY)
    if vix_levels is None:
        return None
    vix_levels.name = 'VIX'
    vix_levels.index = pd.to_datetime(vix_levels.index)

    # Perform inner joins sequentially to ensure all three series align on trading dates
    # Start with HFRX and S&P 500 returns
    merged_data = pd.merge(hfrx_df, sp500_returns, left_index=True, right_index=True, how='inner')
    # Then merge with VIX levels
    final_dataset = pd.merge(merged_data, vix_levels, left_index=True, right_index=True, how='inner')

    print(f"\n--- Prepared Data for {hf_name} ---")
    print(final_dataset.head())
    print(f"Total rows after inner join: {len(final_dataset)}")
    return final_dataset

def run_stationarity_tests(series, name="Series"):
    """
    Performs Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.
    Prints the results and suggests stationarity based on p-values.
    """
    print(f"\n--- Stationarity Tests for {name} ---")

    # ADF Test
    print("Augmented Dickey-Fuller Test:")
    try:
        adf_result = adfuller(series.dropna()) # .dropna() in case of NaNs
        print(f"ADF Statistic: {adf_result[0]:.2f}")
        print(f"P-value: {adf_result[1]:.3f}")
        print("Critical Values:")
        for key, value in adf_result[4].items():
            print(f"   {key}: {value:.2f}")
        if adf_result[1] <= 0.05:
            print(f"{name} is likely stationary (reject H0 of unit root).")
        else:
            print(f"{name} is likely non-stationary (fail to reject H0 of unit root).")
    except Exception as e:
        print(f"ADF Test failed for {name}: {e}")

    # KPSS Test
    print("\nKwiatkowski-Phillips-Schmidt-Shin Test:")
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        print(f"KPSS Statistic: {kpss_result[0]:.2f}")
        print(f"P-value: {kpss_result[1]:.3f}")
        print("Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"   {key}: {value:.2f}")
        if kpss_result[1] > 0.05:
            print(f"{name} is likely stationary (fail to reject H0 of stationarity).")
        else:
            print(f"{name} is likely non-stationary (reject H0 of stationarity).")
    except Exception as e:
        print(f"KPSS Test failed for {name}: {e}")
    print("-" * 40)