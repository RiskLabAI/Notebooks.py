# data_loader.py

import pandas as pd
import os
from fredapi import Fred
from config import Config

def get_fred_data(series_id, start_date, end_date, api_key):
    """
    Downloads historical data for a given FRED series ID.
    """
    fred = Fred(api_key=api_key)
    try:
        data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        if data is None or data.empty:
            print(f"Warning: No data returned for FRED series {series_id} within {start_date} to {end_date}.")
            return None
        # Ensure index is datetime and has no timezone, then convert to date
        data.index = pd.to_datetime(data.index).tz_localize(None).date
        data.name = series_id # Name the series for easier processing later
        return data
    except Exception as e:
        print(f"Error downloading FRED data for series {series_id}: {e}")
        return None

def load_hfrx_data(name, hfrx_data_path):
    """
    Loads HFRX historical data from a CSV file.
    Assumes a specific format for the CSV (header=3, columns 0:2, inverted rows, '%" removed).
    """
    file_path = os.path.join(hfrx_data_path, f'HFRX_historical_{name}.csv')
    try:
        hfrx_df = pd.read_csv(file_path, header=3)
        hfrx_df = hfrx_df.iloc[-7::-1, 0:2] # Custom slicing as per your original code
        hfrx_df.columns = ['Date', 'Daily ROR']
        hfrx_df['Date'] = pd.to_datetime(hfrx_df['Date'])
        hfrx_df = hfrx_df.set_index('Date')
        hfrx_df['Daily ROR'] = hfrx_df['Daily ROR'].str.rstrip("%").astype(float) / 100
        hfrx_df.rename(columns={'Daily ROR': f'{name}_returns'}, inplace=True)
        return hfrx_df
    except FileNotFoundError:
        print(f"Error: HFRX data file not found at {file_path}. Please ensure the 'data' directory and CSV are present.")
        return None
    except Exception as e:
        print(f"Error loading or processing HFRX data for {name}: {e}")
        return None