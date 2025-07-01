# data_loader.py

import yfinance as yf
import pandas as pd
import os
from config import Config

def get_yahoo_finance_data(ticker_symbol, start_date, end_date):
    """
    Downloads historical data for a given ticker symbol from Yahoo Finance.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date, end=end_date)
        data.index = data.index.tz_localize(None).date # Remove timezone and convert to date
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker_symbol}: {e}")
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