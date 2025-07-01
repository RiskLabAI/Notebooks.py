# utils.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import Config
import plotly.express as px

def setup_plotting_style(font_size, font_family):
    """Sets up global Matplotlib plotting style."""
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': font_family,
        'figure.autolayout': True,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'axes.titlesize': font_size,
    })
    # Ensure Times New Roman is available or fallback
    if font_family == 'Times New Roman':
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['font.sans-serif'] = ['Times New Roman'] + plt.rcParams['font.sans-serif']
        plt.rcParams['font.monospace'] = ['Times New Roman'] + plt.rcParams['font.monospace']
        plt.rcParams['font.family'] = 'serif'


def save_figure(fig, filename, dpi, fig_dir):
    """Saves a Matplotlib figure to the specified directory."""
    os.makedirs(fig_dir, exist_ok=True)
    filepath = os.path.join(fig_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {filepath}")

def plot_conditional_correlation(df, title, filename, y_label='Conditional Correlation', y_range=None):
    """
    Generates and saves a conditional correlation plot using Matplotlib.
    Handles 'exact correlation' and confidence bounds.
    """
    fig, ax = plt.subplots(figsize=(10, 6)) # Adjust size for journal quality

    # Map column names to desired labels
    labels = {
        'estimated correlation': 'Estimated Correlation',
        'lower bound': 'Lower Bound',
        'upper bound': 'Upper Bound',
        'exact correlation': 'Exact Correlation',
        'VIX': 'VIX', # For real data plots
        r'$\hat{\beta}_{YW}$': r'$\hat{\beta}_{YW}$',
        r'$\hat{\beta}_{WY}$': r'$\hat{\beta}_{WY}$',
        r'$\beta_{YW}$': r'$\beta_{YW}$',
        r'$\beta_{WY}$': r'$\beta_{WY}$'
    }

    # Determine x-axis label dynamically
    x_label = 'X'
    if 'VIX' in df.columns:
        x_label = 'VIX'
    elif 'X' in df.columns:
        x_label = 'X'

    # Plot lines with appropriate colors and styles
    if 'estimated correlation' in df.columns:
        ax.plot(df[x_label], df['estimated correlation'], label=labels['estimated correlation'], color='blue', linewidth=1.5)
    if 'exact correlation' in df.columns:
        ax.plot(df[x_label], df['exact correlation'], label=labels['exact correlation'], color='red', linestyle='--', linewidth=1.5)
    if 'lower bound' in df.columns and 'upper bound' in df.columns:
        ax.plot(df[x_label], df['lower bound'], label=labels['lower bound'], color='green', linestyle=':', linewidth=1)
        ax.plot(df[x_label], df['upper bound'], label=labels['upper bound'], color='purple', linestyle=':', linewidth=1)
        ax.fill_between(df[x_label], df['lower bound'], df['upper bound'], color='gray', alpha=0.1, label='95% CI')

    # For Fig 4 specifically, plot all beta and rho values
    if r'$\hat{\beta}_{YW}$' in df.columns:
        ax.plot(df[x_label], df[r'$\hat{\beta}_{YW}$'], label=labels[r'$\hat{\beta}_{YW}$'], color='cyan', linestyle='-', linewidth=1.0)
    if r'$\hat{\beta}_{WY}$' in df.columns:
        ax.plot(df[x_label], df[r'$\hat{\beta}_{WY}$'], label=labels[r'$\hat{\beta}_{WY}$'], color='magenta', linestyle='-', linewidth=1.0)
    if r'$\beta_{YW}$' in df.columns:
        ax.plot(df[x_label], df[r'$\beta_{YW}$'], label=labels[r'$\beta_{YW}$'], color='orange', linestyle='--', linewidth=1.0)
    if r'$\beta_{WY}$' in df.columns:
        ax.plot(df[x_label], df[r'$\beta_{WY}$'], label=labels[r'$\beta_{WY}$'], color='brown', linestyle='--', linewidth=1.0)


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if y_range:
        ax.set_ylim(y_range)

    ax.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white', framealpha=0.8)
    ax.grid(True, linestyle=':', alpha=0.6)

    save_figure(fig, filename, Config.DPI, Config.FIG_DIR)
    plt.close(fig) # Close the figure to free up memory

def calculate_confidence_interval_percentile(data_matrix, lower_percentile, upper_percentile):
    """
    Calculates percentile-based confidence intervals for each column (or point) in a data matrix.
    data_matrix: (n_bootstrap_samples, n_test_points)
    """
    sorted_data = np.sort(data_matrix, axis=0)
    lower_idx = int(data_matrix.shape[0] * (lower_percentile / 100))
    upper_idx = int(data_matrix.shape[0] * (upper_percentile / 100))

    # Ensure indices are within bounds
    lower_idx = max(0, min(lower_idx, data_matrix.shape[0] - 1))
    upper_idx = max(0, min(upper_idx, data_matrix.shape[0] - 1))

    return sorted_data[lower_idx, :], sorted_data[upper_idx, :]