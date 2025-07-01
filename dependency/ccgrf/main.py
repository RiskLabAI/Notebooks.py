# main.py

import os
import numpy as np
from config import Config
from utils import setup_plotting_style
from simulation import (
    run_gaussian_simulation,
    run_nonlinear_simulation_eq14,
    run_nonlinear_simulation_eq15,
    run_nonlinear_simulation_eq16
)
from real_data_analysis import analyze_hedge_fund_correlation

def main():
    # Setup plotting style once
    setup_plotting_style(Config.FONT_SIZE, Config.FONT_FAMILY)

    # Create figs directory if it doesn't exist
    os.makedirs(Config.FIG_DIR, exist_ok=True)

    print("Starting Conditional Correlation Analysis...")

    # --- Run Simulations ---
    print("\n--- Running Simulation Tests ---")

    # Gaussian Simulations (Fig 1a and 1b)
    print("\nRunning Gaussian Simulation: Case 1...")
    run_gaussian_simulation('case1', Config.GAUSSIAN_N_SAMPLES)
    print("\nRunning Gaussian Simulation: Case 2...")
    run_gaussian_simulation('case2', Config.GAUSSIAN_N_SAMPLES)

    # Non-Linear Simulations
    # Eq 14 (Table output)
    print("\nRunning Non-Linear Simulation: Equation 14 (Table Output)...")
    run_nonlinear_simulation_eq14()

    # Eq 15 (Fig 2)
    print("\nRunning Non-Linear Simulation: Equation 15...")
    run_nonlinear_simulation_eq15()

    # Eq 16 (Fig 3a, 3b, and Fig 4)
    print("\nRunning Non-Linear Simulation: Equation 16 (Simulation and Bootstrap CI, All Params)...")
    run_nonlinear_simulation_eq16()

    # --- Run Real Data Analysis ---
    print("\n--- Running Real Data Analysis ---")
    # Loop through each hedge fund type for analysis
    for hf_name in Config.HEDGE_FUND_NAMES:
        analyze_hedge_fund_correlation(hf_name)

    print("\nConditional Correlation Analysis Completed.")

if __name__ == "__main__":
    main()