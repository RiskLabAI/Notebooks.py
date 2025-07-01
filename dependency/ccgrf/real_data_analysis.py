# real_data_analysis.py

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from config import Config
from utils import plot_conditional_correlation, calculate_confidence_interval_percentile
from preprocessing import prepare_hedgefund_and_sp500, run_stationarity_tests
import matplotlib.pyplot as plt

np.random.seed(Config.RANDOM_SEED)

def conditional_correlation_grf_bootstrap(X, Y, W, x_test, n_bootstrap_samples, n_estimators, min_samples_leaf):
    """
    Performs bootstrap resampling to estimate conditional correlation and its confidence intervals.
    X, Y, W are numpy arrays.
    x_test: array of conditioning variable values for prediction.
    """
    n = X.shape[0]
    n_predict_points = x_test.shape[0]
    all_bootstrap_correlations = []

    print(f"\nStarting bootstrap for real data with {n_bootstrap_samples} samples...")
    for i in range(n_bootstrap_samples):
        if (i + 1) % 10 == 0:
            print(f"  Bootstrap sample {i+1}/{n_bootstrap_samples}")

        # Resample with replacement
        indices = np.random.choice(n, n, replace=True) # Resample original size, as per standard bootstrap
        x_selected = X[indices]
        y_selected = Y[indices]
        w_selected = W[indices]

        # Fit CausalForest for Y on W conditional on X
        forest_y_w = CausalForest(
            n_estimators=n_estimators,
            criterion="mse",
            min_samples_leaf=min_samples_leaf,
            random_state=Config.RANDOM_SEED + i # Vary seed for different trees in bootstrap
        )
        forest_y_w.fit(x_selected, y_selected, w_selected)

        # Fit CausalForest for W on Y conditional on X
        forest_w_y = CausalForest(
            n_estimators=n_estimators,
            criterion="mse",
            min_samples_leaf=min_samples_leaf,
            random_state=Config.RANDOM_SEED + i # Use same seed variation for consistency if desired, or different
        )
        forest_w_y.fit(x_selected, w_selected, y_selected)

        # Predict point estimates
        # Corrected line: Unpack only one value as interval=False
        pred_y_w = forest_y_w.predict(x_test, interval=False)
        # Corrected line: Unpack only one value as interval=False
        pred_w_y = forest_w_y.predict(x_test, interval=False)

        # Calculate conditional correlation
        corr_est = np.sign(pred_y_w) * np.sqrt(np.abs(pred_y_w * pred_w_y))
        all_bootstrap_correlations.append(np.squeeze(corr_est))

    all_bootstrap_correlations = np.array(all_bootstrap_correlations)
    return all_bootstrap_correlations

def analyze_hedge_fund_correlation(hf_name):
    """
    Analyzes the conditional correlation between a hedge fund index and S&P 500,
    conditioned on VIX, using real market data.
    """
    print(f"\n--- Analyzing Real Data for {hf_name} ---")
    full_df = prepare_hedgefund_and_sp500(
        hf_name,
        Config.DATA_START_DATE,
        Config.DATA_END_DATE,
        Config.HFRX_DATA_PATH
    )

    if full_df is None:
        print(f"Skipping analysis for {hf_name} due to data loading issues.")
        return

    # Run stationarity tests as mentioned in the paper
    run_stationarity_tests(full_df[f'{hf_name}_returns'], name=f'{hf_name} Returns')
    run_stationarity_tests(full_df['S&P500_returns'], name='S&P 500 Returns')

    # Prepare data for GRF: X (VIX), Y (HF returns), W (S&P 500 returns)
    # The paper's text and original code implies VIX(t-1) as X for Y(t), W(t)
    # We need to align the VIX as the conditioning variable from the *previous day*
    # compared to the returns of HF and S&P 500.
    # The `prepare_hedgefund_and_sp500` outputs all on the same `Date` index.
    # So we need to shift VIX for the `X` input.
    df_for_grf = pd.DataFrame()
    df_for_grf['X_var'] = full_df['VIX'].shift(1) # VIX from previous day
    df_for_grf['Y_var'] = full_df[f'{hf_name}_returns'] # HF returns (current day)
    df_for_grf['W_var'] = full_df['S&P500_returns'] # S&P 500 returns (current day)
    df_for_grf = df_for_grf.dropna() # Drop the first row due to shift(1)

    X_vals = df_for_grf[['X_var']].values
    Y_vals = df_for_grf[['Y_var']].values
    W_vals = df_for_grf[['W_var']].values

    # Define the test range for VIX values
    x_test = np.arange(Config.TEST_RANGE_REAL_DATA[0], Config.TEST_RANGE_REAL_DATA[1], Config.TEST_RANGE_REAL_DATA[2]).reshape(-1, 1)

    # Perform bootstrap estimation for conditional correlation
    all_boot_correlations = conditional_correlation_grf_bootstrap(
        X_vals, Y_vals, W_vals, x_test,
        Config.N_BOOTSTRAP_REAL_DATA,
        Config.N_ESTIMATORS,
        Config.MIN_SAMPLES_LEAF_REAL_DATA
    )

    # Calculate mean and confidence intervals from bootstrap results
    estimated_mean_corr = np.mean(all_boot_correlations, axis=0)
    lower_bound_ci, upper_bound_ci = calculate_confidence_interval_percentile(all_boot_correlations, 2.5, 97.5) # 95% CI

    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame({
        'VIX': np.squeeze(x_test),
        'estimated correlation': estimated_mean_corr,
        'lower bound': lower_bound_ci,
        'upper bound': upper_bound_ci
    })

    # Determine filename from config
    filename = Config.REAL_DATA_FIG_NAMES.get(hf_name)
    if filename is None:
        filename = f'{hf_name}_ConditionalCorrelation.png' # Fallback
        print(f"Warning: No specific filename configured for {hf_name}. Using default: {filename}")

    plot_conditional_correlation(
        plot_df,
        f'Correlation between Market and {hf_name} Daily Indexes under VIX',
        filename,
        y_label='Conditional Correlation',
        y_range=[-0.8, 1] # Adjusted y-range as per your original plots
    )