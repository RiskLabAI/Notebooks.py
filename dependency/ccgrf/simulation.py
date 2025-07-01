# simulation.py

import numpy as np
import pandas as pd
from econml.grf import CausalForest
from config import Config
from utils import plot_conditional_correlation
from utils import calculate_confidence_interval_percentile
import matplotlib.pyplot as plt
import os

np.random.seed(Config.RANDOM_SEED)

def run_gaussian_simulation(case_type, n_samples):
    """
    Runs a Gaussian simulation and estimates conditional correlation.
    case_type: 'case1' or 'case2'
    """
    mu = np.array([0, 0, 0])
    if case_type == 'case1':
        sigma = np.array([[1, 0.5, 0.5],
                          [0.5, 1, 0.5],
                          [0.5, 0.5, 1]])
        exact_corr_value = (0.5 - 0.5*0.5) / (np.sqrt(1-0.5**2) * np.sqrt(1-0.5**2)) # Formula for partial correlation
        exact_corr_value = 0.3333 # As stated in your paper
        y_range = [0.15, 0.75]
        filename = Config.GAUSSIAN_FIG_NAMES['case1']
        title = 'Conditional Correlation for Gaussian Case 1'
    elif case_type == 'case2':
        sigma = np.array([[1, 0.0, 0.0],
                          [0.0, 1, 0.5],
                          [0.0, 0.5, 1]])
        exact_corr_value = 0.5 # As stated in your paper (rho_YW directly if X is uncorrelated)
        y_range = [-0.15, 0.5]
        filename = Config.GAUSSIAN_FIG_NAMES['case2']
        title = 'Conditional Correlation for Gaussian Case 2'
    else:
        raise ValueError("Invalid case_type for Gaussian simulation. Use 'case1' or 'case2'.")

    sample = np.random.multivariate_normal(mu, sigma, n_samples)
    x = sample[:, 0:1] # Conditioning variable
    w = sample[:, 1]    # One of the correlated variables
    y = sample[:, 2]    # The other correlated variable

    forest = CausalForest(
        n_estimators=Config.N_ESTIMATORS,
        criterion="mse",
        min_samples_leaf=Config.MIN_SAMPLES_LEAF_SIMULATION,
        random_state=Config.RANDOM_SEED
    )
    forest.fit(x, y, w)

    x_test = np.arange(Config.TEST_RANGE_GAUSSIAN[0], Config.TEST_RANGE_GAUSSIAN[1], Config.TEST_RANGE_GAUSSIAN[2]).reshape(-1, 1)
    pred, lb, ub = forest.predict(x_test, interval=True, alpha=0.05) # 95% CI

    data = {
        'X': np.squeeze(x_test),
        'estimated correlation': np.squeeze(pred),
        'lower bound': np.squeeze(lb),
        'upper bound': np.squeeze(ub),
        'exact correlation': np.ones(len(x_test)) * exact_corr_value
    }
    df = pd.DataFrame(data)

    plot_conditional_correlation(df, title, filename, y_range=y_range)


def run_nonlinear_simulation_eq14():
    """
    Runs the non-linear simulation based on Equation 14.
    Outputs results in a table format as per the paper.
    """
    n = Config.NON_LINEAR_N_SAMPLES
    W_t = np.random.normal(size=n) # i.i.d. standard normal r.v.
    X_t = np.random.binomial(1, 0.5, size=n) # i.i.d. Bernoulli r.v. with p=0.5

    # Equation 14: Y_t = W_t * (2*X_t - 1)
    Y_t = W_t * (2 * X_t - 1)

    # Prepare data for GRF
    x = X_t.reshape(-1, 1)
    w = W_t
    y = Y_t

    # Define test points for X=0 and X=1
    x_test_values = np.array([[0], [1]])

    # Run GRF for beta_YW(x)
    forest_yw = CausalForest(
        n_estimators=Config.N_ESTIMATORS,
        criterion="mse",
        min_samples_leaf=Config.MIN_SAMPLES_LEAF_SIMULATION,
        random_state=Config.RANDOM_SEED
    )
    forest_yw.fit(x, y, w)
    pred_yw = forest_yw.predict(x_test_values, interval=False)

    # Run GRF for beta_WY(x)
    forest_wy = CausalForest(
        n_estimators=Config.N_ESTIMATORS,
        criterion="mse",
        min_samples_leaf=Config.MIN_SAMPLES_LEAF_SIMULATION,
        random_state=Config.RANDOM_SEED
    )
    forest_wy.fit(x, w, y)
    pred_wy = forest_wy.predict(x_test_values, interval=False)

    # Calculate conditional correlation using the derived formula
    estimated_rho = np.sign(pred_yw) * np.sqrt(np.abs(pred_yw * pred_wy))

    # For confidence intervals, we need to run bootstrap for this specific case
    # This simulation has unit conditional variances, so a single GRF prediction's CI could be used
    # or follow the paper's explicit CI for beta_YW, then apply continuous mapping.
    # The paper's Table 1 uses a 95% CI computed using Section 2.2 (which means using GRF's own CI for beta if V=1).
    # Since V(Y|X)=V(W|X)=1, then rho_YW|X=x = beta_YW(x) here.
    # So we can use the CI directly from forest_yw.predict
    pred_yw_ci, lb_yw, ub_yw = forest_yw.predict(x_test_values, interval=True, alpha=0.05)

    results = {
        'x': [0, 1],
        'Estimated Rho': np.squeeze(estimated_rho),
        'Lower Bound (from beta_YW CI)': np.squeeze(lb_yw),
        'Upper Bound (from beta_YW CI)': np.squeeze(ub_yw)
    }
    df_results = pd.DataFrame(results)

    print("\n--- Results for Non-Linear Simulation (Equation 14) ---")
    print("Table 1: Results when estimating ρ_{Y,W|X=x} for x = {0,1} in (14)")
    print(df_results.to_string(index=False))
    print("-------------------------------------------------------")


def run_nonlinear_simulation_eq15():
    """
    Runs the non-linear simulation based on Equation 15.
    Plots estimated vs exact conditional correlation with CI.
    """
    n = Config.NON_LINEAR_N_SAMPLES
    W_t = np.random.normal(size=n) # i.i.d. standard normal random sequence
    X_t = np.random.uniform(low=0, high=10, size=n) # i.i.d. uniform [0,10] sequence
    epsilon_t = np.random.normal(size=n) # independent standard normal random sequence

    # Equation 15: Y_t = W_t * e^(-X_t) + sqrt(1 - e^(-2*X_t)) * epsilon_t
    Y_t = W_t * np.exp(-X_t) + np.sqrt(1 - np.exp(-2 * X_t)) * epsilon_t

    # In this case, V(Y_t | X_t) = V(W_t | X_t) = 1, hence rho_YW|X=x = beta_YW(x)
    x = X_t.reshape(-1, 1)
    w = W_t
    y = Y_t

    forest = CausalForest(
        n_estimators=Config.N_ESTIMATORS,
        criterion="mse",
        min_samples_leaf=Config.MIN_SAMPLES_LEAF_SIMULATION,
        random_state=Config.RANDOM_SEED
    )
    forest.fit(x, y, w)

    x_test = np.arange(Config.TEST_RANGE_NON_LINEAR[0], Config.TEST_RANGE_NON_LINEAR[1], Config.TEST_RANGE_NON_LINEAR[2]).reshape(-1, 1)
    pred, lb, ub = forest.predict(x_test, interval=True, alpha=0.05) # 95% CI

    data = {
        'X': np.squeeze(x_test),
        'estimated correlation': np.squeeze(pred),
        'lower bound': np.squeeze(lb),
        'upper bound': np.squeeze(ub),
        'exact correlation': np.exp(-np.squeeze(x_test)) # Exact correlation is e^(-X_t)
    }
    df = pd.DataFrame(data)

    plot_conditional_correlation(
        df,
        'Conditional Correlation for Nonlinear case in Equation (15)',
        Config.NON_LINEAR_FIG_NAMES['nonlinear_eq15'],
        y_range=[-0.2, 1]
    )


def run_nonlinear_simulation_eq16():
    """
    Runs the non-linear simulation based on Equation 16.
    Generates two plots:
    1. Estimated vs exact conditional correlation with CI (simulation-based).
    2. Estimated vs exact conditional correlation with CI (bootstrap-based).
    3. True and estimates for all: rho, beta_YW, beta_WY.
    """
    n = Config.NON_LINEAR_NON_UNIT_VARIANCE_N_SAMPLES
    W_t = np.random.normal(size=n)
    X_t = np.random.uniform(low=0, high=10, size=n)
    epsilon_t = np.random.normal(size=n)

    # Equation 16: Y_t = W_t * e^(-X_t) + epsilon_t
    Y_t = W_t * np.exp(-X_t) + epsilon_t

    # Define exact theoretical values for plotting
    x_test_range_plot = np.arange(Config.TEST_RANGE_NON_LINEAR[0], Config.TEST_RANGE_NON_LINEAR[1], Config.TEST_RANGE_NON_LINEAR[2])
    exact_rho = np.exp(-x_test_range_plot) / np.sqrt(np.exp(-2*x_test_range_plot) + 1)
    exact_beta_yw = np.exp(-x_test_range_plot)
    exact_beta_wy = np.exp(-x_test_range_plot) / (np.exp(-2*x_test_range_plot) + 1)

    # --- Simulation-based CI (as per your original code's "final" array) ---
    print("\nRunning Simulation-based CI for Equation 16...")
    all_sim_correlations = []
    for i in range(Config.N_BOOTSTRAP_SIMULATION): # Using N_BOOTSTRAP_SIMULATION as number of re-estimations
        if i % 10 == 0:
            print(f"  Simulation {i+1}/{Config.N_BOOTSTRAP_SIMULATION}")
        # Generate new sample for each simulation run
        sim_W_t = np.random.normal(size=n)
        sim_X_t = np.random.uniform(low=0, high=10, size=n)
        sim_epsilon_t = np.random.normal(size=n)
        sim_Y_t = sim_W_t * np.exp(-sim_X_t) + sim_epsilon_t

        sim_x = sim_X_t.reshape(-1, 1)
        sim_w = sim_W_t
        sim_y = sim_Y_t

        forest_sim_yw = CausalForest(n_estimators=Config.N_ESTIMATORS, criterion="mse", min_samples_leaf=30, random_state=Config.RANDOM_SEED + i) # Vary seed for diversity
        forest_sim_yw.fit(sim_x, sim_y, sim_w)

        forest_sim_wy = CausalForest(n_estimators=Config.N_ESTIMATORS, criterion="mse", min_samples_leaf=30, random_state=Config.RANDOM_SEED + i)
        forest_sim_wy.fit(sim_x, sim_w, sim_y)

        x_test_sim = np.arange(Config.TEST_RANGE_NON_LINEAR[0], Config.TEST_RANGE_NON_LINEAR[1], Config.TEST_RANGE_NON_LINEAR[2]).reshape(-1, 1)
        pred_sim_yw, _, _ = forest_sim_yw.predict(x_test_sim, interval=False)
        pred_sim_wy, _, _ = forest_sim_wy.predict(x_test_sim, interval=False)

        current_corr = np.sign(pred_sim_yw) * np.sqrt(np.abs(pred_sim_yw * pred_sim_wy))
        all_sim_correlations.append(np.squeeze(current_corr))

    all_sim_correlations = np.array(all_sim_correlations)
    mean_sim_corr = np.mean(all_sim_correlations, axis=0)
    lower_sim_ci, upper_sim_ci = calculate_confidence_interval_percentile(all_sim_correlations, 2.5, 97.5) # For 95% CI

    df_sim_ci = pd.DataFrame({
        'X': x_test_range_plot,
        'estimated correlation': mean_sim_corr,
        'exact correlation': exact_rho,
        'lower bound': lower_sim_ci,
        'upper bound': upper_sim_ci
    })
    plot_conditional_correlation(
        df_sim_ci,
        'Conditional Correlation (Simulation-based CI) for Nonlinear case in Equation (16)',
        Config.NON_LINEAR_FIG_NAMES['nonlinear_eq16_sim_ci'],
        y_range=[-0.2, 1]
    )

    # --- Percentile Bootstrap CI (as per your original code's "final_bootstrap" array) ---
    print("\nRunning Bootstrap-based CI for Equation 16...")
    all_bootstrap_correlations = []
    bootstrap_sample_size = Config.NON_LINEAR_NON_UNIT_VARIANCE_N_SAMPLES # Using the same N for the base dataset
    for i in range(Config.NON_LINEAR_NON_UNIT_VARIANCE_BOOTSTRAP_SAMPLES):
        if i % 20 == 0:
            print(f"  Bootstrap sample {i+1}/{Config.NON_LINEAR_NON_UNIT_VARIANCE_BOOTSTRAP_SAMPLES}")
        indices = np.random.choice(n, bootstrap_sample_size, replace=True)
        boot_x = X_t[indices].reshape(-1, 1)
        boot_w = W_t[indices]
        boot_y = Y_t[indices]

        forest_boot_yw = CausalForest(n_estimators=500, criterion="mse", min_samples_leaf=30, random_state=Config.RANDOM_SEED + i)
        forest_boot_yw.fit(boot_x, boot_y, boot_w)

        forest_boot_wy = CausalForest(n_estimators=500, criterion="mse", min_samples_leaf=30, random_state=Config.RANDOM_SEED + i)
        forest_boot_wy.fit(boot_x, boot_w, boot_y)

        x_test_boot = np.arange(Config.TEST_RANGE_NON_LINEAR[0], Config.TEST_RANGE_NON_LINEAR[1], Config.TEST_RANGE_NON_LINEAR[2]).reshape(-1, 1)
        pred_boot_yw, _, _ = forest_boot_yw.predict(x_test_boot, interval=False)
        pred_boot_wy, _, _ = forest_boot_wy.predict(x_test_boot, interval=False)

        current_corr = np.sign(pred_boot_yw) * np.sqrt(np.abs(pred_boot_yw * pred_boot_wy))
        all_bootstrap_correlations.append(np.squeeze(current_corr))

    all_bootstrap_correlations = np.array(all_bootstrap_correlations)
    mean_boot_corr = np.mean(all_bootstrap_correlations, axis=0)
    lower_boot_ci, upper_boot_ci = calculate_confidence_interval_percentile(all_bootstrap_correlations, 2.5, 97.5) # For 95% CI

    df_boot_ci = pd.DataFrame({
        'X': x_test_range_plot,
        'estimated correlation': mean_boot_corr,
        'exact correlation': exact_rho,
        'lower bound': lower_boot_ci,
        'upper bound': upper_boot_ci
    })
    plot_conditional_correlation(
        df_boot_ci,
        'Conditional Correlation (Bootstrap-based CI) for Nonlinear case in Equation (16)',
        Config.NON_LINEAR_FIG_NAMES['nonlinear_eq16_boot_ci'],
        y_range=[-0.2, 1]
    )

    # --- Plotting all parameters (rho, beta_YW, beta_WY) ---
    # Here, we run GRF once on the full dataset 'n' to get estimates of betas
    # and then calculate estimated rho from those single estimates.
    # This is for a single run to show the curves, not for CI calculation.

    x_full = X_t.reshape(-1, 1)
    w_full = W_t
    y_full = Y_t

    forest_yw_single = CausalForest(n_estimators=Config.N_ESTIMATORS, criterion="mse", min_samples_leaf=100, random_state=Config.RANDOM_SEED)
    forest_yw_single.fit(x_full, y_full, w_full)
    pred_yw_single, _, _ = forest_yw_single.predict(x_test_range_plot.reshape(-1,1), interval=False)

    forest_wy_single = CausalForest(n_estimators=Config.N_ESTIMATORS, criterion="mse", min_samples_leaf=100, random_state=Config.RANDOM_SEED)
    forest_wy_single.fit(x_full, w_full, y_full)
    pred_wy_single, _, _ = forest_wy_single.predict(x_test_range_plot.reshape(-1,1), interval=False)

    estimated_rho_single = np.sign(pred_yw_single) * np.sqrt(np.abs(pred_yw_single * pred_wy_single))

    df_all_params = pd.DataFrame({
        'X': x_test_range_plot,
        'estimated correlation': np.squeeze(estimated_rho_single),
        'exact correlation': exact_rho,
        r'$\hat{\beta}_{YW}$': np.squeeze(pred_yw_single),
        r'$\hat{\beta}_{WY}$': np.squeeze(pred_wy_single),
        r'$\beta_{YW}$': exact_beta_yw,
        r'$\beta_{WY}$': exact_beta_wy
    })
    plot_conditional_correlation(
        df_all_params,
        'Conditional Correlation and Betas for Nonlinear case in Equation (16)',
        Config.NON_LINEAR_FIG_NAMES['nonlinear_eq16_all_params'],
        y_range=[-0.2, 1]
    )