import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import math
import random
import time
import sys
# Quantum and clustering packages
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector, DensityMatrix
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skew, kurtosis
import scipy.stats as stats
from scipy.linalg import eigh
import yfinance as yf
import datetime
import RiskLabAI.utils

from RiskLabAI.optimization import hrp, recursive_bisection



def compute_financial_features(returns, t, window=5):
    """
    Compute a set of six financial features for an asset at time index t
    using a rolling window of past returns (including the current return).

    Parameters:
        returns (np.ndarray): 1D array of asset returns.
        t (int): Current time index.
        window (int): Rolling window size (default is 5).

    Returns:
        np.ndarray: A vector of six features.
    """
    ret = returns[t]
    
    start = max(0, t - window + 1)
    window_returns = returns[start:t+1]
    f1 = ret
    f2 = np.mean(window_returns)
    f3 = np.std(window_returns, ddof=1)
    f4 = np.prod(1 + window_returns) - 1
    f5 = stats.skew(window_returns)
    f6 = stats.kurtosis(window_returns)
    
    return np.array([f1, f2, f3, f4, f5, f6])


##############################################
# Quantum Feature Map and Distance Functions
##############################################
def quantum_feature_map_density(x, x_min, x_max, alpha=2.0):
    """
    Encodes a high-dimensional observation x (a vector of length P)
    into a quantum state using a three‐round circuit.
    """
    n = len(x)
    pi = math.pi
    theta = np.zeros(n)
    for i in range(n):
        if x_max[i] > x_min[i]:
            theta[i] = alpha * pi * (x[i] - x_min[i]) / (x_max[i] - x_min[i])
        else:
            theta[i] = 0
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.barrier()
    # 1st encoding: ry rotations
    for i in range(n):
        qc.t(i)
        qc.ry(theta[i], i)
    qc.barrier()
    entanglement_pairs = [(0, 1), (0, 5), (2, 4), (0, 4), (2, 3), (1, 2)]
    for (i, j) in entanglement_pairs:
        if i < n and j < n:
            qc.iswap(i, j)
    qc.barrier()
    # 2nd encoding: rx rotations
    for i in range(n):
        qc.t(i)
        qc.rx(theta[i], i)
    qc.barrier()
    for (i, j) in entanglement_pairs:
        if i < n and j < n:
            qc.iswap(i, j)
    qc.barrier()
    # 3rd encoding: ry rotations again
    for i in range(n):
        qc.t(i)
        qc.ry(theta[i], i)
    qc.barrier()
    for (i, j) in entanglement_pairs:
        if i < n and j < n:
            qc.iswap(i, j)
    qc.barrier()
    # Final transformation: T then Hadamard
    for i in range(n):
        qc.t(i)
        qc.h(i)
    state = Statevector.from_instruction(qc)
    rho = DensityMatrix(state)
    return rho

def average_density_matrix(asset_data, alpha=2.0):
    """
    For asset_data (T observations × P features), compute the average density matrix.
    """
    T, P = asset_data.shape
    x_min = np.min(asset_data, axis=0)
    x_max = np.max(asset_data, axis=0)
    rho_sum = None
    for t in range(T):
        x = asset_data[t]
        rho = quantum_feature_map_density(x, x_min, x_max, alpha)
        if rho_sum is None:
            rho_sum = rho.data
        else:
            rho_sum += rho.data
    avg_rho = rho_sum / T
    return avg_rho

def frobenius_distance(rho1, rho2, scale=1.5):
    """
    Compute the scaled Frobenius distance between two density matrices.
    """
    diff = rho1 - rho2
    return scale * (0.5 * np.sqrt(np.trace(diff.conj().T @ diff)))

def compute_distance_matrix(rho_list):
    """
    Compute a distance matrix from a list of density matrices.
    """
    N = len(rho_list)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = frobenius_distance(rho_list[i], rho_list[j])
            D[i, j] = d
            D[j, i] = d
    return D

def quantum_ordering(D):
    """
    Order assets based on quantum distances using hierarchical clustering.
    """
    condensed_D = squareform(D)
    Z = linkage(condensed_D, method='ward')
    order = leaves_list(Z)
    return order

##############################################
# Synthetic Data Generation and Yahoo Finance
##############################################
def generate_valid_correlation_matrix(N, low=0.05, high=0.95, seed=None):
    """Generate a valid random correlation matrix."""
    np.random.seed(seed)
    A = np.random.uniform(low, high, size=(N, N))
    A = (A + A.T) / 2  # Ensure symmetry
    np.fill_diagonal(A, 1.0)
    eigvals, eigvecs = np.linalg.eigh(A)
    min_eigval = np.min(eigvals)
    if min_eigval < 0:
        A += np.eye(N) * (-min_eigval + 1e-3)
    return A

def simulate_asset_features_regimes(N=10, T=1000, P=6, seed=42):
    """
    Generate synthetic asset returns and features under two market regimes.
    """
    np.random.seed(seed)
    regimes = np.zeros(T, dtype=int)
    regimes[0] = np.random.choice([0, 1], p=[0.8, 0.2])
    for t in range(1, T):
        if regimes[t-1] == 0:
            regimes[t] = 0 if np.random.rand() < 0.95 else 1
        else:
            regimes[t] = 1 if np.random.rand() < 0.80 else 0
    mu_normal = np.random.uniform(0.001, 0.01, size=N)
    sigma_normal = np.random.uniform(0.1, 0.3, size=N)
    mu_crisis = np.random.uniform(-0.01, 0, size=N)
    sigma_crisis = np.random.uniform(0.04, 0.08, size=N)
    corr_normal = generate_valid_correlation_matrix(N, low=0.0, high=0.4)
    corr_crisis = generate_valid_correlation_matrix(N, low=0.7, high=0.99)
    breakpoints = np.arange(100, T, 100)
    num_breaks = min(3, len(breakpoints))
    structural_breaks = np.sort(np.random.choice(breakpoints, size=num_breaks, replace=False))
    asset_returns = np.zeros((T, N))
    current_corr = corr_normal
    for t in range(T):
        if regimes[t] == 0:
            mu_t = mu_normal
            sigma_t = sigma_normal
            corr_t = current_corr
        else:
            mu_t = mu_crisis
            sigma_t = sigma_crisis
            corr_t = current_corr
        if t % np.random.randint(50, 100) == 0 and t > 50:
            noise = np.random.uniform(-0.005, 0.005, size=N)
            mu_t += noise
            sigma_t *= np.random.uniform(0.8, 1.2, size=N)
        if t in structural_breaks:
            current_corr = generate_valid_correlation_matrix(N, low=0.1, high=0.9, seed=seed+t)
        cov_t = np.diag(sigma_t) @ corr_t @ np.diag(sigma_t)
        min_eig = np.min(np.real(eigh(cov_t, eigvals_only=True)))
        if min_eig < 1e-6:
            cov_t += np.eye(N) * max(-min_eig + 1e-4, 1e-6)
        try:
            L = np.linalg.cholesky(cov_t)
        except np.linalg.LinAlgError:
            print(f"Warning: Cholesky failed at t={t}, applying SVD fix.")
            U, S, Vt = np.linalg.svd(cov_t)
            S = np.maximum(S, 1e-6)
            cov_t = U @ np.diag(S) @ Vt
            L = np.linalg.cholesky(cov_t)
        if regimes[t] == 0:
            z = np.random.randn(N)
        else:
            z = stats.t.rvs(df=4, size=N) * np.random.choice([-1, 1], size=N)
        asset_returns[t] = mu_t + L @ z

    # # Generate nonlinear asset features for each asset.
    # data = np.zeros((N, T, P))
    # for i in range(N):
    #     for t in range(T):
    #         r = asset_returns[t, i]
    #         f0 = r
    #         f1 = r ** 2
    #         f2 = abs(r) ** 1.5
    #         f3 = np.log1p(abs(r)) * np.sign(r)
    #         f4 = np.tanh(r)
    #         f5 = np.sign(r) * np.sqrt(abs(r) + 1e-6)
    #         data[i, t, :] = np.array([f0, f1, f2, f3, f4, f5])
    # return data, asset_returns, regimes

    features = np.zeros((N, T, P))
    for i in range(N):
        # Extract the return series for asset i
        asset_series = asset_returns[:, i]
        for t in range(T):
            features[i, t, :] = compute_financial_features(asset_series, t)
    return features, asset_returns, regimes

def fetch_stock_data(tickers, start_date, end_date, P=6):
    """
    Fetches stock data from Yahoo Finance, extracts 'Close' prices,
    computes daily returns, and ensures data integrity.
    """
    # Download stock data
    price_data = yf.download(tickers, start=start_date, end=end_date)

    # Handle MultiIndex issue and extract 'Close' prices
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data.xs('Close', level=0, axis=1)

    # Drop tickers (columns) that are entirely NaN
    price_data = price_data.dropna(axis=1, how='all')
    if price_data.empty:
        raise ValueError("No valid ticker data found after filtering out missing values. Check tickers or date range.")

    # Identify which tickers did not return data
    downloaded_tickers = list(price_data.columns)
    missing_tickers = [ticker for ticker in tickers if ticker not in downloaded_tickers]
    if missing_tickers:
        print(f"Warning: No data found for the following tickers: {missing_tickers}")

    # Drop rows with all NaNs
    price_data = price_data.dropna(how='all')
    if price_data.empty:
        raise ValueError("All data is missing after dropping NaNs. Check tickers or date range.")

    # Compute daily returns
    asset_returns = price_data.pct_change().dropna()
    if asset_returns.empty:
        raise ValueError("Asset returns are empty after computing percentage change. Check data.")

    # Convert returns to NumPy array and generate features
    asset_returns_np = asset_returns.to_numpy()
    T, N = asset_returns_np.shape  # Time steps (T) and assets (N)

    print(f"Successfully downloaded data for {N} tickers (out of {len(tickers)-1})")

    features = np.zeros((N, T, P))
    for i in range(N):
        for t in range(T):
            r = asset_returns_np[t, i]
            f0 = r
            f1 = r ** 2
            f2 = abs(r) ** 1.5
            f3 = np.log1p(abs(r)) * np.sign(r)
            f4 = np.tanh(r)
            f5 = np.sign(r) * np.sqrt(abs(r) + 1e-6)
            features[i, t, :] = np.array([f0, f1, f2, f3, f4, f5])


    features = np.zeros((N, T, P))
    for i in range(N):
        asset_series = asset_returns_np[:, i]
        for t in range(T):
            features[i, t, :] = compute_financial_features(asset_series, t)


    # Dummy regime classification (all zeros)
    regimes = np.zeros(T, dtype=int)

    return features, asset_returns_np, regimes, price_data

##############################################
# Performance Metrics: Sharpe Ratio, PSR, MinTRL
##############################################
def sharpe_ratio(returns, rf=0.01, annualization_factor=252):
    rf = rf / annualization_factor  # Convert risk-free rate to daily (or other frequency)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    daily_sr = (mean_ret - rf) / std_ret if std_ret != 0 else 0
    annual_sr = daily_sr * np.sqrt(annualization_factor)
    return annual_sr

def probabilistic_sharpe_ratio(returns, target_SR=0, rf=0.01, annualization_factor=252):
    rf = rf / annualization_factor  # Convert risk-free rate to daily
    target_SR = target_SR / np.sqrt(annualization_factor)  # Ensure target_SR is in daily terms
    n = len(returns)
    sr = sharpe_ratio(returns, rf=0.01, annualization_factor=252)  # annual
    sr_daily = sr / np.sqrt(annualization_factor)
    sample_skew = skew(returns, bias=False)
    sample_kurt = kurtosis(returns, fisher=False, bias=False)

    denom = np.sqrt(1 - sample_skew * sr_daily + ((sample_kurt - 1) / 4) * sr_daily**2)
    if denom <= 0:
        psr = 1.0 if sr_daily > target_SR else 0.0
    else:
        z = (sr_daily - target_SR) * np.sqrt(n - 1) / denom
        psr = norm.cdf(z)

    return sr, psr, sample_skew, sample_kurt

def min_track_record_length(returns, target_SR, prob=0.95):
    """
    Compute the Minimum Track Record Length (MinTRL).
    """
    n = len(returns)
    sr, _, sample_skew, sample_kurt = probabilistic_sharpe_ratio(returns, target_SR, rf=0.01, annualization_factor=252)
    if sr <= target_SR:
        return np.inf
    factor = 1 - sample_skew * sr + ((sample_kurt - 1) / 4) * sr**2
    min_trl = 1 + factor * (norm.ppf(prob) / (sr - target_SR))**2
    return min_trl

##############################################
# Additional Functions: Markowitz and Equal Weights
##############################################
def markowitz_weights(cov: np.ndarray) -> np.ndarray:
    """
    Compute inverse-variance (Markowitz) portfolio weights.
    """
    inv_diag = 1 / np.diag(cov)
    w = inv_diag / np.sum(inv_diag)
    return w

def equal_weights(n: int) -> np.ndarray:
    """
    Compute equal weights for n assets.
    """
    return np.ones(n) / n


##############################################
# Helper Functions (Classical HRP Ordering)
##############################################
def distance_corr(corr_matrix: pd.DataFrame) -> np.ndarray:
    """
    Compute the distance matrix based on correlation.
    The distance is computed as: sqrt((1 - corr) / 2)
    """
    return np.sqrt((1 - corr_matrix) / 2.0)

def quasi_diagonal(linkage_matrix: np.ndarray) -> list:
    """
    Return a sorted list of original items to reorder the correlation matrix.
    This procedure quasi-diagonalizes the linkage matrix.
    """
    linkage_matrix = linkage_matrix.astype(int)
    sorted_items = pd.Series([linkage_matrix[-1, 0], linkage_matrix[-1, 1]])
    num_items = linkage_matrix[-1, 3]

    while sorted_items.max() >= num_items:
        # Set new indices with spacing to allow for insertion
        sorted_items.index = range(0, sorted_items.shape[0] * 2, 2)
        # Find clusters that are not original items
        dataframe = sorted_items[sorted_items >= num_items]
        i = dataframe.index
        j = dataframe.values - num_items
        # Replace the cluster index with its two children from the linkage matrix
        sorted_items[i] = linkage_matrix[j, 0]
        dataframe = pd.Series(linkage_matrix[j, 1], index=i + 1)
        sorted_items = sorted_items._append(dataframe)
        sorted_items = sorted_items.sort_index()
        sorted_items.index = range(sorted_items.shape[0])
    return sorted_items.tolist()

def classical_hrp_ordering(asset_returns: np.ndarray) -> list:
    """
    Order assets based on the correlation matrix of asset returns
    using single linkage and the quasi-diagonalization procedure.
    """
    # Compute the correlation matrix from asset returns
    corr = np.corrcoef(asset_returns, rowvar=False)
    corr_df = pd.DataFrame(corr)
    
    # Compute distance matrix based on correlation
    distance = distance_corr(corr_df)
    
    # Perform single linkage clustering on the distance matrix
    link = linkage(distance, method="single")
    
    # Obtain the ordering via quasi-diagonalization
    sorted_items = quasi_diagonal(link)
    # Map sorted indices back to the DataFrame's index
    sorted_items = corr_df.index[sorted_items].tolist()
    
    return sorted_items

def hrp_recursive_bisection(cov: np.ndarray, sorted_items: list) -> pd.Series:
    """
    Compute the HRP portfolio weights using recursive bisection.
    A wrapper that converts a numpy covariance matrix to a pandas DataFrame
    and calls recursive_bisection from RiskLabAI.optimization.
    """
    cov_df = pd.DataFrame(cov)
    return recursive_bisection(cov_df, sorted_items)

##############################################
# New Functions for Kernel-based HRP Ordering
##############################################
def gaussian_mmd_distance(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute the Maximum Mean Discrepancy (MMD) distance between two sets of features X and Y
    using a Gaussian kernel. X and Y are arrays of shape (num_samples, num_features).
    """
    # Compute squared norms for each row in X and Y
    X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
    Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)
    
    # Compute kernel matrices
    K_XX = np.exp(- (X_norm + X_norm.T - 2 * np.dot(X, X.T)) / (2 * sigma**2))
    K_YY = np.exp(- (Y_norm + Y_norm.T - 2 * np.dot(Y, Y.T)) / (2 * sigma**2))
    K_XY = np.exp(- (X_norm + Y_norm - 2 * np.dot(X, Y.T)) / (2 * sigma**2))
    
    mmd_sq = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return np.sqrt(max(mmd_sq, 0))

def compute_kernel_distance_matrix(features_list: list, sigma: float = 1.0) -> np.ndarray:
    """
    Compute an N x N distance matrix where the (i,j) entry is the Gaussian MMD distance
    between the features of asset i and asset j.
    """
    N = len(features_list)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = gaussian_mmd_distance(features_list[i], features_list[j], sigma)
            D[i, j] = d
            D[j, i] = d
    return D

def kernel_based_hrp_ordering(features_list: list, sigma: float = 1.0) -> list:
    """
    Compute the ordering of assets based on an MMD distance matrix computed using a Gaussian Kernel.
    The MMD distance is computed between the features of each asset.
    """
    D = compute_kernel_distance_matrix(features_list, sigma)
    # Explicitly force symmetry by taking the upper triangle and mirroring it
    D_sym = np.triu(D) + np.triu(D, k=1).T
    # Replace any NaN or infinite values (if present)
    D_sym = np.nan_to_num(D_sym, nan=0.0, posinf=0.0, neginf=0.0)
    # Optionally, round to remove small floating-point discrepancies
    D_sym = np.round(D_sym, decimals=8)
    # Convert the symmetric distance matrix to a condensed distance matrix for linkage
    D_condensed = squareform(D_sym)
    link = linkage(D_condensed, method="single")
    sorted_items = quasi_diagonal(link)
    return sorted_items

