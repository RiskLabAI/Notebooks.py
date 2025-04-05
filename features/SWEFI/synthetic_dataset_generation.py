import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.datasets as Datasets


def generate_ar_time_series(n, ar_params, ma_params, sigma=1.0):
    """
    Generate an autoregressive (AR) time series.

    This function generates a time series based on an autoregressive (AR) model with specified parameters.
    The AR model is defined by the given AR parameters, while the moving average (MA) parameters and standard deviation of the white noise are used to simulate the time series.

    Parameters:
    - n (int): The length of the time series to generate. Must be a positive integer.
    - ar_params (list of float): A list of AR parameters [phi1, phi2, ..., phi_p], where p is the order of the AR process.
      The length of this list determines the order of the AR process. Each parameter phi_i should be a float representing the coefficient for the corresponding lag.
    - ma_params (list of float): A list of MA parameters [theta1, theta2, ..., theta_q], where q is the order of the MA process.
      The length of this list determines the order of the MA process. Each parameter theta_i should be a float representing the coefficient for the corresponding lag.
    - sigma (float, optional): The standard deviation of the white noise. Defaults to 1.0. This controls the variance of the noise term in the time series.

    Returns:
    - time_series (numpy.ndarray): The generated AR time series as a NumPy array of length `n`.

    Example:
    >>> generate_ar_time_series(n=100, ar_params=[0.5, -0.2], ma_params=[0.3], sigma=1.0)
    array([ 0.41716091, -0.78926768,  0.29385976, ...,  0.07809191, -0.39316153])

    Notes:
    - The function uses NumPy for numerical operations and random number generation.
    - Ensure that the AR and MA parameters are provided as lists of appropriate length, matching the orders of the AR and MA processes respectively.
    - The generated time series will have a length specified by the parameter `n` and will be affected by the specified AR and MA parameters as well as the standard deviation of the noise.

    Raises:
    - ValueError: If `n` is not a positive integer.
    - ValueError: If `ar_params` or `ma_params` is not a list of floats.
    """

    # Create the AR coefficients for the model
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]

    # Generate the ARMA process
    arma_process = sm.tsa.ArmaProcess(ar, ma)
    time_series = arma_process.generate_sample(nsample=n, scale=sigma)

    return time_series


def create_lagged_data(time_series, k):
    """
    Create a DataFrame with the time series and its k lags.

    This function generates a DataFrame where each row represents a time step in the input time series. 
    The DataFrame includes the original time series as well as its k lagged versions. Lagged versions are 
    created by shifting the time series by one or more time steps. The resulting DataFrame contains the 
    original time series in the first column and its k lagged versions in subsequent columns.

    Parameters:
    - time_series (array-like): The input time series data. This can be a list, NumPy array, or any 
      structure that supports indexing.
    - k (int): The number of lags to generate. This determines how many lagged columns will be 
      added to the DataFrame. For instance, if k=3, the DataFrame will include the time series and 
      its 3 lagged versions.

    Returns:
    - numpy.ndarray: A NumPy array where each row corresponds to a time step in the time series, and 
      each column represents the original time series or its lagged versions. The array has shape 
      (n-k, k+1), where n is the length of the original time series.

    Example:
    >>> import numpy as np
    >>> time_series = np.array([1, 2, 3, 4, 5])
    >>> k = 2
    >>> create_lagged_data(time_series, k)
    array([[ 3,  2,  1],
           [ 4,  3,  2],
           [ 5,  4,  3]])

    Notes:
    - The function uses `np.roll` to create lagged versions of the time series. The rolling operation 
      will introduce NaN values at the beginning of the series, which are subsequently removed.
    - The resulting DataFrame has the original time series in the first column and the k lagged 
      versions in the following columns.
    - Make sure the length of the time series is greater than k to avoid an empty DataFrame.

    """
    data = {'y': time_series}
    for i in range(1, k + 1):
        data[f'lag_{i}'] = np.roll(time_series, i)

    df = pd.DataFrame(data)
    df = df.iloc[k:]  # Drop the initial rows with NaN values due to the roll

    return df.to_numpy()

def generate_cross_sectional_dataset(
    n_informative: int = 5, 
    n_redundant: int = 25,  
    n_noise: int = 5, 
    n_samples: int = 10000,  
    random_state: int = 41,  
    sigma_std: float = 0.1, 
    n_clusters_per_class: int = 2,
) -> tuple:
    """
    Generates a synthetic cross-sectional dataset for classification tasks.

    This function creates a dataset with a specified number of informative, redundant,
    and noisy features. The dataset is created using scikit-learn's `make_classification` function,
    with additional redundant features generated as noisy combinations of the informative features.

    Parameters
    ----------
    n_informative : int, optional, default=5
        The number of informative features in the dataset. These features are used to generate the target labels.

    n_redundant : int, optional, default=25
        The number of redundant features. These are linear combinations of the informative features with added noise.

    n_noise : int, optional, default=5
        The number of noise features. These features are random and do not contribute to the target labels.

    n_samples : int, optional, default=10000
        The number of samples (rows) in the generated dataset.

    random_state : int, optional, default=41
        The seed used by the random number generator for reproducibility of the results.

    sigma_std : float, optional, default=0.1
        The standard deviation of the Gaussian noise added to the redundant features.

    n_clusters_per_class : int, optional, default=2
        The number of clusters per class for the classification problem.

    Returns
    -------
    tuple
        A tuple containing:
        - X : pd.DataFrame
            A DataFrame with shape (n_samples, n_features) where n_features is the sum of 
            `n_informative`, `n_redundant`, and `n_noise` features. The columns are named according 
            to their type (informative, redundant, or noise).
        - y : pd.Series
            A Series with shape (n_samples,) containing the target labels.

    Notes
    -----
    - The total number of features used to generate the dataset will be the sum of 
      `n_informative`, `n_redundant`, and `n_noise`.
    - Redundant features are generated as noisy versions of the informative features.
    - Informative features contribute directly to the class labels, while redundant and noise 
      features do not.

    Examples
    --------
    >>> X, y = generate_cross_sectional_dataset()
    >>> X.head()
       I0   I1   I2   I3   I4  N0  N1  N2  N3  N4  R0 (from I2)  R1 (from I1)  ...
    0  1.2  3.4  5.6  7.8  9.0  0.1  0.2  0.3  0.4  0.5       5.7           3.3  ...
    1  1.1  3.5  5.7  7.9  9.1  0.2  0.3  0.4  0.5  0.6       5.6           3.4  ...
    
    >>> y.head()
    0    0
    1    1
    dtype: int64
    """
    n_features = n_informative + n_redundant + n_noise
    np.random.seed(random_state)
    X, y = Datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features-n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=random_state,
        n_clusters_per_class=n_clusters_per_class,
    )

    columns = [f"I{i}" for i in range(n_informative)]
    columns += [f"N{i}" for i in range(n_noise)]
    X, y = pd.DataFrame(X, columns=columns), pd.Series(y)
    i = np.random.choice(range(n_informative), size=n_redundant)
    for k, j in enumerate(i):
        X[f"R{k} (from I{j})"] = X[f"I{j}"] + np.random.normal(size=X.shape[0]) * sigma_std

    return X, y

def generate_multivariate_time_series_dataset(
    n_informative: int = 10,  
    n_redundant: int = 10,  
    n_noise: int = 20,
    n_samples: int = 10000, 
    n_time_steps: int = 2, 
    random_state: int = 41,  
    sigma_std: float = 0.1,  
    time_series_params = [[0.3], [0.3], 0.1], 
):
    """
    Generate a multivariate time series dataset with informative, redundant, and noise features.

    This function creates a dataset of time series data where each feature can be categorized as 
    informative, redundant, or noise. Informative features are generated using autoregressive 
    processes, redundant features are linear combinations of informative features with added noise, 
    and noise features are generated using a simpler autoregressive process.

    Parameters:
    - n_informative (int): Number of informative features to generate. Default is 10.
    - n_redundant (int): Number of redundant features to generate. Default is 10.
    - n_noise (int): Number of noise features to generate. Default is 20.
    - n_samples (int): Number of samples (data points) in the dataset. Default is 10000.
    - n_time_steps (int): Number of time steps for each time series. Default is 2.
    - random_state (int): Seed for the random number generator to ensure reproducibility. Default is 41.
    - sigma_std (float): Standard deviation of noise added to redundant features. Default is 0.1.
    - time_series_params (list): Parameters for generating informative features using AR processes. 
      The list should contain:
        - List of autoregressive coefficients for the AR process.
        - List of moving average coefficients for the AR process (not used in this function).
        - Variance of the noise in the AR process.

    Returns:
    - X (pd.DataFrame): DataFrame containing the generated features. The columns are labeled as:
        - "I{i} lag{j}" for informative features.
        - "R{i} from I{k} lag{j}" for redundant features indicating their origin.
        - "N{i} lag{j}" for noise features.
      The features are reshaped into a 2D array where each row corresponds to a sample.
    - y (pd.Series): Series containing the binary target variable. The target is generated as 
      a binary classification based on the linear combination of informative features plus some 
      noise.

    Example:
    >>> X, y = generate_multivariate_time_series_dataset()
    >>> X.head()
       I0 lag0  I0 lag1  I1 lag0  I1 lag1  ...  N19 lag0  N19 lag1
    0      0.12     -0.34     0.45      0.67  ...     -0.02      0.01
    1      0.10     -0.33     0.44      0.66  ...     -0.01      0.03
    2      0.14     -0.31     0.46      0.65  ...     -0.03      0.02
    3      0.13     -0.35     0.48      0.68  ...     -0.01      0.04

    >>> y.head()
    0    1
    1    0
    2    1
    3    1
    dtype: int64
    """

    n_lag = n_time_steps - 1

    informative_features = np.zeros(shape=(n_samples, n_informative, n_time_steps))
    for k in range(n_informative):
        time_series = generate_ar_time_series(n_samples + n_lag, time_series_params[0], time_series_params[1], 1)
        informative_features[:, k, :] = create_lagged_data(time_series, n_lag)

    np.random.seed(random_state)
    i = np.random.choice(range(n_informative), size=n_redundant)
    np.random.seed()
    
    linear_redundant_features = np.zeros((n_samples, n_redundant, n_time_steps))
    linear_redundant_features_from_ = []

    for k, j in enumerate(i):
        linear_redundant_features[:, k, :] = informative_features[:, j, :] + np.random.normal(size=(n_samples, n_time_steps)) * sigma_std
        linear_redundant_features_from_.append(j)

    # Generate noise features
    noise_features = np.zeros(shape=(n_samples, n_noise, n_time_steps))
    for k in range(n_noise):
        time_series = generate_ar_time_series(n_samples + n_lag, 0.5, 0.5, 1)
        noise_features[:, k, :] = create_lagged_data(time_series, n_lag)

    # Concatenate all features
    X = np.concatenate([
        informative_features,
        linear_redundant_features,
        noise_features],
        axis=1)

    columns = [f"I{i} lag{j}" for i in range(n_informative) for j in range(n_time_steps)]
    columns += [f"R{i} from I{k} lag{j}" for i, k in enumerate(linear_redundant_features_from_) for j in range(n_time_steps)]
    columns += [f"N{i} lag{j}" for i in range(n_noise) for j in range(n_time_steps)]

    X = X.reshape((n_samples, -1))
    X = pd.DataFrame(X, columns=columns)

    I = X.iloc[:, :n_informative * n_time_steps]
    I = (I - I.mean()) / I.std()
    weights = np.ones(shape=len(I.columns))

    # Calculate linear combination of features plus some noise
    linear_combination = I @ weights
    # Generate binary target variable based on probabilities
    y = (linear_combination > 0).astype(int)

    y = pd.Series(y)

    return X, y

