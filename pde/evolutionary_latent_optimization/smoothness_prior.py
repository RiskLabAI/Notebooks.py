import numpy as np
from scipy.special import roots_genlaguerre
from numpy.linalg import pinv
from rbf_volatility_surface import RBFVolatilitySurface


class RBFQuadraticSmoothnessPrior:
    def __init__(
        self,
        maturity_times,
        strike_prices,
        maturity_std,
        strike_std,
        n_roots,
        smoothness_controller,
        random_state=None,
    ):
        """
        Initialize the RBFQuadraticSmoothnessPrior class.

        Parameters:
        - maturity_times: Array of maturity times T_j.
        - strike_prices: Array of strike prices K_j.
        - maturity_std: Standard deviation for time to maturity.
        - strike_std: Standard deviation for strike prices.
        - n_roots: Number of roots for Generalized Gauss-Laguerre Quadrature.
        - smoothness_controller: Smoothness control parameter.
        - random_state: Random seed for reproducibility.
        """
        self.maturity_times = np.array(maturity_times)
        self.strike_prices = np.array(strike_prices)
        self.maturity_std = maturity_std
        self.strike_std = strike_std
        self.n_roots = n_roots
        self.smoothness_controller = smoothness_controller
        self.random_generator = np.random.default_rng(random_state)

        # Precompute the roots and weights for the generalized Gauss-Laguerre quadrature with α = -1/2
        self.roots, self.weights = roots_genlaguerre(self.n_roots, -0.5)

        # Initialize covariance and lambda matrix placeholder
        self.covariance_matrix = None
        self.lambda_matrix = None

    def compute_psi(
        self, 
        t, 
        k, 
        t_j, 
        k_j, 
        t_k, 
        k_k
    ):
        """
        Compute the polynomial factor Ψ(T, K) for the lambda matrix elements.

        Parameters:
        - t, k: Transformed time and strike price using Gauss-Laguerre quadrature.
        - t_j, k_j: Centers of the RBFs for maturity and strike for the first RBF.
        - t_k, k_k: Centers of the RBFs for maturity and strike for the second RBF.
        """
        term_1 = ((k - k_j) ** 2 / self.strike_std ** 4 - 1 / self.strike_std ** 2) * (
            (k - k_k) ** 2 / self.strike_std ** 4 - 1 / self.strike_std ** 2
        )
        term_2 = ((t - t_j) * (t - t_k)) / self.maturity_std ** 4
        return term_1 + term_2

    def calculate_lambda_matrix(self):
        """
        Calculate the lambda matrix (Λ) based on the smoothness prior distribution of the RBF surface parameters,
        leveraging symmetry to avoid redundant calculations.
        """
        n = len(self.maturity_times)
        lambda_matrix = np.zeros((n, n))

        for j in range(n):
            for k in range(
                j, n
            ):  # Only compute for upper triangle (including diagonal)
                delta_t = self.maturity_times[j] - self.maturity_times[k]
                delta_k = self.strike_prices[j] - self.strike_prices[k]

                # Exponential factor in the lambda matrix formula
                exp_factor = np.exp(
                    -(delta_t ** 2) / (4 * self.maturity_std ** 2)
                    - delta_k ** 2 / (4 * self.strike_std ** 2)
                )

                # Average points for the RBFs
                t_avg = (self.maturity_times[j] + self.maturity_times[k]) / 2
                k_avg = (self.strike_prices[j] + self.strike_prices[k]) / 2

                # Vectorized Gauss-Laguerre quadrature integration
                t_sqrt_roots = np.sqrt(self.roots) * self.maturity_std  # Precompute sqrt(roots) * maturity_std
                k_sqrt_roots = np.sqrt(self.roots) * self.strike_std  # Precompute sqrt(roots) * strike_std

                # Generate all possible combinations of t_val_pos, t_val_neg, k_val_pos, k_val_neg
                t_vals_pos = t_avg + t_sqrt_roots
                t_vals_neg = t_avg - t_sqrt_roots
                k_vals_pos = k_avg + k_sqrt_roots
                k_vals_neg = k_avg - k_sqrt_roots

                # Initialize psi_vals for all combinations and apply conditions
                psi_vals = np.zeros((self.n_roots, self.n_roots, 4))

                # First combination: t_val_pos, k_val_pos
                mask_pos_pos = (t_vals_pos[:, np.newaxis] >= 0) & (k_vals_pos[np.newaxis, :] >= 0)
                psi_vals[:, :, 0] = np.where(
                    mask_pos_pos,
                    self.compute_psi(
                        t_vals_pos[:, np.newaxis],
                        k_vals_pos[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Second combination: t_val_pos, k_val_neg
                mask_pos_neg = (t_vals_pos[:, np.newaxis] >= 0) & (k_vals_neg[np.newaxis, :] >= 0)
                psi_vals[:, :, 1] = np.where(
                    mask_pos_neg,
                    self.compute_psi(
                        t_vals_pos[:, np.newaxis],
                        k_vals_neg[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Third combination: t_val_neg, k_val_pos
                mask_neg_pos = (t_vals_neg[:, np.newaxis] >= 0) & (k_vals_pos[np.newaxis, :] >= 0)
                psi_vals[:, :, 2] = np.where(
                    mask_neg_pos,
                    self.compute_psi(
                        t_vals_neg[:, np.newaxis],
                        k_vals_pos[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Fourth combination: t_val_neg, k_val_neg
                mask_neg_neg = (t_vals_neg[:, np.newaxis] >= 0) & (k_vals_neg[np.newaxis, :] >= 0)
                psi_vals[:, :, 3] = np.where(
                    mask_neg_neg,
                    self.compute_psi(
                        t_vals_neg[:, np.newaxis],
                        k_vals_neg[np.newaxis, :],
                        self.maturity_times[j],
                        self.strike_prices[j],
                        self.maturity_times[k],
                        self.strike_prices[k]
                    ),
                    0
                )

                # Compute psi_val by summing and subtracting according to the original logic
                psi_val_combined = psi_vals[:, :, 0] - psi_vals[:, :, 1] - psi_vals[:, :, 2] + psi_vals[:, :, 3]

                # Compute the integral sum
                integral_sum = np.sum(
                    self.weights[:, np.newaxis] * self.weights[np.newaxis, :] * psi_val_combined / \
                    np.sqrt(self.roots[:, np.newaxis] * self.roots[np.newaxis, :])
                )

                # Update the lambda matrix
                lambda_matrix[j, k] = (
                    exp_factor * self.maturity_std * self.strike_std / 4 * integral_sum
                )

                if j != k:
                    lambda_matrix[k, j] = lambda_matrix[j, k]  # Use symmetry

        lambda_matrix += 1e-6 * np.eye(lambda_matrix.shape[0])

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(lambda_matrix)

        # Clip negative eigenvalues to zero
        eigvals_clipped = np.clip(eigvals, a_min=0, a_max=None)

        # Reconstruct the matrix with the clipped eigenvalues
        lambda_matrix_positive = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        lambda_matrix_positive += 1e-6 * np.eye(lambda_matrix_positive.shape[0])
        lambda_matrix_positive = (lambda_matrix_positive + lambda_matrix_positive.T) / 2  # Ensure symmetry

        self.lambda_matrix = lambda_matrix_positive            

        return self.lambda_matrix    

    def prior_covariance(self):
        """
        Calculate the covariance matrix of the smoothness prior distribution.

        Returns:
        - covariance_matrix: The covariance matrix as gamma^2 * Lambda^(-1).
        """
        # Check if the covariance matrix has been calculated
        if self.lambda_matrix is None:
            self.calculate_lambda_matrix()

        inverse_lambda = pinv(self.lambda_matrix)    

        # Covariance matrix is gamma^2 * Λ^(-1)
        self.covariance_matrix = self.smoothness_controller ** 2 * (inverse_lambda + inverse_lambda.T) / 2
        return self.covariance_matrix

    def sample_smooth_surfaces(
        self, 
        n_samples
    ):
        """
        Sample smooth surface parameters from the Gaussian smoothness prior distribution.

        Parameters:
        - n_samples: Number of samples to generate.

        Returns:
        - samples: A (n_samples, N) array where each row represents a sample of the RBF coefficients.
        """
        # Check if the covariance matrix has been calculated
        if self.covariance_matrix is None:
            self.prior_covariance()

        # Generate samples using multivariate normal distribution
        samples = self.random_generator.multivariate_normal(
            mean=np.zeros(len(self.maturity_times)),
            cov=self.covariance_matrix,
            size=n_samples,
        )

        return samples

    def log_likelihood(
        self,
        data_implied_volatilities,
        data_maturity_times,
        data_strike_prices,
        risk_free_rate,
        underlying_price,
        noise_level=0.0,
    ):
        """
        Calculate the log-likelihood of the observed implied volatilities based on the smoothness prior.

        Parameters:
        - data_implied_volatilities: Array of observed implied volatilities.
        - data_maturity_times: Array of maturity times for the data.
        - data_strike_prices: Array of strike prices for the data.
        - risk_free_rate: Risk-free interest rate.
        - underlying_price: Current price of the underlying asset.
        - noise_level: Noise level in the observations, default is 0.

        Returns:
        - log_likelihood_value: The log-likelihood value based on the provided data and smoothness parameter.
        """
        # Number of data points
        num_data_points = len(data_implied_volatilities)

        # If the lambda matrix is not calculated, calculate it
        if self.lambda_matrix is None:
            self.calculate_lambda_matrix()

        # Calculate the constant term (constant_volatility) using the static method
        constant_volatility = RBFVolatilitySurface.calculate_constant_volatility(
            data_implied_volatilities=data_implied_volatilities,
            data_maturity_times=data_maturity_times,
            data_strike_prices=data_strike_prices,
            risk_free_rate=risk_free_rate,
            underlying_price=underlying_price,
        )

        # Expand dimensions to enable broadcasting
        time_diff = (data_maturity_times[:, np.newaxis] - self.maturity_times[np.newaxis, :]) ** 2
        strike_diff = (data_strike_prices[:, np.newaxis] - self.strike_prices[np.newaxis, :]) ** 2

        # Compute the RBF values for all pairs of (data_maturity_times, data_strike_prices) and (maturity_times, strike_prices)
        rbf_evaluations = np.exp(
            -time_diff / (2 * self.maturity_std ** 2) - strike_diff / (2 * self.strike_std ** 2)
        )

        covariance_matrix = (
            noise_level ** 2 * np.eye(num_data_points)
            + self.smoothness_controller ** 2 * rbf_evaluations @ pinv(self.lambda_matrix) @ rbf_evaluations.T
        )

        # Calculate log determinant of the covariance matrix
        log_det_covariance_matrix = np.linalg.slogdet(covariance_matrix)[1]

        # Inverse of the covariance matrix
        covariance_matrix_inv = pinv(covariance_matrix)

        # Calculate the difference between the observed volatilities and the constant volatility
        volatility_difference = data_implied_volatilities - constant_volatility

        # Calculate the log-likelihood
        log_likelihood_value = (
            -num_data_points / 2 * np.log(2 * np.pi) - 0.5 * log_det_covariance_matrix
            - 0.5 * volatility_difference.T @ covariance_matrix_inv @ volatility_difference
        )

        return log_likelihood_value

    def square_root_expected_squared_frobenius_norm(
        self,
        data_implied_volatilities,
        data_maturity_times,
        data_strike_prices,
        risk_free_rate,
        underlying_price
    ):
        """
        Calculate the square root of the expected squared Frobenius norm of the volatility surface.

        Parameters:
        - data_implied_volatilities: Array of observed implied volatilities.
        - data_maturity_times: Array of maturity times for the data.
        - data_strike_prices: Array of strike prices for the data.
        - risk_free_rate: Risk-free interest rate.
        - underlying_price: Current price of the underlying asset.

        Returns:
        - sqrt_expected_frobenius_norm: The square root of the expected Frobenius norm squared.
        """
        # Number of data points
        num_data_points = len(data_implied_volatilities)

        # If the lambda matrix is not calculated, calculate it
        if self.lambda_matrix is None:
            self.calculate_lambda_matrix()

        # Calculate the constant term (constant_volatility) using the static method
        constant_volatility = RBFVolatilitySurface.calculate_constant_volatility(
            data_implied_volatilities=data_implied_volatilities,
            data_maturity_times=data_maturity_times,
            data_strike_prices=data_strike_prices,
            risk_free_rate=risk_free_rate,
            underlying_price=underlying_price,
        )

        # Expand dimensions to enable broadcasting
        time_diff = (data_maturity_times[:, np.newaxis] - self.maturity_times[np.newaxis, :]) ** 2
        strike_diff = (data_strike_prices[:, np.newaxis] - self.strike_prices[np.newaxis, :]) ** 2

        # Compute the RBF values for all pairs of (data_maturity_times, data_strike_prices) and (maturity_times, strike_prices)
        rbf_evaluations = np.exp(
            -time_diff / (2 * self.maturity_std ** 2) - strike_diff / (2 * self.strike_std ** 2)
        )

        # Compute the expected squared Frobenius norm
        trace_term = np.trace(
            rbf_evaluations.T @ rbf_evaluations @ pinv(self.lambda_matrix)
        )
        expected_squared_frobenius_norm = (
            constant_volatility ** 2 * num_data_points
            + self.smoothness_controller ** 2 * trace_term
        )

        # Return the square root of the expected squared Frobenius norm
        sqrt_expected_squared_frobenius_norm = np.sqrt(expected_squared_frobenius_norm)

        return sqrt_expected_squared_frobenius_norm
