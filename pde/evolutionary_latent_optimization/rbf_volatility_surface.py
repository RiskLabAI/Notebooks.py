import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product


class RBFVolatilitySurface:
    def __init__(
        self,
        coefficients,
        maturity_times,
        strike_prices,
        strike_std,
        maturity_std,
        constant_volatility,
    ):
        """
        Initialize the RBFVolatilitySurface class.

        Parameters:
        - coefficients: Array of RBF coefficients ω_j.
        - maturity_times: Array of RBF centers for time to maturity T_j.
        - strike_prices: Array of RBF centers for strike price K_j.
        - maturity_std: Standard deviation (spread) for time to maturity in the RBF.
        - strike_std: Standard deviation (spread) for strike prices in the RBF.
        - constant_volatility: Constant term φ_0 representing the weighted average of Black-Scholes implied volatilities.
        """
        self.coefficients = np.array(coefficients)
        self.maturity_times = np.array(maturity_times)
        self.strike_prices = np.array(strike_prices)
        self.strike_std = strike_std
        self.maturity_std = maturity_std
        self.constant_volatility = constant_volatility

    def implied_volatility_surface(
        self, 
        time_to_maturity, 
        strike_price
    ):
        """
        Calculate the implied volatility surface at a given pair (T, K).

        Parameters:
        - time_to_maturity: The time to maturity T.
        - strike_price: The strike price K.

        Returns:
        - The implied volatility σ(T, K) at (T, K).
        """

        rbf_values = np.exp(
            - ((time_to_maturity - self.maturity_times) ** 2) / (2 * self.maturity_std ** 2)
            - ((strike_price - self.strike_prices) ** 2) / (2 * self.strike_std ** 2)
        )
        rbf_sum = np.dot(self.coefficients, rbf_values)

        # Volatility surface σ(T, K) = φ_0 + Σ_j ω_j φ_j(T, K)
        return self.constant_volatility + rbf_sum

    def implied_volatility_tensor(
        self, 
        time_to_maturity_tensor, 
        strike_price_tensor
    ):
        """
        Calculate the implied volatility surface at given tensors of time to maturity and strike prices
        in the PyTorch environment, ensuring correct device handling.

        Parameters:
        - time_to_maturity_tensor: Tensor of times to maturity.
        - strike_price_tensor: Tensor of strike prices.

        Returns:
        - Tensor of implied volatilities, placed on the same device as the input tensors.
        """
        # Ensure that the tensors are on the same device as the inputs
        device = time_to_maturity_tensor.device

        # Move numpy arrays to torch tensors on the appropriate device
        maturity_times_tensor = torch.tensor(self.maturity_times, device=device)
        strike_prices_tensor = torch.tensor(self.strike_prices, device=device)
        coefficients_tensor = torch.tensor(self.coefficients, device=device)

        # Compute RBF values for each time-to-maturity and strike price in the tensor
        # This operation will broadcast time_to_maturity_tensor and strike_price_tensor against all RBF centers
        rbf_values = torch.exp(
            -((time_to_maturity_tensor.unsqueeze(-1) - maturity_times_tensor) ** 2) / (2 * self.maturity_std ** 2)
            -((strike_price_tensor.unsqueeze(-1) - strike_prices_tensor) ** 2) / (2 * self.strike_std ** 2)
        )

        # Compute the weighted sum of the RBF values across all centers using the coefficients
        # Perform a weighted sum across the last dimension (axis -1) to aggregate the RBF values for each input point
        rbf_sum = torch.sum(rbf_values * coefficients_tensor, dim=-1)

        # Compute the implied volatility: constant volatility + RBF sum
        return self.constant_volatility + rbf_sum

    @staticmethod
    def calculate_constant_volatility(
        data_implied_volatilities, 
        data_maturity_times, 
        data_strike_prices, 
        risk_free_rate, 
        underlying_price, 
        epsilon=1e-6
    ):
        """
        Calculate the constant term φ_0 as a weighted average of the Black-Scholes implied volatilities.

        Parameters:
        - data_implied_volatilities: Array of Black-Scholes implied volatilities σ_{BS}(T_i, K_i).
        - data_maturity_times: Array of maturity times T_i.
        - data_strike_prices: Array of strike prices K_i.
        - risk_free_rate: Risk-free rate r.
        - underlying_price: Current spot price of the underlying asset S.
        - epsilon: Small constant to avoid division by zero.

        Returns:
        - The constant volatility φ_0.
        """
        weights = []
        weighted_volatilities = []

        for t_i, k_i, implied_volatility in zip(
            data_maturity_times, data_strike_prices, data_implied_volatilities
        ):
            forward_strike = k_i * np.exp(-risk_free_rate * t_i)
            weight = 1 / ((underlying_price - forward_strike) ** 2 + epsilon)
            weights.append(weight)
            weighted_volatilities.append(weight * implied_volatility)

        constant_volatility = np.sum(weighted_volatilities) / np.sum(weights)
        return constant_volatility


class SurfaceDataset(Dataset):
    def __init__(
        self, 
        sampled_surface_coefficients, 
        maturity_time_list, 
        strike_price_list, 
        strike_std, 
        maturity_std, 
        constant_volatility, 
        strike_infinity
    ):
        """
        Initialize the SurfaceDataset class.

        Parameters:
        - sampled_surface_coefficients: 2D NumPy array where each row is a set of surface coefficients.
        - maturity_time_list: List of unique maturity times for PDE and boundary points.
        - strike_price_list: List of unique strike prices for PDE and boundary points.
        - strike_std: Standard deviation for strike prices in the RBF.
        - maturity_std: Standard deviation for maturity times in the RBF.
        - constant_volatility: Constant volatility term φ_0.
        - strike_infinity: A large strike price used for the boundary condition at infinity.
        """
        # Store the parameters
        self.sampled_surface_coefficients = sampled_surface_coefficients
        self.maturity_time_list = maturity_time_list
        self.strike_price_list = strike_price_list
        self.strike_std = strike_std
        self.maturity_std = maturity_std
        self.constant_volatility = constant_volatility
        self.strike_infinity = strike_infinity

        # Create the full data grid of points
        self._create_data_grid()

    def _create_data_grid(self):
        """
        Create the grid of time-to-maturity and strike price points for PDE and boundary points.
        This will be used to create the full dataset for training the PINN.

        Returns:
        - None.
        """
        # Create the product grids for PDE and boundary conditions
        pde_grid = list(product(self.maturity_time_list, self.strike_price_list))
        boundary_maturity_zero = list(product([0], self.strike_price_list))
        boundary_strike_zero = list(product(self.maturity_time_list, [0]))
        boundary_strike_infinity = list(product(self.maturity_time_list, [self.strike_infinity]))

        # Combine all the points into a single list
        combined_grid = (
            pde_grid
            + boundary_maturity_zero
            + boundary_strike_zero
            + boundary_strike_infinity
        )

        # Extract time to maturity and strike price values from the combined grid
        time_to_maturity, strike_price = zip(*combined_grid)
        self.pde_time_to_maturity, self.pde_strike_price = zip(*pde_grid)

        self.time_to_maturity, self.strike_price = time_to_maturity, strike_price

    def __len__(self):
        """
        Return the number of samples (number of surfaces).
        """
        return len(self.sampled_surface_coefficients)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Parameters:
        - idx: Index of the sampled surface coefficients.

        Returns:
        - A tuple of tensors (time_to_maturity, strike_price, implied_volatility).
        """
        # Get the surface coefficients for the sampled surface
        coefficients = self.sampled_surface_coefficients[idx]

        # Convert to PyTorch tensors
        time_to_maturity_tensor = torch.tensor(self.time_to_maturity, dtype=torch.float32, requires_grad=True)
        strike_price_tensor = torch.tensor(self.strike_price, dtype=torch.float32, requires_grad=True)

        # Initialize the RBFVolatilitySurface for the current sample
        rbf_surface = RBFVolatilitySurface(
            coefficients=coefficients,
            maturity_times=self.pde_time_to_maturity,
            strike_prices=self.pde_strike_price,
            strike_std=self.strike_std,
            maturity_std=self.maturity_std,
            constant_volatility=self.constant_volatility
        )

        # Disable gradient tracking while computing the implied volatility tensor
        with torch.no_grad():
            implied_volatility_tensor = rbf_surface.implied_volatility_tensor(
                time_to_maturity_tensor.detach(),  # detach to ensure grads are not tracked
                strike_price_tensor.detach()       # detach to ensure grads are not tracked
            )

        # Return the tensors needed for training (time_to_maturity, strike_price, implied_volatility)
        return time_to_maturity_tensor, strike_price_tensor, implied_volatility_tensor
