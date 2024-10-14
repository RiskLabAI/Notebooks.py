import numpy as np
import pandas as pd
from scipy.stats import norm


def generate_sabr_call_options(
    alpha,
    beta,
    rho,
    nu,
    maturity_times,
    strike_prices,
    risk_free_rate,
    underlying_price,
):
    """
    Generate a dataset of call option prices and implied volatilities using the SABR model for implied volatility
    and the Black-Scholes model for option pricing.

    Parameters:
    - alpha: Stochastic volatility parameter in the SABR model.
    - beta: Elasticity parameter in the SABR model.
    - rho: Correlation between the asset price and volatility in the SABR model.
    - nu: Volatility of volatility parameter in the SABR model.
    - maturity_times: Array of times to maturity T_i.
    - strike_prices: Array of strike prices K_i.
    - risk_free_rate: Risk-free interest rate r.
    - underlying_price: Current price of the underlying asset S_0.

    Returns:
    - A pandas DataFrame with columns 'time_to_maturity', 'strike_price', 'call_option_price', 'implied_volatility'.
    """

    def forward_price(
        s_0, 
        t_i, 
        r
    ):
        """Calculate forward price F_i using s_0 and r."""
        return s_0 * np.exp(r * t_i)

    def sabr_implied_volatility(
        f_i, 
        k_i, 
        t_i, 
        alpha, 
        beta, 
        rho, 
        nu
    ):
        """Calculate the implied volatility using the SABR model."""
        if f_i == k_i:
            # Handle the ATM case
            return alpha * (
                1
                + (
                    (1 - beta) ** 2 * alpha ** 2 / (24 * f_i ** (2 - 2 * beta))
                    + rho * beta * nu * alpha / (4 * f_i ** (1 - beta))
                    + (2 - 3 * rho ** 2) * nu ** 2 / 24
                )
                * t_i
            )
        else:
            # Calculate the SABR-implied volatility
            z = (nu / alpha) * (f_i * k_i) ** ((1 - beta) / 2) * np.log(f_i / k_i)
            chi_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

            return (
                alpha
                * (
                    1
                    + (
                        (1 - beta) ** 2 * alpha ** 2 / (24 * (f_i * k_i) ** (1 - beta))
                        + rho * beta * nu * alpha / (4 * (f_i * k_i) ** ((1 - beta) / 2))
                        + (2 - 3 * rho ** 2) * nu ** 2 / 24
                    )
                    * t_i
                )
                / (
                    (f_i * k_i) ** ((1 - beta) / 2)
                    * (
                        1
                        + (1 - beta) ** 2 / 24 * np.log(f_i / k_i) ** 2
                        + (1 - beta) ** 4 / 1920 * np.log(f_i / k_i) ** 4
                    )
                )
                * z
                / chi_z
            )


    def black_scholes_call_price(
        s_0, 
        k_i, 
        t_i, 
        r, 
        sigma
    ):
        """Calculate the Black-Scholes call option price."""
        d1 = (np.log(s_0 / k_i) + (r + 0.5 * sigma ** 2) * t_i) / (sigma * np.sqrt(t_i))
        d2 = d1 - sigma * np.sqrt(t_i)

        call_price = s_0 * norm.cdf(d1) - k_i * np.exp(-r * t_i) * norm.cdf(d2)
        return call_price

    # Initialize lists to store the results
    times_to_maturity = []
    strikes = []
    call_prices = []
    implied_volatilities = []

    # Generate the dataset for each pair (T_i, K_i) as pairwise inputs
    for t_i, k_i in zip(maturity_times, strike_prices):
        
        # Step 1: Calculate the forward price
        f_i = forward_price(underlying_price, t_i, risk_free_rate)

        # Step 2: Calculate the implied volatility using the SABR model
        implied_vol = sabr_implied_volatility(f_i, k_i, t_i, alpha, beta, rho, nu) 

        # Step 3: Calculate the Black-Scholes call option price
        call_price = black_scholes_call_price(
            underlying_price, k_i, t_i, risk_free_rate, implied_vol
        )

        # Append results to the lists
        times_to_maturity.append(t_i)
        strikes.append(k_i)
        call_prices.append(call_price)
        implied_volatilities.append(implied_vol)

    # Create a DataFrame with the results
    data = {
        "Time to Maturity": times_to_maturity,
        "Strike Price": strikes,
        "Call Option Price": call_prices,
        "Implied Volatility": implied_volatilities,
    }

    return pd.DataFrame(data)
