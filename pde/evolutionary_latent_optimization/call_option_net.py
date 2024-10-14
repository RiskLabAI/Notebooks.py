import numpy as np
import torch
import torch.nn as nn


class CallOptionPINN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_layers,
        maturity_min,
        maturity_max,
        strike_min,
        strike_max,
        volatility_mean,
        volatility_std,
    ):
        super(CallOptionPINN, self).__init__()

        # Convert constants to tensors and register as buffers
        self.register_buffer('maturity_min', torch.tensor(maturity_min, dtype=torch.float32))
        self.register_buffer('maturity_max', torch.tensor(maturity_max, dtype=torch.float32))
        self.register_buffer('strike_min', torch.tensor(strike_min, dtype=torch.float32))
        self.register_buffer('strike_max', torch.tensor(strike_max, dtype=torch.float32))
        self.register_buffer('volatility_mean', torch.tensor(volatility_mean, dtype=torch.float32))
        self.register_buffer('volatility_std', torch.tensor(volatility_std, dtype=torch.float32))

        # Define the layers of the network
        layers = []
        input_dim = 3  # Inputs: time_to_maturity, strike_price, volatility

        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        # Final output layer (produces call option price)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights with Xavier normal initialization (gain for ELU)
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self, 
        time_to_maturity, 
        strike_price, 
        volatility
    ):
        # Ensure constants are on the same device and dtype
        maturity_min = self.maturity_min.to(time_to_maturity.device).type_as(
            time_to_maturity
        )
        maturity_max = self.maturity_max.to(time_to_maturity.device).type_as(
            time_to_maturity
        )
        strike_min = self.strike_min.to(strike_price.device).type_as(strike_price)
        strike_max = self.strike_max.to(strike_price.device).type_as(strike_price)
        volatility_mean = self.volatility_mean.to(volatility.device).type_as(volatility)
        volatility_std = self.volatility_std.to(volatility.device).type_as(volatility)

        # Normalize the inputs:
        time_to_maturity_norm = (time_to_maturity - maturity_min) / (maturity_max - maturity_min)
        strike_price_norm = (strike_price - strike_min) / (strike_max - strike_min)
        volatility_standardized = (volatility - volatility_mean) / volatility_std
        sqrt_2 = torch.sqrt(
            torch.tensor(
                2.0,
                device=volatility_standardized.device,
                dtype=volatility_standardized.dtype,
            )
        )
        volatility_norm = 0.5 * (
            1 + torch.erf(volatility_standardized / sqrt_2).float()
        )

        # Concatenate inputs for the network
        inputs = torch.cat(
            [
                time_to_maturity_norm.unsqueeze(-1),
                strike_price_norm.unsqueeze(-1),
                volatility_norm.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Pass through the network
        call_option_price = self.network(inputs)

        return call_option_price.squeeze(-1)


def pinn_dupire_loss(
    call_option_price,
    time_to_maturity,
    strike_price,
    volatility,
    strike_infinity=2.5,
    risk_free_rate=np.log(1.02),
    underlying_price=1.0,
):
    """
    Compute the Dupire PDE loss and boundary condition losses for the PINN with batched inputs.

    Parameters:
    - call_option_price: Batched tensor of predicted call option prices from the PINN (batch_size, num_points).
    - time_to_maturity: Batched tensor of time to maturities T_i (batch_size, num_points).
    - strike_price: Batched tensor of strike prices K_i (batch_size, num_points).
    - volatility: Batched tensor of volatilities σ(T_i, K_i) (batch_size, num_points).
    - strike_infinity: Large value for the high strike price boundary condition, default is 2.5.
    - risk_free_rate: The risk-free interest rate r, default is log(1.02).
    - underlying_price: The current spot price S_0, default is 1.0.

    Returns:
    - pde_loss: The average Dupire forward PDE loss across all batches.
    - maturity_zero_loss: The average loss for the boundary condition at maturity T=0 across all batches.
    - strike_zero_loss: The average loss for the boundary condition at K=0 across all batches.
    - strike_infinity_loss: The average loss for the boundary condition as K→∞ (approximated by strike_infinity) across all batches.
    """
    # Convert constants to tensors
    risk_free_rate = torch.tensor(risk_free_rate, device=call_option_price.device, dtype=call_option_price.dtype)
    underlying_price = torch.tensor(underlying_price, device=call_option_price.device, dtype=call_option_price.dtype)
    strike_infinity = torch.tensor(strike_infinity, device=call_option_price.device, dtype=call_option_price.dtype)

    # Create masks
    maturity_zero_mask = (time_to_maturity == 0).float()
    strike_zero_mask = (strike_price == 0).float()
    strike_infinity_mask = (strike_price == strike_infinity).float()

    # Compute first derivatives
    call_price_t = torch.autograd.grad(
        call_option_price.sum(),
        time_to_maturity,
        create_graph=True,
    )[0]

    call_price_k = torch.autograd.grad(
        call_option_price.sum(),
        strike_price,
        create_graph=True,
    )[0]

    # Compute second derivative
    call_price_k2 = torch.autograd.grad(
        call_price_k.sum(),
        strike_price,
        create_graph=True,
    )[0]

    # Compute PDE residuals
    pde_residual = (
        call_price_t
        + 0.5 * volatility ** 2 * strike_price ** 2 * call_price_k2
        + risk_free_rate * strike_price * call_price_k
        - risk_free_rate * call_option_price
    )

    # Compute PDE loss
    pde_loss = torch.mean(pde_residual ** 2)

    # Compute boundary condition residuals
    maturity_zero_residual = (
        call_option_price
        - torch.clamp(underlying_price - strike_price, min=0)
    ) * maturity_zero_mask

    strike_zero_residual = (
        call_option_price - underlying_price
    ) * strike_zero_mask

    strike_infinity_residual = (
        call_option_price
    ) * strike_infinity_mask

    # Compute boundary condition losses
    maturity_zero_loss = torch.mean(maturity_zero_residual ** 2)
    strike_zero_loss = torch.mean(strike_zero_residual ** 2)
    strike_infinity_loss = torch.mean(strike_infinity_residual ** 2)

    return pde_loss, maturity_zero_loss, strike_zero_loss, strike_infinity_loss
