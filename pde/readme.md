
# Deep Time Neural Network for PDEs

## Overview

This repository provides implementation and solutions for Partial Differential Equations (PDEs) , utilizing the Deep Time Neural Network architecture.
```bibtex
@article{aghapour2023deep,
  title={Deep-Time Neural Networks: An Efficient Approach for Solving High-Dimensional PDEs},
  author={Aghapour, Ahmad and Arian, Hamid R and Seco, Luis A},
  journal={Available at SSRN},
  year={2023}
}
```

## Structure

The repository is structured into four primary files/folders:

- **`Equation.py`**: Contains definitions and functions related to PDEs.
- **`Model.py`**: Includes the implementation of the Deep Time Neural Network architecture.
- **`Solver.py`**: Features a solver class that accepts PDEs and offers methods to solve them.
- **`DTNN.ipynb`**: A Jupyter Notebook for executing the code and printing results.

## Installation

Before using the code, install the required library with the following command:

```shell
!pip install RiskLabAI
```

## How to Use

First, import the necessary libraries and `RiskLabAI`:

```python
import RiskLabAI
import torch
import numpy as np
```

Now, you can proceed to solve PDEs for option pricing with default risk. Here is an example code:

```python
from RiskLabAI.pde.solver import FBSDESolver
config = {
  "dim": 100,
  "total_time": 1,
  "num_time_interval": 50
}

pde = RiskLabAI.pde.PricingDefaultRisk(config)
device = torch.device('cuda')

fastsolver = FBSDESolver(pde,[150,400,500,400,150],0.001,'DTNN')
losses, inits = fastsolver.solve(500, 128, (40 + torch.rand(1)*10).to(device), device)
```

The `'DTNN'` method is used here; you can replace it with `'Monte-Carlo'` , `'Deep-Time-SetTransformer'`or  `'DeepBSDE'` proposed in the paper.

also you can use `'FBSNN'` model by calling :
```python
from RiskLabAI.pde.solver import FBSNNolver
fastsolver = FBSNNolver(pde,[pde.dim + 1] + [256] * 4 + [1],0.001)
loss , init = fastsolver.solve(iterations,128,(40 + torch.rand(1)*10).to(device),device)
```


# Results

To visualize the results, refer to the `DTNN.ipynb` notebook or execute the code in your preferred environment.
## Option Pricing with Default Risk
The stock price $X_t$ evolves according to the following stochastic differential equation:

$$
\begin{align*}
& dX_t = \bar{\mu} X_t dt + \bar{\sigma}\text{diag}(X_t) dW_t, \nonumber\\
& X_0 = \xi ,  \nonumber\\
& dY_t = ((1-\delta) Q(Y_t) + R)Y_t dt + Z_t^{T}dW_t , \nonumber \\
& Y_T = g\left(X_T\right).
\end{align*}
$$

### Option Price Equation

The defualt rate is described by the function, which is defined as:

$$ Q(y) = I_{\left(-\infty, v^h\right)}(y) \gamma^h+I_{\left[v^l, \infty\right)}(y) \gamma^l 
+I_{\left[v^h, v^l\right)}(y)\left[\frac{\left(\gamma^h-\gamma^l\right)}{\left(v^h-v^l\right)}\left(y-v^h\right)+\gamma^h\right].$$

### Parameters

- Time horizon, $T = 1$
- Default intensity, $\delta = \frac{2}{3}$
- Risk-free interest rate, $R = 0.02$
- Drift coefficient, $\bar{\mu} = 0.02$
- Volatility coefficient, $\bar{\sigma} = 0.2$
- Upper bound of the option price, $v^h = 50$
- Lower bound of the option price, $v^l = 70$
- Default intensity for high price, $\gamma^h = 0.2$
- Default intensity for low price, $\gamma^l = 0.02$
- Terminal condition for the worst-of option with strike price $K = 0$
<p align="center">
  <img src="figs\default_main_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\default_main_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>


## Option Pricing with Different Borrowing and Lending Rates


In this subsection, we explore option pricing under the assumption of different borrowing and lending rates. Incorporating these rates into the model allows us to investigate the potential impact of varying financing conditions on option prices:

$$
\begin{align*}
& dX_t = \bar{\mu} X_t dt + \bar{\sigma} \text{diag}(X_t) dW_t,  \\
& X_0 = \xi,  \\
& dY_t = (R^l Y_t + \frac{(\bar{\mu} - R^l)}{\bar{\sigma}} \sum_{i=1}^d z_i \\
& \quad + (R^l - R^b) \max \left\lbrace 0, \left[ \frac{1}{\bar{\sigma}} \sum_{i=1}^d z_i\right] - Y_t\right \rbrace) dt + Z_t^T dW_t,  \\
& Y_T = g\left(X_T\right).
\end{align*}
$$

The option price, $Y_t$, is governed by a different stochastic differential equation that includes lending rate $R^l$ and borrowing rate $R^b$. The lending rate is typically lower than the borrowing rate, as lending money is considered cheaper. The equation also includes a term representing the excess return scaled by volatility and the sum of the elements $z_i$ from 1 to $d$. The max operator ensures that the term inside the brackets remains non-negative, as it captures the potential payoff resulting from the difference between the lending and borrowing rates.

###  Parameters

For a comparative analysis, we use the parameters as follows:

- Time horizon, $T = 0.5$
- Lending rate, $R^l = 0.04$
- Borrowing rate, $R^b = 0.04$
- Drift coefficient, $\bar{\mu} = 0.06$
- Volatility coefficient, $\bar{\sigma} = 0.2$
- Initial values, $\xi = \left(100, \ldots, 100\right)$

### Terminal Condition

The terminal condition for the option is defined by the payoff function:

$$
\begin{align*}
g(x) = \max & \left\lbrace \left[\max _{1 \leq i \leq 100} x_i\right] - 120, 0 \right \rbrace \nonumber \\
& - 2 \max \left\lbrace \left[\max _{1 \leq i \leq 100} x_i\right] - 150, 0\right\rbrace.
\end{align*}
$$
<p align="center">
  <img src="figs\PricingDiffRate_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\PricingDiffRate_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>

## Black-Scholes-Barenblatt Equation


The Black-Scholes-Barenblatt (BSB) equation extends the classical Black-Scholes model to incorporate transaction costs and assess their impact on financial derivatives pricing. The system of stochastic differential equations for this model is as follows:

$$
\begin{align*}
d X_t & = \bar{\sigma} \  diag(X_t) d W_t, \\
X_0 & = \xi, \\
d Y_t & = r\left(Y_t - \frac{1}{\bar{\sigma}} J^T Z_t\right) dt + Z_t^{\prime} d W_t, \\
Y_T & = g(X_T).
\end{align*}
$$
In this model, the drift component is omitted to emphasize the stochastic nature of stock price dynamics. The pricing equation for the derivative, denoted by $Y_t$, integrates the risk-free rate $r$, and a vector of ones $J$. The term $\frac{1}{\bar{\sigma}}Z_t$ introduces a correction for transaction costs into the dynamics of option pricing.

### Parameter 

Adopting parameter settings from the literature, we set:

- Time horizon, $T = 1$
- Risk-free rate, $r = 0.05$
- Volatility coefficient, $\bar{\sigma} = 0.4$
- Initial state vector, $\xi = (1, 0.5, \ldots, 1, 0.5)$

The terminal payoff condition is defined as:
$$g(x) = ||x||^2. $$
<p align="center">
  <img src="figs\BlackScholesBarenblatt_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\BlackScholesBarenblatt_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>



### Basket Option

The payoff for a European call Basket Option is determined by the weighted sum of multiple underlying asset prices:

$$
\text{Payoff} = \max\left(\sum_{i=1}^n w_i S_i(T) - K, 0\right),
$$

where $S_i(T)$ represents the price of the $i$-th underlying asset at maturity, $w_i$ is the weighting factor for each asset, $n$ is the total number of underlying assets, and $K$ is the strike price. We set each $w_i$ to $\frac{1}{n}$ and $K = 0$.

<p align="center">
  <img src="figs\BlackScholesBarenblatt_basket_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\BlackScholesBarenblatt_basket_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>
### Max-Min Spread Option

For a European call Max-Min Spread Option, the payoff is based on the difference between the maximum and minimum asset prices:

$$
\text{Payoff} = \max\left(\max_{i=1}^{n}(S_i(T)) - \min_{i=1}^{n}(S_i(T)) - K, 0\right).
$$

Here, $K$ is set to 0.

<p align="center">
  <img src="figs\BlackScholesBarenblatt_maxmin_init.png" alt="First Image" width="49%"  height = "250"/>
  <img src="figs\BlackScholesBarenblatt_maxmin_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>

### Best-of Option

The payoff for a European call Best-of Option, which depends on the performance of the best-performing underlying asset, is expressed as:

$$
\text{Payoff} = \max\left(\max_{i=1}^{n}(S_i(T)) - K, 0\right).
$$

Again, $K$ is set to 0 for this analysis.

<p align="center">
  <img src="figs\BlackScholesBarenblatt_bestof_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\BlackScholesBarenblatt_bestof_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>


## Hamilton-Jacobi-Bellman Equation



In the realm of optimal control, the Hamilton-Jacobi-Bellman (HJB) equation serves as a critical condition for the optimality of control strategies. It delineates the value function that minimizes or maximizes the expected cost over a system governed by differential equations. 


For the experiment, we consider the following FBSDEs, which exemplify the application of the HJB equation in a stochastic setting:

$$\begin{align*}
    & dX_t = \sigma dW_t, \quad t \in[0, T], \\
    & X_0 = \xi, \\
    & dY_t =\frac{\|Z_t\|^2}{\sigma^2} dt +  Z_t^\prime dW_t, \quad t \in[0, T), \\
    & Y_T = g(X_T),
\end{align*}$$

### Parameters

- **T**: 1 - The time horizon for the model.
- **σ**: √2 - The volatility coefficient, reflecting the intensity of the stochastic component.
- **ξ**: (0, 0, ..., 0) ∈ $ℝ^{100}$ - The initial state vector, indicating the starting condition.

also 
$$g(x)=\ln \left(0.5\left(1+\|x\|^2\right)\right)$$

with the terminal condition $u(T, x) = g(x)$.
<p align="center">
  <img src="figs\HJB_init.png" alt="First Image" width="49%" height = "250"/>
  <img src="figs\HJB_loss.png" alt="Second Image" width="49%" height = "250"/>
</p>
