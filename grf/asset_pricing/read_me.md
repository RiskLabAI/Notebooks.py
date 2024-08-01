# Improving Asset Pricing via Conditioning: Evidence and Examples

## Overview

This repository provides the implementation and empirical analysis for improving asset pricing models by conditioning on important market factors using Generalized Random Forests (GRFs).

```bibtex
@article{aghapour2024conditional,
  title={Improving Asset Pricing via Conditioning: Evidence and Examples},
  author={Aghapour, Ahmad and Arian, Hamid and Escobar-Anel, Marcos},
  journal={Finance Research Letters},
  year={2024}
}
```

## Structure

The repository is structured into four primary files/folders:

- **`Equation.py`**: Contains definitions and functions related to the pricing models.
- **`Model.py`**: Includes the implementation of the Generalized Random Forests architecture.
- **`Solver.py`**: Features a solver class that accepts asset pricing models and offers methods to solve them.
- **`GRF.ipynb`**: A Jupyter Notebook for executing the code and printing results.

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

Now, you can proceed to solve asset pricing models for various portfolios. Here is an example code:

```python
from RiskLabAI.asset_pricing.solver import GRFSolver
config = {
  "dim": 100,
  "total_time": 1,
  "num_time_interval": 50
}

model = RiskLabAI.asset_pricing.PricingModel(config)
device = torch.device('cuda')

grf_solver = GRFSolver(model, [150, 400, 500, 400, 150], 0.001, 'GRF')
losses, inits = grf_solver.solve(500, 128, (40 + torch.rand(1) * 10).to(device), device)
```

The `'GRF'` method is used here; you can replace it with other methods proposed in the paper, such as `'Monte-Carlo'`, `'Deep-Time-SetTransformer'`, or `'DeepBSDE'`.

# Methodology

## Overview

This repository is dedicated to the exploration and implementation of conditioning asset pricing models using Generalized Random Forests (GRFs). Our focus is on enhancing the performance of traditional models by capturing dynamic and nonlinear market conditions.

## Formalization of Models

We start by defining the continuous asset pricing model as follows:

```math
  \begin{align}
  Y_{i,t} & = \alpha_i + \sum_{j=1}^K \beta_{i,j}Z_{j,t} + \epsilon_{i,t},
  \end{align}
```

The Generalized Random Forests (GRF) method adapts this model to incorporate conditioning variables:

```math
  \begin{align}
  Y_{i,t} & = \alpha_i(X_{i,t}) + \sum_{j=1}^{K} \beta_{i,j}(X_{i,t}) Z_{j,t} + \epsilon_{i,t}(X_{i,t}),
  \end{align}
```

The conditional $\mathrm{R}^2$ (CR2) is defined as:

```math
  \begin{align}
  \mathrm{R}_{i,x}^2 &= 1 - \frac{\sigma^2_{e,i,x}}{\sigma^2_{i,x}} = \frac{\beta_{i \mid X=x}^{T} \Sigma_{Z,x} \beta_{i \mid X=x}} {\sigma^2_{i,x}}.
  \end{align}
```

The main algorithm is implemented using GRF to estimate the conditional factor loadings and compute the conditional $\mathrm{R}^2$.

### Pseudocode

1. **Initialization**
   - Require: The market factors, initial state $x_0$, initial value of returns $Y_{t_0}$.

2. **Main Loop**
   - While k < nIteration:
     - For each time step i from 0 to N-1:
       - Compute conditional factor loadings using GRF.
       - Update returns and state variables.
     - Compute Loss at the end of each full iteration.
     - Update parameters using gradient descent.

# Results

To visualize the results, refer to the `GRF.ipynb` notebook or execute the code in your preferred environment.

## Conditional CAPM

The improvement in fitting asset prices using the conditional CAPM model:

<p align="center">
  <img src="figs/CAPM_init.png" alt="Initial Conditional CAPM" width="49%" height="250"/>
  <img src="figs/CAPM_loss.png" alt="Conditional CAPM Loss" width="49%" height="250"/>
</p>

## Conditional Fama-French 3-Factor Model

The improvement in fitting asset prices using the conditional Fama-French 3-factor model:

<p align="center">
  <img src="figs/3FF_init.png" alt="Initial Conditional 3FF" width="49%" height="250"/>
  <img src="figs/3FF_loss.png" alt="Conditional 3FF Loss" width="49%" height="250"/>
</p>
