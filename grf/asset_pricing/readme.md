# Conditional Non-linear Asset Pricing via Conditioning: Evidence and Examples

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

# Empirical Results

To visualize the results, refer to the `GRF.ipynb` notebook or execute the code in your preferred environment.

## Conditional CAPM

The improvement in fitting asset prices using the conditional CAPM model:

<p align="center">
  <img src="figs/SMALL LoBM.capm.png" alt="Initial Conditional CAPM" width="49%" height="250"/>
  <img src="figs/SMALL HiBM.capm.png" alt="Conditional CAPM Loss" width="49%" height="250"/>
</p>

## Conditional Fama-French 3-Factor Model

The improvement in fitting asset prices using the conditional Fama-French 3-factor model:

<p align="center">
  <img src="figs/SMALL HiBM.3factorbeta.png" alt="Initial Conditional 3FF" width="49%" height="250"/>
  <img src="figs/BIG LoBM.3factorbeta.png" alt="Conditional 3FF Loss" width="49%" height="250"/>
</p>

For more details, additional results, and analysis, please visit [Additional Results and Analysis](https://ahmad-aghapour.github.io/conditional-asset-pricing/).
