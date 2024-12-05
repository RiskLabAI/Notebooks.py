---
# **Hedged Random Forests**
---

```bibtex
@article{beck2023hedging,
  title={Hedging Forecast Combinations With an Application to the Random Forest},
  author={Beck, Elliot and Kozbur, Damian and Wolf, Michael},
  journal={arXiv preprint arXiv:2308.15384},
  year={2023}
}
```

# Hedged Forecast Combinations: Code Design and Mathematical Foundations

This document provides an overview of the **Hedged Forecast Combinations** and **Hedged Random Forest** implementations, along with the mathematical principles that govern their behavior. Each code snippet includes the corresponding mathematical formulation and is displayed in a GitHub-friendly format.

---

## Hedged Forecast Combination

### Mathematical Formulation

The hedged forecast combination minimizes the mean-squared error (MSE):

$$
\text{MSE}(\hat{f}_w) = (w^\top \mu)^2 + w^\top \Sigma w
$$

where:
- $w$ is the vector of weights.
- $\mu$ is the mean vector of forecast errors.
- $\Sigma$ is the covariance matrix of forecast errors.

The optimization problem is:

$$
\begin{aligned}
& \min_w \quad (w^\top \mu)^2 + w^\top \Sigma w \\
\text{subject to} \quad & w^\top \mathbf{1} = 1, \\
& \|w\|_1 \leq \kappa
\end{aligned}
$$

---

### Code Outline

```python
class HedgedForecastCombination(BaseEstimator, RegressorMixin):
    def __init__(self, base_models: List[BaseEstimator], kappa: float = 2.0, shrinkage: str = "ledoit_wolf"):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HedgedForecastCombination":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _solve_optimization(self) -> np.ndarray:
        pass

    def _ledoit_wolf_shrinkage(self, residuals: np.ndarray) -> np.ndarray:
        pass
```

---

## Hedged Random Forest

### Mathematical Formulation

In the context of random forests, the individual trees serve as forecasting models $\mathcal{M}_j(x)$. The residual matrix $R$ is constructed as:

$$
R[:, j] = y - \mathcal{M}_j(x)
$$

The weights $w$ are optimized using the same convex optimization problem as the general case. The output prediction is a weighted combination of the individual trees:

```math
\hat{f}_{\text{HRF}}(x) = \sum_{j=1}^p w_j \mathcal{M}_j(x)
```

---

### Code Outline

```python
class HedgedRandomForestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None, 
                 kappa: float = 2.0, shrinkage: str = "ledoit_wolf"):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HedgedRandomForestRegressor":
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def _solve_optimization(self) -> np.ndarray:
        pass
```

---

## Optimization Problem Solver

### Mathematical Formulation

The optimization problem is solved using quadratic programming:

$$
\begin{aligned}
\text{Objective:} \quad & \min_w \quad (w^\top \hat{\mu})^2 + w^\top \hat{\Sigma} w \\
\text{Constraints:} \quad & w^\top \mathbf{1} = 1, \\
& \|w\|_1 \leq \kappa
\end{aligned}
$$

This ensures the weights are optimal given the estimated mean vector $\hat{\mu}$ and covariance matrix $\hat{\Sigma}$.

---

### Code Outline

```python
def _solve_optimization(self) -> np.ndarray:
    pass
```

---

## Covariance Matrix Estimation

### Mathematical Formulation

To improve stability, the covariance matrix $\hat{\Sigma}$ is estimated using Ledoit-Wolf shrinkage:

$$
\hat{\Sigma} = (1 - \lambda) S + \lambda T
$$

where:
- $S$ is the sample covariance matrix.
- $T$ is the shrinkage target.
- $\lambda$ is the shrinkage intensity.

---

### Code Outline

```python
def _ledoit_wolf_shrinkage(self, residuals: np.ndarray) -> np.ndarray:
    pass
```

---

## Bootstrapping for Random Forest

### Mathematical Formulation

Bootstrap sampling is used to train individual trees. A bootstrap sample is drawn from the training data:

$$
\text{Sample size:} \quad n
$$

where $n$ is the number of training observations.

---

### Code Outline

```python
def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass
```

---

## Prediction with Hedged Weights

### Mathematical Formulation

The final prediction is computed as a weighted combination of individual model predictions:

```math
\hat{f}_w(x) = \sum_{j=1}^p w_j \mathcal{M}_j(x)
```

---

### Code Outline

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    pass
```

Here’s the continuation of the markdown:

---

## Fitting the Models

### Mathematical Formulation

Each base model $\mathcal{M}_j$ is trained on the training dataset $(X, y)$. For random forests, this involves fitting individual decision trees on bootstrap samples of the data.

The residual matrix $R$ is computed as:

$$
R[:, j] = y - \mathcal{M}_j(X), \quad j = 1, \ldots, p
$$

These residuals are used to estimate $\hat{\mu}$ (the mean vector of residuals) and $\hat{\Sigma}$ (the covariance matrix of residuals).

---

### Code Outline

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> Union["HedgedForecastCombination", "HedgedRandomForestRegressor"]:
    pass
```

---

## Cross-Validation and Reproducibility

### Mathematical Formulation

To ensure robust evaluation, the methodology includes cross-validation or repeated train-test splits. The final performance metric, the **root-mean-squared-error (RMSE) ratio**, is calculated as:

```math
\text{RMSE}_{\text{HRF}/\text{RF}} = \frac{\sqrt{\frac{1}{B} \sum_{b=1}^B \text{MSE}_{\text{HRF},b}}}{\sqrt{\frac{1}{B} \sum_{b=1}^B \text{MSE}_{\text{RF},b}}}
```

where:
- $\text{MSE}_{\text{HRF},b}$ and $\text{MSE}_{\text{RF},b}$ are the test set mean-squared errors for the $b$-th iteration of HRF and RF, respectively.
- $B$ is the number of repetitions.

---

### Code Outline

```python
def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_repeats: int = 10) -> float:
    pass
```

---

## Utility Functions

### Covariance Matrix Estimation

As discussed earlier, the Ledoit-Wolf shrinkage method is used to improve numerical stability. Alternatively, the sample covariance matrix can also be used, depending on the user’s choice.

---

### Code Outline

```python
def estimate_covariance(self, residuals: np.ndarray) -> np.ndarray:
    pass
```

---

### Bootstrap Sampling

Random subsampling with replacement is used for constructing bootstrap samples during random forest training.

---

### Code Outline

```python
def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass
```

---

### Predicting with Weighted Models

The weighted prediction is calculated using the hedged weights $w_j$:

```math
\hat{f}_w(x) = \sum_{j=1}^p w_j \mathcal{M}_j(x)
```

---

### Code Outline

```python
def predict(self, X: np.ndarray) -> np.ndarray:
    pass
```
