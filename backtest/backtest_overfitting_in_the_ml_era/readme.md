# Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment

## Overview
This repository contains the Jupyter Notebook associated with our research on the integration of advanced statistical models and machine learning in financial analytics. Our focus is on robust model evaluation, out-of-sample testing methodologies, and specialized cross-validation techniques tailored for financial markets.

## Repository Contents
- `backtest_overfitting_in_the_machine_learning_era.ipynb`: The Jupyter Notebook to reproduce the results.

## Key Findings

- **Advanced Model Evaluation**: This research enhances financial analytics by integrating sophisticated statistical models and machine learning with a comprehensive cross-validation framework, specifically tailored to address non-stationarity, autocorrelation, and regime shifts in financial data.
- **Superiority of CPCV**: The Combinatorial Purged Cross-Validation (CPCV) method demonstrates superior performance in mitigating overfitting, evidenced by lower Probability of Backtest Overfitting (PBO) and higher Deflated Sharpe Ratio (DSR) compared to traditional methods like K-Fold, Purged K-Fold, and Walk-Forward.
- **Novel Variants of CPCV**: Introduction of Bagged CPCV and Adaptive CPCV variants enhances robustness through ensemble approaches and dynamic adjustments based on market conditions, further improving model performance stability.
- **Real-World Validation**: Empirical validation using historical S\&P 500 data confirms the practical applicability and resilience of CPCV and its variants. Sensitivity analyses highlight CPCV's robustness against model specification errors and its consistent performance across various market scenarios.
- **Limitations and Future Research**: Despite CPCV's computational intensity, advancements in parallelization show promise in improving efficiency. Future research should focus on optimizing these methods for real-time applications, investigating data leakage, and expanding the framework to diverse financial instruments and global markets.
- **Practical Implications for Financial Strategy**: Findings emphasize the critical need for advanced cross-validation techniques in financial modeling, crucial for developing resilient trading strategies, enhancing risk management, and meeting regulatory requirements in complex market environments.

<p align="center">
  <img src="https://github.com/RiskLabAI/Notebooks.py/blob/73253c1c854227c67941732174a4d098281a6164/backtest/Backtest%20Overfitting%20in%20the%20Machine%20Learning%20Era/figs/pbo_dsr_comparison.png" style="width: 40%;" />
  <img src="https://github.com/RiskLabAI/Notebooks.py/blob/e0ba18b70c79e8c6f8fc5925fe5bfab147567cbc/backtest/Backtest%20Overfitting%20in%20the%20Machine%20Learning%20Era/figs/pbo_dsr_adf.png" style="width: 40%;" /> 
</p>
<p align="center">
  <img src="https://github.com/RiskLabAI/Notebooks.py/blob/73253c1c854227c67941732174a4d098281a6164/backtest/Backtest%20Overfitting%20in%20the%20Machine%20Learning%20Era/figs/sp500_pbo_dsr_comparison.png" style="width: 40%;" />
  <img src="https://github.com/RiskLabAI/Notebooks.py/blob/e0ba18b70c79e8c6f8fc5925fe5bfab147567cbc/backtest/Backtest%20Overfitting%20in%20the%20Machine%20Learning%20Era/figs/computational_requirements_comparison.png" style="width: 40%;" /> 
</p>

## Installation

Before using the code, install the required library with the following command:

```shell
!pip install RiskLabAI
```

## Usage

This section guides you through the process of using the provided scripts and models to conduct financial market simulations and evaluate model performance.

### Simulating Market Conditions

The simulation starts by defining the parameters for a speculative bubble market regime. We generate two arrays representing bubble drifts and volatilities using the `drift_volatility_burst` function. These arrays are then used as parameters for the 'speculative_bubble' regime.

```python
x = 0.35
bubble_drifts, bubble_volatilities = drift_volatility_burst(
    bubble_length=5 * 252, 
    a_before=x, 
    a_after=-x, 
    b_before=0.6 * x, 
    b_after=0.6 * x, 
    alpha=0.75, 
    beta=0.45,
    explosion_filter_width=0.1
)
```

### Defining Market Regimes

We define a dictionary named `regimes` that contains Heston model parameters for different market conditions: calm, volatile, and speculative bubble. Each market condition has its unique set of parameters that dictate how the simulated prices will behave.

```python
regimes = {
    'calm': {
        'mu': 0.1,
        'kappa': 3.98,
        'theta': 0.029,
        'xi': 0.389645311,
        'rho': -0.7,
        'lam': 121,
        'm': -0.000709,
        'v': 0.0119
    },
    'volatile': {
        'mu': 0.1,
        'kappa': 3.81,
        'theta': 0.25056,
        'xi': 0.59176974,
        'rho': -0.7,
        'lam': 121,
        'm': -0.000709,
        'v': 0.0119
    },
    'speculative_bubble': {
        'mu': list(bubble_drifts),
        'kappa': 1,
        'theta': list(bubble_volatilities),
        'xi': 0,
        'rho': 0,
        'lam': 0,
        'm': 0,
        'v': 0.00000001
    },
}
```

### Creating Transition Matrix and Strategy Parameters

We define the `transition_matrix` that controls the probabilities of switching between different market regimes over time. Additionally, `strategy_parameters` are set to dictate the behavior of the trading strategies in the simulation.

```python
dt = TOTAL_TIME / N_STEPS
transition_matrix = np.array([
    [1 - 1 * dt,   1 * dt - 0.00001,        0.00001],  # State 0 transitions
    [20 * dt,      1 - 20 * dt - 0.00001,   0.00001],  # State 1 transitions
    [1 - 1 * dt,   1 * dt,                      0.0],  # State 2 transitions
])

strategy_parameters = {
    'fast_window' : [5, 20, 50, 70],
    'slow_window' : [10, 50, 100, 140],
    'exponential' : [False],
    'mean_reversion' : [False]
}
```

### Generating Prices

Utilizing the `parallel_generate_prices` function, we synthesize asset prices for each regime based on the defined transition probabilities and strategy parameters.

```python
all_prices, all_regimes = parallel_generate_prices(
    N_PATHS,
    regimes,
    transition_matrix,
    TOTAL_TIME,
    N_STEPS,
    RANDOM_STATE,
    N_JOBS
)
```

### Model Evaluation

For model evaluation, we define a dictionary `models` that contains the setup for three different machine learning models: k-Nearest Neighbors (k-NN), Decision Tree, and XGBoost. Each model is accompanied by a set of parameters to be optimized.

```python
models = {
    'k-NN' : {
        'Model': CustomPipeline.from_existing_pipeline(existing_pipeline=make_pipeline(StandardScaler(), KNeighborsClassifier())),
        'Parameters': {
            'kneighborsclassifier__n_neighbors': [1, 2, 3],
        }
    },
    'Decision Tree' : {
        'Model': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Parameters': {
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
        }
    },
    'XGBoost': {
        'Model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', seed=RANDOM_STATE),
        'Parameters': {
            'n_estimators': [1000],
            'max_depth': [1000000000],
            'learning_rate': [1, 10, 100],
            'subsample': [1.0],
            'colsample_bytree': [1.0],
        }
    },
}
```

### Running the Overfitting Simulation

Finally, we run a parallelized simulation to evaluate the risk of overfitting for each model using the `backtset_overfitting_simulation` function. This process is tracked using the `joblib_progress` bar, which provides a visual indication of the simulation's progress.

```python
with joblib_progress("Overfitting...", total=all_prices.shape[1]):    
    results = Parallel(n_jobs=N_JOBS)(delayed(backtset_overfitting_simulation)(all_prices[column], strategy_parameters, models, STEP_RISK_FREE_RATE, all_prices.shape[0]) for column in all_prices.columns)

```

## Abstract
This research explores the integration of advanced statistical models and machine learning in financial analytics, representing a shift from traditional to advanced, data-driven methods. We address a critical gap in quantitative finance: the need for robust model evaluation and out-of-sample testing methodologies, particularly tailored cross-validation techniques for financial markets. We present a comprehensive framework to assess these methods, considering the unique characteristics of financial data like non-stationarity, autocorrelation, and regime shifts. Through our analysis, we unveil the marked superiority of the Combinatorial Purged (CPCV) method in mitigating overfitting risks, outperforming traditional methods like K-Fold, Purged K-Fold, and especially Walk-Forward, as evidenced by its lower Probability of Backtest Overfitting (PBO) and superior Deflated Sharpe Ratio (DSR) test statistic. Walk-Forward, by contrast, exhibits notable shortcomings in false discovery prevention, characterized by increased temporal variability and weaker stationarity. This contrasts starkly with CPCV's demonstrable stability and efficiency, confirming its reliability for financial strategy development. We introduce novel variants of CPCV, including Bagged CPCV and Adaptive CPCV, which enhance robustness through ensemble approaches and dynamic adjustments based on market conditions. Our empirical validation using historical SP 500 data confirms the practical applicability and resilience of these advanced cross-validation methods. The analysis also suggests that choosing between Purged K-Fold and K-Fold necessitates caution due to their comparable performance and potential impact on the robustness of training data in out-of-sample testing. Our investigation utilizes a Synthetic Controlled Environment incorporating advanced models like the Heston Stochastic Volatility, Merton Jump Diffusion, and Drift-Burst Hypothesis, alongside regime-switching models. This approach provides a nuanced simulation of market conditions, offering new insights into evaluating cross-validation techniques. We also address the computational aspects of these methods, demonstrating that parallelization significantly improves efficiency, making them feasible for large-scale financial datasets. Our study underscores the necessity of specialized validation methods in financial modeling, especially in the face of growing regulatory demands and complex market dynamics. It bridges theoretical and practical finance, offering a fresh outlook on financial model validation. Highlighting the significance of advanced cross-validation techniques like CPCV, our research enhances the reliability and applicability of financial models in decision-making.

## Citing Our Work
If you find our research useful in your work, please consider citing it as follows:

```bibtex
@article{arian_norouzi_seco2024backtest,
  title={Backtest Overfitting in the Machine Learning Era: A Comparison of Out-of-Sample Testing Methods in a Synthetic Controlled Environment},
  author={Arian, Hamid R and Norouzi M, Daniel and Seco, Luis A},
  journal={Available at SSRN},
  year={2024}
}
```

## License
This project is open-sourced under the MIT license.

## Contact
For any queries regarding the research, please reach out to:
- Daniel Norouzi M.
  - Email: [norouzi@risklab.ai](mailto:norouzi@risklab.ai)
