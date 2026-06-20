<div align="center">
  <a href="https://risklab.ai"><img src="utils/risklab_ai.gif" width="80px"/></a>
  <h1>RiskLabAI.py — Tutorial Notebooks</h1>
</div>

Runnable Python tutorials for [`RiskLabAI.py`](https://github.com/RiskLabAI/RiskLabAI.py)
(≥ 2.0.1), the financial-machine-learning library based on López de Prado's
*Advances in Financial Machine Learning* and *Machine Learning for Asset Managers*.
Each notebook is **exhaustive over its module**, runs **top-to-bottom against a
pinned environment**, and mirrors the matching Julia tutorial in
[`Notebooks.jl`](https://github.com/RiskLabAI/Notebooks.jl).

## Flagship tutorials

| # | Notebook | Topic |
|---|----------|-------|
| 1 | [`optimization/portfolio_construction.ipynb`](optimization/portfolio_construction.ipynb) | Robust portfolio construction — covariance **denoising** (Marčenko–Pastur), **HRP**, and **NCO** vs naive Markowitz |
| 2 | [`data/structures/financial_data_structures.ipynb`](data/structures/financial_data_structures.ipynb) | **Financial data structures** — time, tick, volume, dollar, imbalance & run bars (offline AAPL data) |
| 3 | [`data/differentiation/fractional_differentiation.ipynb`](data/differentiation/fractional_differentiation.ipynb) | **Fractional differentiation** — the minimum `d` for stationarity via an ADF sweep (FRED S&P 500) |
| 4 | [`data/labeling/triple_barrier_labeling.ipynb`](data/labeling/triple_barrier_labeling.ipynb) | **Triple-barrier labeling** + meta-labeling + trend scanning (FRED S&P 500) |
| 5 | [`backtest/cross_validation_and_pbo.ipynb`](backtest/cross_validation_and_pbo.ipynb) | **Cross-validation** — purged K-Fold, CPCV, walk-forward, and the Probability of Backtest Overfitting (PBO) |
| 6 | [`features/feature_importance/feature_importance.ipynb`](features/feature_importance/feature_importance.ipynb) | **Feature importance** — MDI, MDA, SFI, orthogonal (PCA) features, and weighted Kendall-τ |
| 7 | [`pde/deep_bsde.ipynb`](pde/deep_bsde.ipynb) | **Deep-BSDE** PDE solver (Han, Jentzen & E 2018) — four financial PDEs, checked vs Monte-Carlo / closed-form references |

## Running the notebooks

The pinned environment (`requirements.txt`) installs the local package editable from
a relative path. From this folder:

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Then open any notebook with Jupyter, or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace <path-to-notebook>
```

Tutorials 3–4 read the S&P 500 from **FRED**; set the `FRED_API_KEY` environment
variable (a free key from <https://fred.stlouisfed.org>). All other data is either
synthetic or the committed `input_data/AAPL_OHLCV_1m_Data.csv` snapshot, so the
notebooks stay reproducible and offline.
