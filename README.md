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

## Setup

`RiskLabAI` installs straight from PyPI — no clone of the library needed.

```bash
git clone https://github.com/RiskLabAI/Notebooks.py.git
cd Notebooks.py
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # RiskLabAI (PyPI) + notebook deps
jupyter notebook                   # launch, then open any tutorial
```

Run a notebook headless:

```bash
jupyter nbconvert --to notebook --execute --inplace <path-to-notebook>
```

**Optional — Deep-BSDE PDE notebook only.** Tutorial 7 (`pde/deep_bsde.ipynb`) needs
PyTorch, kept out of the light core environment:

```bash
pip install -r requirements-pde.txt
```

Tutorials 3–4 read the S&P 500 from **FRED**; set the `FRED_API_KEY` environment
variable (a free key from <https://fred.stlouisfed.org>). All other data is either
synthetic or the committed `input_data/AAPL_OHLCV_1m_Data.csv` snapshot, so the
notebooks stay reproducible and offline.
