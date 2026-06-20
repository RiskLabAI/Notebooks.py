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

## Getting started

New to this? This gets you from a fresh computer to running any tutorial. You only do
**Step 1** once.

### Step 1 — install the tools (once)

**1a. VS Code.** Download and install [Visual Studio Code](https://code.visualstudio.com/Download)
(Windows or macOS) and open it once. Then install these **extensions** (left sidebar →
Extensions icon → search → Install):
- **Python** (publisher: Microsoft)
- **Jupyter** (publisher: Microsoft)

**1b. Anaconda (Python + Jupyter).** Download and install
[Anaconda](https://www.anaconda.com/download) (Windows: run the installer with defaults;
macOS: run the `.pkg`). This gives you Python and Jupyter ready to go.

### Step 2 — run a notebook

1. **Get the notebooks.** Go to <https://github.com/RiskLabAI/Notebooks.py> → green
   **Code** button → **Download ZIP** → unzip somewhere easy. (Or, with git:
   `git clone https://github.com/RiskLabAI/Notebooks.py`.)
2. **Open the folder in VS Code:** File → Open Folder → choose the `Notebooks.py` folder.
3. **Install the packages.** Open a terminal in VS Code (Terminal → New Terminal) and run:
   ```bash
   pip install -r requirements.txt
   ```
   This installs **RiskLabAI** (from PyPI) plus Jupyter, pandas, matplotlib, and the data
   tools. (First time takes a few minutes.)
4. **Open a notebook** (e.g. `optimization/portfolio_construction.ipynb`), click
   **Select Kernel** (top-right) → **Python Environments** → choose the Anaconda/`base`
   Python, then click **Run All**.

### The FRED API key (two notebooks)

The **fractional differentiation** and **triple-barrier labeling** tutorials pull real
data from FRED (the Federal Reserve's free database). To run them:

1. Get a free API key (instant): <https://fredaccount.stlouisfed.org/apikey>
2. Set it as an environment variable, then **open a new terminal** so it takes effect:
   - **Windows:** `setx FRED_API_KEY "your_key_here"`
   - **macOS:** add `export FRED_API_KEY="your_key_here"` to `~/.zshrc`, then `source ~/.zshrc`
3. Verify in a new terminal: `echo %FRED_API_KEY%` (Windows) or `echo $FRED_API_KEY` (macOS).

All other tutorials run fully offline and need no key.

### Optional — the PDE notebook only

The Deep-BSDE notebook (`pde/deep_bsde.ipynb`) needs PyTorch (a large download). Only if
you want to run it:
```bash
pip install -r requirements-pde.txt
```

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
