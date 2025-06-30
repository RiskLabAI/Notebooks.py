# Quantum Hierarchical Risk Parity and Its Classical Counterpart

## Abstract

This paper introduces two advanced alternatives: Quantum Hierarchical Risk Parity (QHRP) and its classical counterpart, Kernel-based HRP (KHRP). QHRP integrates HRP with quantum machine learning, using quantum density matrices and Frobenius distance to model deeper asset relationships. KHRP employs kernel methods to achieve a similar goal classically. In empirical tests, both new methods deliver superior risk-adjusted performance over traditional approaches, demonstrating their potential for more robust portfolio management.

```bibtex
@article{quantum_hrp_2025,
  title={Quantum Hierarchical Risk Parity and Its Classical Counterpart},
  author={Arian, Hamid and Kondratyev, Oleksiy and Norouzi Mobarekeh, Daniel},
  journal={Working Paper},
  year={2025}
}
```

## Methodology

### Quantum Hierarchical Risk Parity (QHRP)

The QHRP methodology integrates quantum machine learning techniques with the Hierarchical Risk Parity algorithm for optimal portfolio construction. The process is structured into five key stages: Quantum Feature Mapping, Density Matrix Construction, Frobenius Distance Calculation, Distance Matrix Construction, and Integration with HRP.

#### QHRP Algorithm Pseudocode:

1.  **Quantum Feature Mapping:**
      - Encode each observation $x_i^k$ into quantum feature maps $\\varphi(x_i^k)$ using unitary transformations $U(x_i^k)$.
      - Construct average density matrices $\\rho_i = \\frac{1}{T} \\sum_{k=1}^{T} |x_i^k\\rangle \\langle x_i^k|$ for each asset $i$.
2.  **Frobenius Distance & Distance Matrix Calculation:**
      - Compute Frobenius distances $D_{i,j} = d_F(\\rho_i, \\rho_j) = \\frac{1}{2} \\sqrt{\\mathrm{Tr}\\left[(\\rho_i - \\rho_j)^2\\right]}$ for all asset pairs $(i, j)$ to form distance matrix $D$.
3.  **Tree Clustering:**
      - Apply hierarchical clustering on $D$ to build dendrogram $\\mathcal{T}$ using minimum distance linkage.
4.  **Quasi-Diagonalization:**
      - Rearrange asset order based on $\\mathcal{T}$ to obtain a quasi-diagonal covariance matrix $\\Sigma'$.
5.  **Recursive Bisection:**
      - Allocate portfolio weights $\\mathbf{w}$ using recursive bisection aligned with hierarchical clusters.

The core equations governing the QHRP process are as follows:

1. **Quantum Feature Map:** A feature vector $\\mathbf{x}$ is mapped to a density matrix $\\varphi(\\mathbf{x})$.
  ```math
  \varphi(\mathbf{x}) := U(\mathbf{x}) | \mathbf{0} \rangle \langle \mathbf{0} | U(\mathbf{x})^\dagger \equiv | \mathbf{x} \rangle \langle \mathbf{x} |
  ```
2. **Average Density Matrix:** For each asset $i$, an average density matrix $\\rho_i$ is constructed.
  ```math
  \rho_i = \frac{1}{T} \sum_{k=1}^{T} |\mathbf{x}_i^k\rangle \langle \mathbf{x}_i^k|
  ```
3. **Frobenius Distance:** The similarity between assets is quantified using the Frobenius distance between their density matrices.
  ```math
  d_F(\rho_i, \rho_j) = \frac{1}{2}\|\rho_i - \rho_j\| = \frac{1}{2} \sqrt{\mathrm{Tr}\left[(\rho_i - \rho_j)^2\right]}
  ```

The resulting distance matrix $D$ is then used in the standard HRP stages of Tree Clustering, Quasi-Diagonalization, and Recursive Bisection to determine the final asset weights.

### Kernel-Based Hierarchical Risk Parity (KHRP)

KHRP serves as a classical counterpart to QHRP. It uses a Gaussian kernel and the Maximum Mean Discrepancy (MMD) to measure asset similarities instead of quantum methods.

#### KHRP Algorithm Pseudocode:

1.  Compute pairwise MMD distances between asset features to form the distance matrix $\\mathbf{D}$.
2.  Perform hierarchical clustering on the condensed form of $\\mathbf{D}$ to obtain an asset ordering $\\pi$.
3.  Rearrange the covariance matrix $\\boldsymbol{\\Sigma}$ according to $\\pi$ to obtain a quasi-diagonal matrix $\\boldsymbol{\\Sigma}'$.
4.  Allocate portfolio weights using recursive bisection with inverse-variance allocation.

The key equations for KHRP are:

1. **Gaussian Kernel:** Measures similarity between two feature vectors $\\mathbf{x}$ and $\\mathbf{y}$.
  ```math
  k(\mathbf{x},\mathbf{y}) \;=\; \exp\!\Bigl(-\frac{\|\mathbf{x}-\mathbf{y}\|^2}{2\sigma^2}\Bigr)
  ```
2. **Maximum Mean Discrepancy (MMD):** The distance between two assets $i$ and $j$ is calculated based on their feature matrices $\\mathbf{X}_i$ and $\\mathbf{X}_j$.
  ```math
  \mathrm{MMD}(\mathbf{X}_i,\mathbf{X}_j)^2 \;=\; {\frac{1}{T_i^2}\sum_{t=1}^{T_i}\sum_{s=1}^{T_i} k(\mathbf{x}_{i,t},\mathbf{x}_{i,s}) + \frac{1}{T_j^2}\sum_{t=1}^{T_j}\sum_{s=1}^{T_j} k(\mathbf{x}_{j,t},\mathbf{x}_{j,s}) - \frac{2}{T_iT_j}\sum_{t=1}^{T_i}\sum_{s=1}^{T_j} k(\mathbf{x}_{i,t},\mathbf{x}_{j,s})}
  ```

## Numerical Experiments

### Experimental Setup and Features

The analysis was conducted on a diversified portfolio of 170 assets from January 1, 2005, to January 1, 2025. QHRP and KHRP use a richer, multi-dimensional feature set to build superior portfolios.

  - **Number of Features ($P$)**: 6
  - **Features**:
    1.  Daily Return (log)
    2.  30-Day Rolling Volatility (annualized)
    3.  14-Day Momentum
    4.  60-Day Momentum
    5.  7-Day Short-Term Reversal
    6.  30-Day Rolling Volume Average (normalized)
  - **Benchmark Models**: Classical HRP and Markowitz optimization used standard daily returns as input.
  - **Kernel Bandwidth ($\\sigma$) for KHRP**: `1.0`

### Quantum Circuit Ansatz

Classical financial data is embedded into a quantum Hilbert space using a quantum feature map, implemented by a parameterized unitary operator $U(\\mathbf{x})$. For our $P=6$ features, we use a 6-qubit circuit. The classical vector is embedded using amplitude encoding, where the feature vector is normalized and mapped to the amplitudes of the quantum state $\\ket{\\psi(\\mathbf{x})}$.

### Results

The out-of-sample performance was evaluated using a walk-forward analysis. The tables below show the key performance metrics.

<div align="center">

| Method             | Mean Sharpe | Std of Sharpe | Mean PSR |
| ------------------ | ----------- | ------------- | -------- |
| **Quantum HRP** | **1.4658** | 3.7234        | **0.6172** |
| Classical HRP      | 1.2190      | 3.8598        | 0.5970   |
| Kernel-based HRP   | 1.2944      | **3.6671** | 0.6055   |
| Markowitz          | 1.3474      | 3.7785        | 0.6093   |
| Equal Weights      | 1.3565      | 3.9065        | 0.6043   |

</div>

*Walk-Forward Out-of-Sample Performance Metrics*

<br>

<div align="center">

| Method             | Aggregated Out-of-Sample Sharpe |
| ------------------ | ------------------------------- |
| **Quantum HRP** | **1.3975** |
| Classical HRP      | 1.1378                          |
| Kernel-based HRP   | 1.2186                          |
| Markowitz          | 1.2557                          |
| Equal Weights      | 1.1983                          |

</div>

*Overall Out-of-Sample Sharpe Ratios from Aggregated Test Returns*

<br>

The results show that **Quantum HRP (QHRP)** consistently achieves the highest risk-adjusted returns (Sharpe Ratio) in rigorous out-of-sample tests, outperforming both traditional methods and its advanced classical counterpart, KHRP.


<p align="center">
<img src="figs/correlation_original.png" alt="Unordered Correlation Matrix" width="49%">
<img src="figs/quantum_distance_unordered.png" alt="Unordered Quantum Distance Matrix" width="49%">
</p>
<p align="center">
<img src="figs/correlation_hrp_ordering.png" alt="Classical HRP Ordered Correlation" width="49%">
<img src="figs/correlation_quantum_hrp_ordering.png" alt="Quantum HRP Ordered Correlation" width="49%">
</p>
<p align="center">
<img src="figs/quantum_distance_ordered.png" alt="Ordered Quantum Distance Matrix" width="49%">
</p>

The figures visualize how hierarchical clustering reveals structure in the asset relationships. Both classical and quantum HRP reorder the covariance matrix to be quasi-diagonal, grouping similar assets. The quantum distance matrix shows that QHRP successfully partitions assets into coherent clusters based on their quantum representations.

## Discussions

The empirical results indicate that QHRP is a practical methodology capable of delivering superior and more robust investment portfolios by advancing beyond correlation-based methods.

  - **Advancement**: By encoding a richer, six-dimensional feature set, QHRP mitigates the dependency on the notoriously unstable correlation matrix. This leads to the highest out-of-sample Sharpe ratios among all tested methods.
  - **Quantum vs. Kernel**: Both QHRP and KHRP improve upon the original HRP by using multiple features. However, the substantial performance leap of QHRP over KHRP suggests that quantum feature mapping, which uses entanglement to model interdependencies, provides a more powerful and effective way to represent asset data.
  - **Limitations and Future Research**:
    1.  **Hardware**: Experiments were run on classical simulators. Testing on physical NISQ devices is a critical next step.
    2.  **Hyperparameters**: A systematic study of feature importance and hyperparameter optimization could unlock further gains.
    3.  **Scalability**: Investigating the scalability of QHRP for even larger institutional portfolios is paramount, an area where future fault-tolerant quantum computers are expected to excel.