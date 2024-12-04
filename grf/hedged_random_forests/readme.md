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

The goal is to forecast (or predict) a random variable \( y \in \mathbb{R} \) based on a set of variables (or attributes) \( x \in \mathbb{R}^d \). Denote a generic forecast by \( \hat{f} \). Then its mean-squared error (MSE) is given by:

\[
\text{MSE}(\hat{f}) \coloneqq \mathbb{E} \left( y - \hat{f}(x) \right)^2.
\]

Letting

\[
\text{Bias}(\hat{f}) \coloneqq \mathbb{E} \left( y - \hat{f}(x) \right)
\quad \text{and} \quad
\text{Var}(\hat{f}) \coloneqq \operatorname{Var} \left( y - \hat{f}(x) \right) = \mathbb{E} \left( \left( y - \hat{f}(x) \right)^2 \right) - \left( \mathbb{E} \left( y - \hat{f}(x) \right) \right)^2,
\]

there exists the well-known decomposition:

\[
\text{MSE}(\hat{f}) = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}).
\]

The oracle that minimizes the MSE is given by the conditional expectation \( \hat{f}_{\text{or}}(x) \coloneqq \mathbb{E}(y \mid x) \) but is not available in practice.

This paper considers combinations of a given set of \( p \) forecasting methods (or forecasting models), denoted by \( \{ \mathcal{M}_j \}_{j=1}^p \). The number of methods, \( p \), is assumed to be exogenous and fixed.

There exists an extensive literature on forecast combinations. The consensus seems to be that simple averaging (or equal weighting), given by

\[
\hat{f}_{\text{AV}}(x) \coloneqq \frac{1}{p} \sum_{j=1}^p \mathcal{M}_j(x),
\]

is hard to beat by more general linear combinations of the kind

\[
\hat{f}_w(x) \coloneqq \sum_{j=1}^p w_j \mathcal{M}_j(x) \quad \text{with} \quad w \coloneqq (w_1, \ldots, w_p)' \quad \text{and} \quad \sum_{j=1}^p w_j = 1.
\]

Nevertheless, our aim is to find a method for selecting a set of weights \( w \) that does improve the (out-of-sample) MSE of simple averaging, at least "on balance".

Denote by \( e_j \coloneqq y - \mathcal{M}_j(x) \) the forecast error made by model \( \mathcal{M}_j \) and collect these errors into the vector \( e \coloneqq (e_1, \ldots, e_p)' \) with expectation vector and covariance matrix

\[
\mu \coloneqq \mathbb{E}(e) \quad \text{and} \quad \Sigma \coloneqq \operatorname{Var}(e).
\]

The MSE of the forecast \( \hat{f}_w \) is then given by

\[
\text{MSE}(\hat{f}_w) = (w' \mu)^2 + w' \Sigma w.
\]

Therefore, the optimal (in terms of the MSE) forecast in the class \( \hat{f}_w \) is the solution of the following optimization problem:

\[
\begin{align}
& \min_w \quad (w' \mu)^2 + w' \Sigma w, \label{eq:opt1} \\
\text{subject to} \quad & w' \mathbf{1} = 1, \label{eq:opt2}
\end{align}
\]

where \( \mathbf{1} \) denotes a vector of ones of appropriate dimension.

In practice, the inputs \( \mu \) and \( \Sigma \) are unknown. A feasible solution is to replace them with sample-based estimates \( \hat{\mu} \) and \( \hat{\Sigma} \), which is an application of the general "plug-in method".

Being agnostic, for the time being, about the nature of the estimators \( \hat{\mu} \) and \( \hat{\Sigma} \), we then solve the feasible optimization problem:

\[
\begin{align}
& \min_w \quad (w' \hat{\mu})^2 + w' \hat{\Sigma} w, \label{eq:fopt1} \\
\text{subject to} \quad & w' \mathbf{1} = 1, \label{eq:fopt2} \\
& \| w \|_1 \leq \kappa, \label{eq:fopt3}
\end{align}
\]

where \( \| w \|_1 \coloneqq \sum_{j=1}^p |w_j| \) denotes the \( L_1 \) norm of \( w \), and \( \kappa \in [1, \infty] \) is a constant chosen by the user.

Assuming that the estimator \( \hat{\Sigma} \) is symmetric and positive semi-definite, the optimization problem \eqref{eq:fopt1}–\eqref{eq:fopt3} is still of convex nature and can be solved easily and quickly in practice, even for large dimensions \( p \). We shall denote the solution to this optimization problem by \( \hat{w} \).

The addition of the constraint \eqref{eq:fopt3} is motivated by the related problem of **portfolio selection** in finance, in which context the constraint is called a "gross-exposure constraint". Adding this type of constraint to the infeasible problem \eqref{eq:opt1}–\eqref{eq:opt2} clearly would result in a (weakly) worse solution for any value \( \kappa \in [1, \infty) \). But in the feasible problem, which must use estimated instead of true inputs, the constraint typically helps. The intuition here is that replacing \( \mu \) and \( \Sigma \) with respective estimates \( \hat{\mu} \) and \( \hat{\Sigma} \) can lead to unstable and underdiversified solutions that look good in sample (or in the training set) but perform badly out of sample, especially when the number of models, \( p \), is not (exceedingly) small relative to the sample size relevant to the estimation of \( \mu \) and \( \Sigma \).

In the extreme case \( \kappa = 1 \), the weights are forced to be non-negative, that is, \( w_j \geq 0 \). Imposing this constraint is standard in the forecast-combination literature but it might well lead to sub-optimal performance because of not giving enough flexibility to the solution of the problem \eqref{eq:fopt1}–\eqref{eq:fopt3}. At the other end of the spectrum, the choice \( \kappa = \infty \) corresponds to removing the constraint \eqref{eq:fopt3}, which may also lead to sub-optimal performance for the reasons mentioned above. Staying away from either extreme, there is ample evidence in the finance literature that choosing \( \kappa \in [1.5, 2.5] \) typically results in improved forecasting performance, and that the exact choice in this interval is not overly critical.

Because the constraint \eqref{eq:fopt3} protects the user against extreme "positions", that is, against weights \( \hat{w}_j \) that are unduly large in absolute value, we call our approach **"hedging forecast combinations"**.

## Theory

The solution to the convex optimization problem \eqref{eq:fopt1}–\eqref{eq:fopt3} is continuous in the inputs \( \hat{\mu} \) and \( \hat{\Sigma} \). Therefore, with the choice \( \kappa = \infty \), its solution, denoted by \( \hat{w} \), would lead to an asymptotically optimal forecast combination \( \hat{f}_{\hat{w}} \) based on consistent estimators \( \hat{\mu} \) and \( \hat{\Sigma} \). Stating this fact in a theorem is possible, but as this is a routine matter we find it outside the scope of the basic research content of this paper. First, this fact has been recognized before. Furthermore, in practical application, the relevant property is the finite-sample performance of the forecast \( \hat{f}_{\hat{w}} \) and, so far, the evidence based on simulation studies and empirical applications to real-life data sets indicates that such forecast combinations, on balance, do not outperform \( \hat{f}_{\text{AV}} \), that is, simple averaging.

Therefore, our goal is isolated to finding a forecast combination \( \hat{f}_{\hat{w}} \) that, on balance, outperforms \( \hat{f}_{\text{AV}} \) in empirical applications to commonly used benchmark data sets.

### Remark: Scale Invariance

The solution \( \hat{w} \) to the optimization problem \eqref{eq:fopt1}–\eqref{eq:fopt3} remains unchanged if \( \hat{\mu} \) and \( \hat{\Sigma} \) are replaced by \( c \hat{\mu} \) and \( c^2 \hat{\Sigma} \), respectively, for any constant \( c \in (0, \infty) \). Therefore, it is not important that the estimators \( \hat{\mu} \) and \( \hat{\Sigma} \) get the "levels" of the true quantities \( \mu \) and \( \Sigma \) right. In particular, the use of in-sample (or training-set) errors in the construction of \( \hat{\mu} \) and \( \hat{\Sigma} \) can still lead to favorable performance of the forecast combination \( \hat{f}_{\hat{w}} \) even if such errors are systematically smaller (in magnitude) compared to out-of-sample errors because of in-sample (or training-set) overfitting. Instead of approximating the actual entries of \( \mu \) and \( \Sigma \), the corresponding estimators \( \hat{\mu} \) and \( \hat{\Sigma} \) only need to approximate the entries relative to each other in order for \( \hat{f}_{\hat{w}} \) to outperform \( \hat{f}_{\text{AV}} \).

### Remark: Importance of Negative Weights

Notably all previous proposals for weighting the random forest that we are aware of, not only in the context of regression but also in the context of classification, impose the "no-short-sales constraint" \( \kappa = 1 \), that is, \( w_j \geq 0 \; \forall j \). As shown in the robustness checks, allowing for negative weights generally improves performance and for certain data sets by a pronounced margin. In the context of finance, a "no-short-sales constraint" can be motivated by legislation (for example, mutual funds are not allowed to short stocks) or by practical considerations (for example, shorting certain assets may not be possible or prohibitively expensive). On the other hand, "short-selling" individual forecast methods \( \mathcal{M}_j \) by assigning them a negative weight is always possible and does not incur any monetary costs. We, therefore, hope that our paper will serve as motivation to the scientific community to allow for negative weights not only in the random forest but also in other forecast-combination applications.

