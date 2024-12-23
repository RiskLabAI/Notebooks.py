import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.covariance import ledoit_wolf
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import cvxpy as cp

from quadratic_inverse_shrinkage import QIS

class HeterogeneousHedgedRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Heterogeneous Hedged Random Forest Regressor.

    This regressor extends the Hedged Random Forest by incorporating partitioning
    of the conditioning variable space using K-means clustering. It optimizes
    weights for each cluster to minimize the mean-squared error (MSE) using a convex
    optimization approach, allowing for negative weights.

    Parameters
    ----------
    n_estimators : int, default=500
        The number of trees in the forest.

    criterion : str, default='squared_error'
        The function to measure the quality of a split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights required to be at a leaf node.

    max_features : int, float, str or None, default=1.0
        The number of features to consider when looking for the best split.

    max_leaf_nodes : int, default=None
        Grow trees with max_leaf_nodes in best-first fashion.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : int, default=None
        The number of jobs to run in parallel.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.

    shrinkage : str, default='ledoit_wolf'
        Method for covariance estimation. Options are 'ledoit_wolf' or 'empirical' or 'quadratic_inverse'.

    kappa : float, default=2.0
        Gross-exposure constraint parameter.

    n_partition : int, default=10
        The number of clusters to partition the conditioning variable space.

    Attributes
    ----------
    random_forest_ : RandomForestRegressor
        The fitted random forest model.

    kmeans_ : KMeans
        The fitted KMeans clustering model.

    weights_ : ndarray of shape (n_partition, n_estimators)
        The optimized weights for each cluster.

    mu_ : ndarray of shape (n_partition, n_estimators)
        The mean vector of in-sample residuals for each cluster.

    Sigma_ : ndarray of shape (n_partition, n_estimators, n_estimators)
        The covariance matrix of in-sample residuals for each cluster.

    Examples
    --------
    >>> from heterogeneous_hedged_random_forests import HeterogeneousHedgedRandomForestRegressor
    >>> model = HeterogeneousHedgedRandomForestRegressor(n_estimators=500, max_depth=10, n_partition=5, kappa=2.0)
    >>> model.fit(X_train, y_train, z_train)
    >>> predictions = model.predict(X_test, z_test)
    """

    def __init__(
        self,
        n_estimators=500,
        criterion='squared_error',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        shrinkage='ledoit_wolf',
        kappa=2.0,
        n_partition=10
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.shrinkage = shrinkage
        self.kappa = kappa
        self.n_partition = n_partition

    def fit(self, X, y, z):
        """
        Fit the heterogeneous hedged random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        z : array-like of shape (n_samples, n_z_features)
            The conditioning variables for clustering.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=True)
        # z = check_array(z, accept_sparse=True)

        # Initialize and fit the Random Forest
        self.random_forest_ = RandomForestRegressor(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start
        )
        self.random_forest_.fit(X, y)

        # Fit Gaussian Mixture Model clustering on z
        self.gmm_ = Pipeline([
            ('scaler', StandardScaler()),
            ('gmm', GaussianMixture(n_components=self.n_partition, random_state=self.random_state))
        ])
        cluster_labels = self.gmm_.fit_predict(z)

        # Extract individual tree predictions on the training data
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        predictions = np.zeros((n_samples, n_estimators))

        for idx, tree in enumerate(self.random_forest_.estimators_):
            predictions[:, idx] = tree.predict(X)

        # Initialize arrays to store weights, mu, and Sigma for each cluster
        self.weights_ = np.zeros((self.n_partition, n_estimators))
        self.mu_ = np.zeros((self.n_partition, n_estimators))
        self.Sigma_ = np.zeros((self.n_partition, n_estimators, n_estimators))

        # Weight Optimization per Cluster
        for k in range(self.n_partition):
            # Get indices for the current cluster
            cluster_indices = np.where(cluster_labels == k)[0]
            if len(cluster_indices) == 0:
                raise ValueError(f"Cluster {k} has no samples.")

            # Compute residuals for the current cluster
            residuals = y[cluster_indices].reshape(-1, 1) - predictions[cluster_indices, :]  # Shape (n_k, p)

            # Estimate conditional mean vector mu_k
            mu_k = residuals.mean(axis=0)  # Shape (p,)

            # Estimate conditional covariance matrix Sigma_k
            if self.shrinkage == 'ledoit_wolf':
                Sigma_k, _ = ledoit_wolf(residuals)
            elif self.shrinkage == 'empirical':
                Sigma_k = np.cov(residuals, rowvar=False)
            elif self.shrinkage == 'quadratic_inverse':
                Sigma_k = QIS(pd.DataFrame(residuals)).values
            else:
                raise ValueError("Invalid shrinkage method. Choose 'ledoit_wolf' or 'empirical' or 'quadratic_inverse'.")

            self.mu_[k] = mu_k
            self.Sigma_[k] = Sigma_k

            # Solve the optimization problem to find weights for cluster k
            self.weights_[k] = self._solve_optimization(mu_k, Sigma_k)

        return self

    def predict(self, X, z):
        """
        Predict using the heterogeneous hedged random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        z : array-like of shape (n_samples, n_z_features)
            The conditioning variables for clustering.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ['weights_', 'random_forest_', 'gmm_'])
        X = check_array(X, accept_sparse=True)
        # z = check_array(z, accept_sparse=True)

        # Get posterior probabilities for each mixture component
        cluster_probabilities = self.gmm_.predict_proba(z)  # Shape (n_samples, n_partition)

        # Collect predictions from base trees
        predictions = np.column_stack([
            tree.predict(X) for tree in self.random_forest_.estimators_
        ])  # Shape (n_samples, n_estimators)

        # Compute w_sample for all samples
        w_sample = cluster_probabilities.dot(self.weights_)  # Shape (n_samples, n_estimators)

        # Compute the final predictions
        y_pred = np.sum(predictions * w_sample, axis=1)  # Shape (n_samples,)

        return y_pred

    def _solve_optimization(self, mu, Sigma):
        """
        Solve the hedged forecast combination optimization problem with ||w||_1 <= kappa.

        Minimizes (w^T mu)^2 + w^T Sigma w
        subject to w^T 1 = 1 and ||w||_1 <= kappa.

        Parameters
        ----------
        mu : ndarray of shape (n_estimators,)
            The mean vector of in-sample residuals for the cluster.

        Sigma : ndarray of shape (n_estimators, n_estimators)
            The covariance matrix of in-sample residuals for the cluster.

        Returns
        -------
        weights : ndarray of shape (n_estimators,)
            The optimized weights for the cluster.
        """
        p = len(mu)
        mu = mu.reshape(-1, 1)  # Shape (p, 1)

        # Define the optimization variable
        w = cp.Variable(p)

        # Define the objective function
        objective = cp.Minimize(cp.square(cp.matmul(w, mu)) + cp.quad_form(w, Sigma))

        # Define the constraints
        constraints = [
            cp.sum(w) == 1,
            cp.norm1(w) <= self.kappa
        ]

        # Define and solve the problem
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError("Optimization did not converge")

        weights = w.value
        return weights
