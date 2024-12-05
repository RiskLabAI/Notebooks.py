import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.covariance import ledoit_wolf
import cvxpy as cp

class HedgedRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Hedged Random Forest Regressor.

    This regressor implements a hedged forecast combination by applying hedged forecast
    combinations to the individual trees of a random forest. It optimizes the weights
    of the trees to minimize the mean-squared error (MSE) using a convex optimization
    approach, allowing for negative weights.

    Parameters
    ----------
    n_estimators : int, default=100
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
        Method for covariance estimation. Options are 'ledoit_wolf' or 'empirical'.

    kappa : float, default=2.0
        Gross-exposure constraint parameter.

    Attributes
    ----------
    fitted_models_ : RandomForestRegressor
        The fitted random forest model.

    weights_ : ndarray of shape (n_estimators,)
        The optimized weights for each tree.

    mu_ : ndarray of shape (n_estimators,)
        The mean vector of in-sample residuals.

    Sigma_ : ndarray of shape (n_estimators, n_estimators)
        The covariance matrix of in-sample residuals.

    Examples
    --------
    >>> from hedged_random_forests import HedgedRandomForestRegressor
    >>> model = HedgedRandomForestRegressor(n_estimators=100, max_depth=10, kappa=2.0)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """

    def __init__(self,
                 n_estimators=100,
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
                 kappa=2.0):
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

    def fit(self, X, y):
        """
        Fit the hedged random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        
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

        # Extract individual tree predictions on the training data
        n_samples = X.shape[0]
        n_estimators = self.n_estimators
        predictions = np.zeros((n_samples, n_estimators))

        for idx, tree in enumerate(self.random_forest_.estimators_):
            predictions[:, idx] = tree.predict(X)

        # Compute residuals (in-sample forecast errors)
        residuals = y.reshape(-1, 1) - predictions

        # Estimate mean vector mu
        self.mu_ = residuals.mean(axis=0)

        # Estimate covariance matrix Sigma
        if self.shrinkage == 'ledoit_wolf':
            self.Sigma_, _ = ledoit_wolf(residuals)
        elif self.shrinkage == 'empirical':
            self.Sigma_ = np.cov(residuals, rowvar=False)
        else:
            raise ValueError("Invalid shrinkage method. Choose 'ledoit_wolf' or 'empirical'.")

        # Solve the optimization problem to find weights
        self.weights_ = self._solve_optimization()

        return self

    def predict(self, X):
        """
        Predict using the hedged random forest regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ['weights_', 'random_forest_'])
        X = check_array(X, accept_sparse=True)

        # Collect predictions from base trees
        predictions = np.column_stack([
            tree.predict(X) for tree in self.random_forest_.estimators_
        ])

        # Compute weighted combination
        y_pred = predictions @ self.weights_
        return y_pred

    def _solve_optimization(self):
        """
        Solve the hedged forecast combination optimization problem with ||w||_1 <= kappa.

        Minimizes (w^T mu)^2 + w^T Sigma w
        subject to w^T 1 = 1 and ||w||_1 <= kappa.

        Returns
        -------
        weights : ndarray of shape (n_estimators,)
            The optimized weights for each tree.
        """
        p = self.n_estimators
        mu = self.mu_.reshape(-1, 1)  # shape (p, 1)
        Sigma = self.Sigma_

        # Define the optimization variable
        w = cp.Variable(p)

        # Define the objective function
        objective = cp.Minimize(cp.square(w @ mu) + cp.quad_form(w, Sigma))

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

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Hedged Random Forest Regressor
hrf = HedgedRandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    shrinkage='ledoit_wolf',
    kappa=2.5,
    random_state=42
)

# Fit the model
hrf.fit(X_train, y_train)

# Make predictions
y_pred = hrf.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f"Hedged Random Forest MSE: {mse:.4f}")

# Compare with standard Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Standard Random Forest MSE: {mse_rf:.4f}")
