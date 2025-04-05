import pandas as pd
import numpy as np
from enum import Enum
from tqdm.notebook import tqdm
from numpy.random import RandomState
from collections import Counter
import itertools
import sklearn.metrics as Metrics
from sklearn.preprocessing import minmax_scale
from pycaret.classification import ClassificationExperiment
from sklearn.model_selection import StratifiedKFold 
from arch.bootstrap import optimal_block_length
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, MovingBlockBootstrap, IndependentSamplesBootstrap, IIDBootstrap
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, f_classif
from scipy.stats import pearsonr, kendalltau
import sklearn
from typing import Union, List
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

def feature_importance_linear_models(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
) -> pd.Series:
    """
    Computes and returns feature importances for a linear model by calculating
    the absolute values of the model's coefficients and scaling them between 0 and 1.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A linear model classifier from scikit-learn. The classifier must have a `coef_` attribute
        after fitting, which contains the model coefficients. Examples include `LogisticRegression`,
        `LinearSVC`, etc.
    
    X : pd.DataFrame or np.ndarray
        The feature matrix used for training the classifier. Should be of shape 
        `(n_samples, n_features)` where `n_samples` is the number of samples and 
        `n_features` is the number of features.
    
    y : pd.Series or np.ndarray
        The target values (class labels) corresponding to the feature matrix `X`. Should be of shape 
        `(n_samples,)`.
    
    Returns
    -------
    pd.Series
        A Pandas Series containing the feature importances scaled between 0 and 1. The index 
        corresponds to the feature names, and the values represent the relative importance of 
        each feature in the model.
    
    Raises
    ------
    AttributeError
        If the classifier does not have a `coef_` attribute, which is required to extract the model 
        coefficients.
    
    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.preprocessing import minmax_scale
    >>>
    >>> # Example Data
    >>> X = pd.DataFrame({
    >>>     'feature1': np.random.rand(100),
    >>>     'feature2': np.random.rand(100),
    >>>     'feature3': np.random.rand(100)
    >>> })
    >>> y = np.random.randint(0, 2, size=100)
    >>>
    >>> # Model
    >>> classifier = LogisticRegression()
    >>>
    >>> # Compute feature importances
    >>> importances = feature_importance_linear_models(classifier, X, y)
    >>> print(importances)
    
    Notes
    -----
    - The function first fits the classifier to the data `X` and `y`.
    - It then calculates the absolute mean value of the model's coefficients, which represent the 
      feature importances.
    - These importances are scaled to the range `[0, 1]` using the `minmax_scale` function to make 
      them easier to interpret.
    """

    classifier.fit(X, y)
    coefficients_importances = np.abs(classifier.coef_).mean(axis=0)
    coefficients_importances = pd.Series(coefficients_importances, classifier.feature_names_in_)

    importances_scaled = minmax_scale(
        coefficients_importances,
        feature_range=(0, 1),
        axis=0,
    )

    return pd.Series(importances_scaled, index=coefficients_importances.index)

def feature_importance_sklearn(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
) -> pd.Series:
    """
    Fits a classifier to the data and computes the feature importances.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A scikit-learn compatible classifier that has a `feature_importances_` attribute after fitting.
    
    X : pd.DataFrame or np.ndarray
        The feature matrix used for training. Rows represent samples and columns represent features.
    
    y : pd.Series or np.ndarray
        The target values (class labels in classification).

    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to feature names and the values are scaled feature importances
        (ranging between 0 and 1).
    
    Notes
    -----
    - The function expects the classifier to have a `feature_importances_` attribute, which is typically available 
      for tree-based models like `RandomForestClassifier`, `GradientBoostingClassifier`, etc.
    - The feature names are derived from the classifier's `feature_names_in_` attribute, which should be automatically
      set if the classifier is fitted with a pandas DataFrame `X` as input.

    Example
    -------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Initialize classifier
    clf = RandomForestClassifier()

    # Get scaled feature importances
    importances = feature_importance_sklearn(clf, X, y)
    print(importances)
    ```
    
    """
    classifier.fit(X, y)
    importances = pd.Series(classifier.feature_importances_, index=classifier.feature_names_in_)
    
    importances = pd.concat({
        "Mean": importances.mean(),
    }, axis=1)

    importances_scaled = minmax_scale(
        importances["Mean"],
        feature_range=(0, 1),
        axis=0,
    )

    return pd.Series(importances_scaled, index=importances.index)

def feature_importance_RFE(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
) -> pd.Series:
    """
    Computes feature importance using Recursive Feature Elimination (RFE).

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A scikit-learn compatible estimator used for ranking features. This can be any model that has a `fit` method.
    
    X : pd.DataFrame or np.ndarray
        The feature matrix used for training. Rows represent samples and columns represent features.
    
    y : pd.Series or np.ndarray
        The target values (class labels in classification or target values in regression).

    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to feature names and the values represent the normalized
        importance of each feature, scaled between 0 and 1.
    
    Notes
    -----
    - The function uses Recursive Feature Elimination (RFE) to rank features based on their importance to the classifier.
    - The returned feature importances are scaled to be within the range [0, 1], with higher values indicating more important features.
    - The function expects the input features `X` to be in a pandas DataFrame or a NumPy array, and if `X` is a DataFrame, 
      the feature names will be extracted automatically.

    Example
    -------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Initialize classifier
    clf = RandomForestClassifier()

    # Get RFE-based feature importances
    importances = feature_importance_RFE(clf, X, y)
    print(importances)
    ```
    """
    rfe = RFE(
        estimator=classifier,
        verbose=0,
        n_features_to_select=1,    
    )

    rfe.fit(X, y)

    inverted_ranking = np.max(rfe.ranking_) - rfe.ranking_ + 1

    normalized_importance = minmax_scale(inverted_ranking)

    return pd.Series(normalized_importance, index=rfe.feature_names_in_)


def feature_importance_SFI(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
    n_splits: int = 5,
    score_sample_weights: np.ndarray = None,  
    train_sample_weights: np.ndarray = None, 
) -> pd.Series:
    """
    Computes feature importances using a Single Feature Importance (SFI) technique, which involves training
    and evaluating a classifier on each feature individually using cross-validation.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A scikit-learn compatible classifier that has a `fit` and `predict` method.
    
    X : pd.DataFrame or np.ndarray
        The feature matrix used for training. Rows represent samples and columns represent features.
    
    y : pd.Series or np.ndarray
        The target values (class labels in classification).

    n_splits : int, default=5
        Number of splits/folds to be used in Stratified K-Fold cross-validation.
    
    score_sample_weights : np.ndarray, optional
        Sample weights for scoring in cross-validation. If not provided, all samples are weighted equally.

    train_sample_weights : np.ndarray, optional
        Sample weights for training the classifier in cross-validation. If not provided, all samples are weighted equally.

    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to feature names and the values are scaled feature importances
        (ranging between 0 and 1).

    Notes
    -----
    - The function implements a Single Feature Importance (SFI) technique by training a separate model for each feature
      and evaluating its performance using Stratified K-Fold cross-validation.
    - The final importance for each feature is the mean accuracy score across all folds, normalized to a range of 0 to 1.

    Example
    -------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Initialize classifier
    clf = RandomForestClassifier()

    # Compute Single Feature Importances
    importances = feature_importance_SFI(clf, X, y)
    print(importances)
    ```
    
    """
    if train_sample_weights is None:
        train_sample_weights = np.ones(X.shape[0])
    if score_sample_weights is None:
        score_sample_weights = np.ones(X.shape[0])

    cv_generator = StratifiedKFold(n_splits=n_splits)
    feature_names = X.columns
    importances = []
    for feature_name in feature_names:
        scores = []

        for train, test in cv_generator.split(X, y):
            x_train, y_train, weights_train = X.iloc[train, :][[feature_name]], y.iloc[train], train_sample_weights[train]
            x_test, y_test, weights_test = X.iloc[test, :][[feature_name]], y.iloc[test], score_sample_weights[test]

            feature_train, label_train, sample_weights_train = (
                x_train, y_train, weights_train,
            )

            feature_test, label_test, sample_weights_test = (
                x_test, y_test, weights_test,
            )

            try:
                classifier.fit(feature_train, label_train, sample_weight=sample_weights_train)
            except:
                classifier.fit(feature_train, label_train)

            prediction = classifier.predict(feature_test)
            score = Metrics.accuracy_score(label_test, prediction, sample_weight=sample_weights_test)
            scores.append(score)

        importances.append({
            "FeatureName": feature_name,
            "Mean": np.mean(scores),
        })

    importances = pd.DataFrame(importances)
    importances_scaled = minmax_scale(
        importances["Mean"],
        feature_range=(0, 1),
        axis=0,
    )

    return pd.Series(importances_scaled, index=importances.FeatureName)


def feature_importance_MDI(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
) -> pd.Series:
    """
    Fits a classifier to the data and computes the Mean Decrease in Impurity (MDI) based feature importances.
    
    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A scikit-learn compatible classifier that has a `feature_importances_` attribute after fitting.
        Typically, this applies to tree-based models like `RandomForestClassifier`, `GradientBoostingClassifier`, etc.
    
    X : pd.DataFrame or np.ndarray
        The feature matrix used for training. Rows represent samples and columns represent features.
    
    y : pd.Series or np.ndarray
        The target values (class labels in classification).
    
    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to feature names and the values are scaled feature importances
        based on Mean Decrease in Impurity (ranging between 0 and 1).
    
    Notes
    -----
    - The function relies on the `feature_importances_` attribute of the fitted classifier, which measures the importance
      of each feature in reducing the impurity (e.g., Gini impurity or entropy) across the splits of a decision tree.
    - The feature names are derived from the classifier's `feature_names_in_` attribute, which should be automatically
      set if the classifier is fitted with a pandas DataFrame `X` as input.
    - The feature importances are scaled using `minmax_scale` to range between 0 and 1 for easier comparison.

    Example
    -------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Initialize classifier
    clf = RandomForestClassifier()

    # Get scaled feature importances
    importances = feature_importance_MDI(clf, X, y)
    print(importances)
    ```
    """
    classifier.fit(X, y)
    importances = pd.Series(classifier.feature_importances_, index=classifier.feature_names_in_)
    
    importances_scaled = minmax_scale(
        importances,
        feature_range=(0, 1),
        axis=0,
    )

    return pd.Series(importances_scaled, index=importances.index)


def feature_importance_MDA(
    classifier: sklearn.base.BaseEstimator,
    X: Union[pd.DataFrame, np.ndarray], 
    y: Union[pd.Series, np.ndarray],
    n_repeats: int = 5,
) -> pd.Series:
    """
    Computes feature importances using the Mean Decrease Accuracy (MDA) method via permutation importance.

    Parameters
    ----------
    classifier : sklearn.base.BaseEstimator
        A scikit-learn compatible classifier that implements the `fit` method.
    
    X : Union[pd.DataFrame, np.ndarray]
        The feature matrix used for training. Rows represent samples and columns represent features.
        If a DataFrame is provided, the column names are used as feature names in the output.
    
    y : Union[pd.Series, np.ndarray]
        The target values (class labels in classification).

    n_repeats : int, default=5
        The number of times to permute a feature to compute the permutation importance.

    Returns
    -------
    pd.Series
        A pandas Series where the index corresponds to feature names (if `X` is a DataFrame) or feature indices 
        (if `X` is a NumPy array), and the values are scaled feature importances ranging between 0 and 1.
    
    Notes
    -----
    - This function uses the permutation importance method to evaluate feature importance, which shuffles feature
      values to measure the decrease in model accuracy.
    - The function scales the feature importances to a [0, 1] range using min-max scaling.
    - If `X` is a NumPy array, the index of the returned Series will be the feature indices (0, 1, 2, ...).
    - If `X` is a pandas DataFrame, the column names will be used as feature names in the returned Series.

    Example
    -------
    ```python
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    import pandas as pd

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Initialize classifier
    clf = RandomForestClassifier()

    # Get scaled permutation importances
    importances = feature_importance_MDA(clf, X, y, n_repeats=10)
    print(importances)
    ```

    """
    classifier.fit(X, y)
    importances = permutation_importance(classifier, X, y, n_repeats=n_repeats, random_state=43).importances_mean

    importances = minmax_scale(
        importances,
        feature_range=(0, 1),
        axis=0,
    )

    return pd.Series(importances, index=X.columns)


class UAMeasure(Enum):
    """
    An enumeration that defines different statistical measures used for 
    analyzing the relationship or association between variables.

    Attributes
    ----------
    SPEARMAN : str
        Spearman's rank correlation coefficient, a non-parametric measure of rank correlation.
    
    PEARSON : str
        Pearson's correlation coefficient, a measure of linear correlation between two variables.
    
    KENDAL_TAU : str
        Kendall's Tau, a non-parametric measure of the strength and direction of association between two variables.
    
    MUTUAL_INFORMATION : str
        Mutual Information, a measure of the amount of information obtained about one variable through another.
    
    ANOVA_F : str
        ANOVA F-statistic, used in ANOVA to determine whether there are significant differences between group means.
    
    Examples
    --------
    You can use this enumeration to specify the type of statistical measure you want to apply in your analysis.

    ```python
    measure = UAMeasure.SPEARMAN
    print(measure.value)  # Output: "Spearman"
    ```

    This enumeration can be useful in functions or classes that require specifying the type of statistical 
    analysis to perform, ensuring consistency and reducing errors related to string literals.

    """

    SPEARMAN = "Spearman"
    PEARSON = "Pearson"
    KENDAL_TAU = "Kendal-Tau"
    MUTUAL_INFORMATION = "Mutual Information"
    ANOVA_F = "ANOVA F-Stat"

def feature_importance_univariate_analysis(measurements, X, y):
    """
    Computes and scales feature importance scores using various univariate statistical measures.

    Parameters
    ----------
    measurements : list of str
        A list of strings specifying which univariate analysis measures to compute. Valid options are:
        - 'spearman': Spearman's rank correlation coefficient
        - 'pearson': Pearson correlation coefficient
        - 'kendall_tau': Kendall's Tau correlation coefficient
        - 'mutual_information': Mutual information between features and target
        - 'anova_f': ANOVA F-values

    X : pd.DataFrame
        A DataFrame with shape (n_samples, n_features), where each column represents a feature and each row represents a sample.

    y : pd.Series or np.ndarray
        A one-dimensional array-like structure with shape (n_samples,), representing the target values or class labels.

    Returns
    -------
    pd.DataFrame
        A DataFrame where:
        - Rows correspond to the feature names from `X`.
        - Columns correspond to the univariate analysis measures specified in the `measurements` list.
        - Values are scaled feature importance scores between 0 and 1.

    Notes
    -----
    - The function uses the following univariate measures:
        - **Spearman's Rank Correlation**: Measures the strength and direction of the association between two ranked variables.
        - **Pearson Correlation Coefficient**: Measures the linear relationship between two variables.
        - **Kendall's Tau**: Measures the ordinal association between two variables.
        - **Mutual Information**: Measures the amount of information obtained about one variable through another. For classification problems, mutual information is computed between each feature and the target.
        - **ANOVA F-value**: Measures the feature's contribution to explaining the variance in the target variable.
    - Ensure that `X` is a DataFrame and `y` is a one-dimensional array-like structure.
    - The `mutual_info_classif` function is used with `n_neighbors=51`. Adjust this parameter according to your specific problem if needed.
    - If a measure is not included in the `measurements` list, it will not be computed.

    Example
    -------
    ```python
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np

    # Load dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Compute feature importance using Spearman and ANOVA F-values
    measures = ['spearman', 'anova_f']
    importance_df = feature_importance_univariate_analysis(measures, X, y)
    print(importance_df)
    ```

    """
    ua = pd.DataFrame(
        data=None,
        index=X.columns
    )

    if UAMeasure.SPEARMAN.value in measurements:
        spearmans = minmax_scale(X.apply(lambda feature: (np.abs(np.corrcoef(feature, y)[0, 1])), axis=0))
        ua[UAMeasure.SPEARMAN.value] = spearmans

    if UAMeasure.PEARSON.value in measurements:
        pearsons = minmax_scale(X.apply(lambda feature: np.abs(pearsonr(feature, y).statistic), axis=0))
        ua[UAMeasure.PEARSON.value] = pearsons
        
    if UAMeasure.KENDAL_TAU.value in measurements:
        kendalltaus = minmax_scale(X.apply(lambda feature: np.abs(kendalltau(feature, y).statistic), axis=0))
        ua[UAMeasure.KENDAL_TAU.value] = kendalltaus

    if UAMeasure.MUTUAL_INFORMATION.value in measurements:
        mutual_informations = minmax_scale(mutual_info_classif(X, y, n_neighbors=51))
        ua[UAMeasure.MUTUAL_INFORMATION.value] = mutual_informations

    if UAMeasure.ANOVA_F.value in measurements:
        f_values = minmax_scale(f_classif(X, y)[0])
        ua[UAMeasure.ANOVA_F.value] = f_values
    
    return ua


def full_name(klass):
    """
    Returns the full module path and class name of a given class.

    Parameters
    ----------
    klass : type
        The class for which the full name is to be retrieved. This should be a class object, not an instance.

    Returns
    -------
    str
        A string representing the full name of the class in the format 'module.class_name'.

    Examples
    --------
    >>> from datetime import datetime
    >>> full_name(datetime)
    'datetime.datetime'

    >>> import numpy as np
    >>> full_name(np.ndarray)
    'numpy.ndarray'

    Notes
    -----
    - This function works by accessing the `__module__` and `__name__` attributes of the class.
    - Ensure that the input `klass` is indeed a class and not an instance of a class, as this function expects a class object.

    """
    return ".".join([klass.__module__, klass.__name__])

from enum import Enum

class FIModel(Enum):
    """
    Enum class representing different types of feature importance (FI) models.

    Attributes
    ----------
    COEFFICIENT_BASED : str
        Describes feature importance based on model coefficients with optional shrinkage and selection.
    TREE_BASED : str
        Refers to feature importance derived from tree-based models.
    PERMUTATION_BASED : str
        Represents feature importance based on permutation methods.
    SINGLE_FEATURE_BASED : str
        Indicates feature importance calculated by evaluating single features individually.
    RFE_BASED : str
        Refers to feature importance determined using Recursive Feature Elimination (RFE) methods.

    Notes
    -----
    - **COEFFICIENT_BASED**: This method uses the coefficients of models (e.g., linear regression) to gauge feature importance. It may involve additional techniques like shrinkage (e.g., Lasso) and selection to enhance the model's interpretability.
    - **TREE_BASED**: Involves feature importance scores from tree-based models (e.g., decision trees, random forests). These scores reflect how important each feature is in predicting the target variable, typically calculated as the reduction in impurity or gain.
    - **PERMUTATION_BASED**: Measures feature importance by permuting (shuffling) feature values and observing the impact on model performance. Features that cause a significant drop in performance when shuffled are considered more important.
    - **SINGLE_FEATURE_BASED**: Evaluates each feature individually to assess its contribution to the model's performance. This approach might involve metrics like correlation or univariate tests.
    - **RFE_BASED**: Uses Recursive Feature Elimination (RFE) to rank features based on their contribution to the model. RFE iteratively removes the least important features and builds the model until the desired number of features is reached.

    Example
    -------
    ```python
    # Example of accessing an enum member and its value
    fi_model = FIModel.COEFFICIENT_BASED
    print(fi_model.name)  # Output: 'COEFFICIENT_BASED'
    print(fi_model.value) # Output: 'Coefficient-Based + Shrinkage & Selection'
    ```

    """
    COEFFICIENT_BASED = "Coefficient-Based + Shrinkage & Selection"
    TREE_BASED = "Tree-Based"
    PERMUTATION_BASED = "Permutation-Based"
    SINGLE_FEATURE_BASED = "Single Feature-Based"
    RFE_BASED = "Recursive Feature Elimination-Based"


class SWEFI:
    """
    A class for computing and analyzing feature importance using various models and univariate statistical methods.

    This class leverages different machine learning models to compute feature importance scores. It also allows for univariate analysis of feature importance based on statistical measures. It supports bootstrapping methods for robustness and computes Stability Weighted Feature Importance (SWEFI) scores based on feature importance data.

    Attributes
    ----------
    percentage : float, optional
        The percentage of top features to consider for SWEFI score computation. Default is None.
    clfx : ClassificationExperiment
        An instance of the ClassificationExperiment class used for model comparison and tuning.
    X : pd.DataFrame
        The feature matrix with shape (n_samples, n_features).
    y : pd.Series or np.ndarray
        The target values with shape (n_samples,).
    model_to_methods : dict
        A dictionary mapping model names to feature importance methods.
    learning_models : list
        A list of selected models for feature importance computation.
    measurements : list
        A list of univariate analysis measurements to compute.
    model_method_pairs : list
        A list of tuples containing models and their corresponding feature importance methods.
    models_fi : pd.DataFrame
        DataFrame storing feature importance values computed for each model and feature importance method.
    measurements_fi : pd.DataFrame
        DataFrame storing feature importance values computed from univariate analysis measurements.
    models_index : pd.MultiIndex
        MultiIndex for the models_fi DataFrame.
    measurements_index : pd.MultiIndex
        MultiIndex for the measurements_fi DataFrame.
    n_iterationn : int
        The number of bootstrap iterations used.
    swefi : pd.DataFrame
        DataFrame containing SWEFI scores for features.
    stability_scores : pd.Series
        Series of stability scores for each feature.

    Methods
    -------
    __init__(X, y, n_fold=10)
        Initializes the SWEFI instance and sets up the ClassificationExperiment.
    
    select_models(select_n_model=None)
        Selects models for feature importance computation based on a given criterion.
    
    select_univariate_analysis_measurements(measurements=['Pearson', 'Spearman', 'Kendal'])
        Specifies the univariate analysis measurements to use for feature importance computation.
    
    fine_tune_selected_models(hpo_n_fold=4, hpo_n_iter=25, hpo_metric='Accuracy', hpo_search_library='scikit-optimize', hpo_search_algorithm='bayesian')
        Performs hyperparameter optimization for selected models.
    
    _initialize_model_feature_importance_data(models, n_iteration=5)
        Initializes a DataFrame for storing feature importance data for different models and methods.
    
    _initialize_univariate_analysis_feature_importance_data(measurements, n_iteration=5)
        Initializes a DataFrame for storing univariate analysis feature importance data.
    
    stationary_bootstrap(X, y, n_iteration=5)
        Generates stationary bootstrap samples for the given data.
    
    iid_bootstrap(X, y, n_iteration=5)
        Generates iid bootstrap samples for the given data.
    
    compute_feature_importance_data(bootstrap_method, n_iteration=10, n_repeats=10)
        Computes feature importance data using the specified bootstrap method.
    
    compute_swefi_scores(percentage=0.5, weight='linear')
        Computes the Stability Weighted Feature Importance (SWEFI) scores for features.
    
    get_swefi_scores()
        Returns the SWEFI scores DataFrame.
    
    get_inner_models_feature_importances()
        Returns feature importances computed for models.
    """

    def __init__(self, X: pd.DataFrame, y: Union[pd.Series, np.array], n_fold:int =10):
        """
        Initializes the SWEFI class and sets up the ClassificationExperiment.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix with shape (n_samples, n_features).
        
        y : pd.Series or np.ndarray
            The target values with shape (n_samples,).
        
        n_fold : int, optional
            Number of folds for cross-validation. Default is 10.

        Notes
        -----
        - The `ClassificationExperiment` instance is set up with various parameters for model comparison and tuning.
        - Data normalization is performed using z-score normalization.
        """

        self.percentage = None
        self.clfx = ClassificationExperiment()
        self.clfx.setup(data = X, target = y, fold=n_fold, train_size=0.99, session_id = 123, n_jobs = -1, normalize=True, normalize_method='zscore')
        self.X = X
        self.y = y

    def select_models(self, select_n_model:Union[int, float]=100):
        """
        Selects models for feature importance computation based on a given criterion.

        Parameters
        ----------
        select_n_model : int or float, optional
            The number of models to select or a percentage of models to select. If float, it is interpreted as a percentage of total models to select.

        Returns
        -------
        self : SWEFI
            The updated SWEFI instance with selected models.
        
        Notes
        -----
        - Models are selected based on predefined methods for computing feature importance.
        - External models can be included or excluded based on the provided criteria.
        """

        model_to_methods = self.clfx.models()

        model_to_methods['FI Methods'] = pd.Series({
            'lr': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
            'knn': [FIModel.PERMUTATION_BASED.value, ],
            'nb': [FIModel.SINGLE_FEATURE_BASED.value],
            'dt': [FIModel.SINGLE_FEATURE_BASED.value],
            'svm': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value,], 
            'rbfsvm': [FIModel.PERMUTATION_BASED.value,],
            'gpc': [FIModel.PERMUTATION_BASED.value,],
            'mlp': [FIModel.PERMUTATION_BASED.value,],
            'ridge': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
            'rf': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value],
            'qda': [FIModel.PERMUTATION_BASED.value, ],
            'ada': [FIModel.PERMUTATION_BASED.value, ],
            'gbc': [FIModel.PERMUTATION_BASED.value, ],
            'lda': [FIModel.COEFFICIENT_BASED.value, FIModel.PERMUTATION_BASED.value, FIModel.SINGLE_FEATURE_BASED.value, FIModel.RFE_BASED.value],
            'et': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value],
            'xgboost': [FIModel.TREE_BASED.value, FIModel.RFE_BASED.value ],
            'lightgbm': [FIModel.PERMUTATION_BASED.value,],
            'dummy': [FIModel.PERMUTATION_BASED.value],
        })

        model_to_methods = model_to_methods[['Reference', 'FI Methods']].set_index('Reference').squeeze().to_dict()
        
        model_to_methods.update({
            "sklearn.svm._classes.SVC": [FIModel.COEFFICIENT_BASED.value, FIModel.RFE_BASED.value],
        })
        
        self.model_to_methods = model_to_methods

        exclude = set(['dummy', 'svm', 'knn', 'nb', 'lightgbm', 'ada', 'gpc', 'rbfsvm'])
        all_models = set(self.clfx.models().index.tolist()) 
        external_models = [SVC(probability=True, kernel='linear'), ]

        include = list(all_models - exclude) + external_models
        
        if type(select_n_model) == float:
            self.learning_models = self.clfx.compare_models(n_select=round(len(self.clfx.models()) * select_n_model),
                                                             include=include,
                                                             )
        else:
            self.learning_models = self.clfx.compare_models(n_select=select_n_model,
                                                             include=include,
                                                             )
            
        return self

    def select_univariate_analysis_measurements(self, measurements:List[str]=['Pearson', 'Spearman', 'Kendal']):
        """
        Specifies the univariate analysis measurements to use for feature importance computation.

        Parameters
        ----------
        measurements : list of str, optional
            A list of univariate analysis measurements to compute. Allowed values are 'Pearson', 'Spearman', 'Kendal'. Default is ['Pearson', 'Spearman', 'Kendal'].

        Returns
        -------
        self : SWEFI
            The updated SWEFI instance with specified univariate analysis measurements.
        """
        self.measurements = measurements

        return self

    def fine_tune_selected_models(self, hpo_n_fold:int=4, hpo_n_iter:int=25, hpo_metric:str='Accuracy', hpo_search_library:str='scikit-optimize', hpo_search_algorithm='bayesian'):
        """
        Performs hyperparameter optimization for selected models.

        Parameters
        ----------
        hpo_n_fold : int, optional
            Number of folds for cross-validation during hyperparameter optimization. Default is 4.
        
        hpo_n_iter : int, optional
            Number of iterations for hyperparameter optimization. Default is 25.
        
        hpo_metric : str, optional
            Metric to optimize during hyperparameter tuning. Default is 'Accuracy'.
        
        hpo_search_library : str, optional
            Library to use for hyperparameter optimization. Default is 'scikit-optimize'.
        
        hpo_search_algorithm : str, optional
            Search algorithm to use for hyperparameter optimization. Default is 'bayesian'.

        Returns
        -------
        self : SWEFI
            The updated SWEFI instance with fine-tuned models.
        
        Notes
        -----
        - Custom configurations for specific models can be applied during tuning.
        - Models are either tuned with default settings or custom configurations based on their type.
        """

        learning_models = self.learning_models
        model_to_methods = self.model_to_methods


        model_to_custom_config = {
            "sklearn.ensemble._forest.ExtraTreesClassifier": None,
            "sklearn.neighbors._classification.KNeighborsClassifier": None,
            "sklearn.ensemble._forest.RandomForestClassifier": None,
            "xgboost.sklearn.XGBClassifier": None,
            "lightgbm.sklearn.LGBMClassifier": {},
            "sklearn.neural_network._multilayer_perceptron.MLPClassifier": None,
            "sklearn.ensemble._gb.GradientBoostingClassifier": {},
            "sklearn.naive_bayes.GaussianNB": {},
            "sklearn.ensemble._weight_boosting.AdaBoostClassifier": {},
            "sklearn.linear_model._ridge.RidgeClassifier": {
                "alpha": [1],
                "solver": ["lsqr"],
            },
            "sklearn.discriminant_analysis.LinearDiscriminantAnalysis": {
                "shrinkage": [0.0,],
                "solver": ["lsqr"],
            },
            "sklearn.linear_model._logistic.LogisticRegression": {
                "C": [1,],
                "solver": ['liblinear'],
            },
            "sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis": {},
            "sklearn.tree._classes.DecisionTreeClassifier": None,
            "sklearn.svm._classes.SVC": {
                'probability': [True, ],
                'C': [1.0],
                'kernel': [
                    "linear",
                ],
            },
        }

        tuned_learning_models = []
        for model in learning_models:
            print("-" * 80)
            print(model.__class__.__name__)

            if model_to_custom_config[full_name(model.__class__)] is None:
                tlm, tuner = self.clfx.tune_model(
                    model,
                    return_tuner=True,
                    fold=hpo_n_fold, 
                    verbose=True, 
                    tuner_verbose=True, 
                    optimize=hpo_metric, 
                    search_library=hpo_search_library, search_algorithm=hpo_search_algorithm,
                    n_iter=hpo_n_iter,
                    choose_better=True
                )

            elif len(model_to_custom_config[full_name(model.__class__)]) == 0:
                print("No hyper-tuning ...")
                tlm = model

            else:
                print("Custom config ...")
                tlm, tuner = self.clfx.tune_model(
                    model,
                    return_tuner=True,
                    fold=hpo_n_fold, 
                    verbose=True, 
                    tuner_verbose=True, 
                    optimize=hpo_metric, 
                    search_library='scikit-learn', search_algorithm='grid',
                    custom_grid=model_to_custom_config[full_name(model.__class__)], 
                    n_iter=hpo_n_iter,
                    choose_better=True
                )


            tuned_learning_models.append(tlm)

        self.finalized_tuned_learning_models = [self.clfx.finalize_model(estimator=tlm, model_only=True,) for tlm in tuned_learning_models]

        model_method_pairs = []

        for tlm in tuned_learning_models:
            methods = model_to_methods[full_name(tlm.__class__)]
            for method in methods:
                model_method_pairs.append((tlm, method))

        self.model_method_pairs = model_method_pairs
        
        return self

    def _initialize_model_feature_importance_data(self, models:List, n_iteration:int=5):
        """
        Initializes a DataFrame for storing feature importance data for different models and methods.

        Parameters
        ----------
        models : list of models
            A list of models for which to compute feature importance.
        
        n_iteration : int, optional
            Number of bootstrap iterations. Default is 5.

        Returns
        -------
        data : pd.DataFrame
            An empty DataFrame initialized for storing feature importance data.
        
        index : pd.MultiIndex
            MultiIndex for the DataFrame.
        """

        feature_names = self.X.columns
        index_columns = ["Model Name", "Feature Importance Model", "Subset Index"]
        to_apply_product_list = [
            models,
            [i for i in range(n_iteration)],
        ]
        tuples = [(type(model).__name__, proper_fi_model, iteration) for (model, proper_fi_model), iteration in
                  itertools.product(*to_apply_product_list)]
        
        index = pd.MultiIndex.from_tuples(
            tuples,
            names=index_columns
        )

        data = pd.DataFrame({}, columns=feature_names, index=index)
        return data, index
    
    def _initialize_univariate_analysis_feature_importance_data(self, measurements:List[str], n_iteration:int=5):
        """
        Initializes a DataFrame for storing univariate analysis feature importance data.

        Parameters
        ----------
        measurements : list of str
            A list of univariate analysis measurements to compute.
        
        n_iteration : int, optional
            Number of bootstrap iterations. Default is 5.

        Returns
        -------
        data : pd.DataFrame
            An empty DataFrame initialized for storing univariate analysis feature importance data.
        
        index : pd.MultiIndex
            MultiIndex for the DataFrame.
        """
        feature_names = self.X.columns
        index_columns = ["Model Name", "Feature Importance Model", "Subset Index"]
        to_apply_product_list = [
            measurements,
            [i for i in range(n_iteration)],
        ]
        tuples = [("Univariate Analysis", ua_measurement, iteration) for (ua_measurement), iteration in
                  itertools.product(*to_apply_product_list)]
        
        index = pd.MultiIndex.from_tuples(
            tuples,
            names=index_columns
        )

        data = pd.DataFrame({}, columns=feature_names, index=index)
        return data, index

    @staticmethod
    def stationary_bootstrap(X: pd.DataFrame, y:Union[pd.Series, np.ndarray], n_iteration:int=5):
        """
        Generates stationary bootstrap samples for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix with shape (n_samples, n_features).
        
        y : pd.Series or np.ndarray
            The target values with shape (n_samples,).
        
        n_iteration : int, optional
            Number of bootstrap iterations. Default is 5.

        Yields
        ------
        X_boot : pd.DataFrame
            The feature matrix for the bootstrap sample.
        
        y_boot : pd.Series or np.ndarray
            The target values for the bootstrap sample.
        
        Notes
        -----
        - Uses the StationaryBootstrap class to generate bootstrap samples.
        """
        optimal_block_size = round(optimal_block_length(X).mean(axis=0)["stationary"])
        random_state = RandomState(1234)
        bootstraper = StationaryBootstrap(optimal_block_size, X=X, y=y, random_state=random_state)
        for bootstraped_data in bootstraper.bootstrap(n_iteration):
            yield bootstraped_data[1]['X'], bootstraped_data[1]['y']

    @staticmethod
    def iid_bootstrap(X:pd.DataFrame, y:Union[pd.Series, np.ndarray], n_iteration:int=5):
        """
        Generates iid bootstrap samples for the given data.

        Parameters
        ----------
        X : pd.DataFrame
            The feature matrix with shape (n_samples, n_features).
        
        y : pd.Series or np.ndarray
            The target values with shape (n_samples,).
        
        n_iteration : int, optional
            Number of bootstrap iterations. Default is 5.

        Yields
        ------
        X_boot : pd.DataFrame
            The feature matrix for the bootstrap sample.
        y_boot : pd.Series or np.ndarray
            The target values for the bootstrap sample.

        Notes
        -----
        - Uses the IIDBootstrap class to generate bootstrap samples.
        """
        random_state = RandomState(1234)
        bootstraper = IIDBootstrap(X=X, y=y, random_state=random_state)
        for bootstraped_data in bootstraper.bootstrap(n_iteration):
            yield bootstraped_data[1]['X'], bootstraped_data[1]['y']


    def compute_feature_importance_data(self, bootstrap_method, n_iteration:int=10, n_repeats:int=10):
        """
        Computes feature importance data using the specified bootstrap method.

        Parameters
        ----------
        bootstrap_method : callable
            The method used for bootstrapping (e.g., stationary_bootstrap or iid_bootstrap).
        
        n_iteration : int, optional
            Number of bootstrap iterations. Default is 10.
        
        n_repeats : int, optional
            Number of repeats for permutation-based methods. Default is 10.

        Returns
        -------
        self : SWEFI
            The updated SWEFI instance with computed feature importance data.
        
        Notes
        -----
        - Feature importance is computed using different methods based on the model type.
        - Results are normalized before storing.
        """
        X = self.X
        y = self.y

        X = (X-X.mean()) / X.std()

        model_method_pairs = self.model_method_pairs
        models_fi, models_index = self._initialize_model_feature_importance_data(model_method_pairs, n_iteration=n_iteration)

        measurements = self.measurements
        measurements_fi, measurements_index = self._initialize_univariate_analysis_feature_importance_data(measurements, n_iteration=n_iteration)

        for iteration, (X_train, y_train) in tqdm(enumerate(
            bootstrap_method(X, y, n_iteration)
        ), total=n_iteration):

            for model, proper_fi_model in model_method_pairs:

                if proper_fi_model == FIModel.TREE_BASED.value:
                    try:
                        result = feature_importance_MDI(model, X_train, y_train, ).values
                    except:
                        result = feature_importance_sklearn(model, X_train, y_train,).values

                elif proper_fi_model == FIModel.PERMUTATION_BASED.value:
                    result = feature_importance_MDA(model, X_train, y_train, n_repeats=n_repeats).values

                elif proper_fi_model == FIModel.COEFFICIENT_BASED.value:
                    result = feature_importance_linear_models(model, X_train, y_train,).values

                elif proper_fi_model == FIModel.SINGLE_FEATURE_BASED.value:
                    result = feature_importance_SFI(model, X_train, y_train, n_splits=n_repeats).values
                
                elif proper_fi_model == FIModel.RFE_BASED.value:
                    result = feature_importance_RFE(model, X_train, y_train,).values

                # Normalize
                result = (result - result.min()) / (result.max() - result.min()) 

                models_fi.loc[type(model).__name__, proper_fi_model, iteration] = result

            ua_result = feature_importance_univariate_analysis(measurements, X_train, y_train).T.values
            measurements_fi.loc[pd.IndexSlice["Univariate Analysis", :, iteration]] = ua_result
            
        self.models_fi = models_fi
        self.models_index = models_index

        self.measurements_fi = measurements_fi
        self.measurements_index = measurements_index

        self.n_iterationn = n_iteration

        return self

    
    def compute_swefi_scores(self, percentage:float=0.5, weight:str='linear'):
        """
        Computes the Stability Weighted Feature Importance (SWEFI) scores for features.

        Parameters
        ----------
        percentage : float, optional
            The percentage of top features to consider for SWEFI score computation. Default is 0.5.
        
        weight : str, optional
            Weighting method for SWEFI score calculation. Choices are 'linear', 'logarithmic', 'exponential', 'harmonic', 'power-2', 'power-0.5', 'entropy'. Default is 'linear'.

        Returns
        -------
        self : SWEFI
            The updated SWEFI instance with computed SWEFI scores.
        
        Notes
        -----
        - SWEFI scores are computed using different weighting methods.
        - The final SWEFI DataFrame contains both mean and standard deviation of SWEFI scores.
        """
        features = self.X.columns

        fi = pd.concat([self.models_fi, self.measurements_fi])
        index = self.models_index.tolist() + self.measurements_index.tolist()
        k = round(len(features) * percentage)
        important_features_together = []

        for idx in index:
            important_features_in_current_step = fi.loc[idx].squeeze().sort_values(ascending=False)[:k].index.tolist()
            important_features_together.extend(important_features_in_current_step)

        feature_ranked_as_k_top_important = dict(zip(features, [0] * len(features)))
        n_times_that_feature_ranked_as_important = Counter(important_features_together)
        dict.update(feature_ranked_as_k_top_important, n_times_that_feature_ranked_as_important)
        feature_ranked_as_k_top_important = pd.Series(feature_ranked_as_k_top_important)
        stability_scores = feature_ranked_as_k_top_important / (self.n_iterationn * (len(self.model_method_pairs) + len(self.measurements)))

        if weight == 'linear':
            weights = stability_scores
            weights = weights / weights.sum()
            swefi = (fi * weights)

        elif weight == 'logarithmic':
            weights = np.log(stability_scores + 1)
            weights = weights / weights.sum()
            swefi = (fi * weights)

        elif weight == 'exponential':
            weights = np.exp(stability_scores)
            weights = weights / weights.sum()
            swefi = (fi * weights) 

        elif weight == 'harmonic':
            swefi = (2 * fi * stability_scores) / (fi + stability_scores)

        elif weight == 'power-2':
            weights = stability_scores ** 2
            weights = weights / weights.sum()
            swefi = (fi * weights)

        elif weight == 'power-0.5':
            weights = (stability_scores ** 0.5)
            weights = weights / weights.sum()
            swefi = (fi * weights)

        elif weight == 'entropy':
            weights = (-stability_scores * np.log(stability_scores))
            weights = weights / weights.sum()            
            swefi = (fi * weights)

        else:
            raise NotImplementedError(f"Weight {weight} not implemented.")

        swefi = pd.concat(
            [swefi.mean() / swefi.mean().sum(), swefi.std() * (swefi.shape[0]) ** -0.5 / swefi.mean().sum()],
            axis=1,
        ).rename(columns={
            0: 'mean(SWEFI)',
            1: 'std(SWEFI)',
        }).sort_values(by='mean(SWEFI)')

        self.stability_scores = stability_scores
        self.swefi = swefi

        return self
    

    def get_swefi_scores(self):
        """
        Returns the SWEFI scores DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing SWEFI scores for features.
        """
        return self.swefi
    

    def get_inner_models_feature_importances(self):
        """
        Returns feature importances computed for models.

        Returns
        -------
        pd.DataFrame
            DataFrame containing normalized feature importances computed for each model.
        
        Notes
        -----
        - Feature importances are normalized to sum to 1 for each model.
        """
        models_fi = self.models_fi.groupby(level=[0, 1]).sum()
        models_fi = models_fi.div(models_fi.sum(axis=1).values, axis=0).transpose()
        return models_fi