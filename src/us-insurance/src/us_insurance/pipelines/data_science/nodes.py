import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from math import sqrt
from sklearn.base import BaseEstimator
import numpy as np


def train_lr(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def optimize_hyperparameters(model_name, param_grid, X_train, y_train, cv, cv_scoring):
    model = globals()[model_name]()

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        model,
        param_grid,
        scoring=cv_scoring,
        n_jobs=-1,
        refit=True,
        cv=cv,
    )
    grid_search.fit(X_train, np.ravel(y_train))

    # Get the best hyperparameters and model
    best_params = grid_search.best_params_
    # best_params = len(grid_search.cv_results_)
    best_model = grid_search.best_estimator_

    return best_params, best_model


def train_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = Ridge()
    regressor.fit(X_train, y_train)
    return regressor


def train_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Trains the RandomForest regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(regressor: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the model metrics.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", r2)
    logger.info("Model has RMSE of %.3f on test data.", rmse)
    logger.info("Model has MAE of %.3f on test data.", mae)
    return {"r2_score": r2, "mae": mae, "rmse_score": rmse}
