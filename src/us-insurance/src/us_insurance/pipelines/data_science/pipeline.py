from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=optimize_hyperparameters,
                inputs=[
                    "params:linear_regression",
                    "params:linreg_param_grid",
                    "X_test",
                    "y_test",
                    "params:cv",
                    "params:cv_scoring",
                ],
                outputs=["lin_regressor_best_params", "lin_regressor"],
                name="optimize_lin_regressor",
            ),
            node(
                func=optimize_hyperparameters,
                inputs=[
                    "params:ridge_regression",
                    "params:ridge_regression_param_grid",
                    "X_test",
                    "y_test",
                    "params:cv",
                    "params:cv_scoring",
                ],
                outputs=["ridge_regressor_best_params", "ridge_regressor"],
                name="optimize_ridge_regressor",
            ),
            node(
                func=optimize_hyperparameters,
                inputs=[
                    "params:rf_regressor",
                    "params:rf_regressor_param_grid",
                    "X_test",
                    "y_test",
                    "params:cv",
                    "params:cv_scoring",
                ],
                outputs=["rf_regressor_best_params", "rf_regressor"],
                name="optimize_rf_regressor",
            ),
            node(
                func=evaluate_model,
                inputs=["lin_regressor", "X_test", "y_test"],
                outputs="lr_metrics",
                name="evaluate_lr",
            ),
            node(
                func=evaluate_model,
                inputs=["rf_regressor", "X_test", "y_test"],
                outputs="rf_metrics",
                name="evaluate_rf",
            ),
            node(
                func=evaluate_model,
                inputs=["ridge_regressor", "X_test", "y_test"],
                outputs="ridge_metrics",
                name="evaluate_ridge",
            ),
        ]
    )
