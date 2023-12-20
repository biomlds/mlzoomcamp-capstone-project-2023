"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=make_scaler,
                inputs=["X_train_raw", "params:num_features"],
                outputs="scaler",
                name="make_scaler",
            ),
            node(
                func=make_label_encoder,
                inputs=["insurance", "params:cat_features"],
                outputs="label_encoder",
                name="make_label_encoder",
            ),
            node(
                func=label_encoding,
                # inputs=["label_encoder", "insurance", "params:cat_features"],
                inputs=["label_encoder", "insurance", "params:cat_features"],
                outputs="insurance_enc",
                name="label_encoding",
            ),
            node(
                func=make_X_y,
                inputs=["insurance_enc", "params:target"],
                outputs=["X", "y"],
                name="make_X_y",
            ),
            node(
                func=split_data,
                inputs=["X", "y", "params:test_size", "params:random_state"],
                outputs=["X_train_raw", "X_test_raw", "y_train", "y_test"],
                name="split_data",
            ),
            node(
                func=standardization,
                inputs=["scaler", "X_test_raw", "params:num_features"],
                outputs="X_test",
                name="scale_X_test",
            ),
            node(
                func=standardization,
                inputs=["scaler", "X_train_raw", "params:num_features"],
                outputs="X_train",
                name="scale_X_train",
            ),
        ]
    )
