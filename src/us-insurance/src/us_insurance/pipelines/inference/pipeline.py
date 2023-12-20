"""
This is a boilerplate pipeline 'inference'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *

# from src.us_insurance.pipelines.data_processing.nodes import (
#     label_encoding,
#     standardization,
# )
# from ..data_processing.nodes import (
#     label_encoding,
#     standardization,
# )


# from src.us_insurance.pipelines.data_science.nodes import
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=label_encoding,
                inputs=[
                    "label_encoder",
                    "params:rf_regressor_prediction",
                    "params:cat_features",
                ],
                outputs="inference_enc",
                name="inference_label_encoding",
            ),
            node(
                func=standardization,
                inputs=["scaler", "inference_enc", "params:num_features"],
                outputs="inference_inference_std",
                name="inference_scale",
            ),
            node(
                func=predict,
                inputs=["rf_regressor", "inference_inference_std"],
                outputs="inference_prediction",
                name="inference_prediction",
            ),
        ]
    )
