import pandas as pd
from sklearn.base import BaseEstimator

# import logging


def label_encoding(ohe, X, cat_features):
    df = pd.DataFrame(X, index=[0])
    df_enc = df.copy()
    # df_enc[cat_features] = ohe.transform(df[cat_features])
    tmp = pd.DataFrame(
        ohe.transform(df[cat_features]), columns=ohe.get_feature_names_out()
    )
    df_enc = pd.concat([df.drop(columns=cat_features), tmp], axis=1)
    return df_enc


def standardization(scaler, df, num_features):
    df_scaled = df.copy()
    df_scaled[num_features] = scaler.transform(df[num_features])
    return df_scaled


def predict(regressor: BaseEstimator, X: dict) -> float:
    """Predict .

    Args:
        regressor: Trained model.
        X: Testing data of independent features.
    """
    X = pd.DataFrame(X, index=[0])
    y_pred = regressor.predict(X)

    # logger = logging.getLogger(__name__)
    # logger.info(f"INFERENCE: Y_pred={y_pred}")

    return y_pred
