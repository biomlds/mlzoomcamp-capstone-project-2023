from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def make_X_y(df, target):
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def make_scaler(X_train, num_features):
    scaler = StandardScaler()
    scaler.fit(X_train[num_features])
    return scaler


def make_label_encoder(df, cat_features):
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(df[cat_features])
    return ohe


def standardization(scaler, df, num_features):
    df_scaled = df.copy()
    df_scaled[num_features] = scaler.transform(df[num_features])
    return df_scaled


def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def label_encoding(ohe, df, cat_features):
    df_enc = df.copy()
    # df_enc[cat_features] = ohe.transform(df[cat_features])
    tmp = pd.DataFrame(
        ohe.transform(df[cat_features]), columns=ohe.get_feature_names_out()
    )
    df_enc = pd.concat([df.drop(columns=cat_features), tmp], axis=1)
    return df_enc


def print_text(s):
    print(s)
    return {"string": s}
