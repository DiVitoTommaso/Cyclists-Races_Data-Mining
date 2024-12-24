import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np


class ColumnRemoverTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X = X.drop(columns=self.columns)
        return X


class BMICalculatorTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        X["BMI"] = X["weight"] / (X["height"] / 100) ** 2
        return X


def get_encoder(onehot_columns):
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(), onehot_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def get_general_preprocessor(columns, imputer):
    return ColumnTransformer(
        transformers=[
            (
                "general",
                imputer,
                columns,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def get_scaler_encoder(columns_to_scale, columns_to_encode):
    return ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                columns_to_scale,
            ),
            ("onehot", OneHotEncoder(), columns_to_encode),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def get_preprocessor_svm(
    general_imputer, profile_imputer, general_columns, profile_columns
):
    return ColumnTransformer(
        transformers=[
            (
                "general",
                general_imputer,
                general_columns,
            ),
            ("profile", profile_imputer, profile_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
