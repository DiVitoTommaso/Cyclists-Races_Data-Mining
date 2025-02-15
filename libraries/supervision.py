import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "torch"
import keras


class ColumnRemoverTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that removes specified columns from a DataFrame.
    """

    def __init__(self, columns):
        # Store the list of columns to be removed
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Assert that the input X is a pandas DataFrame
        assert isinstance(X, pd.DataFrame)

        # Drop the specified columns from the DataFrame
        X = X.drop(columns=self.columns)
        return X


class BMICalculatorTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that calculates the BMI and adds it as a new column.
    The formula for BMI is: BMI = weight / (height in meters)^2.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # Assert that the input X is a pandas DataFrame
        assert isinstance(X, pd.DataFrame)

        # Calculate BMI and add it as a new column to the DataFrame
        X["BMI"] = X["weight"] / (X["height"] / 100) ** 2

        return X


def get_encoder(onehot_columns):
    """
    Returns a ColumnTransformer object that one-hot encodes the specified columns.
    Used by the RandomForestClassifier.
    """
    return ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_columns),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def get_general_preprocessor(columns, imputer):
    """
    Returns a ColumnTransformer object that imputes missing values in the specified columns.
    Used only for numerical columns. Used by RandomForest and XGB.
    """
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
    """
    Returns a ColumnTransformer object that scales and one-hot encodes the specified columns.
    Used by SVM and NN.
    """
    return ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                columns_to_scale,
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), columns_to_encode),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def get_preprocessor_svm(
    general_imputer, profile_imputer, general_columns, profile_columns
):
    """
    Returns a ColumnTransformer object that imputes missing values in the specified columns.
    Used by SVM and NN.
    """
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


def create_model(input_shape):
    """returns untrained keras model. Used in grid search of NN"""
    model = keras.models.Sequential(
        [
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(32, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model


class SklearnKerasClassifier(BaseEstimator):
    """
    Wrapper for Keras model to be used in sklearn pipelines.
    The scikeras wrapper has a bug, and cannot be used to save model after training.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.model.predict(X, verbose=0).flatten() > 0.5).astype(int)

    def predict_proba(self, X):
        predictions = self.model.predict(X, verbose=0)
        return np.hstack([1 - predictions, predictions])

    def score(self, X, y):
        return self.model.evaluate(X, y)
