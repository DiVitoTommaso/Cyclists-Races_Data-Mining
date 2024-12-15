import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
