import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnRemoverTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that removes specified columns from a DataFrame.

    :param columns: List of column names to remove from the DataFrame
    """

    def __init__(self, columns):
        # Store the list of columns to be removed
        self.columns = columns

    def fit(self, X, y=None):
        """
        The fit method is a no-op for this transformer. It is required by scikit-learn,
        but doesn't need to do anything since the transformer just removes columns.

        :param X: Input data (ignored in this transformer)
        :param y: Target data (ignored in this transformer)
        :return: self (the transformer object)
        """
        return self

    def transform(self, X):
        """
        Removes the specified columns from the DataFrame.

        :param X: DataFrame with columns to be removed
        :return: Transformed DataFrame with specified columns removed
        """
        # Assert that the input X is a pandas DataFrame
        assert isinstance(X, pd.DataFrame)

        # Drop the specified columns from the DataFrame
        X = X.drop(columns=self.columns)

        return X


class BMICalculatorTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that calculates the BMI (Body Mass Index) and adds it as a new column.
    The formula for BMI is: BMI = weight / (height in meters)^2.
    """

    def fit(self, X, y=None):
        """
        The fit method is a no-op for this transformer, as no fitting is required for BMI calculation.

        :param X: Input data (ignored in this transformer)
        :param y: Target data (ignored in this transformer)
        :return: self (the transformer object)
        """
        return self

    def transform(self, X):
        """
        Adds a new column 'BMI' to the DataFrame based on weight and height.

        :param X: DataFrame with 'weight' (kg) and 'height' (cm) columns
        :return: Transformed DataFrame with an added 'BMI' column
        """
        # Assert that the input X is a pandas DataFrame
        assert isinstance(X, pd.DataFrame)

        # Calculate BMI and add it as a new column to the DataFrame
        X["BMI"] = X["weight"] / (X["height"] / 100) ** 2

        return X
