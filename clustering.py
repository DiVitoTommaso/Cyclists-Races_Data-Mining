from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np


def transform(df, categorical_cols=[], standardize_cols=[], minmax_cols=[], samples=0):
    if samples != 0:
        df = df.sample(n=samples, random_state=1804)

    # Preprocess the data
    preprocessor = ColumnTransformer(
        transformers=[
            ("standardize", StandardScaler(), standardize_cols),
            ("minmax", MinMaxScaler(), minmax_cols),
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
        ]
    )
    transformed_data = preprocessor.fit_transform(df)
    return transformed_data, preprocessor


def inverse_transformation(
    transformed_data,
    preprocessor,
    categorical_cols=[],
    standardize_cols=[],
    minmax_cols=[],
):
    # Extract transformed numerical and categorical parts
    standardize_transformer = preprocessor.named_transformers_["standardize"]
    minmax_transformer = preprocessor.named_transformers_["minmax"]
    cat_transformer = preprocessor.named_transformers_["cat"]

    # Separate numerical and categorical data
    standardize_data = transformed_data[:, : len(standardize_cols)]
    minmax_data = transformed_data[
        :, len(standardize_cols) : len(standardize_cols) + len(minmax_cols)
    ]
    cat_data = transformed_data[:, len(standardize_cols) + len(minmax_cols) :]

    # Inverse the transformations
    inverse_standardize_data = standardize_transformer.inverse_transform(
        standardize_data
    )
    inverse_minmax_data = minmax_transformer.inverse_transform(minmax_data)
    inverse_cat_data = cat_transformer.inverse_transform(cat_data)

    # Recombine columns
    return pd.DataFrame(
        np.hstack((inverse_standardize_data, inverse_minmax_data, inverse_cat_data)),
        columns=standardize_cols + minmax_cols + categorical_cols,
    )
