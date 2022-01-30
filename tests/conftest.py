import sys

import pandas as pd
import tensorflow as tf


def get_boston_df():
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.boston_housing.load_data()
    x_train = pd.DataFrame(x_train, columns=features)
    x_valid = pd.DataFrame(x_valid, columns=features)
    for col in ["CHAS", "RAD"]:
        x_train[col] = x_train[col].astype("int").astype("str")
        x_valid[col] = x_valid[col].astype("int").astype("str")
    y_train = pd.Series(y_train, name="MEDV")
    y_valid = pd.Series(y_valid, name="MEDV")
    return x_train, y_train, x_valid, y_valid


sys.modules["pytest"].get_boston_df = get_boston_df
