import sys

import pandas as pd
import tensorflow as tf


def get_reg_df():
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


def get_clf_df():
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.boston_housing.load_data()
    x_train = pd.DataFrame(x_train, columns=features)
    x_valid = pd.DataFrame(x_valid, columns=features)
    x_train["MEDV"] = y_train
    x_valid["MEDV"] = y_valid
    x_train["RAD"] = x_train["RAD"].astype("int").astype("str")
    x_valid["RAD"] = x_valid["RAD"].astype("int").astype("str")
    y_train = x_train["CHAS"]
    y_valid = x_valid["CHAS"]
    x_train = x_train.drop(columns=["CHAS"])
    x_valid = x_valid.drop(columns=["CHAS"])
    return x_train, y_train, x_valid, y_valid


def get_num_df():
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.boston_housing.load_data()
    x_train = pd.DataFrame(x_train, columns=features)
    x_valid = pd.DataFrame(x_valid, columns=features)
    x_train["MEDV"] = y_train
    x_valid["MEDV"] = y_valid
    y_train = x_train["CHAS"]
    y_valid = x_valid["CHAS"]
    x_train = x_train.drop(columns=["CHAS"])
    x_valid = x_valid.drop(columns=["CHAS"])
    x_train = x_train.drop(columns=["RAD"])
    x_valid = x_valid.drop(columns=["RAD"])
    return x_train, y_train, x_valid, y_valid


def get_cat_df():
    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
    (x_train, y_train), (x_valid, y_valid) = tf.keras.datasets.boston_housing.load_data()
    x_train = pd.DataFrame(x_train, columns=features)
    x_valid = pd.DataFrame(x_valid, columns=features)
    x_train["MEDV"] = y_train
    x_valid["MEDV"] = y_valid
    x_train["RAD"] = x_train["RAD"].astype("int").astype("str")
    x_valid["RAD"] = x_valid["RAD"].astype("int").astype("str")
    y_train = x_train["CHAS"]
    y_valid = x_valid["CHAS"]
    x_train = x_train.drop(columns=["CHAS"])
    x_valid = x_valid.drop(columns=["CHAS"])
    for col in x_train.columns.drop("RAD").tolist():
        x_train[col] = pd.cut(x_train[col], bins=10, labels=[str(i) for i in range(10)])
        x_valid[col] = pd.cut(x_valid[col], bins=10, labels=[str(i) for i in range(10)])
    return x_train, y_train, x_valid, y_valid


sys.modules["pytest"].get_reg_df = get_reg_df
sys.modules["pytest"].get_clf_df = get_clf_df
sys.modules["pytest"].get_num_df = get_num_df
sys.modules["pytest"].get_cat_df = get_cat_df
