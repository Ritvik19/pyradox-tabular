import pytest
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import NeuralDecisionTreeConfig
from pyradox_tabular.nn import NeuralDecisionTree


def test_mixed_data_types():
    x_train, y_train, x_valid, y_valid = pytest.get_clf_df()
    data_config = DataConfig(
        numeric_feature_names=[
            "CRIM",
            "ZN",
            "INDUS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT",
            "MEDV",
        ],
        categorical_features_with_vocabulary={
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = NeuralDecisionTreeConfig(depth=2, used_features_rate=0.8, num_classes=2)
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = NeuralDecisionTree.from_config(data_config, model_config, name="mixed_data_types")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)


def test_all_num_type():
    x_train, y_train, x_valid, y_valid = pytest.get_num_df()
    data_config = DataConfig(
        numeric_feature_names=x_train.columns.tolist(),
        categorical_features_with_vocabulary={},
    )
    model_config = NeuralDecisionTreeConfig(depth=2, used_features_rate=0.8, num_classes=2)
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = NeuralDecisionTree.from_config(data_config, model_config, name="all_num_type")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(data_train, validation_data=data_valid)


def test_all_cat_type():
    x_train, y_train, x_valid, y_valid = pytest.get_cat_df()
    data_config = DataConfig(
        numeric_feature_names=[],
        categorical_features_with_vocabulary={
            col: list(sorted(x_train[col].unique().tolist())) for col in x_train.columns
        },
    )
    model_config = NeuralDecisionTreeConfig(depth=2, used_features_rate=0.8, num_classes=2)
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = NeuralDecisionTree.from_config(data_config, model_config, name="all_cat_type")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(data_train, validation_data=data_valid)
