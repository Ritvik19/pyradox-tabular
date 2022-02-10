import pytest
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import TabNetConfig
from pyradox_tabular.nn import TabNet


def test_mixed_data_types():
    x_train, y_train, x_valid, y_valid = pytest.get_reg_df()
    data_config = DataConfig(
        numeric_feature_names=["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"],
        categorical_features_with_vocabulary={
            "CHAS": ["0", "1"],
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = TabNetConfig(num_outputs=1, out_activation=None)
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = TabNet.from_config(data_config, model_config, name="mixed_data_types")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)


def test_all_num_type():
    x_train, y_train, x_valid, y_valid = pytest.get_num_df()
    data_config = DataConfig(
        numeric_feature_names=x_train.columns.tolist(),
        categorical_features_with_vocabulary={},
    )
    model_config = TabNetConfig(num_outputs=1, out_activation="sigmoid")
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = TabNet.from_config(data_config, model_config, name="all_num_type")
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
    model_config = TabNetConfig(num_outputs=1, out_activation="sigmoid")
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = TabNet.from_config(data_config, model_config, name="all_cat_type")
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(data_train, validation_data=data_valid)
