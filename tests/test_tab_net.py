import pytest
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import TabNetConfig
from pyradox_tabular.nn import TabNet
from tensorflow.data import Dataset


def test_tab_net():
    x_train, y_train, x_valid, y_valid = pytest.get_reg_df()
    data_config = DataConfig(
        numeric_feature_names=["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"],
        categorical_features_with_vocabulary={
            "CHAS": ["0", "1"],
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = TabNetConfig(num_outputs=1, out_activation="sigmoid")
    data_train = Dataset.from_tensor_slices(
        ({col: x_train[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_train.values.tolist())
    ).batch(1024)
    data_valid = Dataset.from_tensor_slices(
        ({col: x_valid[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_valid.values.tolist())
    ).batch(1024)

    model = TabNet.from_config(data_config, model_config, name="tabnet")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
