import pytest
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import WideAndDeepNetworkConfig
from pyradox_tabular.nn import WideAndDeepTabularNetwork
from tensorflow.data import Dataset


def test_wide_and_deep_net():
    x_train, y_train, x_valid, y_valid = pytest.get_reg_df()
    data_config = DataConfig(
        numeric_feature_names=["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"],
        categorical_features_with_vocabulary={
            "CHAS": ["0", "1"],
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = WideAndDeepNetworkConfig(num_outputs=1, out_activation=None, hidden_units=[64, 64])
    data_train = Dataset.from_tensor_slices(
        ({col: x_train[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_train.values.tolist())
    ).batch(1024)
    data_valid = Dataset.from_tensor_slices(
        ({col: x_valid[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_valid.values.tolist())
    ).batch(1024)

    model = WideAndDeepTabularNetwork.from_config(data_config, model_config, name="wide_deep_network")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
