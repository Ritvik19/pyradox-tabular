import pytest
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import DeepAndCrossNetworkConfig
from pyradox_tabular.nn import DeepAndCrossTabularNetwork


def test_deep_and_cross_net():
    x_train, y_train, x_valid, y_valid = pytest.get_reg_df()
    data_config = DataConfig(
        numeric_feature_names=["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"],
        categorical_features_with_vocabulary={
            "CHAS": ["0", "1"],
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = DeepAndCrossNetworkConfig(num_outputs=1, out_activation=None, hidden_units=[64, 64])
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = DeepAndCrossTabularNetwork.from_config(data_config, model_config, name="deep_cross_network")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
