import pytest
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import TabTransformerConfig
from pyradox_tabular.nn import TabTransformer


def test_tab_transformer():
    x_train, y_train, x_valid, y_valid = pytest.get_reg_df()
    data_config = DataConfig(
        numeric_feature_names=["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"],
        categorical_features_with_vocabulary={
            "CHAS": ["0", "1"],
            "RAD": ["1", "2", "3", "4", "5", "6", "7", "8", "24"],
        },
    )
    model_config = TabTransformerConfig(
        num_outputs=1, out_activation=None, num_transformer_blocks=3, num_heads=4, mlp_hidden_units_factors=[2, 1]
    )
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = TabTransformer.from_config(data_config, model_config, name="tab_transformer")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
