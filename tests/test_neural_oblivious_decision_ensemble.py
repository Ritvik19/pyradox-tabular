import pytest
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import NeuralObliviousDecisionEnsembleConfig
from pyradox_tabular.nn import NeuralObliviousDecisionEnsemble


def test_neural_oblivious_decision_ensemble():
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
    model_config = NeuralObliviousDecisionEnsembleConfig()
    data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
    data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)

    model = NeuralObliviousDecisionEnsemble.from_config(
        data_config, model_config, name="neural_oblivious_decision_ensemble"
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
