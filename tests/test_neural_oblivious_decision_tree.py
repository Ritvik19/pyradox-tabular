import pytest
from pyradox_tabular.data_config import DataConfig
from pyradox_tabular.model_config import NeuralObliviousDecisionTreeConfig
from pyradox_tabular.nn import NeuralObliviousDecisionTree
from tensorflow.data import Dataset


def test_neural_oblivious_decision_tree():
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
    model_config = NeuralObliviousDecisionTreeConfig()
    data_train = Dataset.from_tensor_slices(
        ({col: x_train[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_train.values.tolist())
    ).batch(1024)
    data_valid = Dataset.from_tensor_slices(
        ({col: x_valid[col].values.tolist() for col in data_config.FEATURE_NAMES}, y_valid.values.tolist())
    ).batch(1024)

    model = NeuralObliviousDecisionTree.from_config(data_config, model_config, name="neural_oblivious_decision_tree")
    model.compile(optimizer="adam", loss="mse")
    model.fit(data_train, validation_data=data_valid)
