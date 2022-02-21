from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


def skip_connection(inputs, units, i, dropout_rate):
    inputs = L.Dense(units, name=f"block_{i+1}_1_dense")(inputs)
    inputs = L.BatchNormalization(name=f"block_{i+1}_1_b_norm")(inputs)
    inputs = L.ReLU(name=f"block_{i+1}_1_relu")(inputs)
    inputs = L.Dropout(dropout_rate, name=f"block_{i+1}_1_dropout")(inputs)
    x = L.Dense(units, name=f"block_{i+1}_2_dense")(inputs)
    x = L.BatchNormalization(name=f"block_{i+1}_2_b_norm")(x)
    x = L.ReLU(name=f"block_{i+1}_2_relu")(x)
    x = L.Dropout(dropout_rate, name=f"block_{i+1}_2_dropout")(x)
    return L.Add(name=f"skip_{i+1}")([inputs, x])


class TabularResNet(NetworkInputs):
    """Tabular Resnet is a ResNet like architecture containing skip connection but instead of Convolutional
    Layers, it consists of Linear Layers.
    """

    @classmethod
    def from_config(cls, data_config, model_config, name):
        """Create a network from configurations.

        Args:
            data_config (DataConfig): configurations for data processing.
            model_config (ModelConfig): configurations for the network.
            name (str): name of the model

        Returns:
            keras.Model: desired neural network.
        """
        inputs = cls.get_inputs(data_config)
        features = cls.encode_inputs(
            inputs,
            data_config,
            use_embeddings=model_config.USE_EMBEDDINGS,
            embedding_dim=model_config.EMBEDDING_DIM,
            prefix=f"{name}_",
            concat_features=True,
        )
        for i, units in enumerate(model_config.HIDDEN_UNITS):
            features = skip_connection(features, units, i, model_config.DROPOUT_RATE)

        outputs = L.Dense(
            units=model_config.NUM_OUT,
            activation=model_config.OUT_ACTIVATION,
            name="outputs",
        )(features)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
