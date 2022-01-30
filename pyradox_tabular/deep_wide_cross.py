from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


class DeepTabularNetwork(NetworkInputs):
    @classmethod
    def from_config(cls, data_config, model_config, name):
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
            features = L.Dense(units, name=f"block_{i+1}_dense")(features)
            features = L.BatchNormalization(name=f"block_{i+1}_b_norm")(features)
            features = L.ReLU(name=f"block_{i+1}_relu")(features)
            features = L.Dropout(model_config.DROPOUT_RATE, name=f"block_{i+1}_dropout")(features)

        outputs = L.Dense(
            units=model_config.NUM_OUT,
            activation=model_config.OUT_ACTIVATION,
            name="outputs",
        )(features)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model


class WideAndDeepTabularNetwork(NetworkInputs):
    @classmethod
    def from_config(cls, data_config, model_config, name):
        inputs = cls.get_inputs(data_config)
        wide = cls.encode_inputs(
            inputs,
            data_config,
            use_embeddings=False,
            embedding_dim=model_config.EMBEDDING_DIM,
            prefix="wide_",
            concat_features=True,
        )
        deep = cls.encode_inputs(
            inputs,
            data_config,
            use_embeddings=True,
            embedding_dim=model_config.EMBEDDING_DIM,
            prefix="deep_",
            concat_features=True,
        )

        for i, units in enumerate(model_config.HIDDEN_UNITS):
            deep = L.Dense(units, name=f"block_{i+1}_dense")(deep)
            deep = L.BatchNormalization(name=f"block_{i+1}_b_norm")(deep)
            deep = L.ReLU(name=f"block_{i+1}_relu")(deep)
            deep = L.Dropout(model_config.DROPOUT_RATE, name=f"block_{i+1}_dropout")(deep)

        merged = L.Concatenate(name="network_concatenate")([wide, deep])
        outputs = L.Dense(units=model_config.NUM_OUT, activation=model_config.OUT_ACTIVATION, name="outputs")(merged)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model


class DeepAndCrossTabularNetwork(NetworkInputs):
    @classmethod
    def from_config(cls, data_config, model_config, name):
        inputs = cls.get_inputs(data_config)
        x0 = cls.encode_inputs(
            inputs,
            data_config,
            use_embeddings=model_config.USE_EMBEDDINGS,
            embedding_dim=model_config.EMBEDDING_DIM,
            prefix=f"{name}_",
            concat_features=True,
        )

        cross = x0
        for i in range(model_config.NUM_CROSS_LAYERS):
            units = cross.shape[-1]
            x = L.Dense(units, name=f"cross_{i+1}_dense")(cross)
            cross = L.Lambda(lambda x: x[0] * x[1] + x[2], name=f"cross_{i+1}")((x0, x, cross))
        cross = L.BatchNormalization(name="cross_b_norm")(cross)

        deep = x0
        for i, units in enumerate(model_config.HIDDEN_UNITS):
            deep = L.Dense(units, name=f"block_{i+1}_dense")(deep)
            deep = L.BatchNormalization(name=f"block_{i+1}_b_norm")(deep)
            deep = L.ReLU(name=f"block_{i+1}_relu")(deep)
            deep = L.Dropout(model_config.DROPOUT_RATE, name=f"block_{i+1}_dropout")(deep)

        merged = L.Concatenate(name="network_concatenate")([cross, deep])
        outputs = L.Dense(units=model_config.NUM_OUT, activation=model_config.OUT_ACTIVATION, name="outputs")(merged)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
