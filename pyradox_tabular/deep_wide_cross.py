from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


class DeepTabularNetwork(NetworkInputs):
    """In principle a neural network can approximate any continuous function and piece wise continuous function.
    However, it is not suitable to approximate arbitrary non-continuous functions as it assumes certain level of
    continuity in its general form.

    Unlike unstructured data found in nature, structured data with categorical features may not have continuity
    at all and even if it has it may not be so obvious.

    Deep Tabular Network use the entity embedding method to automatically learn the representation of categorical
    features in multi-dimensional spaces which reveals the intrinsic continuity of the data and helps neural
    networks to solve the problem.
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
    """The human brain is a sophisticated learning machine, forming rules by memorizing everyday events and
    generalizing those learnings to apply tothings we haven't seen before. Perhaps more powerfully, memorization
    also allows us to further refine our generalized rules with exceptions.

    By jointly training a wide linear model (for memorization) alongside a deep neural network (for generalization)
    Wide and Deep Tabular Networks combine the strengths of both to bring us one step closer to teach computers to
    learn like humans do.
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
    """Feature engineering has been the key to the success of many prediction models. However, the process is
    nontrivial and often requires manual feature engineering or exhaustive searching. DNNs are able to
    automatically learn feature interactions; however, they generate all the interactions implicitly, and are not
    necessarily efficient in learning all types of cross features.

    Deep and Cross Tabular Network explicitly applies feature crossing at each layer, requires no manual feature
    engineering, and adds negligible extra complexity to the DNN model.
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
