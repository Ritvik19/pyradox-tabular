import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer),
        mlp_layers.append(L.Dense(units, activation=activation))
        mlp_layers.append(L.Dropout(dropout_rate))

    return keras.Sequential(mlp_layers, name=name)


class TabTransformer(NetworkInputs):
    """TabTransformer is built upon self-attention based on Transformers. The Transformer layers transform the
    embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy.

    The contextual embeddings learned from TabTransformer are highly robust against both missing and noisy data
    features, and provide better interpretability.
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
        cat_features, num_features = cls.encode_inputs(
            inputs,
            data_config,
            use_embeddings=True,
            embedding_dim=model_config.EMBEDDING_DIM,
            prefix=f"{name}_",
            concat_features=False,
        )
        cat_features = tf.stack(cat_features, axis=1)
        num_features = L.concatenate(num_features)

        if model_config.USE_COLUMN_EMBEDDING:
            num_columns = cat_features.shape[1]
            column_embedding = L.Embedding(input_dim=num_columns, output_dim=model_config.EMBEDDING_DIM)
            column_indices = tf.range(start=0, limit=num_columns, delta=1)
            cat_features = cat_features + column_embedding(column_indices)

        for block_idx in range(model_config.NUM_TRANSFORMER_BLOCKS):
            attention_output = L.MultiHeadAttention(
                num_heads=model_config.NUM_HEADS,
                key_dim=model_config.EMBEDDING_DIM,
                dropout=model_config.DROPOUT_RATE,
                name=f"multihead_attention_{block_idx}",
            )(cat_features, cat_features)
            x = L.Add(name=f"skip_connection1_{block_idx}")([attention_output, cat_features])
            x = L.LayerNormalization(name=f"layer_norm1_{block_idx}", epsilon=1e-6)(x)
            feedforward_output = create_mlp(
                hidden_units=[model_config.EMBEDDING_DIM],
                dropout_rate=model_config.DROPOUT_RATE,
                activation=keras.activations.gelu,
                normalization_layer=L.LayerNormalization(epsilon=1e-6),
                name=f"feedforward_{block_idx}",
            )(x)
            x = L.Add(name=f"skip_connection2_{block_idx}")([feedforward_output, x])
            cat_features = L.LayerNormalization(name=f"layer_norm2_{block_idx}", epsilon=1e-6)(x)

        cat_features = L.Flatten()(cat_features)
        num_features = L.LayerNormalization(epsilon=1e-6)(num_features)
        features = L.concatenate([cat_features, num_features])
        mlp_hidden_units = [factor * features.shape[-1] for factor in model_config.MLP_HIDDEN_UNITS_FACTORS]
        features = create_mlp(
            hidden_units=mlp_hidden_units,
            dropout_rate=model_config.DROPOUT_RATE,
            activation=keras.activations.selu,
            normalization_layer=L.BatchNormalization(),
            name="MLP",
        )(features)

        outputs = L.Dense(units=model_config.NUM_OUT, activation=model_config.OUT_ACTIVATION, name="outputs")(features)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
