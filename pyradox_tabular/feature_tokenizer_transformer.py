import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


class TransformerBlock(L.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = L.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = tf.keras.Sequential([L.Dense(dense_dim, activation="relu"), L.Dense(embed_dim)])
        self.layernorm1 = L.LayerNormalization()
        self.layernorm2 = L.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[: tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm2(proj_input + proj_output)


class FeatureTokenizerTransformer(NetworkInputs):
    """It is a simple adaptation of the Transformer architecture for the tabular domain. In a nutshell,
    Feature Tokenizer Transformer transforms all features (categorical and numerical) to embeddings and
    applies a stack of Transformer layers to the embeddings.
    Thus, every Transformer layer operates on the feature level of one object.
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
            prefix="",
            concat_features=False,
        )
        num_feat_emb = [
            L.Dense(model_config.EMBEDDING_DIM, name=f"{feature_name}_embeddings")
            for _, feature_name in zip(range(len(num_features)), data_config.NUMERIC_FEATURE_NAMES)
        ]
        num_features = [emb(feat) for emb, feat in zip(num_feat_emb, num_features)]

        features = L.Concatenate(axis=1, name="feature_embeddings_stack")(
            [
                L.Reshape((1, 32), name=f"{feat_name}_reshape_2")(feat)
                for feat, feat_name in zip((num_features + cat_features), data_config.FEATURE_NAMES)
            ]
        )

        for _ in range(model_config.NUM_TRANSFORMER_BLOCKS):
            features = TransformerBlock(
                embed_dim=model_config.EMBEDDING_DIM,
                dense_dim=model_config.DENSE_DIM,
                num_heads=model_config.NUM_HEADS,
            )(features)
        features = L.GlobalMaxPooling1D()(features)
        outputs = L.Dense(
            units=model_config.NUM_OUT,
            activation=model_config.OUT_ACTIVATION,
            name="outputs",
        )(features)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
