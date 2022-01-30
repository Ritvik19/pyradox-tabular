import tensorflow as tf
from tensorflow.keras import layers as L


class NetworkInputs:
    @staticmethod
    def get_inputs(config):
        return {
            feature_name: L.Input(
                name=feature_name,
                shape=(),
                dtype=(tf.float32 if feature_name in config.NUMERIC_FEATURE_NAMES else tf.string),
            )
            for feature_name in config.FEATURE_NAMES
        }

    @staticmethod
    def encode_inputs(inputs, config, use_embeddings=False, embedding_dim=32, prefix="", concat_features=False):
        cat_features = []
        num_features = []
        for feature_name in inputs:
            if feature_name in config.CATEGORICAL_FEATURE_NAMES:
                vocabulary = config.CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
                lookup = L.StringLookup(
                    vocabulary=vocabulary,
                    mask_token=None,
                    num_oov_indices=0,
                    output_mode="int" if use_embeddings else "binary",
                    name=f"{prefix}{feature_name}_lookup",
                )
                if use_embeddings:
                    encoded_feature = lookup(inputs[feature_name])
                    embedding = L.Embedding(
                        input_dim=len(vocabulary),
                        output_dim=embedding_dim,
                        name=f"{prefix}{feature_name}_embeddings",
                    )
                    encoded_feature = embedding(encoded_feature)
                else:
                    encoded_feature = lookup(
                        L.Reshape((1,), name=f"{prefix}{feature_name}_reshape")(inputs[feature_name])
                    )
                cat_features.append(encoded_feature)
            else:
                encoded_feature = L.Reshape((1,), name=f"{prefix}{feature_name}_reshape")(inputs[feature_name])
                num_features.append(encoded_feature)

        features = (
            L.Concatenate(name=f"{prefix}inputs_concatenate")(cat_features + num_features)
            if concat_features
            else (cat_features, num_features)
        )

        return features
