import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from .base import NetworkInputs


class NeuralDecisionTreeBackbone(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(np.arange(num_features), num_used_features, replace=False)
        self.used_features_mask = one_hot[sampled_feature_indicies]

        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(shape=[self.num_leaves, self.num_classes]),
            dtype="float32",
            trainable=True,
        )

        self.decision_fn = L.Dense(units=self.num_leaves, activation="sigmoid", name="decision")

    def call(self, features):
        batch_size = tf.shape(features)[0]

        features = tf.matmul(features, self.used_features_mask, transpose_b=True)
        decisions = tf.expand_dims(self.decision_fn(features), axis=2)
        decisions = L.concatenate([decisions, 1 - decisions], axis=2)

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2

        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])
            mu = tf.tile(mu, (1, 1, 2))
            level_decisions = decisions[:, begin_idx:end_idx, :]
            mu = mu * level_decisions
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])
        probabilities = keras.activations.softmax(self.pi)
        outputs = tf.matmul(mu, probabilities)
        return outputs


class NeuralDecisionForestBackbone(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.ensemble = []
        self.num_classes = num_classes
        for _ in range(num_trees):
            self.ensemble.append(NeuralDecisionTreeBackbone(depth, num_features, used_features_rate, num_classes))

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, self.num_classes])

        for tree in self.ensemble:
            outputs += tree(inputs)
        outputs /= len(self.ensemble)
        return outputs


class NeuralDecisionTree(NetworkInputs):
    """Deep Neural Decision Trees unifies classification trees with the representation learning functionality
    known from deep convolutional network. These are essentially a stochastic and differentiable decision tree
    model.
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
            use_embeddings=False,
            embedding_dim=0,
            prefix=f"{name}_",
            concat_features=True,
        )
        backbone = NeuralDecisionTreeBackbone(
            num_features=features.shape[-1],
            depth=model_config.DEPTH,
            used_features_rate=model_config.USED_FEATURES_RATE,
            num_classes=model_config.NUM_CLASSES,
        )
        outputs = backbone(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


class NeuralDecisionForest(NetworkInputs):
    """A Deep Neural Decision Forest is an bagging ensemble of Deep Neural Decision Trees."""

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
            use_embeddings=False,
            embedding_dim=0,
            prefix=f"{name}_",
            concat_features=True,
        )
        backbone = NeuralDecisionForestBackbone(
            num_trees=model_config.NUM_TREES,
            num_features=features.shape[-1],
            depth=model_config.DEPTH,
            used_features_rate=model_config.USED_FEATURES_RATE,
            num_classes=model_config.NUM_CLASSES,
        )
        outputs = backbone(features)
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
