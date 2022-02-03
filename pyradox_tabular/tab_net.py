import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from tensorflow_addons.activations import sparsemax

from .base import NetworkInputs


def register_keras_custom_object(cls):
    keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


def glu(x, n_units=None):
    if n_units is None:
        n_units = tf.shape(x)[-1] // 2

    return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])


@register_keras_custom_object
@tf.function
def sparsemax(logits, axis):
    logits = tf.convert_to_tensor(logits, name="logits")

    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat([tf.range(dim_index), [last_index], tf.range(dim_index + 1, last_index), [dim_index]], 0),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]
    z = tf.reshape(logits, [obs, dims])
    z_sorted, _ = tf.nn.top_k(z, k=dims)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe


class TransformBlock(keras.Model):
    def __init__(self, features, momentum=0.9, virtual_batch_size=None, block_name="", **kwargs):
        super().__init__(**kwargs)
        self.features = features
        self.momentum = momentum
        self.virtual_batch_size = virtual_batch_size

        self.transform = L.Dense(self.features, use_bias=False, name=f"transformblock_dense_{block_name}")
        self.bn = L.BatchNormalization(
            axis=-1, momentum=momentum, virtual_batch_size=virtual_batch_size, name=f"transformblock_bn_{block_name}"
        )

    def call(self, inputs, training=None):
        x = self.transform(inputs)
        x = self.bn(x, training=training)
        return x


class TabNetBackbone(keras.Model):
    def __init__(
        self,
        num_features,
        feature_dim=16,
        output_dim=12,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        batch_momentum=0.98,
        virtual_batch_size=None,
        epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_features is None:
            raise ValueError("If `feature_columns` is None, then `num_features` cannot be None.")

        if num_decision_steps < 1:
            raise ValueError("Num decision steps must be greater than 0.")

        if feature_dim <= output_dim:
            raise ValueError("To compute `features_for_coef`, feature_dim must be larger than output dim")

        feature_dim = int(feature_dim)
        output_dim = int(output_dim)
        num_decision_steps = int(num_decision_steps)
        relaxation_factor = float(relaxation_factor)
        sparsity_coefficient = float(sparsity_coefficient)
        batch_momentum = float(batch_momentum)
        epsilon = float(epsilon)

        if relaxation_factor < 0.0:
            raise ValueError("`relaxation_factor` cannot be negative !")

        if sparsity_coefficient < 0.0:
            raise ValueError("`sparsity_coefficient` cannot be negative !")

        if virtual_batch_size is not None:
            virtual_batch_size = int(virtual_batch_size)

        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.epsilon = epsilon

        if num_decision_steps > 1:
            features_for_coeff = feature_dim - output_dim
            print(f"[TabNet]: {features_for_coeff} features will be used for decision steps.")

        self.input_features = None
        self.input_bn = None

        self.transform_f1 = TransformBlock(
            2 * self.feature_dim, self.batch_momentum, self.virtual_batch_size, block_name="f1"
        )

        self.transform_f2 = TransformBlock(
            2 * self.feature_dim, self.batch_momentum, self.virtual_batch_size, block_name="f2"
        )

        self.transform_f3_list = [
            TransformBlock(2 * self.feature_dim, self.batch_momentum, self.virtual_batch_size, block_name=f"f3_{i}")
            for i in range(self.num_decision_steps)
        ]

        self.transform_f4_list = [
            TransformBlock(2 * self.feature_dim, self.batch_momentum, self.virtual_batch_size, block_name=f"f4_{i}")
            for i in range(self.num_decision_steps)
        ]

        self.transform_coef_list = [
            TransformBlock(self.num_features, self.batch_momentum, self.virtual_batch_size, block_name=f"coef_{i}")
            for i in range(self.num_decision_steps - 1)
        ]

        self._step_feature_selection_masks = None
        self._step_aggregate_feature_selection_mask = None

    def call(self, inputs, training=None):
        if self.input_features is not None:
            features = self.input_features(inputs)
            features = self.input_bn(features, training=training)

        else:
            features = inputs

        batch_size = tf.shape(features)[0]
        self._step_feature_selection_masks = []
        self._step_aggregate_feature_selection_mask = None

        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])

        total_entropy = 0.0
        entropy_loss = 0.0

        for ni in range(self.num_decision_steps):
            transform_f1 = self.transform_f1(masked_features, training=training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training=training)
            transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
            transform_f3 = (glu(transform_f3, self.feature_dim) + transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
            transform_f4 = (glu(transform_f4, self.feature_dim) + transform_f3) * tf.math.sqrt(0.5)

            if ni > 0 or self.num_decision_steps == 1:
                decision_out = tf.nn.relu(transform_f4[:, : self.output_dim])
                output_aggregated += decision_out
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)

                if self.num_decision_steps > 1:
                    scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1, tf.float32)

                aggregated_mask_values += mask_values * scale_agg

            features_for_coef = transform_f4[:, self.output_dim :]

            if ni < (self.num_decision_steps - 1):
                mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                mask_values *= complementary_aggregated_mask_values
                mask_values = sparsemax(mask_values, axis=-1)

                complementary_aggregated_mask_values *= self.relaxation_factor - mask_values
                total_entropy += tf.reduce_mean(
                    tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon), axis=1)
                ) / (tf.cast(self.num_decision_steps - 1, tf.float32))

                entropy_loss = total_entropy

                masked_features = tf.multiply(mask_values, features)

                mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                self._step_feature_selection_masks.append(mask_at_step_i)

            else:
                entropy_loss = 0.0

        self.add_loss(self.sparsity_coefficient * entropy_loss)

        agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
        self._step_aggregate_feature_selection_mask = agg_mask

        return output_aggregated

    @property
    def feature_selection_masks(self):
        return self._step_feature_selection_masks

    @property
    def aggregate_feature_selection_mask(self):
        return self._step_aggregate_feature_selection_mask


class TabNet(NetworkInputs):
    """TabNet uses sequential attention to choose which features to reason from at each decision step, enabling
    interpretability and better learning as the learning capacity is used for the most salient features.

    It employs a single deep learning architecture for feature selection and reasoning.
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
        backbone = TabNetBackbone(
            num_features=features.shape[-1],
            feature_dim=model_config.FEATURE_DIM,
            output_dim=model_config.OUTPUT_DIM,
            num_decision_steps=model_config.NUM_DECISION_STEPS,
            relaxation_factor=model_config.RELAXATION_FACTOR,
            sparsity_coefficient=model_config.SPARSITY_COEFFICIENT,
            batch_momentum=model_config.BATCH_MOMENTUM,
            virtual_batch_size=model_config.VIRTUAL_BATCH_SIZE,
            epsilon=model_config.EPSILON,
        )
        activation = backbone(features)
        outputs = L.Dense(
            units=model_config.NUM_OUT,
            activation=model_config.OUT_ACTIVATION,
            name="outputs",
        )(activation)
        model = keras.Model(inputs=inputs, outputs=outputs, name=name)
        return model
