import tensorflow as tf


class DeepNetworkConfig:
    """Configurations for the deep network.

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        hidden_units (list): list of hidden units for each hidden layer.
        dropout_rate (float, optional): dropout rate, Defaults to 0.3.
        use_embeddings (bool, optional): whether to use embeddings, Defaults to True.
        embedding_dim (int, optional): embedding dimension, Defaults to 32.
    """

    def __init__(
        self, num_outputs, out_activation, hidden_units, dropout_rate=0.3, use_embeddings=True, embedding_dim=32
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.USE_EMBEDDINGS = use_embeddings
        self.EMBEDDING_DIM = embedding_dim


class WideAndDeepNetworkConfig:
    """Configurations for the wide and deep network.

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        hidden_units (list): list of hidden units for each hidden layer.
        dropout_rate (float, optional): dropout rate, Defaults to 0.3.
        embedding_dim (int, optional): embedding dimension, Defaults to 0.
    """

    def __init__(self, num_outputs, out_activation, hidden_units, dropout_rate=0.3, embedding_dim=32):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.EMBEDDING_DIM = embedding_dim


class DeepAndCrossNetworkConfig:
    """Configurations for the deep and cross network.

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        hidden_units (list): list of hidden units for each hidden layer.
        n_cross (int, optional): number of cross features, Defaults to 2.
        dropout_rate (float, optional): dropout rate, Defaults to 0.3.
        use_embeddings (bool, optional): whether to use embeddings, Defaults to True.
        embedding_dim (int, optional): embedding dimension, Defaults to 32.
    """

    def __init__(
        self,
        num_outputs,
        out_activation,
        hidden_units,
        n_cross=2,
        dropout_rate=0.3,
        use_embeddings=True,
        embedding_dim=32,
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.USE_EMBEDDINGS = use_embeddings
        self.EMBEDDING_DIM = embedding_dim
        self.NUM_CROSS_LAYERS = n_cross


class TabTransformerConfig:
    """Configurations for the tab transformer.

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        num_transformer_blocks (int): number of transformer blocks.
        num_heads (int): number of heads for each transformer block.
        mlp_hidden_units_factors (list): list of factors for each hidden layer.
        use_column_embedding (bool, optional): whether to use column embedding, Defaults to True.
        embedding_dim (int, optional): embedding dimension, Defaults to 32.
        dropout_rate (float, optional): dropout rate, Defaults to 0.3.
    """

    def __init__(
        self,
        num_outputs,
        out_activation,
        num_transformer_blocks,
        num_heads,
        mlp_hidden_units_factors,
        use_column_embedding=True,
        embedding_dim=32,
        dropout_rate=0.3,
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.NUM_TRANSFORMER_BLOCKS = num_transformer_blocks
        self.NUM_HEADS = num_heads
        self.MLP_HIDDEN_UNITS_FACTORS = mlp_hidden_units_factors
        self.USE_COLUMN_EMBEDDING = use_column_embedding
        self.EMBEDDING_DIM = embedding_dim
        self.DROPOUT_RATE = dropout_rate


class NeuralDecisionTreeConfig:
    """Configurations for neural decision tree

    Args:
        depth (int): depth of the tree.
        used_features_rate (float): fraction of the features to be used.
        num_classes (int): number of classes.
    """

    def __init__(self, depth, used_features_rate, num_classes):
        self.DEPTH = depth
        self.USED_FEATURES_RATE = used_features_rate
        self.NUM_CLASSES = num_classes


class NeuralDecisionForestConfig:
    """Configurations for neural decision forest

    Args:
        num_trees (int): number of trees in the forest.
        depth (int): depth of the tree.
        used_features_rate (float): fraction of the features to be used.
        num_classes (int): number of classes.
    """

    def __init__(self, num_trees, depth, used_features_rate, num_classes):
        self.NUM_TREES = num_trees
        self.DEPTH = depth
        self.USED_FEATURES_RATE = used_features_rate
        self.NUM_CLASSES = num_classes


class NeuralObliviousDecisionTreeConfig:
    """Configurations for neural oblivious decision tree

    Args:
        num_outputs (int, optional): Defaults to 1.
        n_trees (int, optional): Defaults to 3.
        depth (int, optional): Defaults to 4.
        threshold_init_beta (float, optional): Defaults to 1.0.
    """

    def __init__(self, num_outputs=1, n_trees=3, depth=4, threshold_init_beta=1.0):
        self.NUM_OUT = num_outputs
        self.N_TREES = n_trees
        self.DEPTH = depth
        self.THRESHOLD_INIT_BETA = threshold_init_beta


class NeuralObliviousDecisionEnsembleConfig:
    """Configurations for neural oblivious decision ensemble

    Args:
        num_outputs (int, optional): Defaults to 1.
        n_layers (int, optional): Defaults to 1.
        link ([type], optional): Defaults to tf.identity.
        n_trees (int, optional): Defaults to 3.
        tree_depth (int, optional): Defaults to 4.
        threshold_init_beta (int, optional): Defaults to 1.
    """

    def __init__(self, num_outputs=1, n_layers=1, link=tf.identity, n_trees=3, tree_depth=4, threshold_init_beta=1):
        self.NUM_OUT = num_outputs
        self.N_LAYERS = n_layers
        self.LINK = link
        self.N_TREES = n_trees
        self.TREE_DEPTH = tree_depth
        self.THRESHOLD_INIT_BETA = threshold_init_beta


class TabNetConfig:
    """Configurations for tabnet

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        feature_dim (int, optional): dimensionality of the hidden representation in feature transformation block,
            Defaults to 16.
        output_dim (int, optional): dimensionality of the outputs of each decision step, which is later mapped
            to the final output, Defaults to 12.
        num_decision_steps (int, optional): number of decision steps, Defaults to 5.
        relaxation_factor (float, optional): relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps, Defaults to 1.5.
        sparsity_coefficient (float, optional): strength of the sparsity regularization, Defaults to 1e-5.
        batch_momentum (float, optional): momentum for ghost batch normalization, Defaults to 0.98.
        virtual_batch_size (int, optional): batch size in ghost batch normalization, Defaults to None.
        epsilon ([type], optional): a small number for numerical stability of the entropy calculations,
            Defaults to 1e-5.
    """

    def __init__(
        self,
        num_outputs,
        out_activation,
        feature_dim=16,
        output_dim=12,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        batch_momentum=0.98,
        virtual_batch_size=None,
        epsilon=1e-5,
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.FEATURE_DIM = feature_dim
        self.OUTPUT_DIM = output_dim
        self.NUM_DECISION_STEPS = num_decision_steps
        self.RELAXATION_FACTOR = relaxation_factor
        self.SPARSITY_COEFFICIENT = sparsity_coefficient
        self.BATCH_MOMENTUM = batch_momentum
        self.VIRTUAL_BATCH_SIZE = virtual_batch_size
        self.EPSILON = epsilon


class FeatureTokenizerTransformerConfig:
    """Configurations for feature tokenizer transformer

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        num_transformer_blocks (int, optional): number of transformer blocks, Defaults to 2.
        num_heads (int, optional): number of heads in each transformer block, Defaults to 8.
        embedding_dim (int, optional): dimensionality of the embedding, Defaults to 32.
        dense_dim (int, optional): dimensionality of the dense layer, Defaults to 16.
    """

    def __init__(
        self,
        num_outputs,
        out_activation,
        num_transformer_blocks=2,
        num_heads=8,
        embedding_dim=32,
        dense_dim=16,
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.NUM_TRANSFORMER_BLOCKS = num_transformer_blocks
        self.NUM_HEADS = num_heads
        self.EMBEDDING_DIM = embedding_dim
        self.DENSE_DIM = dense_dim


class TabularResNetConfig:
    """Configurations for the tabular resnet.

    Args:
        num_outputs (int): number of cells in output layer.
        out_activation (str): activation function for output layer.
        hidden_units (list): list of hidden units for each hidden layer.
        dropout_rate (float, optional): dropout rate, Defaults to 0.3.
        use_embeddings (bool, optional): whether to use embeddings, Defaults to True.
        embedding_dim (int, optional): embedding dimension, Defaults to 32.
    """

    def __init__(
        self, num_outputs, out_activation, hidden_units, dropout_rate=0.3, use_embeddings=True, embedding_dim=32
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.USE_EMBEDDINGS = use_embeddings
        self.EMBEDDING_DIM = embedding_dim
