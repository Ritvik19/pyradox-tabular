class DeepNetworkConfig:
    def __init__(
        self, num_outputs, out_activation, hidden_units, dropout_rate=0.3, use_embeddings=False, embedding_dim=0
    ):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.USE_EMBEDDINGS = use_embeddings
        self.EMBEDDING_DIM = embedding_dim


class WideAndDeepNetworkConfig:
    def __init__(self, num_outputs, out_activation, hidden_units, dropout_rate=0.3, embedding_dim=32):
        self.NUM_OUT = num_outputs
        self.OUT_ACTIVATION = out_activation
        self.HIDDEN_UNITS = hidden_units
        self.DROPOUT_RATE = dropout_rate
        self.EMBEDDING_DIM = embedding_dim


class DeepAndCrossNetworkConfig:
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
    def __init__(self, depth, used_features_rate, num_classes):
        self.DEPTH = depth
        self.USED_FEATURES_RATE = used_features_rate
        self.NUM_CLASSES = num_classes


class NeuralDecisionForestConfig:
    def __init__(self, num_trees, depth, used_features_rate, num_classes):
        self.NUM_TREES = num_trees
        self.DEPTH = depth
        self.USED_FEATURES_RATE = used_features_rate
        self.NUM_CLASSES = num_classes
