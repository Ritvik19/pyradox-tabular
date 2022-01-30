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
