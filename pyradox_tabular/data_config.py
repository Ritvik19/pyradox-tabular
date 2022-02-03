class DataConfig:
    def __init__(self, numeric_feature_names, categorical_features_with_vocabulary):
        """Configuration for data processing.

        Args:
            numeric_feature_names (list): list of numeric feature names.
            categorical_features_with_vocabulary (dict): dictionary of categorical feature names and their vocabulary.
        """
        self.NUMERIC_FEATURE_NAMES = numeric_feature_names
        self.CATEGORICAL_FEATURES_WITH_VOCABULARY = categorical_features_with_vocabulary
        self.CATEGORICAL_FEATURE_NAMES = list(self.CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
        self.FEATURE_NAMES = self.NUMERIC_FEATURE_NAMES + self.CATEGORICAL_FEATURE_NAMES
