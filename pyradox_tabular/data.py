from tensorflow.data import Dataset


class DataLoader:
    """DataLoader class for loading data from a DataFrame."""

    @classmethod
    def from_df(cls, X, y=None, batch_size=1024):
        """Generates tf.data.Dataset from a pandas.DataFrame.

        Args:
            X (pd.DataFrame): Input data.
            y (pd.Series/pd.DataFrame, optional): Target data. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 1024.

        Returns:
            tf.data.Dataset
        """
        return (
            Dataset.from_tensor_slices(({col: X[col].values.tolist() for col in X.columns}, y.values.tolist())).batch(
                batch_size
            )
            if y is not None
            else Dataset.from_tensor_slices({col: X[col].values.tolist() for col in X.columns}).batch(batch_size)
        )
