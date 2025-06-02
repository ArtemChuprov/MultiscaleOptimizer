import numpy as np


class FullScaler:
    """
    Min–max scaler for X and Y.

    Tracks global min/max across sequential calls to handle streaming data.
    """

    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.max_d = None

    def y_fit(self, Y):
        """
        Update Y min/max from a new batch.

        Args:
            Y (ndarray): New output values, shape (n_samples,) or (n_samples,1).
        """
        Y = Y.reshape(-1)
        Y_min = Y.min()
        Y_max = Y.max()

        if (self.y_min is not None) and (self.y_max is not None):
            self.y_min = min(self.y_min, Y_min)
            self.y_max = max(self.y_max, Y_max)
        else:
            self.y_min = Y_min
            if Y_min == Y_max:
                Y_max = 1.001 * Y_min
            self.y_max = Y_max

    def x_fit(self, X):
        """
        Update X min/max from a new batch.

        Args:
            X (ndarray): New input points, shape (n_samples, dim).
        """
        if self.max_d is not None:
            assert (
                self.max_d == X.shape[-1]
            ), f"Polynom is already working for shape {self.max_d}, not {X.shape[-1]}!"
        else:
            self.max_d = X.shape[-1]

        X = X.reshape(-1, self.max_d)
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)

        if (self.x_min is not None) and (self.x_max is not None):
            self.x_min = np.minimum(X_min, self.x_min)
            self.x_max = np.maximum(X_max, self.x_max)
        else:
            self.x_min = X_min
            self.x_max = X_max

    def x_scale(self, X):
        """
        Linearly scale X to [–1, 1] per feature.

        Args:
            X (ndarray): Points to scale, shape (..., dim).

        Returns:
            ndarray: Scaled points.
        """
        assert (self.x_min is not None) and (
            self.x_max is not None
        ), "You should fit X coord before using scaling!"
        shift = (self.x_min + self.x_max) / 2
        scale = (self.x_max - self.x_min) / 2
        return (X - shift) / scale

    def x_inv_scale(self, X):
        """
        Inverse transform of x_scale.

        Args:
            X (ndarray): Scaled points, shape (..., dim).

        Returns:
            ndarray: Original-scale points.
        """
        assert (self.x_min is not None) and (
            self.x_max is not None
        ), "You should fit X coord before using scaling!"
        shift = (self.x_min + self.x_max) / 2
        scale = (self.x_max - self.x_min) / 2
        return X * scale + shift

    def y_scale(self, Y):
        """
        Linearly scale Y to [0, 1].

        Args:
            Y (ndarray): Values to scale.

        Returns:
            ndarray: Scaled values.
        """
        assert (self.y_min is not None) and (
            self.y_max is not None
        ), "You should fit X coord before using scaling!"
        shift = self.y_min
        scale = self.y_max - self.y_min
        return (Y - shift) / scale

    def y_inv_scale(self, Y):
        """
        Inverse transform of y_scale.

        Args:
            Y (ndarray): Scaled values.

        Returns:
            ndarray: Original-scale values.
        """
        assert (self.y_min is not None) and (
            self.y_max is not None
        ), "You should fit X coord before using scaling!"
        shift = self.y_min
        scale = self.y_max - self.y_min
        return Y * scale + shift
