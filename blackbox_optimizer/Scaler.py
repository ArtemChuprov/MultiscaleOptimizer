import numpy as np


class FullScaler:
    def __init__(self):
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.max_d = None

    def y_fit(self, Y):
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
        assert (self.x_min is not None) and (
            self.x_max is not None
        ), "You should fit X coord before using scaling!"
        shift = (self.x_min + self.x_max) / 2
        scale = (self.x_max - self.x_min) / 2
        return (X - shift) / scale

    def x_inv_scale(self, X):
        assert (self.x_min is not None) and (
            self.x_max is not None
        ), "You should fit X coord before using scaling!"
        shift = (self.x_min + self.x_max) / 2
        scale = (self.x_max - self.x_min) / 2
        return X * scale + shift

    def y_scale(self, Y):
        assert (self.y_min is not None) and (
            self.y_max is not None
        ), "You should fit X coord before using scaling!"
        shift = self.y_min
        scale = self.y_max - self.y_min
        return (Y - shift) / scale

    def y_inv_scale(self, Y):
        assert (self.y_min is not None) and (
            self.y_max is not None
        ), "You should fit X coord before using scaling!"
        shift = self.y_min
        scale = self.y_max - self.y_min
        return Y * scale + shift
