import numpy as np
import math
import numpy as np


def generate_terms_recursive(dimensions, max_degree, current_combination=[]):
    """
    Generate all exponent tuples for monomials up to total degree.

    Args:
        dimensions (int): Number of input dimensions.
        max_degree (int): Maximum total degree per term.

    Returns:
        ndarray of shape (n_terms, dimensions): Each row is a tuple of exponents.
    """
    if len(current_combination) == dimensions:
        yield tuple(current_combination)
        return
    start_degree = 0
    max_possible_degree = max_degree - sum(current_combination)
    for degree in range(start_degree, max_possible_degree + 1):
        yield from generate_terms_recursive(
            dimensions, max_degree, current_combination + [degree]
        )


def generate_polynomial_terms(dimensions, max_degree):
    return np.array(list(generate_terms_recursive(dimensions, max_degree)))


class PolynomWorker:
    def __init__(self, point_dim: int, max_d: int = 2):
        """
        Handles polynomial basis evaluation and gradient computation.

        Attributes:
            max_d (int): Maximum total degree.
            point_dim (int): Number of input dimensions.
            polynom_scheme (ndarray): List of exponent tuples for each term.
            weights (ndarray): Learned weights per polynomial term.
            grad_scheme (list): Exponent arrays for gradient of each input dimension.
        """
        self.max_d = max_d
        self.point_dim = point_dim

        self.polynom_scheme = generate_polynomial_terms(point_dim, max_d)
        self.weights = np.zeros(self.polynom_scheme.shape[0])

        self.grad_scheme = []
        for k in range(self.point_dim):
            pows = self.polynom_scheme.copy()
            pows[:, k] -= 1
            pows[pows < 0] = 0

            self.grad_scheme.append(pows)

    def get_polynom(self, point: np.array) -> np.array:
        """
        Evaluate all monomial terms at a given point.

        Args:
            point (ndarray): Input vector, shape (point_dim,).

        Returns:
            list of float: Values of each polynomial term.
        """
        assert len(point) == self.point_dim, "Wrong point dimension for polynom!"
        res = []
        for j in self.polynom_scheme:
            res.append(np.prod(point**j))
        return res

    def grad_polynom(self, point: np.array) -> np.array:
        """
        Compute gradient of the polynomial trend at a given point.

        Args:
            point (ndarray): Input vector, shape (point_dim,).

        Returns:
            ndarray: Gradient vector, shape (point_dim,).
        """
        assert len(point) == self.point_dim, "Wrong point dimension for polynom grad!"
        res = np.zeros(len(point))
        for k in range(len(point)):
            pows = self.grad_scheme[k]
            A = point**pows
            A = np.prod(A, axis=1)
            A = A * self.polynom_scheme[:, k]
            res[k] = A @ self.weights
        return res
