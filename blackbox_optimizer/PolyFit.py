import numpy as np
import math
import numpy as np


def generate_terms_recursive(dimensions, max_degree, current_combination=[]):
    """
    A recursive helper function to generate polynomial terms.
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
    """
    Generate polynomial terms for a point of N dimensions up to the specified maximum degree (max_degree).

    :param dimensions: Number of dimensions (N).
    :param max_degree: Maximum polynomial power (max_d).
    :return: A list of tuples representing polynomial terms.
    """
    return np.array(list(generate_terms_recursive(dimensions, max_degree)))


class PolynomWorker:
    def __init__(self, point_dim: int, max_d: int = 2):
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
        assert len(point) == self.point_dim, "Wrong point dimension for polynom!"
        res = []
        for j in self.polynom_scheme:
            res.append(np.prod(point**j))
        return res

    def grad_polynom(self, point: np.array) -> np.array:
        assert len(point) == self.point_dim, "Wrong point dimension for polynom grad!"
        res = np.zeros(len(point))
        for k in range(len(point)):
            pows = self.grad_scheme[k]
            A = point**pows
            A = np.prod(A, axis=1)
            A = A * self.polynom_scheme[:, k]
            res[k] = A @ self.weights
        return res
