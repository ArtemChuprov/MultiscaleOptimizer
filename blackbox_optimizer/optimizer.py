import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import basinhopping
import math
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process
from .poly_fit import PolynomWorker
from .scaler import FullScaler
import traceback


def basin_scheduler(N, L, dim):
    steps = int(np.sqrt(N))
    step_size = L / N ** (1 / dim)
    return (steps, step_size)


# def points_controller(X, y, optimizer):
#     max_points = optimizer.


class RBF:
    """
    Radial Basis Function surrogate optimizer.

    Uses a combination of RBF interpolation, a polynomial trend, and density-based
    exploration to suggest new points for expensive black-box functions.

    Attributes:
        reg_coeff (float): Regularization coefficient.
        dx (float): Finite-difference step for numerical gradients.
        epsilon (float): Perturbation size around best point.
        lr (float): Learning rate (not used directly).
        dropout (float): Probability of exploring around current best.
        bounds (ndarray): Array of shape (dim, 2) with lower/upper bounds.
        alphas (ndarray): Exploration weights for each process.
        num_processes (int): Number of parallel workers.
        scaler (FullScaler): Scales X and y to [–1,1] or [0,1].
        polynom (PolynomWorker): Fits a low-degree polynomial trend.
        main_func (callable): Objective function f(x, proc_ind, func_kwargs).
        func_kwargs (dict): Extra arguments for main_func.
        minimizer_kwargs (dict): Passed to SciPy's minimizer.
        centers (list): Evaluated sample points.
        y (ndarray): Objective values at sample points (scaled).
        weights (ndarray): Combined RBF + polynomial weights.
    """

    def __init__(
        self,
        X=None,
        y=None,
        lr=0.01,
        dx=0.0001,
        epsilon: float = 0.05,
        rcond: int = None,
        bounds=None,
        main_func=None,
        num_processes=2,
        max_alpha=10,
        dropout: float = 0,
        func_kwargs: dict = dict(),
        minimizer_kwargs: dict = dict(),
        discrete=False,
        max_polynom_dim=None,
        checker_point=None,
        reg_coeff: float = 0,
        start_steps: int = 1,
    ):
        """
        Initialize the RBF optimizer.

        Args:
            X (ndarray, optional): Initial sample points, shape (n_samples, dim).
            y (ndarray, optional): Observed values at X, shape (n_samples, 1).
            lr (float): (Unused) learning rate placeholder.
            dx (float): Finite-difference step size.
            epsilon (float): Neighborhood radius for local refinement.
            rcond (int, optional): rcond parameter for least squares.
            bounds (ndarray): Array of shape (dim,2) specifying [min,max] per dimension.
            main_func (callable): Function to evaluate: f(x, proc_ind, func_kwargs) → float.
            num_processes (int): Number of parallel evaluations per iteration.
            max_alpha (float): Maximum exploration weight.
            dropout (float): Chance to re-exploit around current best.
            func_kwargs (dict): Extra args passed to main_func.
            minimizer_kwargs (dict): Extra args passed to basinhopping’s local minimizer.
            discrete (bool): Round proposed points to integers if True.
            max_polynom_dim (int, optional): Degree of polynomial trend.
            checker_point (ndarray, optional): Point to validate interpolation.
            reg_coeff (float): Regularization coefficient for system solve.
            start_steps (int): Number of random initial evaluations if X, y not provided.
        """
        self.reg_coeff = reg_coeff
        self.dx = dx
        self.epsilon = epsilon
        self.N = 0
        self.lr = lr
        self.y_min = None
        self.dropout = dropout
        self.bounds = bounds
        if num_processes > 1:
            self.alphas = np.linspace(0, max_alpha, num_processes - 1) / np.arange(
                num_processes - 1, 0, -1
            )
            self.alphas = np.append(self.alphas, None)
        else:
            self.alphas = np.array([max_alpha])
        self.num_processes = num_processes
        self.scaler = FullScaler()
        self.scaler.x_fit(X=bounds.T)
        # self.scaler.y_fit(Y=y)
        if max_polynom_dim is None:
            max_polynom_dim = 2

        self.polynom = PolynomWorker(max_d=max_polynom_dim, point_dim=bounds.shape[0])

        self.main_func = main_func
        self.func_kwargs = func_kwargs
        self.minimizer_kwargs = minimizer_kwargs

        self.centers = []
        self.y = []
        self.P = []
        np.random.seed(0)
        self.discrete = discrete
        self.rcond = rcond
        self.checker_point = checker_point

        if (X is None) or (y is None):
            for i in range(start_steps):
                self.on_start()
        else:
            self.fit(X, y)

    def _basis_function(self, center, data_point):
        """
        Compute the RBF kernel φ(r) = r² log(r) between two points.

        Args:
            center (ndarray): Center point, shape (dim,).
            data_point (ndarray): Query point, shape (dim,).

        Returns:
            float: Kernel value φ(||center−data_point||).
        """
        R = np.linalg.norm((center - data_point))
        if R == 0:
            return 0
        return np.log(R) * R**2

    def grad_basis_function(self, center, data_point):
        r_ = data_point - center
        r = np.linalg.norm(r_) + 1e-8
        return r_ * (2 * np.log(r) + 1)

    def grad_density_function(self, data_point):
        points = self.centers
        distances = cdist(points, [data_point]).reshape(-1)
        min_ind = np.argmin(distances)

        r_min = data_point - points[min_ind]
        return r_min / (np.linalg.norm(r_min) + 1e-12)

    def grad_explore(self, data_point, alpha, scale_X=False):
        if scale_X:
            data_point = self.scaler.x_scale(data_point)
        grad_1 = np.sum(
            [
                self.grad_basis_function(self.centers[i], data_point) * self.weights[i]
                for i in range(self.N)
            ],
            axis=0,
        )
        grad_2 = self.polynom.grad_polynom((data_point))
        grad_3 = self.grad_density_function(data_point)

        if alpha is not None:
            res = grad_1 + grad_2 - alpha * grad_3
        else:
            res = -grad_3
        # print("grad: ", res)
        return res

    def fit(self, X, y):
        X = self.scaler.x_scale(X)
        if len(self.centers) == 0:
            self.centers = X
            self.y = y
            self.P = np.array([self.polynom.get_polynom((p)) for p in X])
            self.m = self.P.shape[1]
        else:
            self.centers = np.append(self.centers, X, axis=0)
            self.y = self.scaler.y_inv_scale(self.y)
            self.y = np.append(self.y, y, axis=0)
            self.P = np.append(
                self.P,
                np.array([self.polynom.get_polynom((p)) for p in X]),
                axis=0,
            )
        self.scaler.y_fit(Y=y)
        self.y_min = np.min(self.y)
        self.y = self.scaler.y_scale(self.y)

        # self.y = self.scaler.y_scale(self.y).reshape(-1, 1)
        # print("Current y: ", self.y)

        num_centers = len(self.centers)
        self.N = num_centers

        K = np.zeros((num_centers, num_centers))
        for i in range(num_centers):
            for j in range(num_centers):
                K[i, j] = self._basis_function(self.centers[i], self.centers[j])

        A = np.hstack([K, self.P])
        B = np.hstack([self.P.T, np.zeros((self.m, self.m))])
        C = np.vstack([A, B])

        regularization_matrix = self.reg_coeff * np.eye(C.shape[1])
        # Add regularization term to the matrix A
        C_reg = np.vstack([C, regularization_matrix])

        # Add zeros to the bottom of the B vector
        B_reg = np.concatenate(
            [np.append(self.y, np.zeros(self.m)), np.zeros(C.shape[1])]
        )

        # Solve the regularized system
        self.weights, residuals, rank, singular_values = np.linalg.lstsq(
            C_reg, B_reg, rcond=None
        )

        # self.weights, residuals, rank, singular_values = np.linalg.lstsq(
        #     C ,
        #     np.append(self.y, np.zeros(self.m)),
        #     rcond=self.rcond,
        # )

        self.polynom.weights = self.weights[-self.m :]

        # assert (
        #     np.round((C @ self.weights)[: -self.m], 3) == np.round(self.y.T[0], 3)
        # ).all(), f"{X}\n{y}"

    def density_function(self, x):
        points = self.centers
        distances = cdist(points, [x])
        dist_res = distances.min() / 2

        return dist_res

    def predict(self, X, scale_X=False, scale_Y=False):
        # X = self.scaler.x_scale(X)
        if scale_X:
            X = self.scaler.x_scale(X)
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            K_i = np.array(
                [
                    self._basis_function(self.centers[j], X[i])
                    for j in range(len(self.centers))
                ]
            )
            P_i = self.polynom.get_polynom((X[i]))
            F_i = np.append(K_i, P_i)
            y_pred[i] = np.sum(self.weights.reshape(-1) * F_i)

        if not scale_Y:
            y_pred = self.scaler.y_inv_scale(y_pred)
        return y_pred

    def explore(self, x, alpha, scale_X=False):
        if alpha is not None:
            res = self.predict(
                np.array([x]), scale_Y=True, scale_X=scale_X
            ) - alpha * self.density_function(x)
        else:
            res = -self.density_function(x)
        return res

    def generate_point(self, alpha, iterate_best=False, random=False):
        seed = list(self.alphas).index(alpha) + 2 * len(self.centers)
        np.random.seed(seed)
        if not iterate_best:
            x0 = np.random.uniform(
                low=np.array([bound[0] for bound in self.bounds]),
                high=np.array([bound[1] for bound in self.bounds]),
            )
            if random:
                return x0
            x0 = self.scaler.x_scale(x0)
            print(f"alpha: {alpha}")
            niter, step_size = basin_scheduler(
                N=len(self.centers), L=2, dim=len(self.centers[1])
            )
            result = basinhopping(
                func=self.explore,
                x0=x0,
                niter=niter,
                stepsize=step_size,
                seed=0,
                T=0,
                minimizer_kwargs={
                    "args": (alpha),
                    **self.minimizer_kwargs,
                    **dict(
                        bounds=[(-1, 1) for bound in self.bounds],
                        jac=self.grad_explore,
                    ),
                },
            )

            new_point = result.x
            if (alpha is not None) and alpha > 0:
                new_point += np.random.uniform(
                    low=-0.01, high=0.01, size=self.centers.shape[1]
                )
        else:
            new_point = self.centers[self.y.argmin()] + np.random.uniform(
                low=-self.epsilon, high=self.epsilon, size=self.centers.shape[1]
            )
        # adjusting to boundaries
        new_point = np.minimum(new_point, np.array([1] * self.centers.shape[1]))
        new_point = np.maximum(new_point, np.array([-1] * self.centers.shape[1]))
        return new_point

    def on_start(self):
        n_bytes = self.num_processes * (self.scaler.max_d + 1) * 8
        # create the shared memory
        for i in range(8):
            try:
                self.sm = SharedMemory(
                    name=f"StepCalculations_{i}", create=True, size=n_bytes
                )
                break
            except FileExistsError:
                print("trying another sahred memory name")
                continue
        self.step_data = np.ndarray(
            (self.num_processes, (self.scaler.max_d + 1)),
            dtype=np.double,
            buffer=self.sm.buf,
        )

        processes = []
        for proc_ind in range(self.num_processes):
            new_process = Process(
                target=self.parallel_main_func,
                args=tuple([proc_ind, False, True]),
            )
            new_process.start()
            processes.append(new_process)

        for proc_ind in range(self.num_processes):
            processes[proc_ind].join()

        # remove rows having all zeroes
        data = np.array(self.step_data)
        data = data[~np.all(np.round(data, 7) == 0, axis=1)]
        # print("new batch")
        # print(data)

        X, y = data[:, :-1], data[:, -1]

        self.fit(X=X, y=y.reshape(-1, 1))

        self.sm.close()
        self.sm.unlink()

    def parallel_main_func(self, proc_ind, iterate_best=False, random=False):
        if random and not iterate_best:
            x = self.generate_point(
                alpha=self.alphas[proc_ind], iterate_best=iterate_best, random=True
            )
        else:
            x = self.scaler.x_inv_scale(
                self.generate_point(
                    alpha=self.alphas[proc_ind], iterate_best=iterate_best
                )
            )
        if self.discrete:
            x = np.round(x).astype(np.int32)
        try:
            y = self.main_func(x, proc_ind, self.func_kwargs)
            shared_mem = SharedMemory(name=self.sm.name, create=False)
            data = np.ndarray(
                (self.num_processes, self.scaler.max_d + 1),
                dtype=np.double,
                buffer=shared_mem.buf,
            )
            # print("new res: ", y)
            data[proc_ind] = np.append(x, y)
            shared_mem.close()
        except Exception as e:
            print(f"Error on {proc_ind} proc happaned: {e}")
            # Print the full traceback
            print("Full traceback:")
            traceback.print_exc()

    def find_optimum(self, max_iter=10, min_val=0.03):
        n_bytes = self.num_processes * (self.scaler.max_d + 1) * 8
        # create the shared memory
        for i in range(3):
            try:
                self.sm = SharedMemory(
                    name=f"StepCalculations_{i}", create=True, size=n_bytes
                )
                break
            except FileExistsError:
                continue
        self.step_data = np.ndarray(
            (self.num_processes, (self.scaler.max_d + 1)),
            dtype=np.double,
            buffer=self.sm.buf,
        )

        for i in range(max_iter):
            processes = []
            for proc_ind in range(self.num_processes):
                iterate_best = np.random.uniform(0, 1) < self.dropout
                new_process = Process(
                    target=self.parallel_main_func,
                    args=tuple([proc_ind, iterate_best]),
                )
                new_process.start()
                processes.append(new_process)

            for proc_ind in range(self.num_processes):
                processes[proc_ind].join()

            # remove rows having all zeroes
            data = np.array(self.step_data)
            data = data[~np.all(np.round(data, 7) == 0, axis=1)]
            # print(self.step_data)

            X, y = data[:, :-1], data[:, -1]
            if self.checker_point is not None:
                y0 = self.predict(self.checker_point.reshape(1, -1), scale_X=True)
                y_test = self.predict(X, scale_X=True).min()
                # assert y0 >= y_test, f"True center is already {y0} while have {y_test}"
            if np.min(y) < self.y_min:
                print(f"New best result {np.min(y)} at {X[np.argmin(y)]}")
            # print("X = ", X)
            # print("y = ", y)
            self.fit(X=X, y=y.reshape(-1, 1))
            print(f"Iteration {i} is over!")
            print(self.y.reshape(-1))
            print(f"Current best result: {self.scaler.y_inv_scale(self.y).min()}")
            if np.min(y) <= min_val:
                break

        self.sm.close()
        self.sm.unlink()

        print(f"Iterations took {i+1} step")
        result = min(self.y)
        res_point = self.centers[np.argmin(self.y)]
        print(
            f"Final result: {self.scaler.y_inv_scale(result)} at point {self.scaler.x_inv_scale(res_point)}"
        )
        x_min_estimated = self.generate_point(alpha=0)
        y_min_estimated = self.predict(x_min_estimated.reshape(1, -1))
        print(
            f"Stable result: {y_min_estimated} at point {self.scaler.x_inv_scale(x_min_estimated)}"
        )
        self.minimum_value = np.min(self.y)
        self.minimum_point = res_point
        return res_point
