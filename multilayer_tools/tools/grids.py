import numpy as np

class MultiDimGrid:
    def __init__(self, bounds):
        self.bounds = bounds

    def create_uniform_grid(self, steps, arange=False):
        X = []
        assert type(steps) in [int, list], 'Steps type should be int or list!'
        if isinstance(steps, int):
            for bound_pair in self.bounds:
                if arange:
                    X.append(np.arange(bound_pair[0], bound_pair[1], steps))
                else:
                    X.append(np.linspace(bound_pair[0], bound_pair[1], steps))

        else:
            for i in range(len(steps)):
                bound_pair = self.bounds[i]
                step = steps[i]
                if arange:
                    X.append(np.arange(bound_pair[0], bound_pair[1], step))
                else:
                    X.append(np.linspace(bound_pair[0], bound_pair[1], step))

        X = np.meshgrid(*X)

        positions = np.vstack(list(map(np.ravel, X))).T
        return positions
    
    def create_random_grid(self, points_number: int = 5):
        low, high = self.bounds.T
        positions = np.random.uniform(
            low=low, high=high, size=[points_number, len(low)]
        )
        return positions


