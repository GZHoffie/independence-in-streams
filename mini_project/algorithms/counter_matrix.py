from mini_project.utils import Estimator, _choose_prime
import numpy as np


class CounterMatrix(Estimator):
    """
    The class that uses a counter matrix to estimate the metrics.
    """
    def __init__(self, A: int) -> None:
        super().__init__(input_type=int)
        self.C = np.zeros((A, A), dtype=int)
        self.N = 0
        self.A = A

        # Generate hash functions
        self.p = _choose_prime(10 * A)
        self.param_x = self._generate_random_hash_parameters()
        self.param_y = self._generate_random_hash_parameters()


    def _generate_random_hash_parameters(self):
        """
        Generate random parameters for the hash functions. We only generate two
        functions for X and Y respectively.

        Returns:
            A list of two integers, for the parameters in hash function.
        """
        return [np.random.randint(1, self.p), np.random.randint(0, self.p)]
    
    def _calculate_hash_functions(self, i, j):
        """
        Calculate the value of hash functions. That is, based on the sample (i, j),
        map it to a pair (x, y), where x and y are within [0, A-1].

        Returns:
            x, y (int): A pair of integer indicating the place in counter matrix.
        """
        x = (self.param_x[0] * i + self.param_x[1]) % self.p % self.A
        y = (self.param_y[0] * j + self.param_y[1]) % self.p % self.A
        return x, y
    
    def _read_item(self, i, j):
        x, y = self._calculate_hash_functions(i, j)
        self.C[x, y] += 1
        self.N += 1

    def compute(self) -> float:
        p_x = np.sum(self.C, axis=1, keepdims=True)
        p_y = np.sum(self.C, axis=0, keepdims=True)
        observed = self.C / self.N
        expected = np.dot(p_x, p_y) / self.N ** 2
        
        return np.linalg.norm(observed - expected)




        