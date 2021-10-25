from mini_project.utils import Estimator, _choose_prime
import numpy as np


class CounterMatrix(Estimator):
    """
    The class that uses a counter matrix to estimate the metrics.
    """
    def __init__(self, A: int, n: int) -> None:
        super().__init__(input_type=int)
        self.C = np.zeros((A, A), dtype=int)
        self.m = 0
        self.n = n
        self.A = A

        # Generate hash functions
        self.p = _choose_prime(10 * A)
        self.param_x = self._generate_random_hash_parameters()
        self.param_y = self._generate_random_hash_parameters()

        # Store marginal distribution
        self.p_x = np.sum(self.C, axis=0)
        self.p_y = np.sum(self.C, axis=1)


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
        self.m += 1
    
    def _compute_unbiased_s(self, i, j):
        """
        Compute the unbiased estimator for s_i,j = Pr[X=i, Y=j].
        """
        x, y = self._calculate_hash_functions(i, j)
        
        prob_collision = 1 / self.A ** 2
        return (self.C[x, y] / self.m - prob_collision) / (1 - prob_collision)
    
    def _compute_unbiased_p(self, i):
        """
        Compute the unbiased estimator for p_i = Pr[X=i].
        """
        x = (self.param_x[0] * i + self.param_x[1]) % self.p % self.A
        prob_collision = 1 / self.A
        return (self.p_x[x] / self.m - prob_collision) / (1 - prob_collision)
    
    def _compute_unbiased_q(self, j):
        """
        Compute the unbiased estimator for q_j = Pr[Y=j].
        """
        y = (self.param_y[0] * j + self.param_y[1]) % self.p % self.A
        prob_collision = 1 / self.A
        return (self.p_y[y] / self.m - prob_collision) / (1 - prob_collision)
    
    def compute(self) -> float:
        self.p_x = np.sum(self.C, axis=1)
        self.p_y = np.sum(self.C, axis=0)

        l2_estimate = 0
        for i in range(self.n):
            for j in range(self.n):
                l2_estimate += (self._compute_unbiased_s(i, j) - self._compute_unbiased_p(i) * self._compute_unbiased_q(j)) ** 2
        
        return np.sqrt(l2_estimate)




        