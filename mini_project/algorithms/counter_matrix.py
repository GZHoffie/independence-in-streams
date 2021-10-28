from mini_project.utils import Estimator, _choose_prime
import numpy as np


class CounterMatrix(Estimator):
    """
    The class that uses a counter matrix to estimate the metrics.

    Args:
        A (int): set the size of counter matrix to be A * A
    """
    def __init__(self, A: int, metric: str = "l2") -> None:
        super().__init__(input_type=int)
        self.C = np.zeros((A, A), dtype=int)   # Counter matrix
        self.A = A                             # Size of counter matrix
        self.metric = metric

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
        super()._read_item(i, j)
        x, y = self._calculate_hash_functions(i, j)
        self.C[x, y] += 1

    def compute(self) -> float:
        p_x = np.sum(self.C, axis=1, keepdims=True)
        p_y = np.sum(self.C, axis=0, keepdims=True)
        observed = self.C / self.N
        expected = np.dot(p_x, p_y) / self.N ** 2
        
        if self.metric == "l1":
            return np.sum(np.absolute(observed - expected))
        else:
            return np.linalg.norm(observed - expected) ** 2 / (1-1/self.A) ** 2


class L2Estimator(Estimator):
    """
    Estimator for L2 difference that uses multiple counter matrices and return the mean
    norm.

    Args:
        A (int): size of counter matrix
        B (int): number of counter matrices
    """
    def __init__(self, A: int, B: int) -> None:
        super().__init__(input_type=int)

        self.C_list = []
        for _ in range(B):
            self.C_list.append(CounterMatrix(A, metric="l2"))
        
    def _read_item(self, i, j):
        for C in self.C_list:
            C._read_item(i, j)
    
    def compute(self) -> float:
        res = [C.compute() for C in self.C_list]
        
        return np.sqrt(np.mean(res))


class L1Estimator(Estimator):
    """
    Estimator for L1 difference that uses multiple counter matrices and return the largest
    norm (since result from counter matrix is always underestimated).

    Args:
        A (int): size of counter matrix
        B (int): number of counter matrices
    """
    def __init__(self, A: int, B: int) -> None:
        super().__init__(input_type=int)

        self.C_list = []
        for _ in range(B):
            self.C_list.append(CounterMatrix(A, metric="l1"))
        
    def _read_item(self, i, j):
        for C in self.C_list:
            C._read_item(i, j)
    
    def compute(self) -> float:
        res = [C.compute() for C in self.C_list]
        
        return np.max(res)




        