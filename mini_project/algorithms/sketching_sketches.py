from typing import overload
from mini_project.utils import Estimator, _choose_prime
import numpy as np

class L2Estimator(Estimator):
    """
    The class for estimating L2 difference of two distributions. We use the property
    of 4-wise independent vector of -1 and 1's to get an unbiased estimation of the norm
    ||r-s||. Here, instead of storing the vectors, we use random hash functions to map
    a pair (i, j) to either -1 or 1, so that we can save space.
    """
    def __init__(self, A: int, B: int, n: int = 10000) -> None:
        """
        To reduce error, the user should specify A=O(ε^(-2)) and B=O(log(1/δ))
        so that we can achive an (1+ε)-multiplicative error with probability at least
        1-δ.

        Args:
            A (int): number of experiments in each group. We take the mean in each group.
            B (int): number of groups we run. We take the median of the mean of the groups.
            n (int): Range of X and Y should be [1, n]. Not very important in our implementation,
                only used to determine the prime used in hash functions.
        """
        super().__init__(input_type=int)

        # Number of experience run
        self.A = A
        self.B = B

        # Number of items in the stream
        self.N = 0

        # Matrices to store intermediate values
        self.t_1 = np.zeros((A, B), dtype=int)
        self.t_2 = np.zeros((A, B), dtype=int)
        self.t_3 = np.zeros((A, B), dtype=int)

        # Use polynomial of degree 3 to generate 4-independent hash functions
        self.p = _choose_prime(10 * n)
        self.param_x = self._generate_random_hash_parameters()
        self.param_y = self._generate_random_hash_parameters()

    
    def _generate_random_hash_parameters(self):
        """
        Generate random parameters for the hash functions. To generate 4-wise independent
        hash functions, we use polynomial of degree 3 
        (ref: https://en.wikipedia.org/wiki/K-independent_hashing).
        For example, the hash function for x would be

            h(x) = {[(x_3 * x^3 + x_2 * x^2 + x_1 * x + x_0) mod p] mod 2} * 2 - 1
        
        where x_3, x_2, x_1 and x_0 are generate separately for each experiment.
        """
        parameters = {}
        for i in range(4):
            if i == 0:
                parameters["x_" + str(i)] = np.random.randint(0, self.p, size=(self.A, self.B))
            else:
                parameters["x_" + str(i)] = np.random.randint(1, self.p, size=(self.A, self.B))
        
        return parameters
    
    def _calculate_hash_functions(self, i, j):
        """
        Calculate the value of hash functions. That is, based on the sample (i, j),
        map it to a pair (i', j'), where i' and j' are either 1 or -1.

            h_x(i) = {[(x_3 * i^3 + x_2 * i^2 + x_1 * i + x_0) mod p] mod 2} * 2 - 1
            h_y(j) = {[(y_3 * j^3 + y_2 * j^2 + y_1 * j + y_0) mod p] mod 2} * 2 - 1

        Args:
            i (int): a sample in the stream that follows distribution X.
            j (int): a sample in the stream that follows distribution Y.
        
        Returns:
            x_i (np.array): shape (A, B), containing either -1 or 1.
            y_j (np.array): shape (A, B), containing either -1 or 1.
        """
        x_i = (self.param_x["x_3"] * (i ** 3) + self.param_x["x_2"] * (i ** 2) +\
              self.param_x["x_1"] * i + self.param_x["x_0"]) % self.p % 2 * 2 - 1
        y_j = (self.param_y["x_3"] * (j ** 3) + self.param_y["x_2"] * (j ** 2) +\
              self.param_y["x_1"] * j + self.param_y["x_0"]) % self.p % 2 * 2 - 1
            
        return x_i, y_j
    

    def _read_item(self, i: int, j: int):
        super()._read_item(i, j)
        x_i, y_j = self._calculate_hash_functions(i, j)
        self.t_1 += x_i * y_j
        self.t_2 += x_i
        self.t_3 += y_j
    

    def compute(self) -> float:
        # Calculate estimator Upsilon
        Upsilon = (self.t_1 / self.N - self.t_2 * self.t_3 / self.N ** 2) ** 2

        # Calculate mean of each group
        means = np.mean(Upsilon, axis=0)
        print("variance:", np.var(means))

        # Calculate the median of all means
        med = np.median(means)
        return np.sqrt(med)


class L1Estimator(Estimator):
    """
    The class for estimating L1 difference of two distributions. We use the s number of distributions
    x_1 ... x_s following Cauchy distribution that are independent, and one distribution following
    T-truncated-Cauchy for the estimation.
    """

    def __init__(self, delta: float, s: int, n: int = 10000) -> None:
        """
        To reduce error, the user should specify A=O(ε^(-2)) and B=O(log(1/δ))
        so that we can achive an (1+ε)-multiplicative error with probability at least
        1-δ.

        Args:
            delta (float): params for (O(ln n), δ)-approx. Determines number of experiments in each
                           group (also space usage). We take the mean in each group.
            s (int): number of groups we run. We take the median values of |t_1_r/m - t_2_r*t_3_r/m^2|
                     for all r in each group
            n (int): Range of X and Y should be [1, n].
        """
        super().__init__(input_type=int)

        # Number of experience run
        self.A = int(np.ceil(np.log(1/delta)))
        self.B = s
        self.n = n + 1

        # Define T for T-truncated-cauchy
        self.T = 100 * self.n

        # Number of items in the stream
        self.N = 0

        # Matrices to store intermediate values
        self.t_1 = np.zeros((self.A, self.B), dtype=float)
        self.t_2 = np.zeros((self.A, self.B), dtype=float)
        self.t_3 = np.zeros(self.A, dtype=float)

        # Get cauchy
        (self.x_cauchy, self.y_cauchy) = self._get_cauchy()

    def _get_cauchy(self):
        x_cauchy = np.zeros((self.A, self.B, self.n), dtype=float)
        y_t_cauchy = np.zeros((self.A, self.n), dtype=float)

        for i in range(self.A):
            x_cauchy[i] = self._get_x_cauchy()
            y_t_cauchy[i] = self._get_t_trukcated_cauchy()

        return (x_cauchy, y_t_cauchy)

    def _get_x_cauchy(self):
        x_cauchy = np.zeros((self.B, self.n))
        for i in range(self.B):
            x_cauchy[i] = np.random.standard_cauchy(self.n)
        return x_cauchy

    def _truncate(self, x):
        return np.where(x <= -self.T, -self.T, 0) \
            + np.where((x > -self.T) & (x < self.T), x, 0) \
            + np.where(x >= self.T, self.T, 0)

    def _get_t_trukcated_cauchy(self):
        y_cauchy = np.random.standard_cauchy(self.n)

        return self._truncate(y_cauchy)

    def _read_item(self, i: int, j: int) -> None:
        super()._read_item(i, j)
        self.t_1 += self.x_cauchy[:, :, i] * np.dot(self.y_cauchy[:, j].reshape(self.A, 1), np.ones((1, self.B)))
        self.t_2 += self.x_cauchy[:, :, i]
        self.t_3 += self.y_cauchy[:, j]

    def compute(self) -> float:
        # Calculate estimator Upsilon
        upsilon = np.zeros((self.A, self.B), dtype=float)
        for a in range(self.A):
            for b in range(self.B):
                upsilon[a, b] = (self.t_1[a, b]/self.N - self.t_2[a, b]*self.t_3[a]/self.N ** 2) ** 2

        return np.median(np.median(upsilon, axis=0))
