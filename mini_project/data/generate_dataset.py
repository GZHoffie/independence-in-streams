from mini_project.utils import DataGenerator, ANSWER_DIR
from mini_project.algorithms.exact import ExactEstimator
import pickle
import os
import numpy as np

class DiscreteSampleGenerator(DataGenerator):
    """
    A data generator that generates discrete distributed samples of X and Y.
    Both X and Y should take values between 1 and n.
    """
    def __init__(self, n: int = 1000, N: int = 100000, 
                 independent: bool = False, distribution: str = "random") -> None:
        """
        Create a discrete sample generator.

        Args:
            n (int): range of the random variable X and Y.
            N (int): length of the stream (to be output into a file)
            independent (bool): Whether X and Y are independent or not.
            distribution (str): The distribution of X and Y. Currently support "random" and "zipfian". "random" distribution
                means the joint probability is randomly generated, and "zipfian" means that the probability follows a Zipf's
                law (https://en.wikipedia.org/wiki/Zipf%27s_law).
        """
        super().__init__(output_type=int, N=N)
        self.n = n
        self.independent = independent
        self.distribution = distribution
        assert distribution in ["random", "zipfian"], f"the distribution '{distribution}' is not supported."

        self.p_x, self.p_y, self.prob_table = self._initialize_probability_table()
        self.ground_truth = self._compute_ground_truth()
        
    
    def _initialize_probability_table(self):
        """
        Initialize a probability distribution based on the parameters specified.
        If self.independent is True, then we generate an independent probability table.
        Otherwise, the probability table is randomly generated (may be independent or not).

        Returns:
            p_x, p_y (np.array) of shape self.n * 1, 1 * self.n respectively, containing 
                the marginal probabilities.
            probability table (np.array) of shape self.n * self.n, with all elements sum to 1.
        """
        if self.independent:
            # Randomly generate marginal distribution
            if self.distribution == "random":
                p_x = np.random.random((self.n, 1))
                p_y = np.random.random((1, self.n))
            elif self.distribution == "zipfian":
                p_x = 1/(np.arange(self.n) + 1)
                p_y = 1/(np.arange(self.n) + 1)
                np.random.shuffle(p_x)
                np.random.shuffle(p_y)
                p_x = np.reshape(p_x, (self.n, 1))
                p_y = np.reshape(p_y, (1, self.n))
            
            # Normalize
            p_x = p_x / np.sum(p_x)
            p_y = p_y / np.sum(p_y)

            # choose probability distribution as dot product (independent on each element)
            return p_x, p_y, np.dot(p_x, p_y)
        else:
            # Randomly generate the whole matrix, then normalize
            if self.distribution == "random":
                prob_table = np.random.random((self.n, self.n))
            elif self.distribution == "zipfian":
                prob_table = 1/(np.arange(self.n ** 2) + 1)
                np.random.shuffle(prob_table)
                prob_table = np.reshape(prob_table, (self.n, self.n))

            
            prob_table /= np.sum(prob_table)
            p_x = np.sum(prob_table, axis=1, keepdims=True)
            p_y = np.sum(prob_table, axis=0, keepdims=True)
            return p_x, p_y, prob_table
    
    def _compute_ground_truth(self):
        """
        Compute the l1 and l2 difference based on the generated probability table.
        """
        ground_truth = {}

        # Compute expected value
        expected = np.dot(self.p_x, self.p_y)

        # Calculate l1 difference
        difference = self.prob_table - expected
        ground_truth["l1"] = np.sum(np.absolute(difference))
        ground_truth["l2"] = np.linalg.norm(difference)
        ground_truth["independent"] = self.independent

        return ground_truth

    
    def _generate_item(self):
        """
        Randomly generate a sample (i, j) based on self.prob_table.
        """
        i = np.random.choice(np.arange(self.n), p=self.p_x.flatten())
        # Calculate conditional probability p(y|x=i)
        py_x = self.prob_table[i, :]
        py_x /= np.sum(py_x)
        j =np.random.choice(np.arange(self.n), p=py_x.flatten())
        
        return i + 1, j + 1
    
    def write_file(self, file_name: str):
        super().write_file(file_name)
        estimator = ExactEstimator(self.n, metric=["l1", "l2", "independent"])
        estimator.read_from_file(file_name)
        l1, l2, independent = estimator.compute()
        answer = {"l1": l1, "l2": l2, "independent": independent}
        print(answer)

        with open(os.path.join(ANSWER_DIR, file_name + '.pickle'), 'wb') as p:
            pickle.dump(answer, p, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    generator = DiscreteSampleGenerator(n=10000, N=1000000, independent=True, distribution="zipfian")
    generator.write_file("10000-1000000-independent-zipfian")
    generator = DiscreteSampleGenerator(n=10000, N=1000000, independent=False, distribution="zipfian")
    generator.write_file("10000-1000000-dependent-zipfian")
    generator = DiscreteSampleGenerator(n=10000, N=1000000, independent=True, distribution="random")
    generator.write_file("10000-1000000-independent-random")
    generator = DiscreteSampleGenerator(n=10000, N=1000000, independent=False, distribution="random")
    generator.write_file("10000-1000000-dependent-radom")
