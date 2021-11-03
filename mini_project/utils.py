import os
import warnings
import pickle


TEST_DATA_DIR = "mini_project/test/test_data/"
GROUND_TRUTH_DIR = "mini_project/test/ground_truth/"
ANSWER_DIR = "mini_project/test/answer/"

class Estimator:
    """
    Base class for the correlation estimators.

    Args:
        input_type (class): Usually int or float, indicating the type of input.
    """
    def __init__(self, input_type=int) -> None:
        self.input_type = input_type   # Input type (int or float)
        self.N = 0                     # Length of the stream
    

    def _read_item(self, i, j):
        """
        Read a pair of samples i and j from the stream, and store the information.

        Args:
            i (int or float): a sample in the stream that follows distribution X.
            j (int or float): a sample in the stream that follows distribution Y.
        """
        self.N += 1
        pass


    def read_from_file(self, file_name: str):
        """
        Read the stream from a file. For each line, there should be 2 numbers, which are
        the samples from X and Y distributions, respectively.

        Args:
            file_name (string): the path to the data file.
        """
        with open(TEST_DATA_DIR + file_name + ".txt", 'r') as f:
            for line in f:
                i, j = [self.input_type(x) for x in line.split()]
                self._read_item(i, j)


    def compute(self) -> float:
        """
        Compute the metric given the data.

        Returns:
            The estimated value of the metric specified.
        """
        pass

    def reset(self):
        """
        Clear out all the stored data and be ready for the next stream.
        """
        pass


class DataGenerator:
    """
    Base class for the data generators.

    Args:
        output_type (class): Usually int or float, indicating the type of output.
        N (int): the length of the stream
        overwrite (bool): if set to False, will not generate new dataset if the specified
            file path exist
    """
    def __init__(self, output_type=int, N=100000, overwrite=True) -> None:
        self.output_type = output_type   # Input type (int or float)
        self.N = N                       # Length of the stream
        self.ground_truth = None
        self.overwrite = overwrite


    def _generate_item(self):
        """
        Generate a pair of samples i and j.

        Returns:
            i (int or float): a sample in the stream that follows distribution X.
            j (int or float): a sample in the stream that follows distribution Y.
        """
        pass



    def write_file(self, file_name: str):
        """
        Write the stream to a file. For each line, there should be 2 numbers, which are
        the samples from X and Y distributions, respectively.

        Args:
            file_name (string): the path to the data file.
        """
        # Write data
        if os.path.exists(TEST_DATA_DIR + file_name + ".txt") and not self.overwrite:
            warnings.warn(f"The path {file_name}.txt already exists in {TEST_DATA_DIR}. Skipping the function.")
            return
        with open(TEST_DATA_DIR + file_name + ".txt", 'w') as f:
            for _ in range(self.N):
                i, j = self._generate_item()
                f.write(str(i) + " " + str(j) + "\n")
        
        with open(GROUND_TRUTH_DIR + file_name + ".pickle", 'wb') as p:
            pickle.dump(self.ground_truth, p, protocol=pickle.HIGHEST_PROTOCOL)



def check_error(estimator: Estimator, file_name: str, metric: str = "l2"):
    """
    Evaluate the estimator using a stream, and check on the corresponding stream.

    Args:
        estimator (utils.Estimator): An estimator to be tested.
        file_name (string): A path to the file in which the stream is stored.
        metric (string): one of the metrics indicated above.
    """
    # with open(GROUND_TRUTH_DIR + file_name + ".pickle", 'rb') as p:
    #     ground_truth = pickle.load(p)
    
    with open(ANSWER_DIR + file_name + ".pickle", 'rb') as p:
        answer = pickle.load(p)
    
    # assert metric in ground_truth, f"the metric {metric} is not computed in the"\
    #     f"specified ground truth file {GROUND_TRUTH_DIR}{file_name}.pickle."
    assert metric in answer, f"the metric {metric} is not computed in the"\
        f"specified ground truth file {ANSWER_DIR}{file_name}.pickle."
    
    print(answer)
    
    estimator.read_from_file(file_name)
    res = estimator.compute()
    print("Estimator result:", res)
    print("Answer:", answer[metric])

    if metric != "independent":
        # Return multiplicative error
        return abs(1 - res/answer[metric])
    else:
        # Return 0 if correctly identified independence/dependence and 1 otherwise
        return int(res != answer[metric])
   

"""
Utils used to generate random hash functions.

hash_primes: list of prime numbers that are good to generate random hash functions.
_choose_prime: A function that returns a prime number larger than a number in the prime list.

Reference: https://gcc.gnu.org/onlinedocs/gcc-4.8.5/libstdc++/api/a00971_source.html
"""
hash_primes = [
    5,
    11,
    23,
    47,
    97,
    199,
    409,
    823,
    1741,
    3469,
    6949,
    14033,
    28411,
    57557,
    116731,
    236897,
    480881,
    976369,
    1982627,
    4026031,
    8175383,
    16601593,
    33712729,
    68460391,
    139022417,
    282312799,
    573292817,
    1164186217,
    2364114217,
    4294967291,
    8589934583,
    17179869143,
    34359738337,
    68719476731,
    137438953447,
    274877906899,
    549755813881,
    1099511627689,
    2199023255531,
    4398046511093,
    8796093022151,
    17592186044399,
    35184372088777,
    70368744177643,
    140737488355213,
    281474976710597,
    562949953421231,
    1125899906842597,
    2251799813685119,
    4503599627370449,
    9007199254740881,
    18014398509481951,
    36028797018963913,
    72057594037927931,
    144115188075855859,
    288230376151711717,
    576460752303423433,
    1152921504606846883,
    2305843009213693951,
    4611686018427387847,
    9223372036854775783,
    18446744073709551557
]


def _choose_prime(M):
    """
    Return a prime number larger than M in the prime list.
    """
    for i in range(len(hash_primes)):
        if hash_primes[i] > M:
            return hash_primes[i]
    
    return hash_primes[-1]