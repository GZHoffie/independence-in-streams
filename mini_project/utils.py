class Estimator:
    """
    Base class for the correlation estimators.

    Args:
        input_type (class): Usually int or float, indicating the type of input.
    """
    def __init__(self, input_type=int) -> None:
        self.result = 0                # Storing the estimated metric
        self.input_type = input_type   # Input type (int or float)
    

    def _read_item(self, i, j):
        """
        Read a pair of samples i and j from the stream, and store the information.

        Args:
            i (int or float): a sample in the stream that follows distribution X.
            j (int or float): a sample in the stream that follows distribution Y.
        """
        pass


    def read_from_file(self, file_name: str):
        """
        Read the stream from a file. For each line, there should be 2 numbers, which are
        the samples from X and Y distributions, respectively.

        Args:
            file_name (string): the path to the data file.
        """
        with open(file_name, 'r') as f:
            for line in f:
                i, j = [self.input_type(x) for x in line.split()]
                self._read_item(i, j)


    def compute(self) -> float:
        """
        Compute the metric given the data.

        Returns:
            The estimated value of the metric specified.
        """
        return self.result

    def reset(self):
        """
        Clear out all the stored data and be ready for the next stream.
        """
        pass


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