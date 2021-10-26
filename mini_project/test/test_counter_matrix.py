from mini_project.algorithms.counter_matrix import CounterMatrix
from mini_project.utils import check_error

TEST_FILE = "sample"
L2_DIFFERENCE = 0.12
L1_DIFFERENCE = 0.24

def test_l2_estimator():
    """
    Test function for l2 estimator.
    """
    estimator = CounterMatrix(100)
    error = check_error(estimator, TEST_FILE)
    print("multiplicative error:", error)

if __name__ == "__main__":
    test_l2_estimator()