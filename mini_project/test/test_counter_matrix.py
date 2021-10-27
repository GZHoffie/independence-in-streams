from mini_project.algorithms.counter_matrix import L2Estimator
from mini_project.utils import check_error

TEST_FILE = "sample"

def test_l2_estimator():
    """
    Test function for l2 estimator.
    """
    estimator = L2Estimator(10, 10)
    error = check_error(estimator, TEST_FILE)
    print("multiplicative error:", error)

if __name__ == "__main__":
    test_l2_estimator()