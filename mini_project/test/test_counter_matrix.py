from mini_project.algorithms.counter_matrix import L2Estimator, L1Estimator
from mini_project.utils import check_error

TEST_FILE = "sample"


def test_l2_estimator():
    """
    Test function for l2 estimator.
    """
    estimator = L2Estimator(2, 100)
    (res, answer, error) = check_error(estimator, TEST_FILE, metric="l2")
    print("multiplicative error:", error)


def test_l1_estimator():
    estimator = L1Estimator(100, 100)
    (res, answer, error) = check_error(estimator, TEST_FILE, metric="l1")
    print("multiplicative error:", error)


if __name__ == "__main__":
    test_l1_estimator()
    test_l2_estimator()
