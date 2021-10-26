from mini_project.algorithms.counter_matrix import CounterMatrix

TEST_FILE = "mini_project/test/test_data/simple_dataset.txt"
L2_DIFFERENCE = 0.12
L1_DIFFERENCE = 0.24

def test_l2_estimator():
    """
    Test function for l2 estimator.
    """
    estimator = CounterMatrix(10, 2)
    estimator.read_from_file(TEST_FILE)
    l2_estimate = estimator.compute()
    print(l2_estimate)
    print("multiplicative error:", abs(1 - l2_estimate / L2_DIFFERENCE))

if __name__ == "__main__":
    test_l2_estimator()