from mini_project.algorithms.sketching_sketches import L2Estimator

TEST_FILE = "mini_project/test/test_data/simple_dataset.txt"
L2_DIFFERENCE = 0.12
L1_DIFFERENCE = 0.24

def test_l2_estimator():
    """
    Test function for l2 estimator.
    """
    estimator = L2Estimator(100, 100)
    estimator.read_from_file(TEST_FILE)
    l2_estimate = estimator.compute()
    print("multiplicative error:", abs(1 - l2_estimate / L2_DIFFERENCE))

if __name__ == "__main__":
    test_l2_estimator()