from mini_project.utils import Estimator
import numpy as np
from typing import List
from scipy.stats import chi2

SUPPORTED_METRICS = ["l1", "l2", "chisq", "independent"]

class ExactEstimator(Estimator):
    def __init__(self, n: int, metric: List[str] = ["l2"]) -> None:
        """
        Creator for the exact estimator.

        Args:
            n (int): range of the samples should be [1, n].
            metric (str or List[str]): type of metric to be estimated.
        """
        super().__init__(input_type=int)
        self.C = np.zeros((n, n))
        self.n = n
        if isinstance(metric, str):
            metric = [metric]
        for m in metric:
            assert m in SUPPORTED_METRICS, f"metric {m} is not supported."
        self.metric = metric

    
    def _read_item(self, i, j):
        super()._read_item(i, j)
        self.C[i-1, j-1] += 1

    def compute(self) -> float:
        p_x = np.sum(self.C, axis=1, keepdims=True)
        p_y = np.sum(self.C, axis=0, keepdims=True)
        observed = self.C
        expected = np.dot(p_x, p_y) / self.N
        
        res = []
        for m in self.metric:
            if m == "l2":
                res.append(np.linalg.norm(observed / self.N - expected / self.N))
            elif m == "l1":
                res.append(np.sum(np.absolute(observed / self.N - expected / self.N)))
            elif m == "chisq" or m == "independent":
                chisq_stat = np.sum((observed - expected) ** 2 / expected)
                deg_of_freedom = (self.n - 1) ** 2
                p_value = 1 - chi2.cdf(chisq_stat, deg_of_freedom)
                if m == "chisq":
                    res.append(p_value)
                else:
                    res.append(p_value > 0.05)
        
        if len(res) == 0:
            return res[0]
        else:
            return res