# Identifying Correlation in Stream of Samples

This is the repository containing work of CS5234 (Algorithm at Scale) mini project at NUS Computing. In this project, we try to identify the correlation between two random variables given their samples in a data stream.

We implement two algorithms here:

- Sketching of Sketches algorithm ([Reference](https://people.cs.umass.edu/~mcgregor/papers/08-soda.pdf))
- Counter Matrix algorithm (new and proposed by us)

## Installation

To use our package, you need to have python 3.7.

```bash
git clone https://github.com/GZHoffie/CS5234-mini-project.git
cd CS5234-mini-project
pip install -r requirements.txt
```

## Generating Stream of Samples

You can use the following python code to generate random sample streams of length `N` from two discrete distributions, both in the range `[1, n]`. You can specify whether the two distributions are independent or not by setting the parameter `independent`.

```python
from mini_project.data import DiscreteSampleGenerator

generator = DiscreteSampleGenerator(n=1000, N=100000, independent=False)
generator.write_file("sample")
```

Three files will be generated in `mini_project/test` directory,

- `test_data/sample.txt`, containing the data stream. Each line of the file contains two integers, indicating the samples drawn from the distributions.
- `ground_truth/sample.pickle`, storing a `dict` of `l1`-difference, `l2`-difference and independence of the data, based on the distribution that is used to generate samples.
- `answer/sample.pickle`, storing a `dict` of `l1`-difference, `l2`-difference and independence of the data, based on the actual generated data stream. It may differ from ground_truth, especially when `n` is large or `N` is small.

## Running Algorithms

We also provide utilities to easily run algorithms and check the correctness. For example, to use the counter matrix algorithm,

```python
from mini_project.algorithms.counter_matrix import L2Estimator
from mini_project.utils import check_error

TEST_FILE = "sample"

estimator = L2Estimator(10, 5)
error = check_error(estimator, TEST_FILE)
print("multiplicative error:", error)
```

## References

[1] Noga Alon, Yossi Matias, and Mario Szegedy. The space complexity of approximating the frequency moments.*Journal of Computer and System Sciences*, 58(1):137–147, 1999.

[2] Piotr Indyk. Stable distributions, pseudorandom generators, embeddings, and data stream computation.*Journal of the ACM (JACM)*, 53(3):307–323, 2006.

[3] Piotr Indyk and Andrew McGregor. Declaring independence via the sketching of sketches. pages 737–745, 01 2008.

[4] Minhaz Fahim Zibran.  Chi-squared test of independence. *Department of Computer Science, University of Calgary, Alberta, Canada*, pages 1–7, 2007.
