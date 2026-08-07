# Validation fixtures

`generate_fixtures.py` uses scikit-learn to produce reference input/output
pairs for `ml-core`'s `LinearRegression` and `KMeans` to be checked against.
The generated CSVs live in `fixtures/` and are committed to the repo, so the
C++ test suite (`tests/test_validation.cpp`) has no Python or scikit-learn
dependency at build/test time - it just reads the checked-in fixtures.

Re-run the generator only when the fixtures need to change (e.g. a new
scikit-learn version changes its fit output, or more coverage is wanted):

```sh
python3 -m venv .venv && source .venv/bin/activate  # optional
pip install -r validation/requirements.txt
python3 validation/generate_fixtures.py
```

Then re-run the C++ tests and commit the updated `fixtures/*.csv` alongside
any resulting tolerance adjustments.

## Fixtures

- `regression_X.csv` / `regression_y.csv`: a synthetic linear dataset
  (`y = 4.2 + 2.5*x0 - 1.3*x1 + 0.7*x2` plus small Gaussian noise).
- `regression_expected.csv`: intercept and coefficients from fitting
  scikit-learn's `LinearRegression` on that data. The C++ test asserts
  `ml::LinearRegression::fitNormalEquation` matches these within 1e-6.
- `kmeans_X.csv`: points drawn from four well-separated Gaussian blobs
  (`sklearn.datasets.make_blobs`), spaced far apart relative to their spread
  so the inertia-minimizing partition is essentially unique.
- `kmeans_expected_labels.csv` / `kmeans_expected_meta.csv`: labels and
  inertia from scikit-learn's `KMeans` fit on that data. The C++ test
  matches ml-core's clustering against these after finding the best label
  permutation (cluster numbering is arbitrary between implementations) and
  asserts high assignment agreement plus close inertia.
