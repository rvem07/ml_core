# ml-core

A small, dependency-free C++17 machine learning library:

- **`ml::Matrix`** - a dense matrix backed by a single contiguous
  `std::vector<double>` in row-major order, plus the linear-algebra
  operations the rest of the library is built on.
- **`ml::LinearRegression`** - ordinary least squares, fit either in
  closed form (normal equations) or iteratively (batch gradient descent).
- **`ml::KMeans`** - k-means++ initialization followed by Lloyd's
  iteration, with inertia reporting.

## Building

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## API

### `ml::Matrix` ([include/ml/Matrix.hpp](include/ml/Matrix.hpp))

Storage is one `std::vector<double>` of length `rows * cols`, indexed as
`data[i * cols + j]` - never a vector of vectors. This is what makes the
matmul benchmark below meaningful: a single allocation means predictable,
sequential memory layout instead of `rows` independent heap blocks.

| Member | Description |
| --- | --- |
| `Matrix(rows, cols, fill = 0.0)` | Construct, every entry set to `fill`. |
| `Matrix(rows, cols, {v0, v1, ...})` | Construct from a flat row-major initializer list. |
| `Matrix{{1, 2}, {3, 4}}` | Construct from a nested initializer list, one inner list per row. |
| `Matrix::identity(n)` | The `n x n` identity matrix. |
| `rows()`, `cols()` | Dimensions. |
| `operator()(i, j)` | Bounds-checked element access (const and non-const). |
| `data()` | Read-only access to the underlying contiguous buffer. |
| `operator+`, `operator-`, `+=`, `-=` | Elementwise addition/subtraction. |
| `operator*(Matrix)` | Matrix product. |
| `operator*(vector<double>)` | Matrix-vector product. |
| `operator*(double)`, `*=` | Scalar multiply (both operand orders). |
| `transpose()` | Returns the transpose. |
| `operator==`, `!=` | Elementwise comparison with a small tolerance. |
| `Matrix::solve(A, b)` | Solves square `Ax = b` via Gaussian elimination with partial pivoting. |
| `Matrix::solveLeastSquares(A, b)` | Solves `Ax ~= b` via the normal equations `(AᵀA)x = Aᵀb`; works for over/under-determined `A`. |

### `ml::LinearRegression` ([include/ml/LinearRegression.hpp](include/ml/LinearRegression.hpp))

Fits `y = intercept + X * coefficients`, prepending the intercept column
internally so callers pass the raw feature matrix.

| Member | Description |
| --- | --- |
| `fitNormalEquation(X, y)` | Closed-form fit via `Matrix::solveLeastSquares`. |
| `fitGradientDescent(X, y, learningRate = 0.01, iterations = 10000)` | Iterative batch gradient descent from `theta = 0`. |
| `predict(X)` | Predicted `y` for each row of `X`. |
| `mse(X, y)`, `rSquared(X, y)` | Mean squared error / R² on `(X, y)`. |
| `intercept()`, `coefficients()` | The fitted parameters. |
| `isFitted()` | Whether a fit has been run yet. |

### `ml::KMeans` ([include/ml/KMeans.hpp](include/ml/KMeans.hpp))

Rows of the input `Matrix` are samples, columns are features.

| Member | Description |
| --- | --- |
| `KMeans(k, maxIterations = 300, epsilon = 1e-4, seed = random)` | Configure cluster count, iteration cap, convergence threshold, and RNG seed. |
| `fit(X)` | k-means++ init + Lloyd's iteration; converges when the largest centroid movement drops below `epsilon`, or after `maxIterations`. |
| `predict(X)` | Nearest-centroid label for each row of (possibly new) `X`. |
| `labels()` | Cluster label (0..k-1) for each row of the training data. |
| `centroids()` | The final `k x n_features` centroid matrix. |
| `inertia()` | Sum of squared distances from each training point to its centroid (WCSS). |
| `iterationsRun()` | Actual number of Lloyd iterations run (`<= maxIterations`). |

## Validating against scikit-learn

`validation/generate_fixtures.py` uses scikit-learn to fit the same kind of
model (`LinearRegression`, `KMeans`) on synthetic data and writes the
inputs plus scikit-learn's outputs to checked-in CSVs under
`validation/fixtures/`. `tests/test_validation.cpp` loads those fixtures
and asserts that ml-core reproduces them:

- **Regression**: `fitNormalEquation`'s intercept/coefficients match
  scikit-learn's fitted `LinearRegression` to within `1e-6` (both solve the
  same OLS problem in closed form, so they should agree near machine
  precision); `fitGradientDescent` is checked too, at a looser `1e-3`
  tolerance since it's iterative.
- **Clustering**: `KMeans` is fit on scikit-learn's `make_blobs` data
  (well-separated blobs, so the inertia-minimizing partition is
  essentially unique) and compared against scikit-learn's `KMeans.labels_`
  after searching over all `k!` label permutations (cluster numbering is
  arbitrary between implementations) - the test asserts high agreement
  under the best permutation, plus close inertia.

Because the fixtures are committed, this comparison runs with the rest of
`ctest` and needs no Python/scikit-learn at build or test time. See
[validation/README.md](validation/README.md) for how to regenerate them.

## AddressSanitizer build

Valgrind isn't available on Apple Silicon, so memory-error and leak
checking here is done with AddressSanitizer (ASan) instead. The
`ENABLE_ASAN` CMake option compiles every target with
`-fsanitize=address -fno-omit-frame-pointer -g`:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

### macOS / Apple Silicon toolchain note

On this machine (macOS 26.5.2, AppleClang 17.0.0 from Xcode Command Line
Tools), the ASan runtime bundled with the system's `clang++` hangs forever
during its own startup (an infinite recursion inside
`AsanInitFromRtl`/`InitializeShadowMemory` - reproduces even for a trivial
`int main(){}` compiled with `-fsanitize=address`, so it's an OS/toolchain
incompatibility, not an ml-core bug). If you hit the same hang, build with
Homebrew's LLVM instead, which ships its own, unaffected ASan runtime:

```sh
brew install llvm
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON \
    -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
    -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

LeakSanitizer (`ASAN_OPTIONS=detect_leaks=1`) is a silent no-op on macOS
regardless of toolchain - this is a documented upstream platform
limitation of compiler-rt on Darwin, not specific to this project. For
leak checking on macOS, use Apple's own `leaks` tool against a plain
(non-ASan) build instead - `leaks` and ASan can't inspect the same
process, so run it against `build/`, not `build-asan/`:

```sh
leaks --atExit -- ./build/tests/ml_core_tests
```

### Current result

Clean on both checks: **41 test cases / 257 assertions pass under ASan**
(heap/stack buffer overflows, use-after-free, and invalid reads/writes all
instrumented) with no reported errors, and **`leaks --atExit` reports 0
leaks for 0 total leaked bytes**.

## Benchmark: contiguous storage vs. vector-of-vectors

`benchmarks/matmul_benchmark.cpp` compares `ml::Matrix`'s matrix multiply
against a `std::vector<std::vector<double>>` baseline. Both implementations
use the *same* i-k-j loop order over the *same* values, so storage layout -
one contiguous allocation (`data[i * cols + j]`) versus one independent
heap allocation per row - is the only variable under test, not loop order
or access pattern. A checksum comparison confirms both produce the same
result each run.

Build and run it (Release mode matters here - Debug's lack of
optimization would swamp the effect being measured):

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DML_CORE_BUILD_BENCHMARKS=ON
cmake --build build-release -j
./build-release/benchmarks/ml_core_matmul_benchmark
```

### Results

Measured on an Apple Silicon Mac (arm64, Release build):

```
     n    vector<vector>        ml::Matrix     speedup  check
                    (ms)              (ms)         (x)
    64             0.059             0.047        1.26  ok
   128             0.371             0.353        1.05  ok
   256             2.760             2.424        1.14  ok
   384             6.835             6.458        1.06  ok
   512            15.840            15.699        1.01  ok
   768            53.843            52.546        1.02  ok
  1024           127.166           125.234        1.02  ok
```

`ml::Matrix` is never slower and is measurably faster at the smaller
sizes (~5-25%), converging toward parity by n=1024. That gap is real but
more modest than the "vector-of-vectors is disastrous for cache locality"
folklore suggests - and it's worth being honest about why: Apple Silicon's
unified memory architecture (a large shared System Level Cache and an
aggressive prefetcher) hides a lot of the penalty that scattered
allocations would normally cause, and this benchmark's rows happen to be
allocated back-to-back, which lets the allocator place them near each
other in practice. Re-running with each row deliberately allocated out of
index order (simulating the fragmentation a `vector<vector<double>>`
accumulates after real-world resizing/rebuilding) cost only another
5-17% in exploratory testing - a real cost, but still not the multi-x
gap you'd see on hardware with smaller caches or a less forgiving
allocator.

The bigger, more consistent wins for contiguous storage are structural
rather than raw speed: one allocation instead of `rows`, no per-row
pointer indirection, no risk of the layout degrading further after
resizes, and a data layout that's ready to hand off to BLAS-style or
SIMD-vectorized code as-is - none of which a `vector<vector<double>>`
gives you regardless of how the cache numbers land on a given machine.
