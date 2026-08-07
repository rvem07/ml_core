# ml-core

A small, dependency-free C++17 machine learning library: a contiguous
row-major `Matrix`, `LinearRegression` (normal equations and gradient
descent), and `KMeans` (k-means++ init, Lloyd's iteration).

## Building

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`validation/` holds scikit-learn-generated reference fixtures that
`tests/test_validation.cpp` checks ml-core's output against; see
[validation/README.md](validation/README.md) to regenerate them.

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
