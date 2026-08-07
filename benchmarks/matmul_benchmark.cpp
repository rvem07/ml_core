// Compares ml::Matrix's contiguous row-major matrix multiply against a
// naive std::vector<std::vector<double>> baseline, to demonstrate the
// cache-locality benefit of a single contiguous allocation.
//
// Both implementations use the same i-k-j loop order and operate on the
// same underlying values, so storage layout - one contiguous heap block
// (data[i * cols + j]) versus one independent heap allocation per row - is
// the only thing under test, not the access pattern.

#include "ml/Matrix.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using NaiveMatrix = std::vector<std::vector<double>>;

std::vector<double> randomFlat(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> flat(n * n);
    for (double& v : flat) v = dist(rng);
    return flat;
}

NaiveMatrix toNaiveMatrix(const std::vector<double>& flat, std::size_t n) {
    NaiveMatrix m(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i][j] = flat[i * n + j];
        }
    }
    return m;
}

ml::Matrix toMlMatrix(const std::vector<double>& flat, std::size_t n) {
    ml::Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m(i, j) = flat[i * n + j];
        }
    }
    return m;
}

// Deliberately mirrors ml::Matrix::operator*'s loop order (see
// src/Matrix.cpp) so the comparison isolates storage layout.
NaiveMatrix matmulNaive(const NaiveMatrix& a, const NaiveMatrix& b) {
    const std::size_t n = a.size();
    NaiveMatrix result(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const double a_ik = a[i][k];
            for (std::size_t j = 0; j < n; ++j) {
                result[i][j] += a_ik * b[k][j];
            }
        }
    }
    return result;
}

double checksum(const NaiveMatrix& m) {
    double sum = 0.0;
    for (const auto& row : m) {
        for (double v : row) sum += v;
    }
    return sum;
}

double checksum(const ml::Matrix& m) {
    double sum = 0.0;
    for (double v : m.data()) sum += v;
    return sum;
}

// Runs `fn` `reps` times and returns the minimum wall-clock time observed,
// which is the standard way to reduce noise from OS scheduling jitter in
// microbenchmarks (the fastest run is the closest to steady-state cost;
// slower runs only ever reflect extra interference, never a "truer" cost).
double minSeconds(int reps, const std::function<void()>& fn) {
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i < reps; ++i) {
        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();
        best = std::min(best, std::chrono::duration<double>(end - start).count());
    }
    return best;
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const std::vector<std::size_t> sizes = {64, 128, 256, 384, 512, 768, 1024};
    constexpr int kReps = 5;

    std::printf("%6s  %16s  %16s  %10s  %s\n", "n", "vector<vector>", "ml::Matrix", "speedup",
                "check");
    std::printf("%6s  %16s  %16s  %10s  %s\n", "", "(ms)", "(ms)", "(x)", "");

    for (std::size_t n : sizes) {
        const std::vector<double> flatA = randomFlat(n, rng);
        const std::vector<double> flatB = randomFlat(n, rng);

        const NaiveMatrix naiveA = toNaiveMatrix(flatA, n);
        const NaiveMatrix naiveB = toNaiveMatrix(flatB, n);
        const ml::Matrix mlA = toMlMatrix(flatA, n);
        const ml::Matrix mlB = toMlMatrix(flatB, n);

        NaiveMatrix naiveResult;
        ml::Matrix mlResult;

        // One untimed warm-up call per size to absorb first-touch page
        // faults and allocator warm-up before the timed reps.
        naiveResult = matmulNaive(naiveA, naiveB);
        mlResult = mlA * mlB;

        const double naiveSeconds =
            minSeconds(kReps, [&]() { naiveResult = matmulNaive(naiveA, naiveB); });
        const double mlSeconds = minSeconds(kReps, [&]() { mlResult = mlA * mlB; });

        const bool matches = std::abs(checksum(naiveResult) - checksum(mlResult)) < 1e-6 * n * n;

        std::printf("%6zu  %16.3f  %16.3f  %10.2f  %s\n", n, naiveSeconds * 1000.0,
                    mlSeconds * 1000.0, naiveSeconds / mlSeconds, matches ? "ok" : "MISMATCH");
    }

    return 0;
}
