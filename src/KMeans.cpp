#include "ml/KMeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ml {

namespace {

// Squared Euclidean distance between row `aRow` of A and row `bRow` of B.
// A and B must have the same column count.
double squaredDistance(const Matrix& A, std::size_t aRow, const Matrix& B, std::size_t bRow) {
    double sum = 0.0;
    for (std::size_t j = 0; j < A.cols(); ++j) {
        const double diff = A(aRow, j) - B(bRow, j);
        sum += diff * diff;
    }
    return sum;
}

}  // namespace

KMeans::KMeans(std::size_t k, std::size_t maxIterations, double epsilon, unsigned int seed)
    : k_(k), maxIterations_(maxIterations), epsilon_(epsilon), rng_(seed) {
    if (k_ == 0) {
        throw std::invalid_argument("KMeans: k must be at least 1");
    }
}

void KMeans::requireFitted() const {
    if (!fitted_) {
        throw std::logic_error("KMeans: model has not been fitted yet");
    }
}

std::size_t KMeans::nearestCentroid(const Matrix& X, std::size_t row, const Matrix& centroids,
                                     double* outDistSq) {
    std::size_t best = 0;
    double bestDistSq = squaredDistance(X, row, centroids, 0);
    for (std::size_t c = 1; c < centroids.rows(); ++c) {
        const double d = squaredDistance(X, row, centroids, c);
        if (d < bestDistSq) {
            bestDistSq = d;
            best = c;
        }
    }
    if (outDistSq != nullptr) {
        *outDistSq = bestDistSq;
    }
    return best;
}

Matrix KMeans::initializeCentroidsKMeansPlusPlus(const Matrix& X) {
    const std::size_t n = X.rows();
    const std::size_t d = X.cols();
    Matrix centroids(k_, d);

    std::uniform_int_distribution<std::size_t> pickUniform(0, n - 1);
    std::size_t firstIdx = pickUniform(rng_);
    for (std::size_t j = 0; j < d; ++j) {
        centroids(0, j) = X(firstIdx, j);
    }

    // nearestDistSq[i] tracks each point's squared distance to the closest
    // centroid chosen so far.
    std::vector<double> nearestDistSq(n, std::numeric_limits<double>::infinity());

    for (std::size_t c = 1; c < k_; ++c) {
        for (std::size_t i = 0; i < n; ++i) {
            const double d2 = squaredDistance(X, i, centroids, c - 1);
            if (d2 < nearestDistSq[i]) {
                nearestDistSq[i] = d2;
            }
        }

        const double total = std::accumulate(nearestDistSq.begin(), nearestDistSq.end(), 0.0);
        std::size_t chosen;
        if (total <= 0.0) {
            // All remaining points coincide with an already-chosen centroid;
            // fall back to a uniform pick so we still get k distinct rows.
            chosen = pickUniform(rng_);
        } else {
            std::uniform_real_distribution<double> pickWeighted(0.0, total);
            double target = pickWeighted(rng_);
            chosen = n - 1;  // fallback for floating-point edge cases
            double cumulative = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                cumulative += nearestDistSq[i];
                if (cumulative >= target) {
                    chosen = i;
                    break;
                }
            }
        }
        for (std::size_t j = 0; j < d; ++j) {
            centroids(c, j) = X(chosen, j);
        }
    }

    return centroids;
}

void KMeans::fit(const Matrix& X) {
    const std::size_t n = X.rows();
    const std::size_t d = X.cols();
    if (n == 0) {
        throw std::invalid_argument("KMeans::fit: X must have at least one row");
    }
    if (k_ > n) {
        throw std::invalid_argument("KMeans::fit: k cannot exceed the number of samples");
    }

    centroids_ = initializeCentroidsKMeansPlusPlus(X);
    labels_.assign(n, 0);

    std::size_t iter = 0;
    for (; iter < maxIterations_; ++iter) {
        // Assignment step.
        for (std::size_t i = 0; i < n; ++i) {
            labels_[i] = nearestCentroid(X, i, centroids_);
        }

        // Update step: mean of each cluster's assigned points.
        Matrix newCentroids(k_, d, 0.0);
        std::vector<std::size_t> counts(k_, 0);
        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t c = labels_[i];
            counts[c] += 1;
            for (std::size_t j = 0; j < d; ++j) {
                newCentroids(c, j) += X(i, j);
            }
        }
        for (std::size_t c = 0; c < k_; ++c) {
            if (counts[c] > 0) {
                for (std::size_t j = 0; j < d; ++j) {
                    newCentroids(c, j) /= static_cast<double>(counts[c]);
                }
            } else {
                // Empty cluster: keep its previous centroid rather than
                // producing NaN or collapsing it into another cluster.
                for (std::size_t j = 0; j < d; ++j) {
                    newCentroids(c, j) = centroids_(c, j);
                }
            }
        }

        // Convergence check: largest centroid movement (Euclidean distance).
        double maxShiftSq = 0.0;
        for (std::size_t c = 0; c < k_; ++c) {
            maxShiftSq = std::max(maxShiftSq, squaredDistance(newCentroids, c, centroids_, c));
        }

        centroids_ = std::move(newCentroids);

        if (std::sqrt(maxShiftSq) < epsilon_) {
            ++iter;
            break;
        }
    }
    iterationsRun_ = iter;

    // Final assignment and inertia against the converged centroids.
    inertia_ = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double distSq = 0.0;
        labels_[i] = nearestCentroid(X, i, centroids_, &distSq);
        inertia_ += distSq;
    }

    fitted_ = true;
}

std::vector<std::size_t> KMeans::predict(const Matrix& X) const {
    requireFitted();
    if (X.cols() != centroids_.cols()) {
        throw std::invalid_argument("KMeans::predict: X's column count does not match the fitted model");
    }
    std::vector<std::size_t> result(X.rows());
    for (std::size_t i = 0; i < X.rows(); ++i) {
        result[i] = nearestCentroid(X, i, centroids_);
    }
    return result;
}

const std::vector<std::size_t>& KMeans::labels() const {
    requireFitted();
    return labels_;
}

const Matrix& KMeans::centroids() const {
    requireFitted();
    return centroids_;
}

double KMeans::inertia() const {
    requireFitted();
    return inertia_;
}

}  // namespace ml
