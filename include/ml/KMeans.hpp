#pragma once

// K-means clustering on top of ml::Matrix. Rows of the input Matrix are
// samples, columns are features. Centroids are likewise stored as a
// k x n_features Matrix.
//
// Initialization uses k-means++ (Arthur & Vassilvitskii, 2007): the first
// centroid is picked uniformly at random, and each subsequent centroid is
// picked with probability proportional to its squared distance from the
// nearest centroid already chosen. This spreads the initial centroids out
// and empirically converges faster/more reliably than picking k random
// points.
//
// Fitting then runs Lloyd's iteration (alternating assignment and centroid
// update) until either the largest centroid movement between iterations
// drops below `epsilon`, or `maxIterations` is reached.

#include <cstddef>
#include <random>
#include <vector>

#include "ml/Matrix.hpp"

namespace ml {

class KMeans {
public:
    // k: number of clusters (must be >= 1).
    // maxIterations: hard cap on Lloyd iterations.
    // epsilon: convergence threshold on the largest per-centroid movement
    //          (in Euclidean distance) between successive iterations.
    // seed: seeds the k-means++ initialization RNG; fix it for reproducible
    //       fits (as the tests do).
    explicit KMeans(std::size_t k, std::size_t maxIterations = 300, double epsilon = 1e-4,
                     unsigned int seed = std::random_device{}());

    // Runs k-means++ initialization followed by Lloyd's iteration on X.
    // Throws std::invalid_argument if X has zero rows or k exceeds the
    // number of samples.
    void fit(const Matrix& X);

    // Assigns each row of X to its nearest centroid. Requires fit() to have
    // been called first; throws std::logic_error otherwise, and
    // std::invalid_argument if X's column count doesn't match the fit.
    std::vector<std::size_t> predict(const Matrix& X) const;

    // Cluster label (0..k-1) assigned to each row of the data passed to fit().
    const std::vector<std::size_t>& labels() const;

    // k x n_features matrix of final centroids.
    const Matrix& centroids() const;

    // Sum of squared distances from each training point to its assigned
    // centroid (a.k.a. within-cluster sum of squares / WCSS).
    double inertia() const;

    // Number of Lloyd iterations actually run (<= maxIterations()).
    std::size_t iterationsRun() const noexcept { return iterationsRun_; }

    std::size_t k() const noexcept { return k_; }
    std::size_t maxIterations() const noexcept { return maxIterations_; }
    double epsilon() const noexcept { return epsilon_; }
    bool isFitted() const noexcept { return fitted_; }

private:
    std::size_t k_;
    std::size_t maxIterations_;
    double epsilon_;
    std::mt19937 rng_;

    Matrix centroids_;
    std::vector<std::size_t> labels_;
    double inertia_ = 0.0;
    std::size_t iterationsRun_ = 0;
    bool fitted_ = false;

    Matrix initializeCentroidsKMeansPlusPlus(const Matrix& X);

    // Index of the centroid nearest to X's given row, optionally reporting
    // the squared distance to it.
    static std::size_t nearestCentroid(const Matrix& X, std::size_t row, const Matrix& centroids,
                                        double* outDistSq = nullptr);

    void requireFitted() const;
};

}  // namespace ml
