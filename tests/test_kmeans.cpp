#include "doctest.h"

#include "ml/KMeans.hpp"
#include "ml/Matrix.hpp"

#include <stdexcept>
#include <vector>

using ml::KMeans;
using ml::Matrix;

namespace {

// Three well-separated 2D clusters, 4 points each, arranged as unit-square
// corners around (0,0), (10,0), and (5,10). Nearest true centers are ~7+
// apart while the farthest two points within a cluster are only sqrt(2)
// apart, so k-means should always recover this exact partition regardless
// of k-means++ seed.
//   rows 0-3: cluster near (0, 0)   -> true centroid (0.5, 0.5)
//   rows 4-7: cluster near (10, 0)  -> true centroid (10.5, 0.5)
//   rows 8-11: cluster near (5, 10) -> true centroid (5.5, 10.5)
Matrix sampleData() {
    return Matrix{
        {0, 0}, {1, 0}, {0, 1}, {1, 1},
        {10, 0}, {11, 0}, {10, 1}, {11, 1},
        {5, 10}, {6, 10}, {5, 11}, {6, 11},
    };
}

// Each cluster is a unit square, so each point sits sqrt(0.5) from its
// centroid: squared distance 0.5 per point, 4 points per cluster, 3
// clusters => exact total inertia of 6.0.
constexpr double kExpectedInertia = 6.0;

bool sameCluster(const std::vector<std::size_t>& labels, std::size_t i, std::size_t j) {
    return labels[i] == labels[j];
}

void checkPartitionMatchesTrueClusters(const std::vector<std::size_t>& labels) {
    REQUIRE(labels.size() == 12);
    // Within each true cluster, all labels must agree...
    for (std::size_t base : {0u, 4u, 8u}) {
        CHECK(sameCluster(labels, base, base + 1));
        CHECK(sameCluster(labels, base, base + 2));
        CHECK(sameCluster(labels, base, base + 3));
    }
    // ...and across true clusters, labels must differ.
    CHECK_FALSE(sameCluster(labels, 0, 4));
    CHECK_FALSE(sameCluster(labels, 0, 8));
    CHECK_FALSE(sameCluster(labels, 4, 8));
}

}  // namespace

TEST_CASE("KMeans: k-means++ + Lloyd's iteration recovers well-separated clusters") {
    KMeans model(3, /*maxIterations=*/300, /*epsilon=*/1e-6, /*seed=*/42);
    model.fit(sampleData());

    CHECK(model.isFitted());
    checkPartitionMatchesTrueClusters(model.labels());
    CHECK(model.inertia() == doctest::Approx(kExpectedInertia).epsilon(1e-6));

    const Matrix& centroids = model.centroids();
    CHECK(centroids.rows() == 3);
    CHECK(centroids.cols() == 2);

    // Whichever centroid row a given true cluster landed on, it should sit
    // at that cluster's hand-computed mean.
    const std::vector<std::size_t>& labels = model.labels();
    const std::vector<std::pair<double, double>> expectedCenters = {
        {0.5, 0.5}, {10.5, 0.5}, {5.5, 10.5},
    };
    const std::vector<std::size_t> representative = {0, 4, 8};
    for (std::size_t g = 0; g < 3; ++g) {
        const std::size_t label = labels[representative[g]];
        CHECK(centroids(label, 0) == doctest::Approx(expectedCenters[g].first).epsilon(1e-6));
        CHECK(centroids(label, 1) == doctest::Approx(expectedCenters[g].second).epsilon(1e-6));
    }
}

TEST_CASE("KMeans: inertia and partition are stable across different k-means++ seeds") {
    for (unsigned int seed : {0u, 1u, 2u, 7u, 123u, 999u}) {
        KMeans model(3, 300, 1e-6, seed);
        model.fit(sampleData());
        INFO("seed = ", seed);
        checkPartitionMatchesTrueClusters(model.labels());
        CHECK(model.inertia() == doctest::Approx(kExpectedInertia).epsilon(1e-6));
    }
}

TEST_CASE("KMeans: predict assigns new points to the correct fitted cluster") {
    KMeans model(3, 300, 1e-6, 42);
    model.fit(sampleData());

    // Points near each true center, unseen during fit.
    Matrix probes{{0.5, 0.5}, {10.5, 0.5}, {5.5, 10.5}};
    std::vector<std::size_t> predicted = model.predict(probes);
    REQUIRE(predicted.size() == 3);

    const std::vector<std::size_t>& trainLabels = model.labels();
    CHECK(predicted[0] == trainLabels[0]);  // near cluster A
    CHECK(predicted[1] == trainLabels[4]);  // near cluster B
    CHECK(predicted[2] == trainLabels[8]);  // near cluster C
}

TEST_CASE("KMeans: converges well before the iteration cap on separated clusters") {
    KMeans model(3, 300, 1e-6, 42);
    model.fit(sampleData());
    CHECK(model.iterationsRun() < model.maxIterations());
    CHECK(model.iterationsRun() > 0);
}

TEST_CASE("KMeans: hits the iteration cap when epsilon can never be satisfied") {
    KMeans model(3, /*maxIterations=*/1, /*epsilon=*/0.0, /*seed=*/42);
    model.fit(sampleData());
    // epsilon=0 means "strictly less than 0" can never trigger early
    // convergence, so the cap of 1 iteration must be hit exactly.
    CHECK(model.iterationsRun() == 1);
}

TEST_CASE("KMeans: constructor rejects k == 0") {
    CHECK_THROWS_AS(KMeans(0), std::invalid_argument);
}

TEST_CASE("KMeans: fit rejects k greater than the number of samples") {
    KMeans model(20, 300, 1e-6, 42);
    CHECK_THROWS_AS(model.fit(sampleData()), std::invalid_argument);
}

TEST_CASE("KMeans: fit rejects an empty dataset") {
    KMeans model(1);
    Matrix empty(0, 2);
    CHECK_THROWS_AS(model.fit(empty), std::invalid_argument);
}

TEST_CASE("KMeans: using the model before fitting throws") {
    KMeans model(3);
    CHECK_THROWS_AS(model.labels(), std::logic_error);
    CHECK_THROWS_AS(model.centroids(), std::logic_error);
    CHECK_THROWS_AS(model.inertia(), std::logic_error);
    CHECK_THROWS_AS(model.predict(sampleData()), std::logic_error);
}

TEST_CASE("KMeans: predict rejects mismatched feature count") {
    KMeans model(3, 300, 1e-6, 42);
    model.fit(sampleData());

    Matrix wrongShape(2, 3, 1.0);
    CHECK_THROWS_AS(model.predict(wrongShape), std::invalid_argument);
}
