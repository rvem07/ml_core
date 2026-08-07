// Cross-checks ml-core's LinearRegression and KMeans against reference
// baselines computed by scikit-learn (see validation/generate_fixtures.py).
// The fixtures themselves are checked-in CSVs under validation/fixtures/,
// so this test has no Python/scikit-learn dependency - it only reads files.

#include "doctest.h"

#include "ml/KMeans.hpp"
#include "ml/LinearRegression.hpp"
#include "ml/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ML_CORE_VALIDATION_FIXTURES_DIR
#error "ML_CORE_VALIDATION_FIXTURES_DIR must be defined by the build system"
#endif

using ml::KMeans;
using ml::LinearRegression;
using ml::Matrix;

namespace {

std::string fixturePath(const std::string& filename) {
    return std::string(ML_CORE_VALIDATION_FIXTURES_DIR) + "/" + filename;
}

// Splits a single CSV line on commas. Fixtures never quote or escape values,
// so this doesn't need to handle either.
std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::ifstream openFixture(const std::string& filename) {
    std::ifstream file(fixturePath(filename));
    if (!file) {
        throw std::runtime_error("validation fixture not found: " + fixturePath(filename) +
                                  " (run validation/generate_fixtures.py)");
    }
    return file;
}

// Reads a numeric CSV (header row skipped) into a Matrix, one row per line.
Matrix loadMatrixCsv(const std::string& filename) {
    std::ifstream file = openFixture(filename);
    std::string line;
    std::getline(file, line);  // header

    std::vector<std::vector<double>> rows;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields = splitCsvLine(line);
        std::vector<double> row;
        row.reserve(fields.size());
        for (const auto& f : fields) row.push_back(std::stod(f));
        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        return Matrix();
    }
    Matrix m(rows.size(), rows.front().size());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::size_t j = 0; j < rows[i].size(); ++j) {
            m(i, j) = rows[i][j];
        }
    }
    return m;
}

// Reads the first column of a numeric CSV (header row skipped) into a vector.
std::vector<double> loadColumnCsv(const std::string& filename) {
    std::ifstream file = openFixture(filename);
    std::string line;
    std::getline(file, line);  // header

    std::vector<double> values;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        values.push_back(std::stod(splitCsvLine(line).front()));
    }
    return values;
}

// Reads all comma-separated values from the single data row of a CSV whose
// header names one field per column (e.g. "intercept,coef_0,coef_1,...").
std::vector<double> loadSingleRowCsv(const std::string& filename) {
    std::ifstream file = openFixture(filename);
    std::string header;
    std::getline(file, header);
    std::string dataLine;
    std::getline(file, dataLine);

    std::vector<double> values;
    for (const auto& field : splitCsvLine(dataLine)) {
        values.push_back(std::stod(field));
    }
    return values;
}

std::vector<std::size_t> loadLabelsCsv(const std::string& filename) {
    std::ifstream file = openFixture(filename);
    std::string line;
    std::getline(file, line);  // header

    std::vector<std::size_t> labels;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        labels.push_back(static_cast<std::size_t>(std::stoul(splitCsvLine(line).front())));
    }
    return labels;
}

// kmeans_expected_meta.csv has a single "k,inertia" data row.
struct KMeansMeta {
    std::size_t k;
    double inertia;
};

KMeansMeta loadKMeansMetaCsv(const std::string& filename) {
    std::ifstream file = openFixture(filename);
    std::string header;
    std::getline(file, header);
    std::string dataLine;
    std::getline(file, dataLine);
    std::vector<std::string> fields = splitCsvLine(dataLine);
    return KMeansMeta{static_cast<std::size_t>(std::stoul(fields.at(0))), std::stod(fields.at(1))};
}

// Cluster label numbering is arbitrary and independent between
// implementations, so we search over all k! relabelings of `predicted` and
// report the fraction of points that agree with `expected` under the best
// one. k is small (a handful of clusters) so brute force is fine.
double bestLabelAgreement(const std::vector<std::size_t>& predicted,
                           const std::vector<std::size_t>& expected, std::size_t k) {
    REQUIRE(predicted.size() == expected.size());

    std::vector<std::size_t> relabeling(k);
    std::iota(relabeling.begin(), relabeling.end(), 0);

    std::size_t bestMatches = 0;
    do {
        std::size_t matches = 0;
        for (std::size_t i = 0; i < predicted.size(); ++i) {
            if (relabeling[predicted[i]] == expected[i]) {
                ++matches;
            }
        }
        bestMatches = std::max(bestMatches, matches);
    } while (std::next_permutation(relabeling.begin(), relabeling.end()));

    return static_cast<double>(bestMatches) / static_cast<double>(predicted.size());
}

}  // namespace

TEST_CASE("validation: LinearRegression normal equation matches scikit-learn") {
    Matrix X = loadMatrixCsv("regression_X.csv");
    std::vector<double> y = loadColumnCsv("regression_y.csv");
    std::vector<double> expected = loadSingleRowCsv("regression_expected.csv");
    REQUIRE(expected.size() == X.cols() + 1);  // intercept + one coefficient per feature

    LinearRegression model;
    model.fitNormalEquation(X, y);

    constexpr double kTolerance = 1e-6;
    CHECK(std::abs(model.intercept() - expected[0]) < kTolerance);

    std::vector<double> coeffs = model.coefficients();
    REQUIRE(coeffs.size() == X.cols());
    for (std::size_t j = 0; j < coeffs.size(); ++j) {
        INFO("feature ", j);
        CHECK(std::abs(coeffs[j] - expected[j + 1]) < kTolerance);
    }
}

TEST_CASE("validation: LinearRegression gradient descent also agrees with scikit-learn") {
    // Gradient descent is iterative rather than a closed-form solve, so it
    // gets a looser (but still tight) tolerance than the normal equation.
    Matrix X = loadMatrixCsv("regression_X.csv");
    std::vector<double> y = loadColumnCsv("regression_y.csv");
    std::vector<double> expected = loadSingleRowCsv("regression_expected.csv");

    LinearRegression model;
    model.fitGradientDescent(X, y, /*learningRate=*/0.01, /*iterations=*/20000);

    constexpr double kTolerance = 1e-3;
    CHECK(std::abs(model.intercept() - expected[0]) < kTolerance);

    std::vector<double> coeffs = model.coefficients();
    REQUIRE(coeffs.size() == X.cols());
    for (std::size_t j = 0; j < coeffs.size(); ++j) {
        INFO("feature ", j);
        CHECK(std::abs(coeffs[j] - expected[j + 1]) < kTolerance);
    }
}

TEST_CASE("validation: KMeans matches scikit-learn's clustering and inertia") {
    Matrix X = loadMatrixCsv("kmeans_X.csv");
    std::vector<std::size_t> expectedLabels = loadLabelsCsv("kmeans_expected_labels.csv");
    KMeansMeta expectedMeta = loadKMeansMetaCsv("kmeans_expected_meta.csv");
    REQUIRE(expectedLabels.size() == X.rows());

    KMeans model(expectedMeta.k, /*maxIterations=*/300, /*epsilon=*/1e-6, /*seed=*/7);
    model.fit(X);

    double agreement = bestLabelAgreement(model.labels(), expectedLabels, expectedMeta.k);
    // The fixture's blobs are spaced far apart relative to their spread, so
    // the optimal partition is essentially unique - expect (near) perfect
    // agreement once label permutation is accounted for.
    CHECK(agreement >= 0.98);

    CHECK(model.inertia() == doctest::Approx(expectedMeta.inertia).epsilon(1e-3));
}
