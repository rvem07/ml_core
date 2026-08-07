#include "doctest.h"

#include "ml/LinearRegression.hpp"
#include "ml/Matrix.hpp"

#include <stdexcept>
#include <vector>

using ml::LinearRegression;
using ml::Matrix;

namespace {

// Data generated exactly from y = 3 + 2*x1 - 1*x2, so both fit paths should
// recover intercept = 3, coefficients = [2, -1] with (near) zero residual.
Matrix sampleX() {
    return Matrix{{1, 5}, {2, 3}, {3, 6}, {4, 2}, {5, 7}};
}

std::vector<double> sampleY() {
    return {0, 4, 3, 9, 6};
}

}  // namespace

TEST_CASE("LinearRegression: normal equation recovers known coefficients") {
    LinearRegression model;
    model.fitNormalEquation(sampleX(), sampleY());

    CHECK(model.isFitted());
    CHECK(model.intercept() == doctest::Approx(3.0).epsilon(0.001));

    std::vector<double> coeffs = model.coefficients();
    REQUIRE(coeffs.size() == 2);
    CHECK(coeffs[0] == doctest::Approx(2.0).epsilon(0.001));
    CHECK(coeffs[1] == doctest::Approx(-1.0).epsilon(0.001));

    CHECK(model.mse(sampleX(), sampleY()) == doctest::Approx(0.0).epsilon(0.001));
    CHECK(model.rSquared(sampleX(), sampleY()) == doctest::Approx(1.0).epsilon(0.001));
}

TEST_CASE("LinearRegression: gradient descent recovers known coefficients") {
    LinearRegression model;
    model.fitGradientDescent(sampleX(), sampleY(), /*learningRate=*/0.01, /*iterations=*/20000);

    CHECK(model.isFitted());
    CHECK(model.intercept() == doctest::Approx(3.0).epsilon(0.01));

    std::vector<double> coeffs = model.coefficients();
    REQUIRE(coeffs.size() == 2);
    CHECK(coeffs[0] == doctest::Approx(2.0).epsilon(0.01));
    CHECK(coeffs[1] == doctest::Approx(-1.0).epsilon(0.01));

    CHECK(model.mse(sampleX(), sampleY()) == doctest::Approx(0.0).epsilon(0.01));
    CHECK(model.rSquared(sampleX(), sampleY()) == doctest::Approx(1.0).epsilon(0.01));
}

TEST_CASE("LinearRegression: normal equation and gradient descent agree") {
    LinearRegression normalEq;
    normalEq.fitNormalEquation(sampleX(), sampleY());

    LinearRegression gd;
    gd.fitGradientDescent(sampleX(), sampleY(), /*learningRate=*/0.01, /*iterations=*/20000);

    CHECK(gd.intercept() == doctest::Approx(normalEq.intercept()).epsilon(0.01));

    std::vector<double> a = normalEq.coefficients();
    std::vector<double> b = gd.coefficients();
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(b[i] == doctest::Approx(a[i]).epsilon(0.01));
    }

    // Both should also agree on predictions for an unseen point.
    Matrix probe{{10, 1}};
    std::vector<double> predA = normalEq.predict(probe);
    std::vector<double> predB = gd.predict(probe);
    REQUIRE(predA.size() == 1);
    REQUIRE(predB.size() == 1);
    CHECK(predB[0] == doctest::Approx(predA[0]).epsilon(0.01));
}

TEST_CASE("LinearRegression: predict on unseen data matches the generating function") {
    LinearRegression model;
    model.fitNormalEquation(sampleX(), sampleY());

    // y = 3 + 2*10 - 1*1 = 22
    Matrix probe{{10, 1}};
    std::vector<double> pred = model.predict(probe);
    REQUIRE(pred.size() == 1);
    CHECK(pred[0] == doctest::Approx(22.0).epsilon(0.001));
}

TEST_CASE("LinearRegression: fit rejects mismatched X/y sizes") {
    LinearRegression model;
    std::vector<double> shortY{0, 4};
    CHECK_THROWS_AS(model.fitNormalEquation(sampleX(), shortY), std::invalid_argument);
    CHECK_THROWS_AS(model.fitGradientDescent(sampleX(), shortY), std::invalid_argument);
}

TEST_CASE("LinearRegression: using the model before fitting throws") {
    LinearRegression model;
    CHECK_THROWS_AS(model.predict(sampleX()), std::logic_error);
    CHECK_THROWS_AS(model.intercept(), std::logic_error);
    CHECK_THROWS_AS(model.coefficients(), std::logic_error);
}

TEST_CASE("LinearRegression: predict rejects wrong feature count") {
    LinearRegression model;
    model.fitNormalEquation(sampleX(), sampleY());

    Matrix wrongShape(2, 3, 1.0);
    CHECK_THROWS_AS(model.predict(wrongShape), std::invalid_argument);
}
