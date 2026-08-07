#pragma once

// Ordinary least squares linear regression on top of ml::Matrix. Supports
// two independent ways of fitting the same model so their results can be
// cross-checked against each other:
//
//   1. fitNormalEquation:  closed-form theta = (X^T X)^-1 X^T y
//   2. fitGradientDescent: iterative batch gradient descent
//
// Both prepend an implicit bias/intercept column of ones to the design
// matrix, so callers pass the raw feature matrix X (n samples x p features)
// without a leading column of ones.

#include <cstddef>
#include <vector>

#include "ml/Matrix.hpp"

namespace ml {

class LinearRegression {
public:
    LinearRegression() = default;

    // Fits theta = (X^T X)^-1 X^T y via Matrix::solveLeastSquares.
    // Throws std::invalid_argument if X.rows() != y.size().
    void fitNormalEquation(const Matrix& X, const std::vector<double>& y);

    // Fits via batch gradient descent, starting from theta = 0.
    // Throws std::invalid_argument if X.rows() != y.size().
    void fitGradientDescent(const Matrix& X, const std::vector<double>& y,
                             double learningRate = 0.01, std::size_t iterations = 10000);

    // Predicts y for each row of X. Throws std::logic_error if not yet fitted,
    // std::invalid_argument if X's column count doesn't match the fit.
    std::vector<double> predict(const Matrix& X) const;

    // Mean squared error of predictions on (X, y).
    double mse(const Matrix& X, const std::vector<double>& y) const;

    // Coefficient of determination R^2 on (X, y).
    double rSquared(const Matrix& X, const std::vector<double>& y) const;

    double intercept() const;
    std::vector<double> coefficients() const;  // excludes the intercept
    bool isFitted() const noexcept { return fitted_; }

private:
    std::vector<double> theta_;  // theta_[0] = intercept, theta_[1..] = feature weights
    bool fitted_ = false;

    static Matrix withBiasColumn(const Matrix& X);
    void requireFitted() const;
};

}  // namespace ml
