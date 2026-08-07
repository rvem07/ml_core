#include "ml/LinearRegression.hpp"

#include <numeric>
#include <stdexcept>

namespace ml {

Matrix LinearRegression::withBiasColumn(const Matrix& X) {
    Matrix Xb(X.rows(), X.cols() + 1);
    for (std::size_t i = 0; i < X.rows(); ++i) {
        Xb(i, 0) = 1.0;
        for (std::size_t j = 0; j < X.cols(); ++j) {
            Xb(i, j + 1) = X(i, j);
        }
    }
    return Xb;
}

void LinearRegression::requireFitted() const {
    if (!fitted_) {
        throw std::logic_error("LinearRegression: model has not been fitted yet");
    }
}

void LinearRegression::fitNormalEquation(const Matrix& X, const std::vector<double>& y) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("LinearRegression::fitNormalEquation: X.rows() must match y.size()");
    }
    const Matrix Xb = withBiasColumn(X);
    theta_ = Matrix::solveLeastSquares(Xb, y);
    fitted_ = true;
}

void LinearRegression::fitGradientDescent(const Matrix& X, const std::vector<double>& y,
                                           double learningRate, std::size_t iterations) {
    if (X.rows() != y.size()) {
        throw std::invalid_argument("LinearRegression::fitGradientDescent: X.rows() must match y.size()");
    }
    const Matrix Xb = withBiasColumn(X);
    const std::size_t n = Xb.rows();
    const std::size_t p = Xb.cols();

    std::vector<double> theta(p, 0.0);
    const Matrix XbT = Xb.transpose();

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        std::vector<double> predictions = Xb * theta;
        std::vector<double> errors(n);
        for (std::size_t i = 0; i < n; ++i) {
            errors[i] = predictions[i] - y[i];
        }
        // gradient = (1/n) * X^T * errors
        std::vector<double> gradient = XbT * errors;
        const double scale = learningRate / static_cast<double>(n);
        for (std::size_t j = 0; j < p; ++j) {
            theta[j] -= scale * gradient[j];
        }
    }

    theta_ = std::move(theta);
    fitted_ = true;
}

std::vector<double> LinearRegression::predict(const Matrix& X) const {
    requireFitted();
    const Matrix Xb = withBiasColumn(X);
    if (Xb.cols() != theta_.size()) {
        throw std::invalid_argument("LinearRegression::predict: X's column count does not match the fitted model");
    }
    return Xb * theta_;
}

double LinearRegression::mse(const Matrix& X, const std::vector<double>& y) const {
    const std::vector<double> predictions = predict(X);
    if (predictions.size() != y.size()) {
        throw std::invalid_argument("LinearRegression::mse: X.rows() must match y.size()");
    }
    double sumSquaredError = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double diff = predictions[i] - y[i];
        sumSquaredError += diff * diff;
    }
    return sumSquaredError / static_cast<double>(y.size());
}

double LinearRegression::rSquared(const Matrix& X, const std::vector<double>& y) const {
    const std::vector<double> predictions = predict(X);
    if (predictions.size() != y.size()) {
        throw std::invalid_argument("LinearRegression::rSquared: X.rows() must match y.size()");
    }
    const double meanY = std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());

    double ssRes = 0.0;
    double ssTot = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double residual = y[i] - predictions[i];
        const double deviation = y[i] - meanY;
        ssRes += residual * residual;
        ssTot += deviation * deviation;
    }
    if (ssTot == 0.0) {
        // All y values identical: define R^2 as 1 if the model is exact, else 0.
        return ssRes == 0.0 ? 1.0 : 0.0;
    }
    return 1.0 - ssRes / ssTot;
}

double LinearRegression::intercept() const {
    requireFitted();
    return theta_[0];
}

std::vector<double> LinearRegression::coefficients() const {
    requireFitted();
    return std::vector<double>(theta_.begin() + 1, theta_.end());
}

}  // namespace ml
