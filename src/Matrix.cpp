#include "ml/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ml {

namespace {
constexpr double kEqualityTolerance = 1e-9;
constexpr double kSingularEpsilon = 1e-12;
}  // namespace

Matrix::Matrix() noexcept : rows_(0), cols_(0), data_() {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values)
    : rows_(rows), cols_(cols), data_(values) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("Matrix: initializer list size does not match rows * cols");
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values)
    : rows_(values.size()), cols_(values.size() == 0 ? 0 : values.begin()->size()), data_() {
    data_.reserve(rows_ * cols_);
    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument("Matrix: all rows must have the same length");
        }
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

Matrix Matrix::identity(std::size_t n) {
    Matrix m(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        m.data_[i * n + i] = 1.0;
    }
    return m;
}

double& Matrix::operator()(std::size_t i, std::size_t j) {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix::operator(): index out of range");
    }
    return data_[i * cols_ + j];
}

double Matrix::operator()(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix::operator(): index out of range");
    }
    return data_[i * cols_ + j];
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix::operator+: dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        result.data_[k] = data_[k] + other.data_[k];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix::operator-: dimension mismatch");
    }
    Matrix result(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        result.data_[k] = data_[k] - other.data_[k];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix::operator*: inner dimensions do not match");
    }
    Matrix result(rows_, other.cols_, 0.0);
    const std::size_t n = cols_;
    const std::size_t p = other.cols_;
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            const double a_ik = data_[i * cols_ + k];
            if (a_ik == 0.0) continue;
            for (std::size_t j = 0; j < p; ++j) {
                result.data_[i * p + j] += a_ik * other.data_[k * p + j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t k = 0; k < data_.size(); ++k) {
        result.data_[k] = data_[k] * scalar;
    }
    return result;
}

std::vector<double> Matrix::operator*(const std::vector<double>& x) const {
    if (cols_ != x.size()) {
        throw std::invalid_argument("Matrix::operator*: vector size does not match column count");
    }
    std::vector<double> y(rows_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < cols_; ++j) {
            sum += data_[i * cols_ + j] * x[j];
        }
        y[i] = sum;
    }
    return y;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (double& v : data_) {
        v *= scalar;
    }
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < cols_; ++j) {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return result;
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }
    for (std::size_t k = 0; k < data_.size(); ++k) {
        if (std::abs(data_[k] - other.data_[k]) > kEqualityTolerance) {
            return false;
        }
    }
    return true;
}

std::vector<double> Matrix::solve(const Matrix& A, const std::vector<double>& b) {
    if (A.rows_ != A.cols_) {
        throw std::invalid_argument("Matrix::solve: A must be square");
    }
    const std::size_t n = A.rows_;
    if (b.size() != n) {
        throw std::invalid_argument("Matrix::solve: b size does not match A's row count");
    }

    // Flat augmented matrix [A | b], n rows by (n + 1) columns, row-major.
    const std::size_t width = n + 1;
    std::vector<double> aug(n * width);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            aug[i * width + j] = A.data_[i * n + j];
        }
        aug[i * width + n] = b[i];
    }

    // Forward elimination with partial pivoting.
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivotRow = k;
        double pivotVal = std::abs(aug[k * width + k]);
        for (std::size_t i = k + 1; i < n; ++i) {
            const double v = std::abs(aug[i * width + k]);
            if (v > pivotVal) {
                pivotVal = v;
                pivotRow = i;
            }
        }
        if (pivotVal < kSingularEpsilon) {
            throw std::runtime_error("Matrix::solve: matrix is singular (or numerically singular)");
        }
        if (pivotRow != k) {
            for (std::size_t j = 0; j < width; ++j) {
                std::swap(aug[k * width + j], aug[pivotRow * width + j]);
            }
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            const double factor = aug[i * width + k] / aug[k * width + k];
            if (factor == 0.0) continue;
            for (std::size_t j = k; j < width; ++j) {
                aug[i * width + j] -= factor * aug[k * width + j];
            }
        }
    }

    // Back substitution.
    std::vector<double> x(n, 0.0);
    for (std::size_t ii = 0; ii < n; ++ii) {
        const std::size_t i = n - 1 - ii;
        double sum = aug[i * width + n];
        for (std::size_t j = i + 1; j < n; ++j) {
            sum -= aug[i * width + j] * x[j];
        }
        x[i] = sum / aug[i * width + i];
    }
    return x;
}

std::vector<double> Matrix::solveLeastSquares(const Matrix& A, const std::vector<double>& b) {
    if (b.size() != A.rows_) {
        throw std::invalid_argument("Matrix::solveLeastSquares: b size does not match A's row count");
    }
    const Matrix At = A.transpose();
    const Matrix AtA = At * A;
    const std::vector<double> Atb = At * b;
    return solve(AtA, Atb);
}

Matrix operator*(double scalar, const Matrix& m) {
    return m * scalar;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) {
    for (std::size_t i = 0; i < m.rows(); ++i) {
        os << "[ ";
        for (std::size_t j = 0; j < m.cols(); ++j) {
            os << m(i, j) << " ";
        }
        os << "]";
        if (i + 1 < m.rows()) os << "\n";
    }
    return os;
}

}  // namespace ml
