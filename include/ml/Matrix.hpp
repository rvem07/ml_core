#pragma once

// Dense matrix backed by a single contiguous std::vector<double> in
// row-major order (element (i, j) lives at data[i * cols + j]). This is
// deliberately not a vector-of-vectors: one contiguous allocation keeps
// rows adjacent in memory, which is friendlier to the cache and lets the
// whole buffer be handed to BLAS-like routines later if needed.

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace ml {

class Matrix {
public:
    // Empty 0x0 matrix.
    Matrix() noexcept;

    // rows x cols matrix, every entry initialized to `fill`.
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);

    // rows x cols matrix populated row-major from a flat initializer list.
    // values.size() must equal rows * cols.
    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values);

    // Matrix from a nested initializer list, one inner list per row, e.g.
    //   Matrix m{{1, 2, 3}, {4, 5, 6}};
    // All rows must have the same length.
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    // Copies/moves are plain data copies/moves of the underlying vector, so
    // the compiler-generated versions are correct and efficient (rule of
    // five, spelled out explicitly for clarity).
    Matrix(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix() = default;

    static Matrix identity(std::size_t n);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    // Bounds-checked element access.
    double& operator()(std::size_t i, std::size_t j);
    double operator()(std::size_t i, std::size_t j) const;

    // Read-only access to the underlying contiguous storage.
    const std::vector<double>& data() const noexcept { return data_; }

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;   // matrix product
    Matrix operator*(double scalar) const;
    std::vector<double> operator*(const std::vector<double>& x) const;  // matrix-vector product

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    Matrix transpose() const;

    // Elementwise comparison with a small absolute tolerance.
    bool operator==(const Matrix& other) const;
    bool operator!=(const Matrix& other) const { return !(*this == other); }

    // Solves the square linear system Ax = b via Gaussian elimination with
    // partial pivoting. Throws std::invalid_argument on dimension mismatch
    // and std::runtime_error if A is (numerically) singular.
    static std::vector<double> solve(const Matrix& A, const std::vector<double>& b);

    // Least-squares solve of Ax ~= b via the normal equations
    // (A^T A) x = A^T b. Works for square, over-, or under-determined A,
    // provided A^T A is non-singular.
    static std::vector<double> solveLeastSquares(const Matrix& A, const std::vector<double>& b);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

Matrix operator*(double scalar, const Matrix& m);

std::ostream& operator<<(std::ostream& os, const Matrix& m);

}  // namespace ml
