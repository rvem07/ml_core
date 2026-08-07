#include "doctest.h"

#include "ml/Matrix.hpp"

#include <stdexcept>
#include <vector>

using ml::Matrix;

TEST_CASE("construction: dims fills with default/given value") {
    Matrix z(2, 3);
    CHECK(z.rows() == 2);
    CHECK(z.cols() == 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(z(i, j) == 0.0);
        }
    }

    Matrix f(2, 2, 7.0);
    CHECK(f(0, 0) == 7.0);
    CHECK(f(1, 1) == 7.0);
}

TEST_CASE("construction: flat initializer list is stored row-major") {
    Matrix m(2, 3, {1, 2, 3, 4, 5, 6});
    CHECK(m(0, 0) == 1);
    CHECK(m(0, 1) == 2);
    CHECK(m(0, 2) == 3);
    CHECK(m(1, 0) == 4);
    CHECK(m(1, 1) == 5);
    CHECK(m(1, 2) == 6);

    CHECK_THROWS_AS(Matrix(2, 3, {1, 2, 3}), std::invalid_argument);
}

TEST_CASE("construction: nested initializer list") {
    Matrix m{{1, 2}, {3, 4}, {5, 6}};
    CHECK(m.rows() == 3);
    CHECK(m.cols() == 2);
    CHECK(m(0, 0) == 1);
    CHECK(m(0, 1) == 2);
    CHECK(m(1, 0) == 3);
    CHECK(m(1, 1) == 4);
    CHECK(m(2, 0) == 5);
    CHECK(m(2, 1) == 6);

    CHECK_THROWS_AS((Matrix{{1, 2}, {3}}), std::invalid_argument);
}

TEST_CASE("construction: identity") {
    Matrix id = Matrix::identity(3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            CHECK(id(i, j) == (i == j ? 1.0 : 0.0));
        }
    }
}

TEST_CASE("operator(): access, mutation, bounds checking") {
    Matrix m(2, 2, 0.0);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    CHECK(m(0, 0) == 1);
    CHECK(m(1, 1) == 4);

    const Matrix& cm = m;
    CHECK(cm(0, 1) == 2);

    CHECK_THROWS_AS(m(2, 0), std::out_of_range);
    CHECK_THROWS_AS(m(0, 2), std::out_of_range);
}

TEST_CASE("copy/move semantics: copies are independent (rule of five)") {
    Matrix a{{1, 2}, {3, 4}};
    Matrix b = a;  // copy
    b(0, 0) = 99;
    CHECK(a(0, 0) == 1);
    CHECK(b(0, 0) == 99);

    Matrix c = std::move(b);  // move
    CHECK(c(0, 0) == 99);
    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
}

TEST_CASE("matrix multiply: hand-computed 2x3 * 3x2") {
    Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix b(3, 2, {7, 8, 9, 10, 11, 12});
    Matrix result = a * b;

    CHECK(result.rows() == 2);
    CHECK(result.cols() == 2);
    CHECK(result(0, 0) == 58);   // 1*7 + 2*9 + 3*11
    CHECK(result(0, 1) == 64);   // 1*8 + 2*10 + 3*12
    CHECK(result(1, 0) == 139);  // 4*7 + 5*9 + 6*11
    CHECK(result(1, 1) == 154);  // 4*8 + 5*10 + 6*12
}

TEST_CASE("matrix multiply: dimension mismatch throws") {
    Matrix a(2, 3, 1.0);
    Matrix b(2, 2, 1.0);
    CHECK_THROWS_AS(a * b, std::invalid_argument);
}

TEST_CASE("matrix multiply: identity is the multiplicative identity") {
    Matrix a{{1, 2}, {3, 4}};
    Matrix id = Matrix::identity(2);
    CHECK(a * id == a);
    CHECK(id * a == a);
}

TEST_CASE("transpose: hand-computed 2x3") {
    Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    Matrix t = a.transpose();
    CHECK(t.rows() == 3);
    CHECK(t.cols() == 2);
    CHECK(t(0, 0) == 1);
    CHECK(t(1, 0) == 2);
    CHECK(t(2, 0) == 3);
    CHECK(t(0, 1) == 4);
    CHECK(t(1, 1) == 5);
    CHECK(t(2, 1) == 6);
}

TEST_CASE("elementwise add/subtract: hand-computed 2x2") {
    Matrix a{{1, 2}, {3, 4}};
    Matrix b{{5, 6}, {7, 8}};

    Matrix sum = a + b;
    CHECK(sum == Matrix({{6, 8}, {10, 12}}));

    Matrix diff = a - b;
    CHECK(diff == Matrix({{-4, -4}, {-4, -4}}));

    CHECK_THROWS_AS(a + Matrix(3, 3, 0.0), std::invalid_argument);
    CHECK_THROWS_AS(a - Matrix(3, 3, 0.0), std::invalid_argument);
}

TEST_CASE("scalar multiply: both operand orders and compound assignment") {
    Matrix a{{1, 2}, {3, 4}};
    Matrix expected{{2, 4}, {6, 8}};

    CHECK(a * 2.0 == expected);
    CHECK(2.0 * a == expected);

    Matrix c = a;
    c *= 2.0;
    CHECK(c == expected);
}

TEST_CASE("matrix-vector multiply: hand-computed") {
    Matrix a{{1, 2}, {3, 4}};
    std::vector<double> x{1, 1};
    std::vector<double> y = a * x;
    REQUIRE(y.size() == 2);
    CHECK(y[0] == doctest::Approx(3.0));
    CHECK(y[1] == doctest::Approx(7.0));

    std::vector<double> wrongSize{1, 2, 3};
    CHECK_THROWS_AS(a * wrongSize, std::invalid_argument);
}

TEST_CASE("equality operator") {
    Matrix a{{1, 2}, {3, 4}};
    Matrix b{{1, 2}, {3, 4}};
    Matrix c{{1, 2}, {3, 5}};
    CHECK(a == b);
    CHECK(a != c);
}

TEST_CASE("solve: 2x2 system via Gaussian elimination") {
    // 2x + y = 3
    //  x + 3y = 5
    // => x = 0.8, y = 1.4
    Matrix A{{2, 1}, {1, 3}};
    std::vector<double> b{3, 5};
    std::vector<double> x = Matrix::solve(A, b);

    REQUIRE(x.size() == 2);
    CHECK(x[0] == doctest::Approx(0.8));
    CHECK(x[1] == doctest::Approx(1.4));
}

TEST_CASE("solve: 3x3 system requiring partial pivoting, known solution") {
    Matrix A{{3, 2, -1}, {2, -2, 4}, {-1, 0.5, -1}};
    std::vector<double> b{1, -2, 0};
    std::vector<double> x = Matrix::solve(A, b);

    REQUIRE(x.size() == 3);
    CHECK(x[0] == doctest::Approx(1.0));
    CHECK(x[1] == doctest::Approx(-2.0));
    CHECK(x[2] == doctest::Approx(-2.0));
}

TEST_CASE("solve: singular matrix throws") {
    Matrix A{{1, 2}, {2, 4}};
    std::vector<double> b{1, 2};
    CHECK_THROWS_AS(Matrix::solve(A, b), std::runtime_error);
}

TEST_CASE("solve: non-square A throws") {
    Matrix A(2, 3, 1.0);
    std::vector<double> b{1, 2};
    CHECK_THROWS_AS(Matrix::solve(A, b), std::invalid_argument);
}

TEST_CASE("solveLeastSquares: line fit through 3 points via normal equations") {
    // Design matrix for y = a + b*x at x = 0, 1, 2; observed y = 1, 2, 2.
    Matrix A{{1, 0}, {1, 1}, {1, 2}};
    std::vector<double> y{1, 2, 2};

    std::vector<double> coeffs = Matrix::solveLeastSquares(A, y);

    // Hand-derived from the normal equations A^T A x = A^T b:
    //   [3 3] [a]   [5]
    //   [3 5] [b] = [6]
    // => b = 0.5, a = 7/6
    REQUIRE(coeffs.size() == 2);
    CHECK(coeffs[0] == doctest::Approx(7.0 / 6.0));
    CHECK(coeffs[1] == doctest::Approx(0.5));
}

TEST_CASE("solveLeastSquares: exact square system matches direct solve") {
    Matrix A{{2, 1}, {1, 3}};
    std::vector<double> b{3, 5};
    std::vector<double> x = Matrix::solveLeastSquares(A, b);

    REQUIRE(x.size() == 2);
    CHECK(x[0] == doctest::Approx(0.8));
    CHECK(x[1] == doctest::Approx(1.4));
}
