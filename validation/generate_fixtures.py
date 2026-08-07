#!/usr/bin/env python3
"""Generates reference fixtures for validating ml-core against scikit-learn.

Writes plain CSVs under validation/fixtures/ so the C++ test suite has no
Python/scikit-learn dependency at build or test time: this script is run
once (and re-run only if the fixtures need to change), and its output is
checked into the repository.

Regression fixtures:
    regression_X.csv        - n_samples x n_features design matrix
    regression_y.csv        - n_samples target values
    regression_expected.csv - intercept + coefficients from sklearn's
                              LinearRegression fit on (X, y)

K-Means fixtures:
    kmeans_X.csv               - n_samples x n_features points
    kmeans_expected_labels.csv - cluster label (0..k-1) per row of kmeans_X,
                                  from sklearn's KMeans fit
    kmeans_expected_meta.csv   - k and inertia_ from that same fit

Run from the repo root:
    python3 validation/generate_fixtures.py
"""

import csv
import pathlib

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def write_csv(path: pathlib.Path, header: list, rows) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_regression_fixtures() -> None:
    rng = np.random.default_rng(42)

    n_samples, n_features = 200, 3
    X = rng.normal(loc=0.0, scale=3.0, size=(n_samples, n_features))

    true_intercept = 4.2
    true_coefficients = np.array([2.5, -1.3, 0.7])
    noise = rng.normal(loc=0.0, scale=0.05, size=n_samples)
    y = true_intercept + X @ true_coefficients + noise

    model = LinearRegression()
    model.fit(X, y)

    write_csv(
        FIXTURES_DIR / "regression_X.csv",
        [f"x{i}" for i in range(n_features)],
        X.tolist(),
    )
    write_csv(FIXTURES_DIR / "regression_y.csv", ["y"], [[v] for v in y.tolist()])
    write_csv(
        FIXTURES_DIR / "regression_expected.csv",
        ["intercept"] + [f"coef_{i}" for i in range(n_features)],
        [[model.intercept_] + model.coef_.tolist()],
    )

    print(f"regression: intercept={model.intercept_!r} coef={model.coef_.tolist()!r}")


def generate_kmeans_fixtures() -> None:
    # Centers spaced far apart relative to cluster_std so the partition that
    # minimizes inertia is essentially unique - any correct k-means
    # implementation should converge to the same clustering (up to label
    # permutation), which is what makes this a meaningful cross-check.
    centers = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0], [20.0, 20.0]])
    X, _ = make_blobs(
        n_samples=300,
        centers=centers,
        cluster_std=1.5,
        n_features=2,
        random_state=7,
    )

    k = len(centers)
    model = KMeans(n_clusters=k, n_init=10, random_state=7)
    model.fit(X)

    write_csv(FIXTURES_DIR / "kmeans_X.csv", ["x0", "x1"], X.tolist())
    write_csv(
        FIXTURES_DIR / "kmeans_expected_labels.csv",
        ["label"],
        [[label] for label in model.labels_.tolist()],
    )
    write_csv(
        FIXTURES_DIR / "kmeans_expected_meta.csv",
        ["k", "inertia"],
        [[k, model.inertia_]],
    )

    print(f"kmeans: k={k} inertia={model.inertia_!r}")


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_regression_fixtures()
    generate_kmeans_fixtures()


if __name__ == "__main__":
    main()
