"""Local optimization of the two-dimensional Rosenbrock function."""

import numpy as np

from fsgo import minimize_fsgo


def objective(x):
    return float(
        100.0 * (x[1] - x[0] ** 2) ** 2
        + (1.0 - x[0]) ** 2
    )


def gradient(x):
    return np.array(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2)
            - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ]
    )


result = minimize_fsgo(
    objective,
    gradient,
    bounds=[(-3.0, 3.0), (-3.0, 3.0)],
    mode="LOCAL",
    x0=[-1.2, 1.0],
    population_size=30,
    K=6000,
    seed=42,
)

print("Best value:", result.fun)
print("Best point:", result.x)
