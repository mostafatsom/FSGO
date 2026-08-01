"""Global optimization with a user-defined fixed Gamma set."""

import numpy as np

from fsgo import minimize_fsgo


def objective(x):
    dimension = x.size
    return float(
        10.0 * dimension
        + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))
    )


def gradient(x):
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


result = minimize_fsgo(
    objective,
    gradient,
    bounds=[(-5.12, 5.12)] * 5,
    gamma_set=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
    population_size=30,
    K=3000,
    seed=42,
)

print("Best value:", result.fun)
print("Best point:", result.x)
