"""Global optimization of the Sphere function."""

import numpy as np

from fsgo import minimize_fsgo


def objective(x):
    return float(np.dot(x, x))


def gradient(x):
    return 2.0 * x


result = minimize_fsgo(
    objective,
    gradient,
    bounds=[(-5.0, 5.0)] * 10,
    mode="GLOBAL",
    population_size=30,
    K=3000,
    seed=42,
)

print("Best value:", result.fun)
print("Best point:", result.x)
print("Objective evaluations:", result.nfev)
print("Gradient evaluations:", result.njev)
