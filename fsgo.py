"""Flexible Stochastic Gradient Optimizer (FSGO).

FSGO generates a population of bounded candidates around the current best
point. Each candidate coordinate independently samples a Gamma value and a
positive or negative direction. The Gamma set remains fixed throughout the
optimization run.
"""

from dataclasses import dataclass
from typing import Callable, Sequence
import math

import numpy as np


@dataclass
class FSGOResult:
    """Result returned by :func:`minimize_fsgo`."""

    x: np.ndarray
    fun: float
    nfev: int
    njev: int
    iterations: int
    history: list[float]


def _precision_places(step: float) -> int:
    """Return the decimal places represented by a power-of-ten step."""

    value = float(step)
    if not math.isfinite(value) or value <= 0 or value > 1:
        raise ValueError("precision_step must be in (0, 1].")

    places = int(round(-math.log10(value)))
    expected = 10.0 ** (-places)

    if not math.isclose(
        value,
        expected,
        rel_tol=0.0,
        abs_tol=max(1e-15, expected * 1e-12),
    ):
        raise ValueError("precision_step must be a power of ten.")

    return places


def _default_gamma(step: float) -> np.ndarray:
    """Build the default fixed Gamma set from the precision step."""

    places = _precision_places(step)
    return np.array(
        [0.0] + [10.0 ** p for p in range(-places, 1)],
        dtype=float,
    )


def _lhs(count, lower, upper, rng):
    """Generate a Latin Hypercube sample inside the supplied bounds."""

    sample = np.empty((count, lower.size), dtype=float)

    for coordinate in range(lower.size):
        strata = rng.permutation(count)
        jitter = rng.random(count)
        unit = (strata + jitter) / count
        sample[:, coordinate] = (
            lower[coordinate]
            + unit * (upper[coordinate] - lower[coordinate])
        )

    return sample


def minimize_fsgo(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[Sequence[float]],
    *,
    mode: str = "GLOBAL",
    gamma_set: Sequence[float] | None = None,
    population_size: int = 30,
    K: int = 15000,
    x0: Sequence[float] | None = None,
    seed: int | None = None,
    precision_step: float = 1e-4,
    record_history: bool = True,
) -> FSGOResult:
    """Minimize a differentiable objective over finite bounds.

    Parameters
    ----------
    fun
        Objective function. It must accept a one-dimensional NumPy array and
        return a scalar value.
    grad
        Analytic gradient of ``fun``. It must return an array with the same
        shape as the candidate point.
    bounds
        ``(lower, upper)`` pair for every decision variable.
    mode
        ``"GLOBAL"`` uses Latin Hypercube initialization. ``"LOCAL"`` starts
        from ``x0``.
    gamma_set
        Fixed Gamma values used for candidate generation. When omitted, the
        values are constructed from ``precision_step``.
    population_size
        Number of candidates generated in each complete generation.
    K
        Number of candidate evaluations after initialization.
    x0
        Starting point required in local mode.
    seed
        Seed used by NumPy's random number generator.
    precision_step
        Power-of-ten grid used to quantize evaluated points.
    record_history
        Store the incumbent objective after initialization and each generation.

    Returns
    -------
    FSGOResult
        Best point, objective value, evaluation counts, generation count, and
        optional objective history.

    Notes
    -----
    Gamma values are sampled independently for every candidate coordinate and
    remain unchanged for the complete run. A candidate replaces the incumbent
    only when it provides a strict objective improvement.
    """

    # Validate the problem definition before creating any random state.
    if not callable(fun) or not callable(grad):
        raise TypeError("fun and grad must be callable.")

    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
        raise ValueError("bounds must have shape (dimension, 2).")
    if np.any(~np.isfinite(bounds)):
        raise ValueError("bounds must be finite.")

    lower, upper = bounds[:, 0], bounds[:, 1]
    if np.any(upper <= lower):
        raise ValueError("Each upper bound must exceed its lower bound.")

    if not isinstance(population_size, int) or population_size <= 0:
        raise ValueError("population_size must be positive.")
    if not isinstance(K, int) or K < population_size:
        raise ValueError("K must be at least population_size.")

    mode = str(mode).upper()
    if mode not in {"GLOBAL", "LOCAL"}:
        raise ValueError("mode must be GLOBAL or LOCAL.")

    places = _precision_places(precision_step)

    # The selected Gamma set is fixed and reused by every generation.
    if gamma_set is None:
        gamma = _default_gamma(precision_step)
    else:
        gamma = np.asarray(gamma_set, dtype=float)
        if gamma.ndim != 1 or gamma.size == 0:
            raise ValueError("gamma_set must be one-dimensional and non-empty.")
        if np.any(~np.isfinite(gamma)):
            raise ValueError("gamma_set must contain finite values.")
        if np.any(gamma < 0) or np.any(gamma > 1):
            raise ValueError("Gamma values must lie in [0, 1].")
        if np.unique(gamma).size != gamma.size:
            raise ValueError("gamma_set must not contain duplicates.")

    rng = np.random.default_rng(seed)
    span = upper - lower
    dimension = lower.size

    def quantize(x):
        """Project a point onto the precision grid and variable bounds."""

        return np.clip(
            np.round(np.asarray(x, dtype=float), places),
            lower,
            upper,
        )

    def evaluate(x):
        """Evaluate the objective and map non-finite values to infinity."""

        value = float(fun(x.copy()))
        return value if np.isfinite(value) else np.inf

    nfev = njev = 0

    # Initialize the incumbent from an LHS population or the supplied point.
    if mode == "GLOBAL":
        population = quantize(_lhs(population_size, lower, upper, rng))
        values = np.array([evaluate(point) for point in population])
        nfev += population_size
        index = int(np.argmin(values))
        best_x = population[index].copy()
        best_fun = float(values[index])
    else:
        if x0 is None:
            raise ValueError("x0 is required in LOCAL mode.")

        best_x = np.asarray(x0, dtype=float)
        if best_x.shape != (dimension,):
            raise ValueError(f"x0 must have shape ({dimension},).")
        if np.any(~np.isfinite(best_x)):
            raise ValueError("x0 must be finite.")

        best_x = quantize(best_x)
        best_fun = evaluate(best_x)
        nfev += 1

    best_u = (best_x - lower) / span
    history = [best_fun] if record_history else []
    generations = K // population_size

    for _ in range(generations):
        gradient = np.asarray(grad(best_x.copy()), dtype=float)
        njev += 1

        if gradient.shape != (dimension,):
            raise ValueError(f"grad must return shape ({dimension},).")
        if np.any(~np.isfinite(gradient)):
            raise ValueError("grad returned non-finite values.")

        # Scale the gradient by the bound widths and normalize its magnitude.
        scaled = gradient * span
        maximum = float(np.max(np.abs(scaled)))
        direction = (
            np.zeros_like(scaled)
            if maximum == 0
            else np.abs(scaled) / maximum
        )

        # Sample one Gamma and one sign for each candidate coordinate.
        gamma_matrix = rng.choice(
            gamma,
            size=(population_size, dimension),
            replace=True,
        )
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(population_size, dimension),
            replace=True,
        )

        # Generate candidates in normalized space, then map them back.
        candidate_u_raw = np.clip(
            best_u + signs * gamma_matrix * direction,
            0.0,
            1.0,
        )
        candidates = quantize(lower + candidate_u_raw * span)
        candidates_u = (candidates - lower) / span
        values = np.array([evaluate(point) for point in candidates])
        nfev += population_size

        index = int(np.argmin(values))
        candidate_fun = float(values[index])

        # Preserve strict best-so-far elitism.
        if candidate_fun < best_fun:
            best_x = candidates[index].copy()
            best_u = candidates_u[index].copy()
            best_fun = candidate_fun

        if record_history:
            history.append(best_fun)

    return FSGOResult(
        x=best_x,
        fun=best_fun,
        nfev=nfev,
        njev=njev,
        iterations=generations,
        history=history,
    )
