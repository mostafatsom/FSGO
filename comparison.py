"""Compare FSGO with Differential Evolution and Random Search.

The comparison uses:

- the same objective-evaluation budget;
- the same Latin Hypercube initial population;
- the same bounds, precision grid, dimensions, and seeds.

FSGO uses an analytic gradient. DE and Random Search are gradient-free, so
gradient evaluations are reported separately rather than included in the
objective-evaluation budget.

Run a small validation comparison:

    python comparison.py --quick

Run the complete comparison:

    python comparison.py --full
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from fsgo import minimize_fsgo


Array = np.ndarray
Objective = Callable[[Array], float]
Gradient = Callable[[Array], Array]


@dataclass(frozen=True)
class Problem:
    name: str
    objective: Objective
    gradient: Gradient
    lower: float
    upper: float
    optimum: float = 0.0


def sphere(x: Array) -> float:
    return float(np.dot(x, x))


def sphere_gradient(x: Array) -> Array:
    return 2.0 * x


def rosenbrock(x: Array) -> float:
    return float(
        np.sum(
            100.0 * (x[1:] - x[:-1] ** 2) ** 2
            + (1.0 - x[:-1]) ** 2
        )
    )


def rosenbrock_gradient(x: Array) -> Array:
    gradient = np.zeros_like(x)
    difference = x[1:] - x[:-1] ** 2

    gradient[:-1] += (
        -400.0 * x[:-1] * difference
        - 2.0 * (1.0 - x[:-1])
    )
    gradient[1:] += 200.0 * difference

    return gradient


def rastrigin(x: Array) -> float:
    return float(
        10.0 * x.size
        + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))
    )


def rastrigin_gradient(x: Array) -> Array:
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)


def ackley(x: Array) -> float:
    dimension = x.size
    mean_square = np.sum(x**2) / dimension
    mean_cosine = np.sum(np.cos(2.0 * np.pi * x)) / dimension

    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(mean_square))
        - np.exp(mean_cosine)
        + 20.0
        + math.e
    )


def ackley_gradient(x: Array) -> Array:
    dimension = x.size
    radius = float(np.sqrt(np.sum(x**2) / dimension))

    if radius == 0.0:
        first_term = np.zeros_like(x)
    else:
        first_term = (
            4.0
            * np.exp(-0.2 * radius)
            * x
            / (dimension * radius)
        )

    mean_cosine = np.sum(np.cos(2.0 * np.pi * x)) / dimension
    second_term = (
        np.exp(mean_cosine)
        * (2.0 * np.pi / dimension)
        * np.sin(2.0 * np.pi * x)
    )

    return first_term + second_term


def griewank(x: Array) -> float:
    indices = np.arange(1, x.size + 1, dtype=float)
    cosine = np.cos(x / np.sqrt(indices))

    return float(
        1.0
        + np.sum(x**2) / 4000.0
        - np.prod(cosine)
    )


def griewank_gradient(x: Array) -> Array:
    indices = np.arange(1, x.size + 1, dtype=float)
    roots = np.sqrt(indices)
    angles = x / roots
    cosine = np.cos(angles)
    sine = np.sin(angles)

    gradient = x / 2000.0

    for index in range(x.size):
        other_product = np.prod(np.delete(cosine, index))
        gradient[index] += (
            sine[index] * other_product / roots[index]
        )

    return gradient


def zakharov(x: Array) -> float:
    indices = np.arange(1, x.size + 1, dtype=float)
    weighted_sum = float(np.dot(0.5 * indices, x))

    return float(
        np.dot(x, x)
        + weighted_sum**2
        + weighted_sum**4
    )


def zakharov_gradient(x: Array) -> Array:
    indices = np.arange(1, x.size + 1, dtype=float)
    weighted_sum = float(np.dot(0.5 * indices, x))

    return (
        2.0 * x
        + indices * (weighted_sum + 2.0 * weighted_sum**3)
    )


PROBLEMS = {
    "sphere": Problem(
        "Sphere",
        sphere,
        sphere_gradient,
        -5.12,
        5.12,
    ),
    "rosenbrock": Problem(
        "Rosenbrock",
        rosenbrock,
        rosenbrock_gradient,
        -5.0,
        10.0,
    ),
    "rastrigin": Problem(
        "Rastrigin",
        rastrigin,
        rastrigin_gradient,
        -5.12,
        5.12,
    ),
    "ackley": Problem(
        "Ackley",
        ackley,
        ackley_gradient,
        -32.768,
        32.768,
    ),
    "griewank": Problem(
        "Griewank",
        griewank,
        griewank_gradient,
        -600.0,
        600.0,
    ),
    "zakharov": Problem(
        "Zakharov",
        zakharov,
        zakharov_gradient,
        -5.0,
        10.0,
    ),
}


def precision_places(step: float) -> int:
    places = int(round(-math.log10(step)))

    if not math.isclose(
        step,
        10.0 ** (-places),
        rel_tol=0.0,
        abs_tol=max(1e-15, step * 1e-12),
    ):
        raise ValueError("precision_step must be a power of ten.")

    return places


def latin_hypercube(
    count: int,
    lower: Array,
    upper: Array,
    rng: np.random.Generator,
) -> Array:
    """Match the Latin Hypercube initialization used by FSGO."""

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


def evaluate_population(
    objective: Objective,
    population: Array,
) -> Array:
    values = np.empty(population.shape[0], dtype=float)

    for index, point in enumerate(population):
        value = float(objective(point.copy()))
        values[index] = value if np.isfinite(value) else np.inf

    return values


def make_bounds(
    problem: Problem,
    dimension: int,
) -> tuple[Array, Array, list[tuple[float, float]]]:
    lower = np.full(dimension, problem.lower, dtype=float)
    upper = np.full(dimension, problem.upper, dtype=float)
    bounds = list(zip(lower, upper))

    return lower, upper, bounds


def initial_population(
    problem: Problem,
    dimension: int,
    seed: int,
    population_size: int,
    precision_step: float,
) -> tuple[Array, Array]:
    lower, upper, _ = make_bounds(problem, dimension)
    places = precision_places(precision_step)
    rng = np.random.default_rng(seed)

    population = latin_hypercube(
        population_size,
        lower,
        upper,
        rng,
    )
    population = np.clip(
        np.round(population, places),
        lower,
        upper,
    )
    values = evaluate_population(problem.objective, population)

    return population, values


def run_fsgo(
    problem: Problem,
    dimension: int,
    seed: int,
    budget: int,
    population_size: int,
    precision_step: float,
) -> dict:
    _, _, bounds = make_bounds(problem, dimension)
    candidate_budget = budget - population_size

    started = time.perf_counter()
    result = minimize_fsgo(
        problem.objective,
        problem.gradient,
        bounds,
        mode="GLOBAL",
        population_size=population_size,
        K=candidate_budget,
        seed=seed,
        precision_step=precision_step,
        record_history=False,
    )
    runtime = time.perf_counter() - started

    return {
        "algorithm": "FSGO",
        "best_value": float(result.fun),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "runtime_seconds": runtime,
    }


def run_de(
    problem: Problem,
    dimension: int,
    seed: int,
    budget: int,
    population_size: int,
    precision_step: float,
    differential_weight: float = 0.8,
    crossover_rate: float = 0.9,
) -> dict:
    lower, upper, _ = make_bounds(problem, dimension)
    places = precision_places(precision_step)
    rng = np.random.default_rng(seed)

    population = latin_hypercube(
        population_size,
        lower,
        upper,
        rng,
    )
    population = np.clip(
        np.round(population, places),
        lower,
        upper,
    )
    values = evaluate_population(problem.objective, population)
    nfev = population_size

    started = time.perf_counter()

    while nfev + population_size <= budget:
        trials = np.empty_like(population)

        for target_index in range(population_size):
            available = np.delete(
                np.arange(population_size),
                target_index,
            )
            first, second, third = rng.choice(
                available,
                size=3,
                replace=False,
            )

            mutant = (
                population[first]
                + differential_weight
                * (population[second] - population[third])
            )
            mutant = np.clip(mutant, lower, upper)

            crossover = rng.random(dimension) < crossover_rate
            crossover[rng.integers(dimension)] = True

            trial = np.where(
                crossover,
                mutant,
                population[target_index],
            )
            trials[target_index] = np.clip(
                np.round(trial, places),
                lower,
                upper,
            )

        trial_values = evaluate_population(
            problem.objective,
            trials,
        )
        nfev += population_size

        improved = trial_values <= values
        population[improved] = trials[improved]
        values[improved] = trial_values[improved]

    runtime = time.perf_counter() - started

    return {
        "algorithm": "Differential Evolution",
        "best_value": float(np.min(values)),
        "nfev": nfev,
        "njev": 0,
        "runtime_seconds": runtime,
    }


def run_random_search(
    problem: Problem,
    dimension: int,
    seed: int,
    budget: int,
    population_size: int,
    precision_step: float,
) -> dict:
    lower, upper, _ = make_bounds(problem, dimension)
    places = precision_places(precision_step)
    rng = np.random.default_rng(seed)

    population = latin_hypercube(
        population_size,
        lower,
        upper,
        rng,
    )
    population = np.clip(
        np.round(population, places),
        lower,
        upper,
    )
    values = evaluate_population(problem.objective, population)
    best_value = float(np.min(values))
    nfev = population_size

    started = time.perf_counter()

    while nfev < budget:
        batch_size = min(population_size, budget - nfev)

        candidates = rng.uniform(
            lower,
            upper,
            size=(batch_size, dimension),
        )
        candidates = np.clip(
            np.round(candidates, places),
            lower,
            upper,
        )
        candidate_values = evaluate_population(
            problem.objective,
            candidates,
        )

        best_value = min(
            best_value,
            float(np.min(candidate_values)),
        )
        nfev += batch_size

    runtime = time.perf_counter() - started

    return {
        "algorithm": "Random Search",
        "best_value": best_value,
        "nfev": nfev,
        "njev": 0,
        "runtime_seconds": runtime,
    }


def average_ranks(rows: list[dict]) -> dict[str, float]:
    rank_totals: dict[str, float] = {}
    rank_counts: dict[str, int] = {}

    blocks: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        key = (
            row["problem"],
            row["dimension"],
            row["seed"],
        )
        blocks.setdefault(key, []).append(row)

    for block in blocks.values():
        ordered = sorted(block, key=lambda row: row["best_value"])
        position = 0

        while position < len(ordered):
            end = position + 1

            while (
                end < len(ordered)
                and math.isclose(
                    ordered[end]["best_value"],
                    ordered[position]["best_value"],
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                end += 1

            average_rank = (
                (position + 1) + end
            ) / 2.0

            for row in ordered[position:end]:
                name = row["algorithm"]
                rank_totals[name] = (
                    rank_totals.get(name, 0.0)
                    + average_rank
                )
                rank_counts[name] = (
                    rank_counts.get(name, 0)
                    + 1
                )

            position = end

    return {
        algorithm: rank_totals[algorithm] / rank_counts[algorithm]
        for algorithm in rank_totals
    }


def pairwise_counts(
    rows: list[dict],
    opponent: str,
) -> tuple[int, int, int]:
    blocks: dict[tuple[str, int, int], dict[str, float]] = {}

    for row in rows:
        key = (
            row["problem"],
            row["dimension"],
            row["seed"],
        )
        blocks.setdefault(key, {})[
            row["algorithm"]
        ] = row["best_value"]

    wins = ties = losses = 0

    for values in blocks.values():
        fsgo_value = values["FSGO"]
        opponent_value = values[opponent]

        if math.isclose(
            fsgo_value,
            opponent_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            ties += 1
        elif fsgo_value < opponent_value:
            wins += 1
        else:
            losses += 1

    return wins, ties, losses


def write_csv(rows: list[dict], output: Path) -> None:
    fields = [
        "problem",
        "dimension",
        "seed",
        "algorithm",
        "initial_best",
        "best_value",
        "gap",
        "nfev",
        "njev",
        "runtime_seconds",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FSGO, Differential Evolution, and Random Search."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Run a small validation comparison.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run the complete 6-function comparison.",
    )
    parser.add_argument(
        "--output",
        default="comparison_results.csv",
        help="Output CSV path.",
    )
    arguments = parser.parse_args()

    population_size = 30
    precision_step = 1e-4

    if arguments.quick:
        problem_names = [
            "sphere",
            "rastrigin",
            "zakharov",
        ]
        dimensions = [10]
        seeds = range(3)
        budget = 3030
    else:
        problem_names = list(PROBLEMS)
        dimensions = [10, 30]
        seeds = range(30)
        budget = 15030

    if (budget - population_size) % population_size != 0:
        raise ValueError(
            "budget - population_size must form complete generations."
        )

    rows: list[dict] = []
    total_blocks = (
        len(problem_names)
        * len(dimensions)
        * len(seeds)
    )
    completed_blocks = 0

    for problem_name in problem_names:
        problem = PROBLEMS[problem_name]

        for dimension in dimensions:
            for seed in seeds:
                _, initial_values = initial_population(
                    problem,
                    dimension,
                    seed,
                    population_size,
                    precision_step,
                )
                initial_best = float(np.min(initial_values))

                results = [
                    run_fsgo(
                        problem,
                        dimension,
                        seed,
                        budget,
                        population_size,
                        precision_step,
                    ),
                    run_de(
                        problem,
                        dimension,
                        seed,
                        budget,
                        population_size,
                        precision_step,
                    ),
                    run_random_search(
                        problem,
                        dimension,
                        seed,
                        budget,
                        population_size,
                        precision_step,
                    ),
                ]

                for result in results:
                    rows.append({
                        "problem": problem.name,
                        "dimension": dimension,
                        "seed": seed,
                        "algorithm": result["algorithm"],
                        "initial_best": initial_best,
                        "best_value": result["best_value"],
                        "gap": abs(
                            result["best_value"]
                            - problem.optimum
                        ),
                        "nfev": result["nfev"],
                        "njev": result["njev"],
                        "runtime_seconds": result[
                            "runtime_seconds"
                        ],
                    })

                completed_blocks += 1
                print(
                    f"[{completed_blocks}/{total_blocks}] "
                    f"{problem.name} D={dimension} seed={seed}"
                )

    output = Path(arguments.output)
    write_csv(rows, output)

    ranks = average_ranks(rows)
    print("\nAverage rank")
    for algorithm, rank in sorted(
        ranks.items(),
        key=lambda item: item[1],
    ):
        print(f"  {algorithm:24s} {rank:.4f}")

    for opponent in [
        "Differential Evolution",
        "Random Search",
    ]:
        wins, ties, losses = pairwise_counts(rows, opponent)
        print(
            f"\nFSGO vs {opponent}: "
            f"{wins} wins, {ties} ties, {losses} losses"
        )

    print(f"\nResults saved to: {output.resolve()}")
    print(
        "Objective evaluations are equal. "
        "Gradient evaluations are reported separately."
    )


if __name__ == "__main__":
    main()
