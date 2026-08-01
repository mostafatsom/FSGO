# FSGO

**Flexible Stochastic Gradient Optimizer**

This repository contains a standalone Python implementation of the Flexible
Stochastic Gradient Optimizer (FSGO), an optimization algorithm introduced in:

> M. Ilchi Ghazaan and M. Sharifi, “A Two-Phase Metamodel-Driven Approach for
> Topology and Size Optimization of Truss Structures,” *International Journal
> of Optimization in Civil Engineering*, 15(2), 181–201, 2025.  
> DOI: `10.22068/ijoce.2025.15.2.630`

The complete optimizer is intentionally kept in one file:
[`fsgo.py`](fsgo.py).

## Scope

This repository implements only the standalone FSGO optimization algorithm.

It does not reproduce the complete two-phase methodology presented in the
associated paper. In particular, it does not include the metamodel-training
workflow, adaptive sampling, Extensive Constraints, structural analysis,
topology-optimization workflow, or truss case studies.

## Installation

Clone or download this repository, open its root directory, and run:

```bash
python -m pip install .
```

For direct use without installation, place `fsgo.py` beside your own script.

## Quick start

```python
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
    seed=42,
)

print(result.fun)
print(result.x)
```

## Algorithm summary

1. `GLOBAL` mode initializes a Latin Hypercube population; `LOCAL` mode starts
   from `x0`.
2. The gradient is evaluated at the current best point.
3. Each candidate coordinate independently samples a Gamma value and a
   positive or negative sign.
4. Candidate steps are generated in normalized bounded coordinates and
   quantized to `precision_step`.
5. The best candidate replaces the incumbent only after a strict improvement.
6. The Gamma set remains fixed throughout the run.

For `precision_step=1e-4`, the automatically generated Gamma set is:

```python
[0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
```

## Examples

| Example | Purpose | Command |
|---|---|---|
| `basic_global.py` | Global optimization of Sphere | `python examples/basic_global.py` |
| `local_mode.py` | Local optimization of Rosenbrock | `python examples/local_mode.py` |
| `custom_gamma.py` | Fixed custom Gamma set on Rastrigin | `python examples/custom_gamma.py` |

## API

```python
minimize_fsgo(
    fun,
    grad,
    bounds,
    *,
    mode="GLOBAL",
    gamma_set=None,
    population_size=30,
    K=15000,
    x0=None,
    seed=None,
    precision_step=1e-4,
    record_history=True,
)
```

| Parameter | Description | Default |
|---|---|---:|
| `fun` | Objective function | required |
| `grad` | Analytic gradient | required |
| `bounds` | Finite `(lower, upper)` pair for every variable | required |
| `mode` | `GLOBAL` or `LOCAL` | `GLOBAL` |
| `gamma_set` | Fixed Gamma values; generated automatically when omitted | `None` |
| `population_size` | Candidates generated per complete generation | `30` |
| `K` | Candidate evaluations after initialization | `15000` |
| `x0` | Initial point required in `LOCAL` mode | `None` |
| `seed` | NumPy random seed | `None` |
| `precision_step` | Power-of-ten quantization grid | `1e-4` |
| `record_history` | Store the best objective after each generation | `True` |

Only complete generations are executed, so the generation count is
`K // population_size`. Global initialization adds `population_size`
objective evaluations, while local initialization adds one.

## Result

`minimize_fsgo` returns an `FSGOResult` object:

```text
x           best point
fun         best objective value
nfev        objective evaluations
njev        gradient evaluations
iterations  completed generations
history     best objective after initialization and each generation
```

## Simple comparison

A reproducible comparison with Differential Evolution and Random Search is
included. The methods use the same bounds, precision grid, seeds, Latin
Hypercube initialization procedure, and objective-evaluation budget.

FSGO uses an analytic gradient. Gradient evaluations are reported separately
and are not counted as objective evaluations.

Quick validation run:

```bash
python comparison.py --quick
```

Complete run:

```bash
python comparison.py --full
```

Raw results are written to `comparison_results.csv`. The comparison is an
example of reproducible evaluation and is not a claim of universal
superiority.

## Testing

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the tests and the basic example on supported Python
versions after every push and pull request.

## Requirements

```text
Python >= 3.10
NumPy >= 1.26
```

## Citation

GitHub reads citation metadata from [`CITATION.cff`](CITATION.cff).

When using FSGO in academic work, cite the associated paper listed at the top
of this README.

## License

MIT License.
