import unittest

import numpy as np

from fsgo import minimize_fsgo


class TestFSGO(unittest.TestCase):
    @staticmethod
    def sphere(x):
        return float(np.dot(x, x))

    @staticmethod
    def sphere_gradient(x):
        return 2.0 * x

    def test_global_evaluation_counts(self):
        result = minimize_fsgo(
            self.sphere,
            self.sphere_gradient,
            bounds=[(-2.0, 2.0)] * 3,
            population_size=6,
            K=60,
            seed=7,
        )

        self.assertTrue(np.isfinite(result.fun))
        self.assertEqual(result.nfev, 66)
        self.assertEqual(result.njev, 10)
        self.assertEqual(result.iterations, 10)

    def test_local_evaluation_counts(self):
        result = minimize_fsgo(
            self.sphere,
            self.sphere_gradient,
            bounds=[(-2.0, 2.0)] * 3,
            mode="LOCAL",
            x0=[1.0, -1.0, 0.5],
            population_size=6,
            K=60,
            seed=7,
        )

        self.assertEqual(result.nfev, 61)
        self.assertEqual(result.njev, 10)

    def test_reproducibility(self):
        arguments = dict(
            fun=self.sphere,
            grad=self.sphere_gradient,
            bounds=[(-2.0, 2.0)] * 3,
            population_size=6,
            K=60,
            seed=11,
        )

        first = minimize_fsgo(**arguments)
        second = minimize_fsgo(**arguments)

        np.testing.assert_array_equal(first.x, second.x)
        self.assertEqual(first.fun, second.fun)
        self.assertEqual(first.history, second.history)

    def test_history_is_best_so_far(self):
        result = minimize_fsgo(
            self.sphere,
            self.sphere_gradient,
            bounds=[(-2.0, 2.0)] * 3,
            population_size=6,
            K=60,
            seed=13,
        )

        differences = np.diff(np.asarray(result.history))
        self.assertTrue(np.all(differences <= 0.0))

    def test_custom_gamma_and_bounds(self):
        result = minimize_fsgo(
            self.sphere,
            self.sphere_gradient,
            bounds=[(-1.0, 1.0)] * 2,
            gamma_set=[0.0, 0.001, 0.01, 0.1, 1.0],
            population_size=4,
            K=40,
            seed=3,
        )

        self.assertTrue(np.all(result.x >= -1.0))
        self.assertTrue(np.all(result.x <= 1.0))


if __name__ == "__main__":
    unittest.main()
