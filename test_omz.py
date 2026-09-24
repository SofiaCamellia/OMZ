"""Independent finite-grid checks; standard-library unittest + NumPy only."""
import itertools
import unittest

import numpy as np

from _oracle import exhaustive
from omz import (TOL, allowed, feasible, envelope, sample_endpoints, joint_kh,
                 global_feasible_pairs_steps, kh_candidates, empirical_mass,
                 categorical_scores, mass_sets, empirical_crps, classify,
                 score_bounds, compare_scores, rank)


class OMZTests(unittest.TestCase):
    def test_exhaustive_partial_sequences(self):
        # All 3^6 partial sequences; k includes no qualifying run and k=1.
        # The extracted oracle enumerates completions and scans explicit runs;
        # it calls neither production DP nor the production complete scan.
        settings, joint_settings = 0, 0
        for pattern in itertools.product((-1, 0, 1), repeat=6):
            p = np.asarray(pattern)
            low, high = p != 0, p != 1
            for k in (1, 2, 3, 4, 7):
                expected, expected_joint = exhaustive(p, k)
                actual = feasible(low, high, k)
                np.testing.assert_array_equal(actual, expected)
                count = np.flatnonzero(actual[3])
                np.testing.assert_array_equal(count, np.arange(count[0], count[-1]+1))
                # E/H monotone extrema by independent complete-profile scans.
                extrema = sample_endpoints(np.where(np.stack((~high, low)), 50., 70.), k)
                for endpoint in (0, 4):
                    values = np.flatnonzero(actual[endpoint])
                    self.assertEqual(extrema[0, endpoint], values[0])
                    self.assertEqual(extrema[1, endpoint], values[-1])
                if k >= 2:
                    joint = joint_kh(low, high, k)
                    self.assertEqual(set(map(tuple, np.argwhere(joint))), expected_joint)
                    np.testing.assert_array_equal(joint.any(0), actual[4, :6])
                    np.testing.assert_array_equal(np.flatnonzero(joint.any(1)), count)
                    product, filtered = kh_candidates(actual, k)
                    self.assertFalse(np.any(joint & ~filtered))
                    self.assertFalse(np.any(filtered & ~product))
                    joint_settings += 1
                settings += 1
        self.assertEqual(settings, 3645)
        self.assertEqual(joint_settings, 2916)

    def test_global_feasibility_against_complete_enumeration(self):
        settings = 0
        for n in range(1, 11):
            for k in range(1, n+3):
                expected = set()
                for bits in itertools.product((0, 1), repeat=n):
                    expected.update(exhaustive(bits, k)[1])
                self.assertEqual(global_feasible_pairs_steps(n, k), expected)
                settings += 1
        self.assertEqual(settings, 75)
        self.assertEqual(len(global_feasible_pairs_steps(101, 3)), 1276)

    def test_full_paper_grid_all_unknown(self):
        joint = joint_kh(np.ones(101, bool), np.ones(101, bool), 3)
        self.assertEqual(set(map(tuple, np.argwhere(joint))),
                         global_feasible_pairs_steps(101, 3))
        self.assertEqual(int(joint.sum()), 1276)

    def test_threshold_missingness_and_error_nesting(self):
        low, high, ambiguous = allowed([58, 59, 60, 61, 62], np.ones(5, bool), 1.)
        np.testing.assert_array_equal(low, [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(high, [0, 1, 1, 1, 1])
        np.testing.assert_array_equal(ambiguous, [0, 1, 1, 0, 0])
        low, high, _ = allowed([np.nan, 60.], np.array([False, True]))
        np.testing.assert_array_equal(low, [True, False])
        np.testing.assert_array_equal(high, [True, True])
        previous = None
        for eps in (0., 1., 2., 5.):
            lo, hi, _ = allowed([59, 61, 59, 61, 59], np.ones(5, bool), eps)
            current = joint_kh(lo, hi)
            if previous is not None:
                self.assertFalse(np.any(previous & ~current))
            previous = current
        # Threshold parameter adaptation must retain scanner/allowed agreement.
        samples = np.array([[.2, .2, .2, .8], [.8, .8, .8, .8]])
        result = sample_endpoints(samples, threshold=.5)
        for curve, row in zip(samples, result):
            lo, hi, _ = allowed(curve, np.ones(4, bool), threshold=.5)
            expected = feasible(lo, hi)
            self.assertEqual(int(expected.sum()), 5)
            self.assertTrue(expected[np.arange(5), row].all())

    def test_absent_is_not_a_numeric_hull_endpoint(self):
        truth = np.zeros((5, 7), bool)
        truth[0, 0] = True
        truth[1, [1, 3, 6]] = True  # 6 is ABSENT for J=6
        truth[2, 6] = True
        truth[3, [0, 2]] = True
        truth[4, [0, 4]] = True
        hull = envelope(truth)
        np.testing.assert_array_equal(np.flatnonzero(hull[1]), [1, 2, 3, 6])
        np.testing.assert_array_equal(np.flatnonzero(hull[2]), [6])
        self.assertFalse(np.any(truth & ~hull))

    def test_observation_specific_joint_gain(self):
        p = np.array([1, 1, 1, -1, 1, 1, 1])
        joint = joint_kh(p != 0, p != 1)
        product, filtered = kh_candidates(feasible(p != 0, p != 1))
        self.assertEqual(set(map(tuple, np.argwhere(joint))), {(1, 6), (2, 4)})
        self.assertEqual(set(map(tuple, np.argwhere(filtered))), {(1, 4), (1, 6), (2, 4)})
        prediction = np.zeros_like(joint)
        prediction[1, 4] = True
        self.assertEqual([int(classify(x.ravel(), prediction.ravel()))
                          for x in (product, filtered, joint)], [2, 2, 1])

    def test_proper_scores_and_mass_set_ties(self):
        p = np.array([[.1, .7, .2], [0., 1., 0.]])
        scores = categorical_scores(p)
        for i in range(2):
            for category in range(3):
                self.assertAlmostEqual(scores[i, category],
                                       np.sum((p[i]-np.eye(3)[category])**2))
        np.testing.assert_allclose(empirical_mass([[0, 2, 2, 2]], 3), [[.25, 0., .75]])
        levels = (.25, .5, .75, 1.)
        equal = np.full(4, .25)
        sets = mass_sets(equal, levels)
        for i, level in enumerate(levels):
            np.testing.assert_array_equal(np.flatnonzero(sets[i]), np.arange(i+1))
            self.assertGreaterEqual(float(equal[sets[i]].sum()), level)
        self.assertFalse(np.any(sets[:-1] & ~sets[1:]))
        x = np.array([[0., 20., 40.], [10., 10., 10.]])
        grid = np.arange(5)*10.
        crps = empirical_crps(x, grid)
        for i in range(2):
            expected = np.abs(x[i, :, None]-grid).mean(0)
            expected -= .5*np.abs(x[i, :, None]-x[i, None, :]).mean()
            np.testing.assert_allclose(crps[i], expected)

    def test_common_truth_and_strict_gamma_certificates(self):
        rng = np.random.default_rng(2)
        for _ in range(100):
            scores = categorical_scores(rng.dirichlet(np.ones(8), 2))
            truth = rng.random(8) < .5
            truth[0] = True
            common, separate = compare_scores(*scores, truth)
            self.assertGreaterEqual(common[0], separate[0]-TOL)
            self.assertLessEqual(common[1], separate[1]+TOL)
            actual = (scores[0]-scores[1])[truth]
            self.assertTrue(np.all((actual >= common[0]-TOL) & (actual <= common[1]+TOL)))
            old_rank = rank(common, 0.)
            for gamma in (0., .01, .05):
                verdict = rank(common, gamma)
                if verdict == -1:
                    self.assertTrue(np.all(actual < -gamma-TOL))
                elif verdict == 1:
                    self.assertTrue(np.all(actual > gamma+TOL))
                if verdict != 0:
                    self.assertEqual(verdict, old_rank)
            np.testing.assert_array_equal(compare_scores(scores[0], scores[0], truth)[0], [0., 0.])
        common, separate = compare_scores([40., 60.], [20., 40.], [True, True])
        np.testing.assert_array_equal(common, [20., 20.])
        np.testing.assert_array_equal(separate, [0., 40.])
        self.assertEqual(rank(common, 5.), 1)
        self.assertEqual(rank(separate, 5.), 0)
        self.assertEqual(rank(common, 20.), 0)
        for gamma in (0., 1., 2., 5.):
            np.testing.assert_array_equal(rank([[-gamma-TOL, -gamma-TOL],
                                                [gamma+TOL, gamma+TOL]], gamma), [0, 0])

    def test_set_refinement_never_reverses_a_certificate(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            fine = rng.random(8) < .3
            fine[0] = True
            coarse = fine | (rng.random(8) < .5)
            prediction = rng.random(8) < .5
            coarse_result = classify(coarse, prediction)
            if coarse_result != 2:
                self.assertEqual(classify(fine, prediction), coarse_result)

    def test_invalid_inputs_are_rejected(self):
        calls = (
            lambda: joint_kh([], []),
            lambda: joint_kh([False, True], [False, True]),
            lambda: joint_kh([True, True], [True, True], 1),
            lambda: feasible([True, True], [True, True], 2.5),
            lambda: allowed([1., 2.], np.ones(2, bool), np.nan),
            lambda: allowed([1., 2.], np.ones(2, bool), threshold=np.inf),
            lambda: empirical_mass(np.zeros((1, 0)), 3),
            lambda: mass_sets([.2, .3], (.9,)),
            lambda: empirical_crps(np.zeros((1, 0)), [0.]),
            lambda: classify(np.zeros(3, bool), np.ones(3, bool)),
            lambda: score_bounds([1., 2.], [False, False]),
            lambda: rank([2., 1.]),
            lambda: rank([0., 1.], np.nan),
            lambda: global_feasible_pairs_steps(1.5, 2),
        )
        for function in calls:
            with self.subTest(call=function), self.assertRaises(ValueError):
                function()


if __name__ == '__main__':
    unittest.main(verbosity=2)
