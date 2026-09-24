"""Synthetic illustration of OMZ v3 algorithms, not paper-number reproduction."""
import numpy as np

from omz import (allowed, feasible, envelope, joint_kh, kh_candidates,
                 sample_endpoints, empirical_mass, mass_sets, classify,
                 empirical_crps, compare_scores, rank)


def main():
    # Low=50 (<60); the unobserved fourth node may be low or high.
    y = np.array([50., 50., 50., np.nan, 50., 50., 50.])
    valid = np.isfinite(y)
    k, spacing_m = 3, 10.
    low, high, _ = allowed(y, valid)
    marginal = feasible(low, high, k)
    joint = joint_kh(low, high, k)
    product, filtered = kh_candidates(marginal, k)
    pairs = lambda a: [tuple(map(int, x)) for x in np.argwhere(a)]
    print('Synthetic pattern: 111?111; k=3; spacing=10 m')
    print('Pairs are (K,H in grid steps); multiply H by 10 for metres.')
    print('Marginal product:', pairs(product))
    print('Global filtered: ', pairs(filtered))
    print('Exact joint:     ', pairs(joint))
    print('Exact H:', np.flatnonzero(marginal[4]).tolist(),
          '; envelope H:', np.flatnonzero(envelope(marginal)[4]).tolist())

    # A legitimate complete predicted curve has K=1,H=4 (five low nodes).
    # It is globally feasible but incompatible with the particular observations.
    prediction_curve = np.array([[50., 50., 50., 50., 50., 70., 70.]])
    endpoints = sample_endpoints(prediction_curve, k)
    n = len(y)
    category = endpoints[:, 3] * n + endpoints[:, 4]
    probability = empirical_mass(category[None, :], joint.size)[0]
    fixed_set = mass_sets(probability, (.9,))[0]
    labels = ('inclusion', 'exclusion', 'indeterminate')
    decisions = [int(classify(t.ravel(), fixed_set))
                 for t in (product, filtered, joint)]
    print('Same fixed 90% predictive set:', pairs(fixed_set.reshape(joint.shape)))
    print('Verdicts [product, global, joint]:', [labels[i] for i in decisions])
    assert decisions == [2, 2, 1]

    # Compare two valid synthetic predictors on the same feasible H values.
    # A always predicts H=0; B always predicts H=20 m. Smaller CRPS is better.
    samples_a = np.zeros((1, 4))
    samples_b = np.full((1, 4), 20.)
    grid_m = np.arange(n) * spacing_m
    score_a = empirical_crps(samples_a, grid_m)[0]
    score_b = empirical_crps(samples_b, grid_m)[0]
    common, separate = compare_scores(score_a, score_b, marginal[4, :n])
    print('CRPS A-B separate bounds (m):', separate.tolist())
    print('CRPS A-B common-truth bounds (m):', common.tolist())
    print('Rank at gamma=5 m [separate, common]:',
          [int(rank(separate, 5.)), int(rank(common, 5.))])
    assert np.array_equal(common, [20., 20.])
    assert np.array_equal(separate, [0., 40.])
    assert rank(separate, 5.) == 0 and rank(common, 5.) == 1
    print('PASS: demonstration assertions; no original experiment data used.')


if __name__ == '__main__':
    main()
