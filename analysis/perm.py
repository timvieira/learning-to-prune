"""
Paired permutation test for reward = F1 - lamdba * runtime (which is not additively
decomposable because F1 is a ratio of expectations)
"""
from __future__ import division
import numpy as np

from arsenal.iterview import iterview
from ldp.prune.example import cgw_prf, cgw_f
from arsenal.terminal import bold, red, blue, yellow, green
from ldp.prune.example import test_statistic


def paired_permutation_test(D1, a, b, tradeoff, threshold=0.05, R=10000, verbose=1):
    """Pair permutation test for the following `reward = F_1 - lambda*pushes`.

    Inputs:
    D1: Dataframe of evaluations.
    a: the name a policy
    b: the name a different policy
    """

    # extract the scores by example for each system
    A = D1[D1.policy == a]
    B = D1[D1.policy == b]
    assert (A.example == B.example).all()
    assert (A.index == B.index).all()

    W = B.want.sum()  # number of thing we want is constant among permutations
    n = len(A.index)

    AC = np.array(A.want_and_got) * 1.0
    AG = np.array(A.got) * 1.0
    A_runtime = np.array(A.pushes) * 1.0

    BC = np.array(B.want_and_got) * 1.0
    BG = np.array(B.got) * 1.0
    B_runtime = np.array(B.pushes) * 1.0

    # observed value of test statistic -- the difference of rewards.
    T_observed = test_statistic(AC, AG, A_runtime,
                                BC, BG, B_runtime,
                                np.zeros(n, dtype=np.int32), W, tradeoff)

    r = 0.0
    for _ in iterview(range(R), msg='perm test'):
        # randomly generate a vector of zeros and ones (uniformly).
        # Note: endpoint not included in np.random.randit (that's why theres a 2).
        flip = np.random.randint(0, 2, size=n).astype(np.int32)
        if test_statistic(AC, AG, A_runtime,
                          BC, BG, B_runtime,
                          flip, W, tradeoff) >= T_observed:
            r += 1
    s = (r+1)/(R+1)

    # observed rewards
    ra = cgw_f(AC.sum(), AG.sum(), W) - tradeoff*A_runtime.mean()
    rb = cgw_f(BC.sum(), BG.sum(), W) - tradeoff*B_runtime.mean()

    if verbose:
        # which system has higher reward? is it significant?
        asig = (red % bold) if ra > rb and s <= 0.05 else '%s'
        bsig = (blue % bold) if rb > ra and s <= 0.05 else '%s'
        any_sig = bold if s <= threshold else yellow

        print asig % 'R(A) = %g (%s)' % (ra, a)
        print bsig % 'R(B) = %g (%s)' % (rb, b)
        print any_sig % 'confidence = %g' % (1-s)
        print

    if s <= threshold:
        return s, -1 if ra > rb else +1
    else:
        return s, 0   # "statistical tie"
