"""Test cases for second-order expectation semiring implementation of the dp-alg
for bit-flip rollouts.

"""
from __future__ import division
import numpy as np
import pylab as pl
from pandas import DataFrame
from arsenal.math import compare
from arsenal.iterview import progress
from arsenal.terminal import colors

from ldp.dp.risk2 import InsideOut
#from ldp.dp.risk import InsideOut
#from ldp.dp.test.onebest2 import nice, rollout, dump_items, compare_hypergraph_to_cky


def relative_error(x, y):
    if x == 0:
        return float(abs(x - y) > 1e-20)
    else:
        return abs(x - y)/abs(x)


def _test_gradient(example, grammar, m):
    for gamma in [1]:
        print 'gamma =', gamma
        print 'prune = %s' % progress(len(example.nodes)-sum(m[x] for x in example.nodes), len(example.nodes))
        if gamma == 1:
            annealed_grammar = grammar
        else:
            annealed_grammar = grammar.anneal(gamma)
        __test_gradient(example, annealed_grammar, m*1, gamma)


def cross_section(example, grammar, steps, M, x):
    M = M.copy()
    ddd = []
    for a in np.linspace(0,1,10):
        M[x] = a
        foo = InsideOut(example, grammar, M*1.0, steps=steps, with_gradient=False)
        ddd.append([a, foo.num, foo.den])
    [xs, y1, y2] = np.array(ddd).T
    pl.figure()
    pl.plot(xs, y1, c='b', alpha=0.5)
    pl.plot([xs[0], xs[-1]], [y1.min(), y1.max()], c='r', alpha=0.5)
    pl.title('xsection num')
    pl.figure()
    pl.plot(xs, y2, c='b', alpha=0.5)
    pl.plot([xs[0], xs[-1]], [y2.min(), y2.max()], c='r', alpha=0.5)
    pl.title('xsection den')
    pl.show()


class Errors:
    data = []
    @classmethod
    def show(cls):
        df = DataFrame(cls.data)
        #print df
        del df['tie']
        M = df.groupby(['action', 'delta']).mean()
        C = df.groupby(['action', 'delta']).count()
        S = df.groupby(['action', 'delta']).sum()
        M['rate'] = M['n_error']
        M['n_error'] = S['n_error']
        M['total'] = C['n_error']
        print
        print colors.yellow % 'Error types'
        print colors.yellow % '================================='
        print M


def __test_gradient(example, grammar, m, gamma):
    """
    Finite-difference test for gradient of numerator and denominator.
    """

    assert gamma == 1, 'gamma = %g no longer supported' % gamma

    M = m*1.0
    steps = 2

#    print 'steps = %s' % steps
#    print colors.cyan % '>>> roll-in'

    test_grad      = 1
#    test_onebest   = 0
    test_linearity = 0
#    test_viterbi   = 0    # test that hg-viterbi alg matches cky
#    show_items     = 0
#    get_marginals  = 0

#    if show_items:
#        get_marginals = 1

#    if test_viterbi:
#        compare_hypergraph_to_cky(grammar, example, m, steps)

    f_c = InsideOut(example, grammar, M*1.0, steps=steps, with_gradient=True)
#    f_c = InsideOut2(example, grammar, M*1.0, steps=steps, with_gradient=True, DEBUG=True, IOSPEEDUP=True)
#    f_c = InsideOut2(example, grammar, M*1.0, steps=steps, with_gradient=True, DEBUG=True, IOSPEEDUP=False)

    est = f_c.est

#    import ldp.dp.risk2
#    io2 = ldp.dp.risk2.InsideOut(example, grammar, M*1.0, steps=steps, with_gradient=True)
#    debug = 0
#    from arsenal.math import assert_equal
#    zoo = []
#    for k,v in io2.est.items():
#        assert_equal(v, est[k], name='est %s' % (k,), throw=0, tol=1e-4)
#        zoo.append([v, est[k]])
#    assert_equal(f_c.val, io2.val, name='rollin', throw=0, tol=1e-4)
#    boo,bar = zip(*zoo)
#    compare(boo, bar, show_regression=1, name='compare to old version', scatter=1)
#    pl.show()
#    return

#    from arsenal.debug import ip; ip()

#    if test_onebest:
#        for k,v in f_c.marginals().iteritems():
#            assert 0 <= v <= 1.000001, [k,v]
#            if 0.05 <= v <= 0.95:   # not entirely saturated
#                print 'tie (rollin):', nice(grammar, k), v

    # initial roll-in
#    if test_onebest:
#        rollout(grammar, example, m)

    old_mask = M.copy()

    del m, M

    data = []

#    for x in iterview(example.nodes, msg='fd'):
    for x in example.nodes:
        d = {'span': x, 'action': 'prune' if old_mask[x] else 'unprune'}

#        print '--------------------------------'
#        print colors.cyan % '>>>', d['action'], x

        ad_num = f_c.A2_r[x[0], x[1]]
        ad_den = f_c.A2_p[x[0], x[1]]

        was = old_mask[x]
        new_mask = old_mask*1
        new_mask[x] = 1-was

        tie = False

#        if test_viterbi:
#            compare_hypergraph_to_cky(grammar, example, new_mask, steps)

        if test_grad:

            S = InsideOut(example, grammar, new_mask*1.0, steps=steps, with_gradient=0)
            #S = InsideOut2(example, grammar, new_mask*1.0, steps=steps, with_gradient=True, DEBUG=True, IOSPEEDUP=True)

            # Note: Since the function is multi-linear (for gamma=1) we can
            # extrapolate as far as we'd like. Thus, we take the full step
            # (change 0->1 and 1->0).
            if was == 1:
                f_m = S
                f_p = f_c
            else:
                f_p = S
                f_m = f_c

            surrogate = S.val

#            if show_items:
#                dump_items(S)

#            if get_marginals:
#                for k,v in S.marginals().iteritems():
#                    assert 0 <= v <= 1.000001, [k,v]
#                    if 0.05 <= v <= 0.95:   # not entirely saturated
#                        print 'tie (unprune, %s):' % (x,), nice(grammar, k), v
#                        tie = True

            fd_num = (f_p.num - f_m.num)
            fd_den = (f_p.den - f_m.den)

            if 0:
                assert fd_num >= 0 and fd_den >= 0, [fd_num, fd_den]
                assert ad_num >= 0 and ad_den >= 0, [ad_num, ad_den]

                # Yup, gradient should always be positive! Why? Well, for
                # unnormalized risk and Z it's always beneficial to increase the
                # score of an edge -- there is no competition among edges (until
                # there is normalization; the gradient of normalized risk would
                # have variation in sign). Thus, the gradient is always
                # positive.
                assert f_c.A2_p[x[0], x[1]] >= 0
                assert f_c.A2_r[x[0], x[1]] >= 0

            d.update({'surrogate': surrogate,
                      'fd_num': fd_num,
                      'fd_den': fd_den})

            if test_linearity:
                cross_section(example, grammar, steps, old_mask*1.0, x)

                # NOTE: this test fails when gamma != 1.
                mid_mask = old_mask*1.0
                mid_mask[x] = 0.5
                mid = InsideOut(example, grammar, mid_mask, steps=steps, with_gradient=False)

                # Multilinearity check. Check that three points form a line
                #assert abs(mid.den - (0.5 * (f_p.den - f_m.den) / 1 + f_m.den)) < 1e-8
                #assert abs(mid.num - (0.5 * (f_p.num - f_m.num) / 1 + f_m.num)) < 1e-8
                assert abs(mid.den - 0.5 * f_p.den) < 1e-8
                assert abs(mid.num - 0.5 * f_p.num) < 1e-8

        show = False

#        if test_onebest:
#            # one-best rollout
#            r1 = rollout(grammar, example, new_mask)
#            d['onebest'] = r1
#            if abs(surrogate - r1) > 0.0001:
#                print colors.red % '** error **', \
#                    'surrogate does not equal onebest'
#                show = True

        estimate = est[x]

        d.update({'estimate': estimate,
                  'ad_num': ad_num,
                  'ad_den': ad_den,
                  'rel-error': relative_error(surrogate, estimate),
                  'abs-error': abs(surrogate-estimate),
                  'tie': tie})

        if abs(f_c.val - surrogate) > 0.001:  # need a big enough change.
            if surrogate < f_c.val:
                d['delta_type'] = 'decr'
            else:
                d['delta_type'] = 'incr'
        else:
            d['delta_type'] = 'same'


        is_error = abs(surrogate - estimate) > 0.001 or not np.isfinite(estimate)

        if is_error:
            print "%s: estimate doesn't match surrogate" % (colors.red % 'error')
            show = True

        # Taylor expansion should match brute-force method.
        #
        # Note: we're not comparing directly to onebest, since it'll just result
        # in confusion.
        Errors.data.append({'action': d['action'],
                            'delta': d['delta_type'],
                            #'zero': float(d['surrogate']==0),
                            'n_error': int(is_error),
                            'tie': tie})

        if show:
            for k, v in sorted(d.items()):
                if isinstance(v, float):
                    print '%30s: %g' % (k, v)
                else:
                    print '%30s: %s' % (k, v)

            #foobar(f_c)

        data.append(d)

    df = DataFrame(data)

#    print df

    if df.empty:
        print '** dataframe empty **'
        return

    Errors.show()

    if 0:
        #scale = 1
        scale = max(np.abs(df.fd_den).max(), np.abs(df.ad_den).max()) or 1
        compare(df.fd_den/scale,
                df.ad_den/scale,
                alphabet=example.nodes,
                show_regression=1, scatter=1,
                name='test_grad denominator')
        #scale = 1
        scale = max(np.abs(df.fd_num).max(), np.abs(df.ad_num).max()) or 1
        compare(df.fd_num/scale,
                df.ad_num/scale,
                scatter=1, show_regression=1,
                alphabet=example.nodes,
                name='test_grad numerator')

#        if test_onebest:
#            compare(df.onebest,
#                    df.estimate,
#                    alphabet=example.nodes,
#                    show_regression=1,
#                    scatter=1,
#                    name='onebest v. estimate')

    if 0:
        compare(df.surrogate, df.estimate,
                alphabet=example.nodes,
                show_regression=1, scatter=1,
                name='surrogate v. estimate')

#    if 1:
#        if test_grad and test_onebest:
#            compare(df.onebest, df.surrogate,
#                    alphabet=example.nodes,
#                    show_regression=1, scatter=1,
#                    name='onebest v. surrogate')

#    goal = {d['span']: d['surrogate'] for d in data}

    if 0:
        pl.ioff()
        pl.show()


if __name__ == '__main__':
    from ldp.dp.test.util import main
    main(_test_gradient)
