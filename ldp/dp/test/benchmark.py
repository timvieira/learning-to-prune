from __future__ import division

import pylab as pl
from arsenal import timers, axman
from arsenal.math import compare
from pandas import DataFrame

from ldp.dp.risk import InsideOut
from ldp.parse.leftchild import pruned_parser

T = timers()

data = []


def test(e, grammar, m):
    """
    Compare runtime of running parser v. inside algorithm.

    Note: This isn't the same rollouts.
    """

    M = m*1.0
    steps = 2

    # TODO: compare_hypergraph_to_cky(grammar, example, m, steps)

    grammar.exp_the_grammar_weights()   # Do exp outside the timing loop. It will be cached in the Grammar object.

    with T['dp'](N=e.N):
        expected_recall = InsideOut(e, grammar, M*1.0, steps=steps, with_gradient=0).val

    with T['parse'](N=e.N):
        state = pruned_parser(e.tokens, grammar, m)
        coarse = grammar.coarse_derivation(state.derivation)
        c,_,w = e.recall(coarse)
        recall = c/w

    data.append({'example': e,
                 'N': e.N,
                 'expected_recall': expected_recall,
                 'recall': recall})

    T.compare()

    with axman('compare runtimes') as ax:
        T.plot_feature('N', ax=ax, loglog=1, show='scatter')

    df = DataFrame(data)
    with axman('compare rewards') as ax:
        compare(df.recall, df.expected_recall, scatter=1, show_regression=1, ax=ax)


if __name__ == '__main__':
    from ldp.dp.test.util import main
    main(test)

    print '=================================='
    print 'DONE!'
    pl.ioff()
    pl.show()
