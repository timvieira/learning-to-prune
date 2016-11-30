"""Changeprop runtime analysis.

How well does changeprop leverage sparsity?

See the IPython notebook. For analysis on top of the data produced by running
this script.

"""
from __future__ import division

import random
import pylab as pl
import numpy as np
import seaborn as sns
from pandas import DataFrame
from arsenal import timers, update_ax

from ldp.prune.example import Example
from ldp.parsing.ptb import ptb
from ldp.parse.grammar import Grammar
from ldp.cp.test.correctness import random_mask, CPParser
from ldp.cp.viterbi import DynamicParser as a1
from ldp.parse.leftchild import pruned_parser


# TODO: No longer needed now that shelve is gone.
class Distribution(object):
    def __init__(self, examples, grammar, maxlength, minlength, aggressive, seed=0):
        """
        Manually load examples (instead of using Setup) so that examples will
        come out paired (as long as the random seed is the same and all that.)
        """
        random.seed(seed)
        np.random.seed(seed)

        # TODO: handle other grammars seed ldp.parse.benchmark. Will need
        # different post-processing routines for some of them.
        if grammar == 'medium':
            grammar = Grammar.load('data/medium')
        else:
            grammar = Grammar.load('data/bubs/wsj_6')

        opts = dict(aggressive=aggressive, maxlength=maxlength, examples=examples,
                    minlength=minlength, seed=seed)
        f = 'tmp/cp-analysis-%%s-%s.csv' % ('-'.join('%s_%s' % (k,v) for k,v in sorted(opts.items())))
        self.filename = f % grammar

        examples = list(ptb('train', minlength=minlength, maxlength=maxlength, n=examples))
        random.shuffle(examples)
        examples = [Example(s, grammar=grammar, gold=t) for (s,t) in examples]
        examples = [(e, random_mask(e, aggressive)) for e in examples]
        self.examples = examples
        self.grammar = grammar


# Note: Use this version of the code so that the testing and plotting code won't
# show up in the profiling results.
def profile_run(examples, grammar, maxlength, minlength, aggressive, seed):

    # TODO: localize the seed to just Distribution

    D = Distribution(examples=examples,
                     grammar=grammar,
                     maxlength=maxlength,
                     minlength=minlength,
                     aggressive=aggressive,
                     seed=seed)

    import yep
    yep.start()

    for i, (example, m) in enumerate(D.examples):
        print 'Example: %s, length: %s' % (i, example.N)
        p = CPParser(a1, 'changeprop', D.grammar)
        p.initial_rollout(example, m)
        for [I,K] in example.nodes:
            p.change(I, K, 1-m[I,K])

    yep.stop()


def run(examples, grammar, maxlength, minlength, aggressive, seed=0):

    D = Distribution(examples=examples,
                     grammar=grammar,
                     maxlength=maxlength,
                     minlength=minlength,
                     aggressive=aggressive,
                     seed=seed)

    T = timers()

    ax = pl.figure().add_subplot(111)

    grammar = D.grammar

    DDD = []
    for i, (example, m) in enumerate(D.examples):
        print 'Example: %s, length: %s' % (i, example.N)

        # TODO: log all of the damn rollouts.
        if 1:
            # non-wallclock measurement of work done of BF v. CP.

            p = CPParser(a1, 'changeprop', grammar)
            Q1 = {}
            state = p.initial_rollout(example, m)
            init_chart = np.array(state.chart)['score'].copy()

            ddd = []
            s = np.sum(init_chart != float('-inf'))
            total = np.sum(init_chart != float('-inf'))
            ddd.append({'s': s, 't': total, 'r': s/total})

            for x in example.nodes:
                [I,K] = x

                p.parser.start_undo()
                p.parser.change(I, K, 1-m[I,K])
                chart = y = p.parser.state().chart

                Q1[x] = y

                chart = np.array(chart)['score']

                diff = chart - init_chart
                diff[np.isnan(diff)] = 0  # (-inf) - (-inf) = nan

                ddd.append({'cky': np.sum(chart != float('-inf')),
                            'neq': np.sum(diff != 0),
                            'same': np.sum(diff == 0),
                            'inc': np.sum(diff > 0),
                            'dec': np.sum(diff < 0)})

                p.parser.rewind()

            df = DataFrame(ddd)
            DDD.append({'ratio': df.cky.sum() / df.neq.sum(),
                        'N': example.N,
                        'cky': df.cky.sum(),
                        'inc': df.inc.sum(),
                        'dec': df.dec.sum(),
                        'neq': df.neq.sum()})

        # TODO: regression on with (a*(inc/cky) + b*(dec/cky) + c). Look at
        # residual to make sure this model fits the data better. The
        # coefficients on inc v. dec features will give a sense of how much more
        # expensive each operation is. It's also interesting to look at the
        # distributions of these features, especially as we vary the pruning
        # rate. We also need to compare all quantities of interest on the
        # different grammars.

        with T['cp']:
            p = CPParser(a1, 'changeprop', grammar)
            p.initial_rollout(example, m)
            for [I,K] in example.nodes:
                p.change(I, K, 1-m[I,K])

        with T['bf']:
            for x in example.nodes:
                m[x] = 1-m[x]   # flip
                pruned_parser(example.tokens, grammar, m)
                m[x] = 1-m[x]   # flip back

        T.compare()

        d = DataFrame(DDD)
        d['bf'] = T.timers['bf'].times
        d['cp'] = T.timers['cp'].times
        d['speedup'] = d.bf/d.cp

        if i >= 3:
            # TODO: I'm having trouble updating joint plots. I don't think I can
            # easily get around this unless I patch seaborn (so it doesn't
            # create a new figure each time).
            with update_ax(ax):
                sns.regplot('ratio', 'speedup', d, ax=ax)

    return d


def main():
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--minlength', type=int, default=5)
    p.add_argument('--maxlength', type=int, default=30)
    p.add_argument('--examples', type=int, required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--grammar', choices=('medium','big'), default='medium')
    p.add_argument('--aggressive', type=float, default=0,
                   help='Pruning rate (zero=no pruning, one=lots of pruning).')

    args = p.parse_args()

    if args.profile:
        profile_run(examples = args.examples,
                    grammar = args.grammar,
                    maxlength = args.maxlength,
                    minlength = args.minlength,
                    aggressive = args.aggressive,
                    seed = args.seed)

    else:
        d = run(examples = args.examples,
                grammar = args.grammar,
                maxlength = args.maxlength,
                minlength = args.minlength,
                aggressive = args.aggressive,
                seed = args.seed)

        filename_base = 'tmp/cp-analysis-' + '-'.join('%s_%s' % (k,v) for k,v in sorted(args.__dict__.items()))
        d.to_csv('%s.csv' % filename_base)
        p = sns.jointplot('ratio', 'speedup', d, kind='reg')
        p.savefig('%s.png' % filename_base)
        print '[info] wrote %s.csv' % filename_base

    print '== DONE =='

    pl.ioff()
    pl.show()


if __name__ == '__main__':
    main()
