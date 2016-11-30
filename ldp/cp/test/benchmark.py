from __future__ import division

import numpy as np
import pylab as pl
from collections import defaultdict
from arsenal import timers, colors, axman

from ldp.cp.test.correctness import random_mask, BruteParser, CPParser
from ldp.cp import boolean, viterbi
from ldp.dp.risk import InsideOut
from ldp.parse import leftchild
from ldp.prune.example import Setup

from ldp.dp import risk
from ldp.dp import risk2


class DP(object):

    def __init__(self, cls, name, grammar):
        self.cls = cls
        self.name = name
        self.parser = None
        self.example = None
        self.grammar = grammar
        self.io = None

    def initial_rollout(self, example, m):
        self.example = example
        self.io = self.cls(example, self.grammar, m*1.0)
        return self.io.val

    def change(self, I, K, now):   # assumes we're flipping.
        return self.io.est[I,K]


def _test_efficiency(example, grammar, aggressive, timer, S=None):
    "Test efficiency and correctness under on-policy roll-in and roll-outs."

    pi = random_mask(example, aggressive)

    m = pi.copy()

    grammar.exp_the_grammar_weights()

    # first parser is the reference
    parsers = [
#        BruteParser(leftchild.pruned_parser, 'brute', grammar),
#        BruteForceAgendaParser(DynamicParser, 'brute-agenda', grammar),
#        CPParser(viterbi.DynamicParser, 'changeprop', grammar),
#        CPParser(boolean.DynamicParser, 'cp-bool', grammar),
        DP(risk.InsideOut, 'dp', grammar),
        DP(risk2.InsideOut, 'dp2', grammar),
    ]

    ref = parsers[0].name
    np.random.shuffle(parsers)

    states = example.nodes
    #S = 10*example.N
    if S is not None:
        np.random.shuffle(states)
        s = min(S, len(states))
        states = states[:s]

    #case = 'unprune'
    #case = 'prune'
    case = 'all'

    if case == 'all':
        for p in parsers:
            with timer[p.name](N=example.N):
                p.initial_rollout(example, m)
                for x in states:
                    [I,K] = x
                    now = 1-pi[I,K]
                    p.change(I, K, now)

    elif case == 'prune':
        # just the prunes
        for p in parsers:
            p.initial_rollout(example, m)
            for x in states:
                [I,K] = x
                now = 1-pi[I,K]
                if now == 0:
                    with timer[p.name](N=example.N):
                        p.change(I, K, now)

    elif case == 'unprune':
        # just the unprunes
        for p in parsers:
            p.initial_rollout(example, m)
            for x in states:
                [I,K] = x
                now = 1-pi[I,K]
                if now == 1:
                    with timer[p.name](N=example.N):
                        p.change(I, K, now)

    # Report current timings
#    Timer.compare_many(*timer.values())
    for t in timer.keys():
        if t != ref:
            #timer[parsers[0].name].compare(timer[t])
            timer[ref].compare(timer[t])


def main():
    "Command-line interface for running test cases."
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--minlength', type=int, default=5)
    p.add_argument('--maxlength', type=int, default=40)
    p.add_argument('--examples', type=int, required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--grammar', choices=('medium','big'), default='medium')
    p.add_argument('--aggressive', type=float, default=0.5,
                   help='Pruning rate (zero=no pruning, one=lots of pruning).')
    p.add_argument('--max-rollouts', type=int,
                   help='Option to do a maximum number of rollouts per sentence.')

    args = p.parse_args()

    np.random.seed(args.seed)

    s = Setup(train=args.examples,
              grammar=args.grammar,
              maxlength=args.maxlength,
              minlength=args.minlength,
              features=False)

    T = timers()
    ax = None

    lengths = defaultdict(int)
    for i, example in enumerate(s.train):
        print colors.yellow % '=============================================================='
        print 'example: %s length: %s' % (i, example.N)

        _test_efficiency(example, s.grammar, args.aggressive, timer=T, S=args.max_rollouts)

        lengths[example.N] += 1

        # TODO: should really report error bars because there are differing
        # numbers of examples for each length.
        with axman('time v. length', xlabel='length', ylabel='time (seconds)') as ax:
            def plot(t):
                df = t.dataframe()
                if not df.empty:
                    d = df.groupby('N').mean()  # total time by length
                    [l] = ax.plot(d.index, d.timer, alpha=0.5, label=t.name, lw=2)
                    c = l.get_color()
                    ax.scatter(d.index, d.timer, alpha=0.5, c=c, lw=0, label=None)
                    if 1:
                        for N, dd in df.groupby('N'):
                            #print len(dd)
                            ax.scatter([N]*len(dd), dd.timer, c=c, lw=1, alpha=0.25, label=None)

            for t in T.values():
                print t
                plot(t)

                t.dataframe().to_csv('tmp/timer-%s.csv' % t.name)

            ax.legend(loc=2)
            ax.grid(True)
            ax.figure.savefig('tmp/goo.png')

    print colors.green % '=============================================================='
    print colors.green % 'DONE'
    print

    pl.ioff()
    pl.show()


if __name__ == '__main__':
    main()
