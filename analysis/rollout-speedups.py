from __future__ import division

import numpy as np
import pylab as pl
from collections import defaultdict
from time import time
from arsenal.timer import Timer, timers
from arsenal.terminal import green, yellow
from arsenal.viz import axman
from path import path
from ldp.cp.test.correctness import random_mask, BruteParser, CPParser
from ldp.cp import boolean, viterbi
from ldp.dp import risk
from ldp.parse import leftchild
from pandas import DataFrame
from ldp.prune.example import Setup

from ldp.cp.test.benchmark import DP
from ldp.prune.features import Features
from arsenal.iterview import iterview


def main():
    "Command-line interface for running test cases."
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--minlength', type=int, default=3)
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

#    lengths = defaultdict(int)

    F = Features(s.grammar, nfeatures=2**22)

    examples = s.train
    np.random.shuffle(examples)

    data = []

    for policy in path('results/*baseline9-*/dump/').glob('new_policy*.npz'):

        import cPickle
        if cPickle.load(file(policy.dirname() / 'args.pkl')).grammar != args.grammar:
            continue
        
        print green % '====================================================================='
        print green % policy

        w = np.load(policy)['coef']

        timer = timers()
        for e in iterview(examples):
            #print yellow % '=============================================================='
            #print 'example: %s length: %s' % (i, e.N)

            pi = F.mask(e, w)
            #print 1-np.mean([pi[x] for x in e.nodes])

            # first parser is the reference that runtime are taken against.
            parsers = [
                BruteParser(leftchild.pruned_parser, 'brute', s.grammar),
                CPParser(viterbi.DynamicParser, 'cp', s.grammar),
                DP(risk.InsideOut, 'dp', s.grammar),
            ]

            ref = parsers[0].name
            np.random.shuffle(parsers)

            states = e.nodes
            #S = None
            ##S = 2*e.N
            #if S is not None:
            #    np.random.shuffle(states)
            #    states = states[:min(S, len(states))]

            T = len(states)
            prune_rate = 1 - sum(pi[x] for x in states)/T

            times = {}
            for p in parsers:
                with timer[p.name](N=e.N):
                    b4 = time()
                    p.initial_rollout(e, pi)
                    for x in states:
                        [I,K] = x
                        now = 1-pi[I,K]
                        p.change(I, K, now)
                    times[p.name] = time() - b4
            row = dict(
                prune_rate=prune_rate, policy=policy,
                example=e.name, N=e.N, T=T,
            )
            row.update(times)
            data.append(row)

        DataFrame(data).to_csv('timing-%s-%s.csv' % (args.grammar, args.examples))

        # Report current timings
        for t in sorted(timer.keys()):
            if t != ref:
                timer[ref].compare(timer[t])

#        lengths[example.N] += 1
#
#        # TODO: should really report error bars because there are differing
#        # numbers of examples for each length.
#        with axman('time v. length', xlabel='length', ylabel='time (seconds)') as ax:
#            def plot(t):
#                df = t.dataframe()
#                if not df.empty:
#                    d = df.groupby('N').mean()  # total time by length
#                    [l] = ax.plot(d.index, d.timer, alpha=0.5, label=t.name, lw=2)
#                    c = l.get_color()
#                    ax.scatter(d.index, d.timer, alpha=0.5, c=c, lw=0, label=None)
#                    if 1:
#                        for N, dd in df.groupby('N'):
#                            #print len(dd)
#                            ax.scatter([N]*len(dd), dd.timer, c=c, lw=1, alpha=0.25, label=None)
#
#            for t in T.values():
#                print t
#                plot(t)
#
#                t.dataframe().to_csv('tmp/timer-%s.csv' % t.name)
#
#            ax.legend(loc=2)
#            ax.grid(True)
#            ax.figure.savefig('tmp/goo.png')
#
#    print green % '=============================================================='
#    print green % 'DONE'
#    print
#
#    pl.ioff()
#    pl.show()


if __name__ == '__main__':
    main()
