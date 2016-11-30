"""Load model file and run timing experiments.

This script will download the best model parameters from the grid (according to
early stopping on dev surrogate reward).

To just download the best params and skip evaluation specify the
``--download-only`` flag and give dummy arguments for the output directory and
split. For example,

  $ python analysis/test-time-evaluation.py --download-only --split test --grammar medium /tmp

"""
from __future__ import division

import os, cPickle
import numpy as np
import pandas as pd
from path import path
from time import time

from random import shuffle
from nltk import Tree

from ldp.parsing.evalb import fpr
from ldp.parsing.util import oneline
from ldp.prune.example import Setup, cgw_f
from ldp.prune.features import Features
from ldp.parse.leftchild_bp import pruned_parser

from arsenal import iterview, colors, wide_dataframe
wide_dataframe()



def get_data(G, policy_name, w, examples, verbose=0):

    data = []
    for eid, e in enumerate(iterview(examples, msg='evalb')):
        if verbose:
            print
            print
            print colors.yellow % e.sentence

        F = Features(G, nfeatures=2**22)

        words = e.sentence.split()

        if 'unpruned' in policy_name:
            # <TIMING BLOCK>
            mask = e.mask
            b1 = time()
            e.tokens = np.asarray(G.encode_sentence(words))
            b2 = time()
            state = pruned_parser(e.tokens, G, mask)
            b3 = time()
            # </TIMING BLOCK>
            coarse = G.coarse_derivation(state.derivation)

        else:
            # <TIMING BLOCK>
            b1 = time()
            e.tokens = np.asarray(G.encode_sentence(words))
            mask = F.mask(e, w)
            b2 = time()
            state = pruned_parser(e.tokens, G, mask)
            b3 = time()
            # </TIMING BLOCK>
            coarse = G.coarse_derivation(state.derivation)

        nodes = e.nodes
        mask_size = sum(mask[x] for x in nodes)
        keep_rate = mask_size/len(nodes) if len(nodes) > 0 else 0

        want_and_got, got, want = e.evalb_unofficial(coarse)
        evalb_avg, _, recall_avg = fpr(want_and_got, got, want)

        if isinstance(coarse, Tree):
            parse = oneline(coarse)
            fail = 0
        else:
            parse = '(FAIL %s)' % ' '.join('(X %s)' % x for x in e.sentence.split())
            fail = 1

        data.append({'example': e,
                     'eid': eid,
                     'N': e.N,
                     'fail': fail,

                     'mask': mask_size,
                     'keep_rate': keep_rate,

                     'parse': parse,
                     'policy': policy_name,
                     'time_total':   b3 - b1,
                     'time_feature': b2 - b1,
                     'time_parse':   b3 - b2,

                     'evalb_avg': evalb_avg,
                     'recall_avg': recall_avg,

                     'want_and_got': want_and_got,
                     'want': want,
                     'got': got,

                     'pushes': state.pushes,
                     'pops': state.pops})

    return data


def main():

    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('directory', type=path)
    p.add_argument('--grammar', choices=('medium','big'), required=1)
    p.add_argument('--split', choices=('dev','test','other','train'), required=1)
    p.add_argument('--download-only', action='store_true')

    args = p.parse_args()

    if 'unpruned' in args.directory:
        policies = [path('unpruned') / 'unpruned']
    else:
        # Grab the best model parameters according to early stopping on each dev
        # reward surrogate.
        policies = []
        for x in (path('results/*-lols10-*/').glob('dump')
                  + path('results/*-baseline9-*/').glob('dump')):
#        for x in (path('results/*-lols11-*/').glob('dump')):

            # only grab models matching the grammar specified.
            if cPickle.load(file(x / 'args.pkl')).grammar != args.grammar:
                continue
            df = pd.read_csv(x / 'log.csv')
            if df.get('dev_new_policy_evalb_corpus') is None:
                # [2016-04-29 Fri] SKIP some of the baselin8 jobs. These guys are
                # using the wrong evaluation.
                assert 'baseline9' in x
                continue

            # identify iteration with best dev reward (surrogate).
            el = df.datetime.map(pd.to_datetime).tolist()
            df['elapsed'] = [(t - el[0]).total_seconds() / (24*60*60) for t in el]
            df = df[df.elapsed <= 6]            # take the best policy <= 6 days of training.

            best = df.ix[df.dev_new_policy_reward.argmax()]

            print colors.yellow % 'best policy:'
            print best[['iteration', 'elapsed']]

            # download model file for that iteration, if we don't already have it.
            policy = x / ('new_policy-%03d.npz' % best.iteration)
            if not policy.exists():
                assert 0 == os.system('rsync --progress "timv@login.clsp.jhu.edu:/export/a11/timv/ldp/%s" %s' % (policy, policy))
            policies.append(policy)

    if args.download_only:
        return

    s = Setup(grammar = args.grammar, train = 0, dev = 0, features = 0)

    examples = list(s.load(args.split))
    shuffle(policies)

    outdir = args.directory
    outdir.mkdir_p()

    for pp, policy in enumerate(policies, start=1):
        print
        print colors.green % '[%s/%s] %s' % (pp, len(policies), policy)

        evaluation_file = outdir / (policy.dirname() / '..').abspath().basename() + '-evaluation.csv.gz'
        print evaluation_file

        if path(evaluation_file).exists():
            last_time = pd.read_csv(evaluation_file)
            [last_policy] = last_time.policy.unique()
            if last_policy == policy:
                print colors.yellow % 'SKIP: evaluation file exists.'
                continue
            print colors.red % 'replace old evaluation.'

        if 'unpruned' in policy:
            w = None
        else:
            w = np.load(policy)['coef']

        d = get_data(s.grammar, policy, w, examples)
        df = pd.DataFrame(d)

        if 1:
            xx = df.groupby('policy').sum()
            yy = df.groupby('policy').mean()
            print 'evalb:     %.3f' % (cgw_f(yy.want_and_got.sum(), yy.got.sum(), yy.want.sum()))
            print 'words/sec: %.1f' % (xx.N / xx.time_total)
            print 'sent/sec:  %.1f' % (1 / yy.time_total)
            print 'sec/sent:  %g' % (yy.time_total)
            #print 'features:  %4.1f%%' % (100 * yy.time_feature / yy.time_total)
            #print 'parse:     %4.1f%%' % (100 * yy.time_parse / yy.time_total)

        pd.DataFrame(df).to_csv(evaluation_file, compression='gzip')


if __name__ == '__main__':
    main()
