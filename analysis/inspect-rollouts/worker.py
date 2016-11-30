"""What are rollouts doing?

Compare baseline labels to LOLS's "labels" via rollouts.

How locally optimal is the current policy?

Inspect rollout information.
This version takes only this iteration's rollouts.

"""

from __future__ import division

# Note: This import has to happen before pylab.
from ldp.prune.example import Reward, AvgReward, Setup, cgw_prf, cgw_f

import os
import numpy as np
import pylab as pl
import cPickle
import random
from path import path
from pandas import DataFrame
from time import time

from ldp.dp.risk import InsideOut
from ldp.cp.viterbi import DynamicParser
from ldp.cp.boolean import DynamicParser as BoolCP
from ldp.parse.leftchild import pruned_parser
from ldp.lols.classifier import SVM, GLM, Perceptron, Adagrad


from arsenal import colors, iterview

from ldp.lols.__main__ import ROLLOUT, CP, BF, HY, DP, ACC


class inspect_rollouts(object):

    def __init__(self, args, setup, policy, output_file, tradeoff, roll_out):
        self.setup = setup
        self.tradeoff = tradeoff
        self.grammar = grammar = setup.grammar
        self.train = setup.train
        self.ACCURACY = args.accuracy
        self.RUNTIME = args.runtime

        self.policy_name = policy
        self.output_file = output_file

        self.nfeatures = nfeatures = setup.nfeatures
        self.policy = GLM(nfeatures, C=np.nan, loss=0)   # dummy

        self.policy._coef = np.load(policy)['coef']

        if roll_out == ROLLOUT.CP:
            Rollouts = CP
        elif roll_out == ROLLOUT.BF:
            Rollouts = BF
        elif roll_out == ROLLOUT.HY:
            Rollouts = HY
        elif roll_out == ROLLOUT.DP:
            Rollouts = DP
        else:
            raise ValueError('Unrecognized rollout option %s' % roll_out)

        tmp = workaround(output_file, context=self)
        for e in iterview(self.train, msg='rollouts'):
            p = Rollouts(grammar, e, self.policy,
                         accuracy=args.accuracy,
                         runtime=args.runtime,
                         tradeoff=tradeoff)
            p.roll_outs(tmp)

        tmp.save()


# [2016-08-05 Fri] This class is a strange workaround. The rollouts method in
# lols module are a bit strange because it buffers the set of rollouts in a list
# instead of processing each independently as they are done (this weirdness was
# introduced to support reward functions which are nonadditive over examples
# such as Corpus-F1).
#
# I built this class to periodically (in seconds) save the output of rollouts.
#
# TODO: This class seems like the basis for something which is generally
# useful. Figure out a nice API, refactor things, and put the resulting class
# into the arsenal.
#
class workaround(object):
    def __init__(self, filename, min_seconds = 60, context=None):
        # Note: `context` option is a bit of a workaround.
        self.filename = filename
        self.last_save = time()
        self.min_time = min_seconds
        self.data = []
        self.context = context

    def append(self, element):
        [e, (I,K), w, (a0, r0), (a1, r1)] = element
        action = a0
        if a0 == 1:  # keep
            r0, r1 = r1, r0   # swap
        del a0, a1, w
        if self.context.ACCURACY == ACC.EVALB_avg:
            acc1 = r1.f1()
            acc0 = r0.f1()
        elif self.context.ACCURACY == ACC.EXPECTED_RECALL_avg:
            acc1 = r1.recall()
            acc0 = r0.recall()
        else:
            acc1 = r1.accuracy
            acc0 = r0.accuracy

        rew0 = acc0 - self.context.tradeoff*r0.runtime
        rew1 = acc1 - self.context.tradeoff*r1.runtime
        assert r0.runtime <= r1.runtime, [r0.runtime, r1.runtime]  # checks that we got prune v. keep correct.

        self.data.append({
            'example': e.name,
            'span_begin': I,
            'span_end': K,
            'policy': action,
            'gold': (I,K) in e.gold_spans,
            'delta_rew': rew1 - rew0,
            'delta_acc': acc1 - acc0,
            'delta_run': r1.runtime - r0.runtime,
        })

        # Periodically save data collected
        if time() - self.last_save > self.min_time:
            self.save()
            self.last_save = time()

    def save(self):
        if self.filename is None:
            return
        df = DataFrame(self.data)
        df.to_csv(self.filename, compression='gzip')
        print
        print 'wrote %s' % self.filename, 'n_points:', len(self.data)
        print


def main():
    from argparse import ArgumentParser

    p = ArgumentParser()
    # output
    p.add_argument('policy', type=path)
    p.add_argument('--baseline', action='store_true',
                   help=("if true use the --init-weights policy logged in"
                         "args.pkl in policy directory instead of the --policy."
                         ))
    p.add_argument('--debug', action='store_true')

    args = p.parse_args()

    results = args.policy.dirname().abspath()

    # In this case, we're inspecting the results of a previous job.
    assert results.exists()
    assert results.endswith('dump')

    prev_args = cPickle.load(file(results / 'args.pkl'))

    if args.baseline:
        # Copy initial policy into results directory (for simplicity).
        init_policy = results / 'init.npz'
        assert prev_args.init_weights.exists()
        assert 0 == os.system('cp %s %s' % (prev_args.init_weights, init_policy))
        args.policy = init_policy

    np.random.seed(prev_args.seed)
    random.seed(prev_args.seed)

    setup = Setup(grammar = prev_args.grammar,
                  train = prev_args.train if not args.debug else 50,
                  dev = 0,
                  minlength = prev_args.minlength,
                  maxlength = prev_args.maxlength)

    if args.debug:
        output_file = None
    else:
        output_file = '%s.inspect_rollouts.csv.gz' % args.policy

    inspect_rollouts(prev_args,
                     setup,
                     policy = args.policy,
                     output_file = output_file,
                     tradeoff = prev_args.tradeoff,
                     roll_out = prev_args.roll_out)

    print
    print colors.green % 'DONE!'


if __name__ == '__main__':
    main()
