import numpy as np
import random
from ldp.prune.example import Setup
from arsenal.terminal import yellow


def main(test):
    "Command-line interface for running test cases."
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('--minlength', type=int, default=5)
    p.add_argument('--maxlength', type=int, default=15)
    p.add_argument('--examples', type=int, default=50)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--grammar', choices=('medium','big'), default='medium')

    p.add_argument('--only-examples', type=int, nargs='+')
    p.add_argument('--skip-examples', type=int, nargs='+')

    # Note: using default='random' so that we occasionally get no-parse on
    # initial roll-in.
    #p.add_argument('--policy',
    #               choices=('unpruned', 'false-positive-only', 'random'),
    #               default='random')
    p.add_argument('--delta', type=float, default=0.2)

    args = p.parse_args()

    assert 0 <= args.delta <= 1

    np.random.seed(args.seed)
    random.seed(args.seed)

    s = Setup(train=args.examples,
              grammar=args.grammar,
              maxlength=args.maxlength,
              minlength=args.minlength,
              features=False)

    for i, example in enumerate(s.train):
        print yellow % '=============================================================='
        print yellow % 'Example %s, length %s: %s' % (i, len(example.tokens), example.sentence)

        # rollout with policy to get pruning mask
        m = example.mask
        m[:,:] = 1
        #if args.policy == 'unpruned' or args.delta == 0.0:
        #    print yellow % '[WARNING] Mask is unpruned only.'

        #elif args.policy == 'false-positive-only':
        #    # This version very few increases in accuracy because we always have
        #    # the gold brackets.
        #    for (I,K) in example.nodes:
        #        if (I,K) not in example.gold_spans:
        #            m[I,K] = (np.random.uniform() > args.delta)

        #elif args.policy == 'random':
        if 1:
            # This version has more increases in accuracy than the one below
            # which is only false positives.
            for (I,K) in example.nodes:
                m[I,K] = (np.random.uniform() > args.delta)

        #else:
        #    raise AssertionError('unrecognized policy option %r' % args.policy)

        if args.only_examples is not None and i not in args.only_examples:
            continue

        if args.skip_examples is not None and i in args.skip_examples:
            continue

        test(example, s.grammar, m)
