"""
Tests for change propagation.
"""

import numpy as np
import pylab as pl

from arsenal import colors
from arsenal.math import assert_equal

from ldp.parse import leftchild
from ldp.prune.example import Setup
from ldp.parsing.util import list2tree
from ldp.cp import boolean
from ldp.cp import viterbi

semizero = -np.inf


def random_mask(example, p):
    """Create a mask which is approximately the oracle mask, it has a pruning rate
    equal to `p` (i.e., true negative rate).

    """
    # rollout with some an arbitrary policy, which isn't too far from the
    # oracle.
    pi = example.mask.copy()
    pi[:,:] = 1
    for x in example.nodes:
        if x in example.gold_spans:
            pi[x] = 1
        else:
            pi[x] = (np.random.uniform() > p)
    return pi


#from ldp.cp.test.util import compare_charts, check_prune, check_viterbi, pops, \
#    check_index_support

#def compare_states(expect, got):
#    name = '%s v. %s' % (expect.name, got.name)
#    [l1, _, p1, c1] = expect.state
#    [l2, _, p2, c2] = got.state
#    c1 = np.asarray(c1)
#    try:
#        c1 = c1['score']
#    except (ValueError, IndexError):
#        pass
#    c2 = np.asarray(c2)['score']
#    assert_equal(p2, pops(c2), 'pops incr self-test (%s)' % name, throw=1, verbose=0)
#    assert_equal(l1, l2, 'llh (%s)' % name, throw=1, verbose=0)
#    assert_equal(p1, p2, 'pops (%s)' % name, throw=0, verbose=0)
#    if 0:
#        compare_charts(c1, c2, name='charts (%s)' % name)
#    if 0:
#        checks(got.grammar, got.parser, got.state, got.mask)
#
#
#def checks(grammar, parser, state, mask):
#    """
#    Self-consistency checks for parser state.
#    """
#    [_,_,p,c] = state
#    c = np.asarray(c)
#    scores = c['score']
#    check_prune(mask, scores)
#    assert_equal(p, pops(scores), 'pops incr self-test', throw=1, verbose=0)
#    if hasattr(parser, 'check_index_support'):
#        parser.check_index_support(False)
#    else:
#        check_index_support(c, parser.begin, parser.end)
#    check_viterbi(c, grammar)


class CPParser(object):

    def __init__(self, cls, name, grammar):
        self.name = name
        self.parser = None
        self.example = None
        self.N = None
        self.cls = cls
        self.grammar = grammar

    def initial_rollout(self, example, m):
        N = example.N
        self.example = example
        p = self.cls(example.tokens, self.grammar, m)
        p.run()
        s = p.state()
        self.parser = p
        self.N = N
        return s

    @property
    def mask(self):
        return self.parser.keep

    def change(self, I, K, now):
        p = self.parser
        p.start_undo()
        p.change(I,K,now)
        s = p.state()
        p.rewind()
        return s    # NOTE: chart will be original values, not the changed ones.


class BruteForceAgendaParser(object):

    def __init__(self, cls, name, grammar):
        self.name = name
        self.parser = None
        self.example = None
        self.N = None
        self.cls = cls
        self.grammar = grammar

    def initial_rollout(self, example, m):
        N = example.N
        self.example = example
        p = self.cls(example.tokens, self.grammar, m.copy())
        p.run()
        s = p.state()
        self.parser = p
        self.N = N
        return s

    @property
    def mask(self):
        return self.parser.keep

    def change(self, I, K, now):
        was = self.mask[I,K]
        self.mask[I,K] = now
        p = self.cls(self.example.tokens, self.grammar, self.mask)
        p.run()
        s = p.state()
        self.mask[I,K] = was
        return s


class BruteParser(object):

    def __init__(self, cls, name, grammar):
        self.name = name
        self.example = None
        self.N = None
        self.cls = cls
        self.grammar = grammar
        self.mask = None
        self._prev_mask = None

    def initial_rollout(self, example, m):
        N = example.N
        self.mask = m
        self.example = example
        s = self.cls(example.tokens, self.grammar, m)
        self.N = N
        return s

    def change(self, I, K, now):
        was = self.mask[I,K]
        self.mask[I,K] = now
        y = self.cls(self.example.tokens, self.grammar, self.mask)
        self.mask[I,K] = was
        return y


def _test_correctness(example, grammar, aggressive):
    "Test correctness under on-policy roll-in and roll-outs."

    pi = random_mask(example, aggressive)

    m = pi.copy()

    # first parser is assumed to be the "gold standard"
    parsers = [
        #BruteForceAgendaParser(viterbi.DynamicParser, 'brute', grammar),
        BruteParser(leftchild.pruned_parser, 'brute', grammar),
        CPParser(viterbi.DynamicParser, 'changeprop', grammar),
    ]

    Q = {p.name: {} for p in parsers}

    for p in parsers:
        p.initial_rollout(example, m)
        for x in example.nodes:
            [I,K] = x
            Q[p.name][x] = p.change(I, K, 1-pi[I,K])

    had_tie = False
    for x in Q['brute']:
        ref = Q['brute'][x]
        got = Q['changeprop'][x]

        r = grammar.coarse_derivation(ref.derivation)
        g = grammar.coarse_derivation(got.derivation)

        # [2016-02-05 Fri] OK, well somehow changeprop is finding parses
        # with multiple unary rewrites.
        if r != g:
            # Derivations weren't equal. Assert they're equally likely (tied).
            assert_equal(ref.likelihood, grammar.llh(list2tree(ref.derivation)), 'ref-llh-check')
            assert_equal(got.likelihood, grammar.llh(list2tree(got.derivation)), 'got-llh-check')
            # we must have a tie because derivations are different and scores are tied.
            had_tie = True

        # At this point we've established that the derivations are the same, may
        # as well check that likelihood is the same.  assert abs(ref[0] -
        # got[0]) < 1e-10, ['llh', ref[0], got[0]]
        assert_equal(ref.likelihood, got.likelihood, 'llh')

        # Note: We allow a little bit of sloppiness in runtime measurement.
        assert_equal(ref.pops, got.pops, 'pops', throw=0, tol=5)

    if had_tie:
        print colors.cyan % 'tie'
    print colors.green % 'ok'


def _test_correctness_boolean(example, grammar, aggressive):
    "Test correctness under on-policy roll-in and roll-outs."

    pi = random_mask(example, aggressive)

    m = pi.copy()

    # first parser is assumed to be the "gold standard"
    parsers = [
        BruteParser(leftchild.pruned_parser, 'brute', grammar),
        CPParser(boolean.DynamicParser, 'cp-bool', grammar),
    ]

    Q = {p.name: {} for p in parsers}

    for p in parsers:
        p.initial_rollout(example, m)
        for x in example.nodes:
            [I,K] = x
            Q[p.name][x] = p.change(I, K, 1-pi[I,K])

    for x in Q['brute']:
        ref = Q['brute'][x]
        got = Q['cp-bool'][x]

        #r = grammar.coarse_derivation(ref.derivation)
        #g = grammar.coarse_derivation(got.derivation)

        # At this point we've established that the derivations are the same, may
        # as well check that likelihood is the same.  assert abs(ref[0] -
        # got[0]) < 1e-10, ['llh', ref[0], got[0]]
        assert (ref.likelihood > semizero) == (got.likelihood > semizero)

        # Note: We allow a little bit of sloppiness in runtime measurement.
        assert_equal(ref.pops, got.pops, 'pops', throw=0, tol=5, verbose=1)

    print colors.green % 'ok'


def main():
    "Command-line interface for running test cases."
    from argparse import ArgumentParser
    p = ArgumentParser()

    p.add_argument('--boolean', action='store_true')

    p.add_argument('--minlength', type=int, default=5)
    p.add_argument('--maxlength', type=int, default=30)
    p.add_argument('--examples', type=int, required=True)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--grammar', choices=('medium','big'), default='medium')
    p.add_argument('--aggressive', type=float, default=0.5,
                   help='Pruning rate (zero=no pruning, one=lots of pruning).')

    args = p.parse_args()

    np.random.seed(args.seed)

    s = Setup(train=args.examples,
              grammar=args.grammar,
              maxlength=args.maxlength,
              minlength=args.minlength,
              features=False)


    test = _test_correctness_boolean if args.boolean else _test_correctness

    for i, example in enumerate(s.train):
        print colors.yellow % '=============================================================='
        print 'example: %s length: %s' % (i, example.N)
        test(example, s.grammar, args.aggressive)

    print colors.green % '=============================================================='
    print colors.green % 'DONE'
    print

    if 0:
        from arsenal.debug import ip; ip()
    else:
        pl.ioff()
        pl.show()


if __name__ == '__main__':
    main()
