#!python
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

"""
Learning to prune.
"""
from __future__ import division

import numpy as np
from collections import defaultdict
from arsenal.iterview import iterview
from nltk import Tree

from ldp.parsing.ptb import ptb
from ldp.parsing.util import check_binary_tree, item_tree, is_item_tree, \
    item_tree_get_items, recall_tree, unbinarize, fix_terminals, \
    item_tree_to_label, collapse_self_transition, oneline
from ldp.parsing.evalb import evalb, evalb_unofficial, _relevant_items
from ldp.parse.grammar import Grammar
from ldp.parse.leftchild import pruned_parser
from ldp.prune.features import Features
from ldp.prune.features import letter_pattern


def cgw_prf(double C, double  G, double W):
    P = C/G if G != 0 else 1
    R = C/W if W != 0 else 1
    F = 2*P*R/(P+R) if P+R != 0 else 0.0
    return (P,R,F)

cpdef double cgw_f(double C, double  G, double W):
    P = C/G if G != 0 else 1
    R = C/W if W != 0 else 1
    F = 2*P*R/(P+R) if P+R != 0 else 0.0
    return F

cdef double flip_reward(double[:] AC, double[:] AG, double[:] A_runtime,
                        double[:] BC, double[:] BG, double[:] B_runtime,
                        int[:] flip, double W, double tradeoff):
    cdef double c, g, r
    cdef int i, n
    c = 0.0
    g = 0.0
    r = 0.0
    n = AC.shape[0]
    for i in range(n):
        if flip[i]:
            c += BC[i]
            g += BG[i]
            r += B_runtime[i]
        else:
            c += AC[i]
            g += AG[i]
            r += A_runtime[i]
    return cgw_f(c, g, W) - tradeoff*r/n

def test_statistic(double[:] AC, double[:] AG, double[:] A_runtime,
                   double[:] BC, double[:] BG, double[:] B_runtime,
                   int[:] flip, double W, double tradeoff):
    "Test statistic for reward."
    fa = flip_reward(AC, AG, A_runtime,
                     BC, BG, B_runtime,
                     flip, W, tradeoff)
    fb = flip_reward(BC, BG, B_runtime,
                     AC, AG, A_runtime,
                     flip, W, tradeoff)
    return abs(fa-fb)


class Setup(object):
    "Holds train/dev data, grammar, features and feature alphabet."

    def __init__(self, train, maxlength=None, minlength=3, dev=0,
                 grammar='medium', nfeatures=2**22, features=True):
        assert isinstance(train, int) or train is None
        assert isinstance(dev, int) or dev is None
        assert grammar in ('medium', 'big', None)

        self.train_size = train
        self.dev_size = dev
        self.maxlength = maxlength if maxlength is not None else 40
        self.minlength = minlength if minlength is not None else 3
        self.nfeatures = nfeatures

        # TODO: some grammar's need specialzed encode/decode methods.
        #
        # - decode: convert input string into token ids (e.g., with unseen
        #   words). crf-parser does feature extraction.
        #
        # - encode: coarsen derivation, unbinarize
        #
        self.grammar_name = grammar
        if grammar is None:
            pass
        elif grammar == 'medium':
            grammar = Grammar.load('data/medium')
        elif grammar == 'big':
            grammar = Grammar.load('data/bubs/wsj_6')
        else:
            raise AssertionError('unrecognized grammar %r' % grammar)

        if features:
            assert grammar is not None
            self.extract_features = Features(grammar=grammar, nfeatures=nfeatures)
        else:
            # no-op
            self.extract_features = lambda x: None

        self.grammar = grammar
        self.train = list(iterview(self._train(), length=self.train_size, msg='train'))
        self.dev = list(iterview(self._dev(), length=self.dev_size, msg='dev'))

        letter_pattern.cache.clear()

    def load(self, name):
        data = ptb(names=name)
        for k, (sentence, tree) in enumerate(data):
            e = Example(sentence, self.grammar, gold=tree)
            e.name = '%s/%s' % (name, k)
            yield e

    def _train(self):
        n = 0
        for k, e in enumerate(self.load('train')):
            if n >= self.train_size:
                break
            if self.minlength <= e.N <= self.maxlength:
                e.features = self.extract_features(e)
                yield e
                n += 1

    def _dev(self):
        n = 0
        for k, e in enumerate(self.load('dev')):
            if n >= self.dev_size:
                break
            if self.minlength <= e.N <= self.maxlength:
                e.features = self.extract_features(e)
                yield e
                n += 1


class Reward(object):

    def __init__(self, accuracy, runtime, **kw):
        self.accuracy = accuracy
        self.runtime = runtime
        self.__dict__.update(kw)

    def __repr__(self):
        return 'Reward(%g, %g)' % (self.accuracy, self.runtime)

    def __cmp__(self, other):
        return cmp((self.accuracy, self.runtime), (other.accuracy, other.runtime))

    def __call__(self, tradeoff):
        return self.accuracy - tradeoff*self.runtime


class AvgReward(object):

    def __init__(self, rs):
        acc = [r.accuracy for r in rs]
        run = [r.runtime for r in rs]
        self.acc = acc
        self.run = run
        self.accuracy = np.mean(acc)
        self.runtime = np.mean(run)
        attrs = defaultdict(list)
        for r in rs:
            for k, v in r.__dict__.items():
                attrs[k].append(v)
        if attrs:
            assert len(set(len(x) for x in attrs.itervalues())) == 1, \
                'attrs must have same length.'
        self.attrs = {k: np.mean(v) for k, v in attrs.items()}

        # Compute corpus-level F-measure
        _, _, self.evalb_corpus = cgw_prf(sum(attrs['C']),
                                          sum(attrs['G']),
                                          sum(attrs['W']))

        # Compute average F-measure
        prfs = np.array([cgw_prf(c,g,w) for c,g,w in zip(attrs['C'], attrs['G'], attrs['W'])])
        _, _, self.evalb_avg = prfs.mean(axis=0)

        # Compute corpus-level expected recall
        C = sum(attrs['expected_C'])
        W = sum(attrs['W_b'])
        R = C/W if W > 0 else 1
        self.expected_recall_corpus = R

        # Compute corpus-level expected recall
        self.expected_recall_avg = np.mean([c/w for c,w in zip(attrs['expected_C'], attrs['W_b'])])

        # ** Note: don't forget to add these guys to attr so that they get logged. **
        self.attrs['expected_recall_avg'] = self.expected_recall_avg
        self.attrs['expected_recall_corpus'] = self.expected_recall_corpus
        self.attrs['evalb_avg'] = self.evalb_avg
        self.attrs['evalb_corpus'] = self.evalb_corpus

    def __call__(self, tradeoff):
        return self.accuracy - tradeoff*self.runtime

    def __repr__(self):
        return 'AvgReward(%s, %s)' % (self.accuracy, self.runtime)


cdef class Example:

    def __init__(self, sentence, grammar, gold):
        self.sentence = sentence

        if grammar is not None:
            self.tokens = grammar.encode_sentence(sentence.split())
        else:
            self.tokens = None

        self.N = len(sentence.split())

        # empty slots, which may be filled in later.
        self.features = None
        self.mle_spans = None
        self.baseline = None
        self.oracle = None
        self.Q = np.zeros((self.N, self.N+1, 2))

        if gold is not None:
            self.set_gold(gold)

    def set_gold(self, gold):
        """Precompute 'stuff' based on the gold tree.

        Method expects the gold tree to be a binary `Tree` object with labels of
        the form (X:str, i:int, k:int). It will also accept labels which are
        X:str.

        """
        N = self.N

        # about the gold tree
        assert check_binary_tree(gold), gold

        u = unbinarize(gold)
        # stored as strings because it's much smaller than `nltk.Tree` objects.
        self.gold_binarized = oneline(gold)
        self.gold_unbinarized = oneline(u)
        self.evalb_items = frozenset(_relevant_items(item_tree(u)))

        #assert not is_item_tree(gold)
        gold = item_tree(gold)

        # binarized
        self.gold_items = frozenset({(X,I,K) for (X,I,K) in item_tree_get_items(gold) if K-I > 1 and K-I != N})
        self.gold_spans = frozenset({(I,K) for (_,I,K) in self.gold_items})

    def nofail(self, d):
        """Reward for finding *some* valid parse.

        Warning: validity is not actually checked, just non-emptiness.

        """
        if isinstance(d, basestring):
            return 0.0                  # failed
        assert isinstance(d, Tree), d   # make sure we didn't pass in the list-of-lists version.
        return len(d) != 0

    def recall(self, d):
        """Evaluate recall of derivation `d`.

        Expects `d` to be a coarse binarized derivation.

        Note: This version gives credit for getting binarized items correct.

        """
        return recall_tree(self.gold_items, d)

    def evalb_official(self, d):
        "Run *coarse* binarized derivation thru evalb script."
        assert check_binary_tree(d), d
        d = collapse_self_transition(d)
        c = unbinarize(fix_terminals(item_tree(d), self.sentence.split()))
        return evalb(self.gold_unbinarized, c)

    def evalb_unofficial(self, d):
        "Run *coarse* (binarized or unbinarized) derivation thru the *unofficial* evalb accuracy measure."
        return evalb_unofficial(self.evalb_items, d)   # note: no need to unbinarize!

    def posacc(self, d):
        "Compute POS tag accuracy."
        if not hasattr(d, 'pos'):
            return 0.0   # failed parse
        # compute tagging accuracy
        pred = zip(*d.pos())[1]
        gold = zip(*Tree.fromstring(self.gold_unbinarized).pos())[1]
        return np.mean([a==b for a, b in zip(pred, gold)])

    @property
    def nodes(self):
        N = self.N
        # nodes: doesn't include width=1 or width=N -- shouldn't prune these.
        nodes = [(I,K) for I in xrange(N) for K in xrange(I+1,N+1) if K-I > 1 and K-I != N]
        nodes.sort(key=lambda x: (x[1]-x[0], x[0]))     # CKY order (bottom-up, left-to-right)
        return nodes

    @property
    def mask(self):
        "Create a fresh copy of the *unpruned* mask."
        return np.ones((self.N, self.N+1), dtype=np.int)

    def __repr__(self):
        # TODO: assign every example an unique descriptor which is stable across
        # processes (memory address and the order in which it was loaded are bad
        # examples -- the sentence doesn't work either because there are
        # duplicates.). Setup assigned reasonable names, so use those names when
        # they're available.
        return 'Example(%s)' % (self.name if hasattr(self, 'name') else repr(self.sentence))
