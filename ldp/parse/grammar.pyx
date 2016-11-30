#!python
#cython: initializedcheck=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: infertypes=True
#cython: cdivision=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from collections import defaultdict

from nltk import Tree
from ldp.parsing.unk import signature
from arsenal.alphabet import Alphabet
from arsenal.timer import timeit
from pandas import read_csv

from cython.operator cimport dereference as deref, preincrement as inc
from numpy cimport ndarray


import re
re_subscript = re.compile(r'_\d+$')

Vt = np.double
Dt = np.int16


LHSUR_dt = \
  np.dtype([('child', Dt),
            ('weight', Vt)])

LHSBR_dt = \
  np.dtype([('left', Dt),
            ('right', Dt),
            ('weight', Vt)])

RHSBR_dt = \
  np.dtype([('weight', Vt),
            ('parent', Dt)])

UR_dt = \
  np.dtype([('parent', Dt),
            ('weight', Vt)])

BR_dt = \
  np.dtype([('right', Dt),
            ('parent', Dt),
            ('weight', Vt)])

RCBR_dt = \
  np.dtype([('left', Dt),
            ('parent', Dt),
            ('weight', Vt)])


cdef class Grammar(object):

    def __init__(self, object rules, object lrules, object sym, object lex, int root,
                 str name='', coarse_alphabet=None, value_domain = 'logprob'):
        cdef int i
        self.name = name
        self.lex = lex
        self.lrules = np.array(lrules)  # TODO: should probably set the dtype to be (double,int,int)
        self.rules = np.array(rules)    # TODO: should probably set the dtype to be (double,int,int,int)
        self.n_lrules = len(lrules)
        self.n_rules = len(rules)
        self.sym = sym
        self.root = root
        self.nsymbols = len(sym)
        self.value_domain = value_domain
        self._index()

        self.as_prob = None

        if coarse_alphabet is None:
            coarse_alphabet = Alphabet()

        # coarse-to-fine mapping
        coarse2fine = defaultdict(list)
        self.fine2coarse = np.empty(self.nsymbols, dtype=object)
        self.fine2coarse_int = np.empty(self.nsymbols, dtype=Dt)
        for k,i in sym.items():
            c = k.split('_')[0]
            coarse2fine[c].append(i)
            self.fine2coarse[i] = c
            self.fine2coarse_int[i] = coarse_alphabet[c]
        self.coarse2fine = dict(coarse2fine)
        self.coarse_alphabet = coarse_alphabet

    #___________________________________
    # Annoying little access methods
    def rules_x_yz(self, x):
        return deref(self.r_x_yz[x])
    def rules_y_xz(self, y):
        return deref(self.r_y_xz[y])
    def rules_z_xy(self, z):
        return deref(self.r_z_xy[z])
    def rules_x_y(self, x):
        return deref(self.r_x_y[x])
    def rules_y_x(self, y):
        return deref(self.r_y_x[y])
    #___________________________________
    #

    def _index(self):

        cdef:
            int x, y, z, i
            ndarray[double, ndim=2] rules
            ndarray[double, ndim=2] lrules
            V_t w
            RCBR  rcbr
            BR    lcbr
            LHSBR lhsbr
            UR    lcur
            LHSUR lhsur
            RHSBR rhsbr

        rules = self.rules
        lrules = self.lrules

        # left-child loop indexing strategy proposed by (Dunlop+'10)
        #
        # sort rules by left child, store array of score, parent and sibling
        #
        # opposed to the usual strategy of indexing binary rules by
        # left-right child pairs.

        for _ in range(len(self.lex)):
            self.preterm.push_back(new URvv())

        # sort by pos tag.
        #lrules.sort(key=lambda r: r[1])
        lrules = lrules[np.argsort(lrules[:,1])]

        for (w,x,y) in self.lrules:
            lcur.parent = x
            lcur.weight = w
            self.preterm[y].push_back(lcur)

        for _ in xrange(self.nsymbols):
            self.r_x_yz.push_back(new LHSBRvv())
            self.r_y_xz.push_back(new BRvv())
            self.r_z_xy.push_back(new RCBRvv())
            self.r_x_y.push_back(new LHSURvv())
            self.r_y_x.push_back(new URvv())

        # sort rules by right child
        #self.rules.sort(key=lambda x: x[3])   # column 3 is right child
        rules = rules[np.argsort(rules[:,3])]

        for _ in range(self.nsymbols**2):
            self.r_yz_x.push_back(new RHSBRvv())

#        for (w, x, y, z) in self.rules:
        for i in range(self.n_rules):

            w = rules[i,0]
            x = <int>rules[i,1]
            y = <int>rules[i,2]
            z = <int>rules[i,3]

            if z == -1:
                lhsur.child = y
                lhsur.weight = w
                self.r_x_y[x].push_back(lhsur)

                lcur.parent = x
                lcur.weight = w
                self.r_y_x[y].push_back(lcur)

            else:
                lhsbr.left = y
                lhsbr.right = z
                lhsbr.weight = w
                self.r_x_yz[x].push_back(lhsbr)

                lcbr.right = z
                lcbr.parent = x
                lcbr.weight = w
                self.r_y_xz[y].push_back(lcbr)

                rcbr.left = y
                rcbr.parent = x
                rcbr.weight = w
                self.r_z_xy[z].push_back(rcbr)

                rhsbr.parent = x
                rhsbr.weight = w
                self.r_yz_x[y*self.nsymbols + z].push_back(rhsbr)

        # NOTE: Weight reference tracking assumes that we won't change the
        # indexes. Adding/remove elements can resize the vector datastructure,
        # which causes elements in memory and pointer to be invalid.

        # TODO: Missing weight_refs for some indexes.

        cdef pair[int,pair[int,int]] R
        for y in range(self.nsymbols):
            r_y_xz_it = self.r_y_xz[y].begin()
            while r_y_xz_it != self.r_y_xz[y].end():
                R.first = deref(r_y_xz_it).parent
                R.second.first = y
                R.second.second = deref(r_y_xz_it).right
                self.weight_refs[R].push_back(&(deref(r_y_xz_it).weight))
                inc(r_y_xz_it)

            r_y_x_it = self.r_y_x[y].begin()
            while r_y_x_it != self.r_y_x[y].end():
                R.first = deref(r_y_x_it).parent
                R.second.first = y
                R.second.second = -1
                self.weight_refs[R].push_back(&(deref(r_y_x_it).weight))
                inc(r_y_x_it)

        cdef pair[int,int] p
        for v in range(self.preterm.size()):
            r_preterm_it = self.preterm[v].begin()
            while r_preterm_it != self.preterm[v].end():
                p.first = deref(r_preterm_it).parent
                p.second = v
                self.weight_refs_lex[p].push_back(&(deref(r_preterm_it).weight))
                inc(r_preterm_it)

    def set_weight(self, int x, int y, int z, double new_weight):
        cdef pair[int,pair[int,int]] r
        r.first = x
        r.second.first = y
        r.second.second = z
        for ref in self.weight_refs[r]:
            ref[0] = new_weight

    def get_weight(self, int x, int y, int z):
        cdef pair[int,pair[int,int]] r
        r.first = x
        r.second.first = y
        r.second.second = z
        vals = []
        for ref in self.weight_refs[r]:
            vals.append(ref[0])
        return vals

    def set_weight_lex(self, int x, int y, double new_weight):
        cdef pair[int,int] r
        r.first = x
        r.second = y
        for ref in self.weight_refs_lex[r]:
            ref[0] = new_weight

    def get_weight_lex(self, int x, int y):
        cdef pair[int,int] r
        r.first = x
        r.second = y
        vals = []
        for ref in self.weight_refs_lex[r]:
            vals.append(ref[0])
        return vals

    def __dealloc__(self):
        for a in self.r_x_yz:
            del a
        for b in self.r_y_xz:
            del b
        for c in self.r_z_xy:
            del c
        for d in self.r_yz_x:
            del d
        for e in self.r_x_y:
            del e
        for f in self.r_y_x:
            del f
        for g in self.preterm:
            del g

    cpdef int token(self, str w, int i):
        "Convert string to integer. Handles unseen words."
        if w not in self.lex:
            w = signature(w, i, w.lower() in self.lex)
        return self.lex[w]

    cpdef long[:] encode_sentence(self, list sentence):
        "Encode sentence as an array of integers (long). Handles unknown words."
        return np.array([self.token(w, i) for i, w in enumerate(sentence)])

    def encode_sentence_string(self, list sentence):
        """Encode sentence is encoded known word symbols. Same as encoded_sentence, but
        doesn't mapt to ints).

        """
        return [self.lex.lookup(i) for i in self.encode_sentence(sentence)]

    def coarse_label(self, int x):
        return self.fine2coarse[x]

    def coarse_derivation(self, t):
        "unintegerize tree, coarsen labels."
        if isinstance(t, int):
            # no need to coarsen here because terminals don't have subscripts.
            return self.lex.lookup(t)
        assert len(t) > 0
        x = t[0]
        X = self.coarse_label(x[0])
        if len(t) == 1:
            return X
        elif len(t) == 2:
            return Tree(X, [self.coarse_derivation(t[1])])
        elif len(t) == 3:
            return Tree(X,
                        [self.coarse_derivation(t[1]),
                         self.coarse_derivation(t[2])])
        else:
            assert False, t

    def llh(self, object t):
        """Compute log-likelihood for tree `t`. Useful for debugging (e.g., why didn't
        this tree beat that one?)

        Assumes input is an item tree where the items are (X,I,K). Be careful
        not to pass in (I,K,X)!

        """

        cdef:
            UR ur
            LHSUR lhsur
            LHSBR lhsbr
            int X, Y, Z
            object left, right

        assert isinstance(t, Tree)
        [X,_,_] = t.label()

        if len(t) == 0:
            return float('-inf')

        elif len(t) == 1:
            [left] = t
            if isinstance(left, Tree):
                # search unary rules
                [Y,_,_] = left.label()
                for lhsur in deref(self.r_x_y[X]):
                    if lhsur.child == Y:
                        return self.llh(left) + lhsur.weight

            else:
                # search preterminal rules
                if isinstance(left, int):
                    Y = left
                else:
                    [Y,_,_] = left
                for ur in deref(self.preterm[Y]):
                    if ur.parent == X:
                        return ur.weight  # base case

        else:
            # search binary rules
            [left, right] = t
            [Y,_,_] = left.label()
            [Z,_,_] = right.label()
            for lhsbr in deref(self.r_x_yz[X]):
                if lhsbr.left == Y and lhsbr.right == Z:
                    return (self.llh(left)
                            + self.llh(right)
                            + lhsbr.weight)

        return float('-inf')

    def anneal(self, gamma):
        """Create an equivalent grammar with the same rules, but annealed weights. This
        is used to make expectations over derivations approach Viterbi
        derivations (maximization)

        """
        assert self.value_domain == 'logprob'
        return Grammar([(w*gamma,x,y,z) for (w,x,y,z) in self.rules],
                       [(w*gamma,x,y) for (w,x,y) in self.lrules],
                       self.sym,
                       self.lex,
                       self.root,
                       name = 'Annealed-%s' % self.name,
                       value_domain = 'logprob',
                       coarse_alphabet = self.coarse_alphabet)

    def exp_the_grammar_weights(self):
        if self.as_prob is None:
            rules = self.rules.copy()
            lrules = self.lrules.copy()
            np.exp(rules[:,0], out=rules[:,0])
            np.exp(lrules[:,0], out=lrules[:,0])
            self.as_prob = Grammar(rules = rules,
                                   lrules = lrules,
                                   sym = self.sym,
                                   lex = self.lex,
                                   root = self.root,
                                   name = 'Exp-%s' % self.name,
                                   value_domain = 'prob',
                                   coarse_alphabet = self.coarse_alphabet)
        return self.as_prob

    @classmethod
    def load(cls, prefix, root=None, **kwargs):

        sym = Alphabet.load(prefix + '.sym.alphabet')
        sym.freeze()
        lex = Alphabet.load(prefix + '.lex.alphabet')
        lex.freeze()

        df = read_csv(prefix + '.gr.csv')
        g = [(w, x, y, z) for (_, w, x, y, z) in df[['score', 'head', 'left', 'right']].itertuples()]

        df = read_csv(prefix + '.lex.csv')
        glex = [(w, x, y) for (_, w, x, y) in df[['score', 'head', 'left']].itertuples()]

        # two possibilities for root symbols
        if root is None:
            root = 'ROOT'
            if root not in sym:
                root = 'ROOT_0'
            assert root in sym
            #print 'root symbol:', root

        name = prefix.split('/')
        name = name[len(name)-1]

        grammar = cls(g, glex, sym, lex, sym[root], name, **kwargs)

        print grammar
        return grammar

    @classmethod
    def tiny_grammar(cls, s):
        "Load a tiny hand-written grammar from a string."
        [(lexicon, rules)] = re.findall('# lexicon\n([\w\W]*)# rules\n([\w\W]*)', s)

        sym = Alphabet()
        lex = Alphabet()
        lrules = []
        grules = []

        for line in lexicon.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            w, tag, word = line.split()
            lrules.append([float(w), sym[tag], lex[word]])

        for line in rules.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.split()

            assert len(line) in [3,4]

            w = float(line[0])
            x = sym[line[1]]
            y = sym[line[2]]
            z = sym[line[3]] if len(line) == 4 else -1

            grules.append([w,x,y,z])

        root = sym['ROOT']
        return cls(grules, lrules, sym, lex, root)

    def __repr__(self):
        return '%s(name=%s rules=%s, lexrules=%s, vocab=%s, nonterm=%s)' \
            % (self.__class__.__name__, self.name,
               self.n_rules, self.n_lrules,
               len(self.lex), len(self.sym))
