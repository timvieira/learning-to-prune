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

"""Dyanmic programming algorithm for rollouts.

Estimate expected recall after changing a single bit in the pruning mask. This
can be used to compute expected recall of changing each bit in time O(Gn^3),
where the naive algorithm (=re-run the parser) takes time O(T*Gn^3) where T is
the number of pruning decisions (typically T=O(n^2), which corresponds to the
number of spans).

Applications: Searn rollouts with static features, static oracle, tied
randomness.

"""

from __future__ import division
import numpy as np
from numpy import empty
from ldp.parse.common cimport Grammar
from cython.operator cimport dereference as deref
from ldp.parse.common cimport tri, Grammar, Dt, D_t
from ldp.prune.example cimport Example

from libc.stdio cimport printf
from libcpp.vector cimport vector
from ldp.parse.containers cimport unordered_set
ctypedef vector[short int] intvec
#ctypedef unordered_set[short int] intvec

#_______________________________________________________________________________
#

from libc.math cimport exp

from numpy cimport double_t

ctypedef double LogVal

cdef inline double to_real(LogVal x) nogil:
    return x

cdef inline LogVal logeq(double x) nogil:
#    return exp(x)
    return x

cdef inline LogVal LogValOne() nogil:
    return 1.0

cdef inline LogVal LogValZero() nogil:
    return 0.0

cdef inline LogVal logval(double x) nogil:
    return x

cdef inline LogVal logval_add(LogVal x, LogVal y) nogil:
    return x + y

cdef inline LogVal logval_div(LogVal x, LogVal y) nogil:
    return x / y

cdef inline LogVal logval_mul(LogVal x, LogVal y) nogil:
    return x * y

cdef inline LogVal _abs(LogVal x) nogil:
    return x if x >= 0 else -x

cdef inline int logval_is_zero(LogVal x) nogil:
    return x == 0.0

#________
#

#cdef packed struct LogVal:
#    double_t ell
#
#cdef inline double to_real(LogVal x) nogil:
#    return x.ell
#
#cdef inline LogVal logeq(double x) nogil:
#    cdef LogVal c
#    c.ell = x #exp(x)
#    return c
#
#cdef inline LogVal LogValOne() nogil:
#    return logval(1.0)
#
#cdef inline LogVal LogValZero() nogil:
#    return logval(0.0)
#
#cdef inline LogVal logval(double x) nogil:
#    cdef LogVal c
#    c.ell = x
#    return c
#
#cdef inline LogVal logval_add(LogVal x, LogVal y) nogil:
#    cdef LogVal c
#    c.ell = x.ell + y.ell
#    return c
#
#cdef inline LogVal logval_div(LogVal x, LogVal y) nogil:
#    cdef LogVal c
#    c.ell = x.ell / y.ell
#    return c
#
#cdef inline LogVal logval_mul(LogVal x, LogVal y) nogil:
#    cdef LogVal c
#    c.ell = x.ell * y.ell
#    return c
#
#cdef inline LogVal _abs(LogVal x) nogil:
#    cdef LogVal c
#    c.ell = x.ell if x.ell >= 0 else -x.ell
#    return c
#
#cdef inline int logval_is_zero(LogVal x) nogil:
#    return x.ell == 0.0

#_______________________________________________________________________________
#

DEF INSIDE  = 1
DEF OUTSIDE = 2
DEF DEBUG   = 4

cdef packed struct Semiring:
    LogVal p
    LogVal r

cdef object Semiring_dt = np.dtype([('p', np.double), ('r', np.double)])

cdef inline int is_zero(Semiring x) nogil:
    return logval_is_zero(x.p) and logval_is_zero(x.r)

cdef inline Semiring SemiringOne() nogil:
    cdef Semiring c
    c.p = LogValOne()
    c.r = LogValZero()
    return c

cdef inline Semiring SemiringZero() nogil:
    cdef Semiring c
    c.p = LogValZero()
    c.r = LogValZero()
    return c

cdef inline Semiring add(Semiring y, Semiring x) nogil:
    cdef Semiring c
    c.p = logval_add(y.p, x.p)
    c.r = logval_add(y.r, x.r)
    return c

cdef inline Semiring mul(Semiring y, Semiring x) nogil:
    cdef Semiring c
    c.p = logval_mul(y.p, x.p)
    c.r = logval_add(logval_mul(y.p, x.r), logval_mul(x.p, y.r))
    return c


cdef class InsideOut:
    cdef:
        vector[intvec*] _span_index
        double _rule_weight_hack
    cdef readonly:
        long[:] sentence
        Grammar grammar
        int steps, N, NT
        double[:,:] est
        Semiring[:,:,:,:] A, B
        Semiring[:,:] A2, B2
        double[:,:] A2_r, A2_p

        int mode, n_items
        object example
        double num, den, val
        LogVal _num, _den, _val, recall_scale
        double[:,:] mask
        D_t[:,:] gold_coarse

    def __init__(self, Example example, Grammar grammar, double[:,:] K, int steps=2, int with_gradient=True):
        cdef Semiring v
        cdef LogVal Z, r, dZ, dr, r1, Z1, eps
        cdef int i,k,x,t,xx

        grammar = grammar.exp_the_grammar_weights()
        assert grammar.value_domain == 'prob', grammar.value_domain

        self.example = example
        self.sentence = example.tokens
        self.grammar = grammar
        self.steps = steps
        self.NT = self.grammar.nsymbols
        self.N = self.sentence.shape[0]

        cdef int N = self.N
        cdef int NT = self.grammar.nsymbols

        self.B = np.empty((N,N+1,NT,steps), dtype=Semiring_dt)
        self.B2 = np.empty((N,N+1), dtype=Semiring_dt)
        if with_gradient:
            self.A = np.empty((N,N+1,NT,steps), dtype=Semiring_dt)
            self.A2_p = np.zeros((N,N+1))
            self.A2_r = np.zeros((N,N+1))
            self.A2 = np.empty((N,N+1), dtype=Semiring_dt)

        self.gold_coarse = np.ones((N,N+1), dtype=Dt)*-1
        for (X,i,k) in self.example.gold_items:
            self.gold_coarse[i,k] = self.grammar.coarse_alphabet[X]

        self.est = np.empty((N,N+1))*np.nan
        self.mask = np.array(K, copy=1)

        self.recall_scale = logval(1.0/len(self.example.gold_items))

        with nogil:

            for IK in range(N*(N+1)//2):
                self._span_index.push_back(new intvec())

            # initialize inside charts
            for i in range(N):
                for k in range(i+1,N+1):
                    for x in range(NT):
                        for t in range(steps):
                            self.B[i,k,x,t] = SemiringZero()
            for i in range(N):
                for k in range(i+1, N+1):
                    self.B2[i,k].p = logval(K[i,k])
                    self.B2[i,k].r = LogValZero()

            self.mode = INSIDE
            self.inside()

            v = self.B[0,N,self.grammar.root,steps-1]
            self.num = to_real(v.r)
            self.den = to_real(v.p)
            self.val = to_real(logval_div(v.r, v.p)) if not logval_is_zero(v.p) else 0.0

            self._num = v.r
            self._den = v.p
            self._val = logval_div(v.r, v.p) if not logval_is_zero(v.p) else LogValZero()

            if with_gradient:
                self._gradient()

    cdef void _gradient(self) nogil:
        cdef int i, k
        cdef double estimate
        cdef LogVal Z, r, Z1, r1, dZ, dr, eps
        cdef int N = self.N
        cdef int NT = self.NT
        cdef int steps = self.steps

        # initialize outside charts
        for i in range(N):
            for k in range(i+1,N+1):
                for x in range(NT):
                    for t in range(steps):
                        self.A[i,k,x,t] = SemiringZero()
        for i in range(N):
            for k in range(i+1, N+1):
                self.A2[i,k] = SemiringZero()

        # initialize root to semione
        self.A[0,N,self.grammar.root,steps-1] = SemiringOne()
        self.mode = OUTSIDE
        self.outside()

        Z = self.B[0,N,self.grammar.root,steps-1].p
        r = self.B[0,N,self.grammar.root,steps-1].r
        for i in range(N):
            for k in range(i+1, N+1):
                eps = logval(1-2*self.mask[i,k])
                dZ = self.A2[i,k].p
                dr = self.A2[i,k].r
                r1 = logval_add(r, logval_mul(eps, dr))
                Z1 = logval_add(Z, logval_mul(eps, dZ))
                # Clip: Numerator and denominator should never go
                # negative. If the extrapolated value goes negative, clip it
                # back to zero.
                if to_real(r1) < 0:
                    r1 = LogValZero()
                if to_real(Z1) < 0:
                    Z1 = LogValZero()

                estimate = to_real(logval_div(r1, Z1)) if not logval_is_zero(Z1) else 0.0

                # Clip: expected recall should in range [0,1].
                if estimate > 1:
                    estimate = 1

                # Important note: the follow if-statement is crucial (at least when
                # running with the inside-outside speed-up). See notes with-in.
                if (1-self.mask[i,k]) == 0:  # prune

                    if logval_is_zero(Z):
                        # Avoids divide-by-zero in relative difference. This is
                        # valid because pruning even more can't possibly change the
                        # estimate.
                        estimate = 0
                    else:
                        # Without the following special handling we get wildly
                        # inaccurate errors when the estimate should be zero.
                        #
                        # if |ad_den - den|/|den| < tol` that means that we're pruning
                        # away (almost) all of the parse forest. Therefore, it's
                        # probably a good idea to avoid the potentially inaccurate
                        # division operation. Furthermore, if the denominator is
                        # supposed to be zero, then the numerator should be too!
                        #

                        if estimate < 0 or estimate > 1:
                            #print '[warn] est<0, using other numerical hack.'
                            estimate = 0

                        if to_real(logval_div(_abs(Z1), _abs(Z))) < 1e-8:
                            #print '[warn] using stability trick.'
                            estimate = 0

                # Take quotient to form the estimate of risk.
                self.est[i,k] = estimate
                self.A2_p[i,k] = to_real(self.A2[i,k].p)
                self.A2_r[i,k] = to_real(self.A2[i,k].r)

    def __dealloc__(self):
        for x in self._span_index:
            del x

    def marginals(self):
        "Compute marginals. Used for debugging."
        cdef int T, i, k, X
        M = {}
        Z = self.B[0, self.N, self.grammar.root, self.steps-1]
        for i in range(self.N):
            for k in range(i+1, self.N+1):
                for X in range(self.NT):
                    for T in range(self.steps):
                        M[i,k,X,T] = to_real(mul(self.B[i,k,X,T], self.A[i,k,X,T]).p / Z['p'])
        return M

    cdef void inside(self) nogil:
        "Apply edge callbacks in topological order."

        cdef:
            int I, J, K, X, Y, Z, w, T, steps
            Semiring rule, left, right, span, was
            intvec* cell

        steps = self.steps

        for I in range(self.N):
            Y = self.sentence[I]
            K = I + 1
            cell = self._span_index[tri(I,K)]
            # add preterminals
            for ur in deref(self.grammar.preterm[Y]):
                X = ur.parent
                self._rule_weight_hack = ur.weight

                if is_zero(self.B[I,K,X,0]):
                    cell.push_back(X)
#                    cell.insert(X)

                self.preterm_callback(I,X,Y)

            self.inside_unary(I, K)

        for w in range(2, self.N+1):
            for I in range(self.N-w + 1):
                K = I + w

                if self.mask[I,K] == 0:
                    continue

                span = self.B2[I,K]
                cell = self._span_index[tri(I,K)]

                XT = 0
                YT = steps-1
                ZT = steps-1

                for J in range(I+1, K):
                    for Y in deref(self._span_index[tri(I,J)]):

                        left = mul(span, self.B[I,J,Y,YT])

                        for br in deref(self.grammar.r_y_xz[Y]):   # left-child loop
                            Z = br.right
                            X = br.parent

                            right = self.B[J,K,Z,ZT]

                            if is_zero(right):        # safely filter right neighbors.
                                continue

                            was = self.B[I,K,X,XT]
                            if is_zero(was):
                                cell.push_back(X)
#                                cell.insert(X)
                            rule.p = logeq(br.weight)
                            rule.r = LogValZero()

                            self.B[I,K,X,XT] = add(was, mul(mul(left, right), rule))

                self.inside_unary(I, K)

    cdef void inside_unary(self, int I, int K) nogil:
        cdef:
            int T, Y, X
            intvec tmp
            intvec* cell

        cell = self._span_index[tri(I,K)]

        for T in range(self.steps-1):

            # free unary rule X->X
            for X in deref(cell):
                self._rule_weight_hack = LogValOne()
                self.unary_callback(I, K, X, T+1, X, T)

            for Y in deref(cell):
                for ur in deref(self.grammar.r_y_x[Y]):
                    X = ur.parent
                    self._rule_weight_hack = ur.weight

                    if self.mode == INSIDE:
                        if is_zero(self.B[I,K,X,T+1]):
                            tmp.push_back(X)
#                            tmp.insert(X)

                    self.unary_callback(I,K,X,T+1,Y,T)

            if self.mode == INSIDE:
                for X in tmp:
                    cell.push_back(X)
#                    cell.insert(X)
                tmp.clear()

    cdef inline void unary_callback(self, int I, int K, int X, int XT, int Y, int YT) nogil:
        cdef Semiring rule = SemiringZero()
        rule.p = logeq(self._rule_weight_hack)

        # Edge reward. We only get credit for the top-level symbol in each span.
        if XT == self.steps-1 and self.grammar.fine2coarse_int[X] == self.gold_coarse[I,K]:
            # NOTE: We use k_e = <p_e, p_e*r_e> to get risk. Hence, the
            # multiplication of edge reward by p_e.
            rule.r = logval_mul(rule.p, self.recall_scale)

        if self.mode == INSIDE:
            self.B[I,K,X,XT] = add(self.B[I,K,X,XT], mul(self.B[I,K,Y,YT], rule))

        elif self.mode == OUTSIDE:
            self.A[I,K,Y,YT] = add(self.A[I,K,Y,YT], mul(self.A[I,K,X,XT], rule))

    cdef inline void preterm_callback(self, int I, int X, int Y) nogil:
        cdef int K
        cdef Semiring rule = SemiringZero()
        K = I+1
        rule.p = logeq(self._rule_weight_hack)
        rule.r = LogValZero()
        if self.mode == INSIDE:
            self.B[I,K,X,0] = rule

#        elif self.mode == OUTSIDE:
#            self.B[I,K,Y,None] = self.B[I,K,Y,None] + self.A[I,K,X,0] * rule

    cdef void outside(self) nogil:
        # Notes:
        #
        # - you can't use the span_index from inside pass because zero inside score
        #   does not imply a zero outside score.
        #
        # - Need to be careful with zeros on the outside pass because the dp-alg is
        #   essentially an exercise in handling zeros correctly (It's very sensitive
        #   to one v. two zeros as the gradient of a product should be, but rarely
        #   matters in practice.)
        #
        cdef:
            int N, I, J, K, X, Y, Z, w, steps
            Semiring left, right, span, rule, d, aa, was
            int left_zero, right_zero, span_zero
            int XT, YT, ZT
            int was_zero

        steps = self.steps
        N = self.sentence.shape[0]
        for w in reversed(range(2, N+1)):
            for I in range(N-w + 1):
                K = I + w

                span = self.B2[I,K]
                span_zero = is_zero(span)

                self.outside_unary(I, K)

                XT = 0
                YT = steps-1
                ZT = steps-1
                for J in range(I+1, K):
                    for Y in deref(self._span_index[tri(I,J)]):
                        left = self.B[I,J,Y,YT]
                        for br in deref(self.grammar.r_y_xz[Y]):   # left-child loop
                            Z = br.right
                            X = br.parent
                            XT = 0
                            aa = self.A[I,K,X,XT]
                            if is_zero(aa):
                                continue

                            right = self.B[J,K,Z,ZT]
                            if is_zero(right) and span_zero:
                                continue

                            rule.p = logeq(br.weight)
                            rule.r = LogValZero()
                            d = mul(aa, rule)

                            self.A2[I,K] = add(self.A2[I,K], mul(mul(d, left), right))
                            if is_zero(span):
                                continue

                            was = self.A[J,K,Z,ZT]
                            self.A[J,K,Z,ZT] = add(was, mul(mul(d, span), left))

                    for Z in deref(self._span_index[tri(J,K)]):
                        right = mul(self.B[J,K,Z,ZT], span)
                        for rr in deref(self.grammar.r_z_xy[Z]):   # right-child loop
                            Y = rr.left
                            X = rr.parent
                            aa = self.A[I,K,X,XT]
                            if is_zero(aa):
                                continue

                            left = self.B[I,J,Y,YT]
                            if is_zero(left) and span_zero:
                                continue

                            rule.p = logeq(rr.weight)
                            rule.r = LogValZero()
                            d = mul(aa, rule)

                            was = self.A[I,J,Y,YT]
                            self.A[I,J,Y,YT] = add(was, mul(d, right))


        for I in range(N):
            K = I + 1
            self.outside_unary(I, K)
            Y = self.sentence[I]
            for ur in deref(self.grammar.preterm[Y]):
                X = ur.parent
                self._rule_weight_hack = ur.weight
                self.preterm_callback(I,X,Y)

    cdef void outside_unary(self, int I, int K) nogil:
        cdef:
            int T, Y, X
            Semiring rule
        for T in reversed(range(self.steps-1)):

            # TODO: why doesn't sparse loop work?
            # - Maybe something to do with containers changing size during iteration?
            for X in range(self.NT):
#            for X in deref(self._span_index_outside[tri(I,K)]):
                if is_zero(self.A[I,K,X,T+1]):
                    continue
                # need the lhs-index unary rules because we're working backwards
                for ur in deref(self.grammar.r_x_y[X]):
                    Y = ur.child
                    self._rule_weight_hack = ur.weight
                    self.unary_callback(I,K,X,T+1,Y,T)

                # free X->X rule to advance the time-step
                self._rule_weight_hack = LogValOne()
                self.unary_callback(I,K,X,T+1,X,T)
