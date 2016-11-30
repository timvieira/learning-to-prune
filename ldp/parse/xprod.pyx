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
* Backtracing

* pair-of-children grammar loop.

* One-step unary rewrites at width >1 constituents.

* Unary closure at width ==1 constituents.

"""

from numpy cimport ndarray
from numpy import empty

from ldp.parse.common cimport V_t, Vt, K_t, semizero, intpair, tri, backtrace_tri
from ldp.parse.grammar cimport BR, UR, Grammar, RHSBR

from cython.operator cimport dereference as deref


def pruned_parser(long[:] sentence,
                  Grammar grammar,
                  K_t[:,:] keep):

    cdef:
        int N, I, J, K, X, Y, Z, pos, w, b, R, rr
        V_t r, s, leftscore, rightscore
        set cell, ys, zs, agenda
        set[:,:] span
        V_t[:,:] chart
        RHSBR rule
        UR ur
        int root_sym, NT
        int pushes
        intpair YZ

    root_sym = grammar.root
    NT = grammar.nsymbols

    b = 0
    pushes = 0
    N = sentence.shape[0]

    agenda = set()

    chart = empty((N*(N+1)/2,NT), dtype=Vt)
    for I in range(N):
        for K in range(I+1,N+1):
            for X in range(NT):
                chart[tri(I,K),X] = semizero

    span = empty((N,N+1), dtype=set)
    for I in range(N):
        for K in range(I+1,N+1):
            span[I,K] = set()

    for I in range(N):
        Y = sentence[I]
        K = I + 1
        cell = span[I,K]

        # add preterminals
        for ur in deref(grammar.preterm[Y]):
            s = ur.weight
            X = ur.parent

            pushes += 1

            if s > chart[tri(I,K),X]:
                if chart[tri(I,K),X] == semizero:
                    b += 1
                chart[tri(I,K),X] = s
                cell.add(X)
                agenda.add(X)

        while agenda:
            Y = agenda.pop()
            leftscore = chart[tri(I,K),Y]

            for ur in deref(grammar.r_y_x[Y]):
                r = ur.weight
                X = ur.parent
                s = r + leftscore

                pushes += 1

                if s > chart[tri(I,K),X]:
                    if chart[tri(I,K),X] == semizero:
                        b += 1
                    agenda.add(X)
                    cell.add(X)
                    chart[tri(I,K),X] = s

    for w in range(2, N+1):
        for I in range(N-w + 1):
            K = I + w

            if not keep[I,K] and K-I != N:
                continue

            cell = span[I,K]

            for J in range(I+1, K):

                ys = span[I,J]
                for Y in ys:

                    leftscore = chart[tri(I,J),Y]

                    zs = span[J,K]
                    for Z in zs:

                        rightscore = chart[tri(J,K),Z]

                        for rule in deref(grammar.r_yz_x[Y*NT + Z]):
                            X = rule.parent
                            r = rule.weight

                            s = leftscore + rightscore + r

                            pushes += 1

                            if s > chart[tri(I,K),X]:

                                if chart[tri(I,K),X] == semizero:
                                    cell.add(X)
                                    agenda.add(X)
                                    b += 1

                                chart[tri(I,K),X] = s

#            while agenda:
#                Y = agenda.pop()
            for Y in agenda:

                leftscore = chart[tri(I,K),Y]

                for ur in deref(grammar.r_y_x[Y]):
                    r = ur.weight
                    X = ur.parent

                    pushes += 1

                    s = r + leftscore
                    if s > chart[tri(I,K),X]:

                        if chart[tri(I,K),X] == semizero:
                            cell.add(X)
#                            agenda.add(X)
                            b += 1

                        chart[tri(I,K),X] = s

    cdef list d = backtrace_tri(chart, grammar, 0, N, root_sym, sentence)

    return ParserState(chart[tri(0,N),root_sym], d, b, pushes, chart)


cdef class ParserState:
    cdef readonly double likelihood
    cdef readonly list derivation
    cdef readonly int pops
    cdef readonly int pushes
    cdef readonly object chart
    def __init__(self, double llh, list derivation, int pops, int pushes, object chart):
        self.likelihood = llh
        self.derivation = derivation
        self.pops = pops
        self.pushes = pushes
        self.chart = chart
