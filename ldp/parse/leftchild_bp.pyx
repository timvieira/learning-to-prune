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

"""
Backtracing, left-child loop, terminal->preterminal rules separated.
"""

from numpy import empty

from ldp.parse.common cimport V_t, Vt, K_t, semizero, tri, follow_backpointers, Cell, Cell_dt
from ldp.parse.grammar cimport Grammar, BR, UR
from cython.operator cimport dereference as deref

from libcpp.vector cimport vector


ctypedef vector[short int] intvec
ctypedef intvec* intvec_ptr


def pruned_parser(long[:] sentence,
                  Grammar grammar,
                  K_t[:,:] keep):

    cdef:
        int N, I, J, K, X, Y, Z, w
        V_t s, leftscore, rightscore
        intvec_ptr cell
        intvec agenda
        vector[intvec_ptr] span
        Cell[:,:,:] chart
        BR br
        UR ur
        int NT
        int pops, pushes

    NT = grammar.nsymbols

    pops = 0
    pushes = 0
    N = sentence.shape[0]

    chart = empty((N,N+1,NT), dtype=Cell_dt)

    with nogil:
        for I in range(N):
            for K in range(I+1,N+1):
                for X in range(NT):
                    chart[I,K,X].score = semizero
                    chart[I,K,X].j = -1
                    chart[I,K,X].y = -1
                    chart[I,K,X].z = -1

        for I in range(N):
            for K in range(I+1,N+1):
                span.push_back(new intvec())

        for I in range(N):
            Y = sentence[I]
            K = I + 1
            cell = span[tri(I,K)]

            # add preterminals
            for ur in deref(grammar.preterm[Y]):
                s = ur.weight
                X = ur.parent
                pushes += 1

                if s > chart[I,K,X].score:
                    if chart[I,K,X].score == semizero:
                        pops += 1
                        cell.push_back(X)
                        agenda.push_back(X)

                    chart[I,K,X].score = s
                    chart[I,K,X].j = K
                    chart[I,K,X].y = -1
                    chart[I,K,X].z = -1

            # unary rules above preterminals
            for Y in agenda:
                leftscore = chart[I,K,Y].score
                for ur in deref(grammar.r_y_x[Y]):
                    X = ur.parent
                    s = ur.weight + leftscore
                    pushes += 1
                    if s > chart[I,K,X].score:
                        # Allow two levels of unaries at width one items
                        if chart[I,K,Y].z == -1 and chart[I,K,Y].y != -1:  # prev is unary
                            if (chart[I,K,chart[I,K,Y].y].z == -1 and
                                chart[I,K,chart[I,K,Y].y].y != -1):        # prev-prev is unary
                                continue
                        if chart[I,K,X].score == semizero:
                            pops += 1
                            cell.push_back(X)
                        chart[I,K,X].score = s
                        chart[I,K,X].j = K
                        chart[I,K,X].y = Y
                        chart[I,K,X].z = -1

            agenda.clear()

        for w in range(2, N+1):
            for I in range(N-w + 1):
                K = I + w
                if not keep[I,K] and K-I != N:
                    continue

                cell = span[tri(I,K)]

                # split point loop (for binary rules)
                for J in range(I+1, K):
                    for Y in deref(span[tri(I,J)]):
                        leftscore = chart[I,J,Y].score
                        for br in deref(grammar.r_y_xz[Y]):
                            Z = br.right
                            rightscore = chart[J,K,Z].score
                            if rightscore <= semizero:
                                continue
                            X = br.parent
                            s = leftscore + rightscore + br.weight
                            pushes += 1
                            if s > chart[I,K,X].score:
                                if chart[I,K,X].score == semizero:
                                    cell.push_back(X)
                                    agenda.push_back(X)
                                    pops += 1
                                chart[I,K,X].score = s
                                chart[I,K,X].j = J
                                chart[I,K,X].y = Y
                                chart[I,K,X].z = Z

                # apply unary rules
                for Y in agenda:
                    leftscore = chart[I,K,Y].score
                    for ur in deref(grammar.r_y_x[Y]):
                        X = ur.parent
                        pushes += 1
                        s = ur.weight + leftscore
                        if s > chart[I,K,X].score and chart[I,K,Y].z != -1:
                            # NOTE: this can accidently create muli-level unary
                            # chain. This is because we don't buffer the updates
                            # before applying them to the chart.
                            if chart[I,K,X].score == semizero:
                                cell.push_back(X)
                                pops += 1
                            chart[I,K,X].score = s
                            chart[I,K,X].j = K
                            chart[I,K,X].y = Y
                            chart[I,K,X].z = -1

                agenda.clear()

        # Deallocate
        for x in span:
            del x

    cdef list d = follow_backpointers(chart, grammar.root, 0, N, sentence)

    return ParserState(chart[0,N,grammar.root].score, d, pops, pushes, chart)


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
