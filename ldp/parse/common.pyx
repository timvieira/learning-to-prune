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
Common parsing utils and datatypes.

"""

import numpy as np

from cython.operator cimport dereference as deref

Vt = np.double
Dt = np.int16
Kt = np.int

semizero = float('-infinity')
Cell_dt = np.dtype([('score', Vt), ('y', Dt), ('z', Dt), ('j', Dt)])


cdef list follow_backpointers(Cell[:,:,:] chart, D_t X, D_t I, D_t K, long[:] sentence):
    "Follow backpointers, returns derivation as list of lists."
    cdef:
        Cell cc
        int Y,Z,J
        list left, right

    cc = chart[I,K,X]
    Y = cc.y
    Z = cc.z
    J = cc.j

    # check for failure.
    if cc.score <= semizero:
        # TODO: find a better way to indicate failure.
        return [(X,I,K)]

#    if J == -1:
#        return [(X,I,K)]

    if Y == -1 and Z == -1:
        # At terminal nodes return the token from sentence`.
        return [(X,I,K), sentence[I]]

    if Y >= 0:
        left = follow_backpointers(chart, Y, I, J, sentence)
        if Z >= 0:
            right = follow_backpointers(chart, Z, J, K, sentence)
            return [(X,I,K), left, right]
        else:
            return [(X,I,K), left]

    return [(X,I,K)]


cdef list backtrace(V_t[:,:,:] chart, Grammar grammar, D_t I, D_t K, D_t X, long[:] sentence):
    """
    Extract derivation from chart without backpointers.
    """
    cdef:
        V_t best, r
        int J, Y, Z

    best = chart[I,K,X]         # already know the maximum

    if best == semizero:        # no parse
        return [(X,I,K)]

    # search unary rules
    for lhsur in deref(grammar.r_x_y[X]):
        r = lhsur.weight
        Y = lhsur.child
        if chart[I,K,Y] + r == best:
            return [(X,I,K),
                    backtrace(chart, grammar, I, K, Y, sentence)]

    if K-I == 1:
        # no unary rules seems to have fired, so it must be a terminal
        return [(X,I,K), sentence[I]]

    # search binary rules
    for lhsbr in deref(grammar.r_x_yz[X]):
        for J in range(I+1,K):            # I < J < K
            r = lhsbr.weight
            Y = lhsbr.left
            Z = lhsbr.right

            if chart[I,J,Y] + chart[J,K,Z] + r == best:
                return [(X,I,K),
                        backtrace(chart, grammar, I, J, Y, sentence),
                        backtrace(chart, grammar, J, K, Z, sentence)]

    # base case no rule appears to be good enough
    return [(X,I,K)]


def backtrace_ties(V_t[:,:] chart, Grammar grammar, D_t I, D_t K, D_t X, long[:] sentence):
    """
    Extract derivation from chart without backpointers.
    """
    cdef:
        V_t best, r
        int J, Y, Z

    best = chart[tri(I,K),X]         # already know the maximum

    if best == semizero:        # no parse
        #yield [(X,I,K)]
        pass
    else:

        # search unary rules
        for lhsur in deref(grammar.r_x_y[X]):
            r = lhsur.weight
            Y = lhsur.child
            if chart[tri(I,K),Y] + r == best:
                for asdf in backtrace_ties(chart, grammar, I, K, Y, sentence):
                    yield [(X,I,K), asdf]

        if K-I == 1:
            # no unary rules seems to have fired, so it must be a terminal
            yield [(X,I,K), sentence[I]]

        else:
            # search binary rules
            for lhsbr in deref(grammar.r_x_yz[X]):
                for J in range(I+1,K):            # I < J < K
                    r = lhsbr.weight
                    Y = lhsbr.left
                    Z = lhsbr.right

                    if chart[tri(I,J),Y] + chart[tri(J,K),Z] + r == best:
                        for abc in backtrace_ties(chart, grammar, I, J, Y, sentence):
                            for xyz in backtrace_ties(chart, grammar, J, K, Z, sentence):
                                yield [(X,I,K), abc, xyz]

            # base case no rule appears to be good enough
            #yield [(X,I,K)]


cdef list backtrace_tri(V_t[:,:] chart, Grammar grammar, int I, int K, int X, long[:] sentence):
    """Extract derivation from chart without backpointers.

    This version assumes the chart is represented as an upper-triangular matrix
    and can be indexed using `tri` the helper function.

    """
    cdef:
        V_t best, r
        int J, Y, Z

    best = chart[tri(I,K),X]         # already know the maximum

    if best == semizero:        # no parse
        return [(X,I,K)]

    # search unary rules
    for lhsur in deref(grammar.r_x_y[X]):
        r = lhsur.weight
        Y = lhsur.child

        if chart[tri(I,K),Y] + r == best:
            return [(X,I,K),
                    backtrace_tri(chart, grammar, I, K, Y, sentence)]

    if K-I == 1:
        # no unary rules seems to have fired, so it must be a terminal
        return [(X,I,K), sentence[I]]

    # search binary rules
    for lhsbr in deref(grammar.r_x_yz[X]):
        r = lhsbr.weight
        Y = lhsbr.left
        Z = lhsbr.right
        for J in range(I+1,K):            # I < J < K
            if chart[tri(I,J),Y] + chart[tri(J,K),Z] + r == best:
                return [(X,I,K),
                        backtrace_tri(chart, grammar, I, J, Y, sentence),
                        backtrace_tri(chart, grammar, J, K, Z, sentence)]

    # base case no rule appears to be good enough
    return [(X,I,K)]
