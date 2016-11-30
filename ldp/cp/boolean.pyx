#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: infertypes=True
#cython: cdivision=True
#cython: initializedcheck=False
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

"""Agenda-based parser which maintains the set of nodes which have been built.

This parser is very similar to change propagation parser except the values of
interest have been simplified to booleans.

Internals:

 - Edge node maintains a value which indicates if it has been built and a
   "witness" (much like backpointers).

 - If the witness disappears we have to back-chain the value (just like when the
   maximum edge gets deleted).

 - Can view this parser as an unweighted one -- we can even provide methods to
   extract a derivation.

"""

from ldp.parse.common cimport Cell, Cell_dt, Dt, D_t, V_t, \
    semizero, intpair

from ldp.cp.viterbi cimport DynamicParser as _DynamicParser
from cython.operator cimport dereference as deref
from libc.math cimport isnan


cdef class DynamicParser(_DynamicParser):

    cdef void initialize(self) nogil:
        cdef:
            int I, K, Y
            V_t s

        for I in range(self.N):
            Y = self.sentence[I]
            K = I + 1
            for ur in deref(self.grammar.preterm[Y]):
                s = 0.0
                X = ur.parent
                self._new_edge(I, K, X, Y=-1, Z=-1, J=K, s=s)

    cdef void bc(self, int I, int K, int X, int allow_unary, Cell& v) nogil:

        cdef:
            int J, Y, Z
            int narrowr, narrowl, maxmid, minmid

        v.score = semizero
        v.y = -1
        v.z = -1
        v.j = -1

        if not self.keep[I,K]:
            return

        for br in deref(self.grammar.r_x_yz[X]):
            Y = br.left
            Z = br.right

            # use an index to speed-up to search for midpoints.
            narrowr = self.minright[Y, I]
            narrowl = self.minleft[Z, K]
            if narrowr >= K or narrowl <= I:
                continue
            minmid = max(self.maxleft[Z, K], narrowr)
            maxmid = min(self.maxright[Y, I], narrowl)

            for J in range(minmid, maxmid+1):
                if not self.keep[I,J] or not self.keep[J,K]:
                    continue

                if self.chart[I,J,Y].score > semizero and self.chart[J,K,Z].score > semizero:
                    v.score = 0.0
                    v.y = Y
                    v.z = Z
                    v.j = J
                    return

        if allow_unary:
            for ur in deref(self.grammar.r_x_y[X]):
                Y = ur.child

                if self.chart[I,K,Y].score > semizero:
                    if self.chart[I,K,Y].z != -1:
                        # This ^ check only allows a single step of unary rewrites
                        # by checking that the node (I,K,Y) has a witness which is a
                        # binary edge.
                        v.score = 0.0
                        v.y = Y
                        v.z = -1
                        v.j = K
                        return


    cpdef run(self):
        self._run()

    cdef void _run(self) nogil:

        cdef:
            int I, J, K, X, Y, Z, B, E, S
            V_t value, neighbor
            Cell tmp

        while True:

            self.agenda.pop()

            if self.agenda.done:
                break

            B = self.agenda.begin
            E = self.agenda.end
            S = self.agenda.sym

            value = self.chart[B,E,S].score

            # One-step back chaining on pop; uses NaN is the distinguished UNK value.
            if isnan(value):
                self.bc(B, E, S, allow_unary=1, v=tmp)
                # uses a different "write_chart" method because the write
                # operation has to skip the agenda.
                if tmp.score <= semizero:
                    self._del_node(B,E,S)
                self.chart[B,E,S] = tmp
                value = tmp.score

            # left neighbors
            J = B; K = E; Z = S

            for P in deref(self.end[J]):
                I = P.first
                Y = P.second

                if not self.keep[I,K]:
                    continue

                if self._change_type == 0:
                    # Double-counting elimination: if neighbor is on the agenda
                    # don't combine now, will combine when it pops.
                    #
                    # Efficiency trick: it's faster to test widths than to set
                    # membership on the agenda. The reason this works is because
                    # we've assumed a (width,left-to-right)-ordering.
                    #
                    if K-J < J-I:
                        if self.agenda.wait(I,J,Y):
                            continue

                neighbor = self.chart[I,J,Y].score
                if isnan(neighbor):
                    continue

                self._combine_binary_boolean(neighbor,value,Y,Z,I,J,K)

            # right neighbors
            I = B; J = E; Y = S

            for P in deref(self.begin[J]):
                K = P.first
                Z = P.second

                if not self.keep[I,K]:
                    continue

                if self._change_type == 0:
                    if K-J >= J-I:
                        if self.agenda.wait(J,K,Z):
                            continue

                neighbor = self.chart[J,K,Z].score
                if isnan(neighbor):
                    continue
                self._combine_binary_boolean(value,neighbor,Y,Z,I,J,K)

            # unary rewrite
            I = B; K = E; Y = S
            if self.keep[I,K]:
                self._combine_unary_boolean(value, Y, I, K)

    cdef inline void _combine_unary_boolean(self, V_t value, int Y, int I, int K) nogil:
        for r in deref(self.grammar.r_y_x[Y]):
            self._new_edge(I, K, r.parent, Y, -1, K, value)

    cdef inline void _combine_binary_boolean(self, V_t left, V_t right, int Y, int Z, int I, int J, int K) nogil:
        for r in deref(self.grammar.r_yz_x[Y*self.grammar.nsymbols + Z]):
            self._new_edge(I, K, r.parent, Y, Z, J, min(left, right))
