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

"""Agenda-based parser which supports change propagation to pruning
masks. Unlike a 'vanilla' agenda-based parser this one allows an item's maximum
value to decrease as a result of a change.

"""

import numpy as np
from numpy import empty

from ldp.parse.common cimport follow_backpointers, \
    Dt, D_t, V_t, K_t, semizero, intpair, Cell, Cell_dt

from ldp.parse.grammar cimport Grammar

from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free
from libcpp.utility cimport pair

from libc.stdio cimport printf
from libc.math cimport isnan

from ldp.parse.grammar cimport Grammar
from ldp.parse.common cimport intpair
from ldp.cp.agenda cimport Agenda2 as Agenda
#from ldp.cp.agenda cimport Agenda

from libcpp.pair cimport pair as pair
from libcpp.vector cimport vector

#ctypedef vector[triple] friends


cdef V_t nan = np.nan


cdef class DynamicParser:

    # NOTE: derived classes don't seem to inherit __cinit__, so we use __init__
    # instead.
    def __init__(self, long[:] sentence, Grammar grammar, K_t[:,:] keep):
        cdef:
            int I, K, X, N, NT

        self._pops = 0
        self.sentence = sentence
        self.grammar = grammar
        self.keep = keep
        self.recording = 0

        N = sentence.shape[0]
        NT = grammar.nsymbols
        self.N = N
        self.NT = NT

        self.chart = empty((N,N+1,NT), dtype=Cell_dt)

        self.agenda = Agenda(N, NT)

        x = empty((NT, N+1), dtype=Dt)
        x.fill(-1)
        self.minleft = x

        x = empty((NT, N+1), dtype=Dt)
        x.fill(-1)
        self.maxright = x

        x = empty((NT, N+1), dtype=Dt)
        x.fill(N+1)
        self.minright = x

        x = empty((NT, N+1), dtype=Dt)
        x.fill(N+1)
        self.maxleft = x

        with nogil:
            # indexes
            self.begin = <pairset**> malloc((N+1) * sizeof(pairset*))
            self.end = <pairset**> malloc((N+1) * sizeof(pairset*))
            for I in range(N+1):
                self.begin[I] = new pairset(100)
                self.end[I] = new pairset(100)

            self._change_type = 0

            for I in range(N):
                for K in range(N+1):
                    for X in range(NT):
                        self.chart[I,K,X].score = semizero
                        self.chart[I,K,X].y = -1
                        self.chart[I,K,X].z = -1
                        self.chart[I,K,X].j = -1

            self.initialize()

    def __dealloc__(self):
        cdef:
            int i, k
        for i in range(self.N+1):
            #free(self.begin[i])
            #free(self.end[i])
            del self.end[i]
            del self.begin[i]
        free(self.begin)
        free(self.end)

    cpdef V_t score(self, int I, int K, int X):
        return self.chart[I, K, X].score

    cdef void initialize(self) nogil:
        cdef:
            int I, K, Y
            V_t s
        for I in range(self.N):
            Y = self.sentence[I]
            K = I + 1
            for ur in deref(self.grammar.preterm[Y]):
                s = ur.weight
                X = ur.parent
                self._new_edge(I, K, X, Y=-1, Z=-1, J=K, s=s)

    cpdef change(self, int I, int K, K_t now):
        assert 0 <= I < K <= self.N, 'Changeprop: Indices (%s,%s) out of bounds. Sentence length = %s.' % (I,K,self.N)
        assert now == 0 or now == 1
        self._change(I,K,now)
        self.run()

    cpdef change_many(self, list changes):
        for (I,K,now) in changes:
            self._change(I,K,now)
        self.run()

    cdef inline void _change(self, int I, int K, K_t a) nogil:

        cdef:
            Cell v

        if self.keep[I,K] == a:
            return

        self._change_type = -1 if a == 0 else +1

        if self.recording:
            # TODO: should check that we haven't altered this key already.
            self.undo_keep.push_back(triple(self.keep[I,K], intpair(I,K)))

        self.keep[I,K] = a

        for X in range(self.NT):
            self.bc(I, K, X, allow_unary=0, v=v)
            self._write_chart(I, K, X, v)

    cdef inline void _write_chart(self, int I, int K, int X, Cell& new) nogil:

        cdef:
            Cell old

        old = self.chart[I,K,X]

        if (old.score == new.score
            and old.j == new.j
            and old.y == new.y
            and old.z == new.z):

            return

        if (new.score != self.chart[I,K,X].score and not (new.score <= semizero and old.score <= semizero)):

            # Only need to put on agenda if the value changed. Otherwise, we
            # still need to record it because the backpointer must've changed.
            self.agenda.set(I,K,X)

            if old.score <= semizero and new.score > semizero:
                self._new_node(I,K,X)

            if old.score > semizero and new.score <= semizero:
                self._del_node(I,K,X)

            # TODO: should this be unindented?
            if self.recording:
                self.record_undo(I,K,X)

        self.chart[I,K,X] = new

#        self.obligation(I, new.j, K, X, new.y, new.z)

#    cdef void obligation(self, int I, int J, int K, int X, int Y, int Z) nogil:
#        cdef triple x,y,z
#
#        x.first = I
#        x.second.first = K
#        x.second.second = X
#
#        y.first = I
#        y.second.first = J
#        y.second.second = Y
#        self.oblig[y].push_back(x)
#
#        z.first = J
#        z.second.first = K
#        z.second.second = Z
#        self.oblig[z].push_back(x)

    cpdef start_undo(self):
        self._start_undo()

    cdef void _start_undo(self) nogil:
        self.undo_keep.clear()
        self.U.clear()
        self.recording = 1

    cpdef rewind(self):
        self._rewind()

    cdef inline void _rewind(self) nogil:
        cdef:
            int I,K,X,a,i
            triple key
            Cell val

        if self.recording:
            for i in reversed(range(self.U.size())):
                keyval = self.U[i]
                key = keyval.first
                new = keyval.second
                I = key.first
                K = key.second.first
                X = key.second.second
                if self.chart[I,K,X].score <= semizero and new.score > semizero:
                    self._new_node(I,K,X)
                if self.chart[I,K,X].score > semizero and new.score <= semizero:
                    self._del_node(I,K,X)
                self.chart[I,K,X] = new

        if self.recording:
            for i in reversed(range(self.undo_keep.size())):
                key = self.undo_keep[i]
                I = key.second.first
                K = key.second.second
                self.keep[I,K] = key.first

        # stop recording after the rewind and clear datastructures.
        self.recording = 0
        self.undo_keep.clear()
        self.U.clear()

    cdef inline void _new_node(self, int I, int K, int X) nogil:
        cdef:
            intpair x

        self._pops += 1

        x.first = K
        x.second = X
        self.begin[I].insert(x)

        x.first = I
        x.second = X
        self.end[K].insert(x)

        # update midpoint filter
        if I > self.minleft[X, K]:
#            if self.undo is not None:
#                k = (X,K)
#                u = self.undo_minleft
#                if k not in u:
#                    u[k] = v
            self.minleft[X, K] = I

        if I < self.maxleft[X, K]:
#            if self.undo is not None:
#                k = (X,K)
#                u = self.undo_maxleft
#                if k not in u:
#                    u[k] = v
            self.maxleft[X, K] = I

        if K < self.minright[X, I]:
#            if self.undo is not None:
#                k = (X,I)
#                u = self.undo_minright
#                if k not in u:
#                    u[k] = v
            self.minright[X, I] = K

        if K > self.maxright[X, I]:
#            if self.undo is not None:
#                k = (X,I)
#                u = self.undo_maxright
#                if k not in u:
#                    u[k] = v
            self.maxright[X, I] = K

    cdef inline void _del_node(self, int I, int K, int X) nogil:
        cdef:
            intpair x
        self._pops -= 1

        x.first = K
        x.second = X
        self.begin[I].erase(x)

        x.first = I
        x.second = X
        self.end[K].erase(x)

    cdef void bc(self, int I, int K, int X, int allow_unary, Cell& v) nogil:

        cdef:
            V_t r, s
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
            r = br.weight

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
                s = self.chart[I,J,Y].score + self.chart[J,K,Z].score + r
                if s > v.score:
                    v.score = s
                    v.y = Y
                    v.z = Z
                    v.j = J

        if allow_unary:
            for ur in deref(self.grammar.r_x_y[X]):
                Y = ur.child
                r = ur.weight
                s = self.chart[I,K,Y].score + r
                if s > v.score:
                    if self.chart[I,K,Y].z != -1:
                        # This ^ check only allows a single step of unary
                        # rewrites by checking that the node (I,K,Y) has a
                        # witness which is a binary edge.
                        v.score = s
                        v.y = Y
                        v.z = -1
                        v.j = K

        return

    cdef inline void _new_edge(self, int I, int K, int X, int Y, int Z, int J, V_t s) nogil:
        cdef:
            V_t was
            Cell v
            triple undokey

        was = self.chart[I,K,X].score

        if not self.keep[I,K]:
            s = semizero

        if s == was or (s <= semizero and was <= semizero):
            pass

        elif s > was:
            v.score = s
            v.y = Y
            v.z = Z
            v.j = J
            self._write_chart(I,K,X,v)

        else:

            # edge score decreased.
            if self.chart[I,K,X].y == Y and self.chart[I,K,X].z == Z and self.chart[I,K,X].j == J:
                # Edge score decreased and this new edge was the maximal hyperedge.

                # Push-time update: in this case, the score of the maximal edge
                # decreased, we need to recompute consequent (because max does not
                # have a subtraction operator).
                #
                # TODO: Should we be lazier about calling bc? we might call it
                # twice for a given item. Can we do it at pop time?
                #
#                self.bc(I, K, X, allow_unary=1, v=v)
#                self._write_chart(I,K,X,v)

                # TODO: use a real triple type instead of a nested pair.
                if self.recording:
                    self.record_undo(I,K,X)

                # lazier version. Zero-out here and at pop time, run `bc`.
                v.score = nan #self.chart[I,K,X].score
                v.y = -1
                v.z = -1
                v.j = -1
                #self._write_chart(I,K,X,v)

#                self._del_node(I,K,X)
                self.chart[I,K,X] = v
                self.agenda.set(I,K,X)    # put notiication on agenda, don't write the chart or update indexes.

    cdef void record_undo(self, int I, int K, int X) nogil:
        cdef:
            triple undo_key
            UndoKeyVal undo_key_val
        if self.recording:
            undo_key.first = I
            undo_key.second.first = K
            undo_key.second.second = X
            undo_key_val.first = undo_key
            undo_key_val.second = self.chart[I,K,X]
            self.U.push_back(undo_key_val)

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

                # [2016-04-26 Tue] we only do this check in the initial
                # roll-in. For bit flips there can never be anything wider on
                # the agenda due ot linearity of the pruning architecture. Note:
                # the general case will have this.
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

                # Notes:
                #
                #  - We do not need a keep[I,J] check because the begin/end
                #    index is tight and eagerly maintained.
                #
                #  - Moved keep[I,K] check earlier to avoid more calls to
                #    agenda.wait.
                #
                # Older version:
                #
                #if not self.keep[I,K]: # or not self.keep[I,J]:
                #    continue

                neighbor = self.chart[I,J,Y].score
                if isnan(neighbor):
                    continue

                self._combine_binary(neighbor,value,Y,Z,I,J,K)

            # right neighbors
            I = B; J = E; Y = S

            for P in deref(self.begin[J]):
                K = P.first
                Z = P.second

                if not self.keep[I,K]:
                    continue

                if self._change_type == 0:
                    # double-counting elimination (comments in left neighbor loop
                    # apply here as well).
                    if K-J >= J-I:     # tested '>=' by trial-and-error and theory.
                        if self.agenda.wait(J,K,Z):
                            continue

                # see notes on left neighbor loop.
                #if not self.keep[I,K]: # or not self.keep[J,K]:
                #    continue

                neighbor = self.chart[J,K,Z].score
                if isnan(neighbor):
                    continue
                self._combine_binary(value,neighbor,Y,Z,I,J,K)

            # unary rewrite
            I = B; K = E; Y = S
            if self.keep[I,K]:
                self._combine_unary(value, Y, I, K)

    cdef inline void _combine_unary(self, V_t value, int Y, int I, int K) nogil:
        for r in deref(self.grammar.r_y_x[Y]):
            self._new_edge(I, K, r.parent, Y, -1, K, value + r.weight)

    cdef inline void _combine_binary(self, V_t left, V_t right, int Y, int Z, int I, int J, int K) nogil:
        for r in deref(self.grammar.r_yz_x[Y*self.NT + Z]):
            self._new_edge(I, K, r.parent, Y, Z, J, left + right + r.weight)

    cpdef object state(self):
        cdef int root = self.grammar.root
        return ParserState(self.chart[0, self.N, root].score,
                           self.derivation(),
                           self._pops,
                           self._pushes,
                           self.chart)

    cpdef list derivation(self):
        """Extract most-likely derivation from chart."""
        return follow_backpointers(self.chart, self.grammar.root, 0, self.N, self.sentence)

    def check_index_support(self, int verbose):
        """Check that that all items with support appear in `begin` and `end` and vice
        versa.

        """
        cdef:
            int I, K, X, N, NT, s, b, e
        N = self.chart.shape[0]
        NT = self.chart.shape[2]
        for I in range(N):
            for K in range(I+1,N+1):
                for X in range(NT):
                    s = self.chart[I,K,X].score > semizero
                    b = self.begin[I].count(intpair(K,X))
                    e = self.end[K].count(intpair(I,X))

                    if s:
                        if not (b and e):
                            print 'bad index should support item/supported/begin/end/score:', ([I,K,X], s, b, e, self.chart[I,K,X].score)
                    else:
                        if not (not b and not e):
                           print 'bad index should NOT support item/supported/begin/end/score:', ([I,K,X], s, b, e, self.chart[I,K,X].score)

                    if s != b or s != e:
                        print '[index support] fail', [s,b,e,(I,K,X)]
        if verbose:
            print 'check index support:', 'ok'



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
