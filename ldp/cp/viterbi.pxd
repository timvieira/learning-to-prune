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

"""Agenda-based parser which supports change propagation to pruning
masks. Unlike a 'vanilla' agenda-based parser this one allows an item's maximum
value to decrease as a result of a change.

"""

from ldp.parse.common cimport follow_backpointers, \
    Cell, Cell_dt, Dt, D_t, V_t, K_t, semizero

from ldp.parse.grammar cimport Grammar
from ldp.parse.common cimport intpair
from ldp.cp.agenda cimport Agenda2 as Agenda
#from ldp.cp.agenda cimport Agenda

from ldp.parse.containers cimport unordered_set
from libcpp.pair cimport pair as pair
from libcpp.vector cimport vector

ctypedef unordered_set[intpair] pairset
ctypedef pair[int,intpair] triple
ctypedef pair[triple,Cell] UndoKeyVal


cdef class DynamicParser:

    cdef D_t[:,:] minleft, maxleft, minright, maxright

    cdef readonly Cell[:,:,:] chart
    cdef pairset** begin
    cdef pairset** end
    cdef Agenda agenda
    cdef readonly K_t[:,:] keep
    cdef long[:] sentence
    cdef Grammar grammar
    cdef readonly int N, NT, _pops
    cdef readonly int _pushes

    cdef vector[triple] undo_keep
    cdef vector[UndoKeyVal] U
    cdef int recording, _change_type

    cdef void initialize(self) nogil
    cdef inline void _change(self, int I, int K, K_t a) nogil
    cdef inline void _write_chart(self, int I, int K, int X, Cell& new) nogil
    cdef inline void _new_node(self, int I, int K, int X) nogil
    cdef inline void _del_node(self, int I, int K, int X) nogil
    cdef inline void _new_edge(self, int I, int K, int X, int Y, int Z, int J, V_t s) nogil
    cdef inline void _combine_unary(self, V_t value, int Y, int I, int K) nogil
    cdef inline void _combine_binary(self, V_t left, V_t right, int Y, int Z, int I, int J, int K) nogil
    cdef void bc(self, int I, int K, int X, int allow_unary, Cell& v) nogil
    cdef inline void _rewind(self) nogil
    cdef void _start_undo(self) nogil
    cdef void _run(self) nogil
    cdef void record_undo(self, int I, int K, int X) nogil

    cpdef rewind(self)
    cpdef run(self)

    cpdef change(self, int I, int K, K_t now)
    cpdef change_many(self, list changes)
    cpdef start_undo(self)
    cpdef object state(self)
    cpdef list derivation(self)
    cpdef V_t score(self, int I, int K, int X)
