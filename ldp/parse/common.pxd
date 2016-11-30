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

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from numpy cimport int_t, double_t, int16_t
from ldp.parse.grammar cimport Grammar
from cython.operator cimport dereference as deref

ctypedef int16_t    D_t
ctypedef double_t   V_t
ctypedef int_t      K_t

cdef V_t semizero

cdef object Cell_dt
cdef object Vt
cdef object Dt
cdef object Kt

ctypedef pair[int,int] intpair


# Cell
cdef packed struct Cell:
    V_t score
    D_t y
    D_t z
    D_t j


cdef inline int tri(int i, int j) nogil:
    return j*(j-1)/2 + i

cdef list backtrace(V_t[:,:,:] chart, Grammar grammar, D_t I, D_t K, D_t X, long[:] sentence)

cdef list backtrace_tri(V_t[:,:] chart, Grammar grammar, int I, int K, int X, long[:] sentence)

cdef list follow_backpointers(Cell[:,:,:] chart, D_t X, D_t I, D_t K, long[:] sentence)
