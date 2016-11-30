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

from numpy cimport int_t, double_t
from libcpp.map cimport map as cpp_map

ctypedef int_t D_t
ctypedef double_t V_t
ctypedef cpp_map[D_t, V_t] Map
ctypedef cpp_map[D_t, V_t].iterator Map_it

cdef class SparseVector:
    cdef readonly D_t[:] keys
    cdef readonly V_t[:] vals
    cdef readonly int length
    cpdef double dot(self, V_t[:] w)
    cdef double _dot(self, V_t[:] w) nogil
    cpdef pluseq(self, V_t[:] w, double coeff)
    cdef void _pluseq(self, V_t[:] w, double coeff) nogil

cdef class SparseBinaryVector:
    cdef D_t* keys
    cdef readonly int length
    cpdef double dot(self, V_t[:] w)
    cdef double _dot(self, V_t[:] w) nogil
    cpdef pluseq(self, V_t[:] w, double coeff)
    cdef void _pluseq(self, V_t[:] w, double coeff) nogil

cdef class SparseVectorHash:
    cdef Map _map
    cdef _to_arrays(self, D_t[:] keys, V_t[:] values)
    cpdef double dot(self, V_t[:] w)
    cdef double _dot(self, V_t[:] w) nogil
    cpdef pluseq(self, V_t[:] w, double coeff)
    cdef void _pluseq(self, V_t[:] w, double coeff) nogil
