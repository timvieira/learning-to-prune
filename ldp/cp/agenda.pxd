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

import numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from ldp.parse.containers cimport unordered_set

ctypedef unordered_set[int] intset


cdef class Agenda:
    """Specialized agenda which does approximately CKY ordering.

    It is designed for efficiency and should be used with caution!

    If an element is pushed which has a smaller width than the current width bad
    things will probably happen.

    """
    cdef vector[intset] q
    cdef int N, NT, begin, end, sym, done
    cdef inline void set(self, int I, int K, int X) nogil
    cdef inline int wait(self, int I, int K, int X) nogil
    cdef inline void pop(self) nogil


#-------------
# IntSet hack

cdef struct IntSet:
    vector[int]* _elements  # unsorted collection of elements
    int* _contains

cdef inline int intset_empty(IntSet self) nogil:
    return self._elements.empty()

cdef inline void intset_add(IntSet self, int x) nogil:
    if not self._contains[x]:
        self._elements.push_back(x)
        self._contains[x] = 1

cdef inline int intset_contains(IntSet self, int x) nogil:
    return self._contains[x]

cdef inline int intset_pop(IntSet self) nogil:
    cdef int x
    x = self._elements.back()
    self._elements.pop_back()
    self._contains[x] = 0
    return x

cdef class Agenda2:
    """Specialized agenda which does approximately CKY ordering.

    It is designed for efficiency and should be used with caution!

    If an element is pushed which has a smaller width than the current width bad
    things will probably happen.

    """
    cdef vector[IntSet] q
    cdef int N, NT, begin, end, sym, done
    cdef inline void set(self, int I, int K, int X) nogil
    cdef inline int wait(self, int I, int K, int X) nogil
    cdef inline void pop(self) nogil
#-------------
