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
Feature extraction for pruning spans.
"""
import numpy as np
from numpy cimport ndarray
from ldp.parse.grammar cimport Grammar
from numpy cimport uint64_t, uint32_t, int32_t

cdef uint32_t hashit(uint64_t key, unsigned int seed)
cpdef uint32_t hash_bytes(bytes key, unsigned int seed)
cdef inline uint64_t pack(uint64_t template, uint64_t a, uint64_t b)


cdef class Features(object):
    cdef:
        Grammar grammar
        int nfeatures
        dict LFS
        int BOS, EOS, seed
        
    cdef inline int32_t hashpack(self, int a, int b, int c)

        
