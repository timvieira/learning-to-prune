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

cdef class Example:

    cdef public:
        object name
        object sentence
        long[:] tokens
        int N
        object[:,:] features
        object baseline, oracle
        frozenset gold_items, gold_spans, mle_spans, evalb_items
        object gold_unbinarized, gold_binarized
        double[:,:,:] Q
