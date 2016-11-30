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

cdef class LazyRegularizedAdagrad:

    cdef public double[:] w   # weight vector
    cdef public double[:] q   # sum of squared weights
    cdef public double eta    # learning rate (assumed constant)
    cdef public double C      # regularization constant
    cdef int[:] u             # time of last update
    cdef int L                # regularizer type in {1,2}
    cdef int d                # dimensionality
    cdef double fudge         # adagrad fudge factor paramter
    cdef public int step      # time step of the optimization algorithm (caller is
                              # responsible for incrementing)

    cdef inline double catchup(self, int k) nogil
    cdef inline void update_active(self, int k, double g) nogil
