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

from libc.math cimport exp, log, log1p


cpdef double log1pexp(double)
cpdef double log1mexp(double)
cpdef double logadd(double, double)
cpdef double sigmoid(double)
cpdef double sign(double)
cpdef double dsigmoid(double)
cpdef double[:] exp_normalize(double[:])
cpdef exp_normalize_inplace(double[:])
cpdef int binary_search(double[:], double)


cdef inline double _logsigmoid(double x) nogil:
    """
    log(sigmoid(x)) = -log(1+exp(-x)) = -log1pexp(-x)
    """
    return -_log1pexp(-x)


cdef inline double _log1pexp(double x) nogil:
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).

    -log1pexp(-x) is log(sigmoid(x))

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= 18:
        return log1p(exp(x))
    elif 18 < x <= 33.3:
        return x + exp(-x)
    else:
        return x


cdef inline double _sigmoid(double x) nogil:
    """
    Numerically-stable sigmoid function.
    """
    cdef double z
    if x >= 0:
        z = exp(-x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = exp(x)
        return z / (1 + z)
