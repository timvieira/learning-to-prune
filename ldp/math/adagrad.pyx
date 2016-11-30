#!/usr/bin/env python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: cdivision=True
#cython: infertypes=True
#distutils: language = c++
#distutils: libraries = ['stdc++']
#distutils: extra_compile_args = -Wno-unused-function -Wno-unneeded-internal-declaration

import numpy as np
from libc.math cimport sqrt

cdef inline double sign(double x) nogil:
    return 1 if x >= 0 else -1

cdef inline double abs(double x) nogil:
    return x if x >= 0 else -x


cdef class LazyRegularizedAdagrad:

    def __init__(self, int d, int L, double C, double eta = 0.1, double fudge = 1e-4):
        self.L = L
        self.d = d
        self.fudge = fudge
        self.u = np.zeros(d, dtype=np.int32)
        self.q = np.zeros(d, dtype=np.double) + fudge
        self.w = np.zeros(d, dtype=np.double)
        self.C = C
        self.eta = eta
        self.step = 1

    def reset(self):
        """ reset the AdaGrad values """
        self.u = np.zeros(self.d, dtype=np.int32)
        self.q = np.zeros(self.d, dtype=np.double) + self.fudge

    def _catchup(self, int k):
        self.catchup(k)

    def _update_active(self, int k, double g):
        self.update_active(k, g)

    def finalize(self):
        for i in range(self.d):
            self.catchup(i)
        return np.asarray(self.w)

    cdef inline double catchup(self, int k) nogil:
        "Lazy L1/L2-regularized adagrad catchup operation."
        cdef int dt
        cdef double sq
        dt = self.step - self.u[k]
        # shortcircuit when weights are up-to-date
        if dt == 0:
            return self.w[k]
        sq = sqrt(self.q[k])
        if self.L == 2:
            # Lazy L2 regularization
            self.w[k] *= (sq / (self.eta * self.C + sq)) ** dt
        elif self.L == 1:
            # Lazy L1 regularization
            self.w[k] = sign(self.w[k]) * max(0, abs(self.w[k]) - self.eta * self.C * dt / sq)
        # update last seen
        self.u[k] = self.step
        return self.w[k]

    cdef inline void update_active(self, int k, double g) nogil:
        cdef double d, z, sq
        self.q[k] += g**2
        sq = sqrt(self.q[k])
        if self.L == 2:
            self.w[k] = (self.w[k]*sq - self.eta*g)/(self.eta*self.C + sq)
        elif self.L == 1:
            z = self.w[k] - self.eta*g/sq
            d = abs(z) - self.eta*self.C/sq
            self.w[k] = sign(z) * max(0, d)
        else:
            self.w[k] -= self.eta*g/sq
        self.u[k] = self.step+1


def test():
    """
    Integration test for Lazily regularized adagrad.
    """
    import numpy as np
    from numpy import sqrt, sign, zeros

    class EagerL1Weights(object):

        def __init__(self, D, C, a, fudge):
            self.w = zeros(D)
            self.g2 = zeros(D) + fudge
            self.C = C
            self.a = a

        def update(self, g):
            # dense weight update
            self.g2 += g**2
            z = self.w - self.a * g / sqrt(self.g2)
            d = np.abs(z) - self.a*self.C / sqrt(self.g2)
            d[d <= 0] = 0  # d = max(0, d)
            self.w = sign(z) * d

    T = 50  # number of iterations
    D = 6   # number of features
    K = 3   # number of active features

    C = .8        # regularization constant
    eta = .3      # stepsize
    fudge = 1e-4  # adagrad fudge factor

    lazy = LazyRegularizedAdagrad(D, L=1, C=C, eta=eta, fudge=fudge)
    eager = EagerL1Weights(D, C=C, a=eta, fudge=fudge)

    for _ in range(T):

        keys = range(D)
        np.random.shuffle(keys)
        keys = keys[:K]

        # dense vector.
        dense = np.zeros(D)
        dense[keys] = 1
        eager.update(dense)

        for k in keys:
            lazy._catchup(k)
            lazy._update_active(k, 1)

        lazy.step += 1

    print
    print 'step=', lazy.step
    w = np.asarray(lazy.finalize())
    print w
    print eager.w
    assert (np.abs(w-eager.w) < 1e-8).all()


if __name__ == '__main__':
    test()
