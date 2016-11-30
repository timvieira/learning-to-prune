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

from numpy cimport ndarray
from libc.math cimport exp, log, log1p, sqrt, abs, expm1

import numpy as np
from numpy import array, empty
from scipy.linalg import norm
from scipy.sparse import dok_matrix


cdef double ninf = float('-infinity')


cpdef double sign(double x):
    if x == 0:
        return 0.0
    elif x < 0:
        return -1.0
    else:
        return 1.0


cdef double sigmoid_2 = sigmoid(2) # 0.8807977077977882

cpdef int sample_sigmoid(double a, double r):
    if a < 0:
        a = -a
        r = 1-r
        if a >= 2:
            if r <= sigmoid_2 or r <= sigmoid(a):
                return 0
            else:
                return 1
        else:
            if r > 0.5 + a/4.0:
                return 1
            elif r <= 0.5 + a/5.252141128658 or r <= sigmoid(a):
                return 0
            else:
                return 1

    if a >= 2:
        if r <= sigmoid_2 or r <= sigmoid(a):
            return 1
        else:
            return 0
    else:
        if r > 0.5 + a/4.0:
            return 0
        elif r <= 0.5 + a/5.252141128658 or r <= sigmoid(a):
            return 1
        else:
            return 0


def cosine(x, y):
    "Cosine similarity between two vectors."
    n = x.dot(y)
    nx = norm(x)
    ny = norm(y)

    if nx == 0.0 or ny == 0.0:
        # if one of the vectors is zero. Division results in nan, but if the
        # other vector is zero also, the the score should be 1.0.
        if nx == 0.0 and ny == 0.0:
            return 1.0
        else:
            return 0.0

    return n / nx / ny


cdef class RunningVariance(object):

    cdef readonly double[:] mean
    cdef readonly double[:] square
    cdef readonly int n

    def __cinit__(self, shape):
        self.mean = np.zeros(shape, dtype=np.double)
        self.square = np.zeros(shape, dtype=np.double)
        self.n = 0

    def update(self, double[:] x):
        cdef:
            int i
            double[:] m, s
        m = self.mean
        s = self.square
        for i in range(x.shape[0]):
            m[i] += x[i]
            s[i] += x[i]*x[i]
        self.n += 1

    def value(self):
        n = self.n
        m = np.asarray(self.mean) / n
        s = np.asarray(self.square) / n
        return (s - m*m) * n / (n-1)


cpdef int binary_search(double[:] A, double x):
    cdef:
        int a, b, m
    a = 0
    b = A.shape[0] - 1
    m = 0
    while True:
        if b < a:
            return b+1
        m = (a + b) // 2
        if A[m] < x:
            a = m + 1
        elif A[m] > x:
            b = m - 1
        else:
            return m


cpdef double logsigmoid(double x):
    """
    log(sigmoid(x)) = -log(1+exp(-x)) = -log1pexp(-x)
    """
    return -log1pexp(-x)


cpdef double log1pexp(double x):
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).

    -log1pexp(-x) is log(sigmoid(x))

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37:
        return exp(x)
    elif -37 <= x <= 18:
        return log1p(exp(x))
    elif 18 < x <= 33.3:
        return x + exp(-x)
    else:
        return x


cpdef double log1mexp(double x):
    """
    Numerically stable implementation of log(1-exp(x))

    Note: function is finite for x < 0.

    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    cdef:
        double a
    if x >= 0:
        return np.nan
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return log(-expm1(-a))
        else:
            return log1p(-exp(-a))


cpdef double logadd(double x, double y):
    cdef double d, r
    # implementation: need separate checks for inf because inf-inf=nan.
    if x == ninf:
        return y
    elif y == ninf:
        return x
    else:
        if y <= x:
            d = y-x
            r = x
        else:
            d = x-y
            r = y
        return r + log1pexp(d)


cpdef double sigmoid(double x):
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


cpdef double dsigmoid(double x):
    """
    Numerically-stable derivative of the sigmoid function.
    """
    cdef double z
    if x >= 0:
        z = exp(-x)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = exp(x)
    return z / (1 + z)**2


cpdef double[:] exp_normalize(double[:] x):
    """
    >>> x = [1, -10, 100, .5]
    >>> exp_normalize(x)
    array([  1.01122149e-43,   1.68891188e-48,   1.00000000e+00,
             6.13336839e-44])
    >>> exp(x) / exp(x).sum()
    array([  1.01122149e-43,   1.68891188e-48,   1.00000000e+00,
             6.13336839e-44])
    """
    cdef:
        int i, n
        double b, z
        double[:] y

    n = x.shape[0]

    y = array(x)

    b = y[0]
    for i in range(1,n):
        if y[i] > b:
            b = y[i]

    for i in range(n):
        y[i] -= b

    z = 0.0
    for i in range(n):
        y[i] = exp(y[i])
        z += y[i]

    for i in range(n):
        y[i] /= z

    return y


cpdef tuple exp_normalize_with_lnz(double[:] x):
    cdef:
        int i, n
        double b, z
        double[:] y

    n = x.shape[0]

    y = array(x)

    b = y[0]
    for i in range(1,n):
        if y[i] > b:
            b = y[i]

    for i in range(n):
        y[i] -= b

    z = 0.0
    for i in range(n):
        y[i] = exp(y[i])
        z += y[i]

    for i in range(n):
        y[i] /= z

    return y, log(z) + b


cpdef exp_normalize_inplace(double[:] y):
    """
    >>> x = [1, -10, 100, .5]
    >>> exp_normalize(x)
    array([  1.01122149e-43,   1.68891188e-48,   1.00000000e+00,
             6.13336839e-44])
    >>> exp(x) / exp(x).sum()
    array([  1.01122149e-43,   1.68891188e-48,   1.00000000e+00,
             6.13336839e-44])
    """
    cdef:
        int i, n
        double b, z

    n = y.shape[0]

    b = y[0]
    for i in range(1,n):
        if y[i] > b:
            b = y[i]

    for i in range(n):
        y[i] -= b

    z = 0.0
    for i in range(n):
        y[i] = exp(y[i])
        z += y[i]

    for i in range(n):
        y[i] /= z


# XXX: untested
def interpolate_inplace(double[:] x, double[:] y, double alpha):
    """
    Compute:

      x = (1-a)*x + a*y

      x += alpha*(y - x)

    """
    cdef:
        int i, n

    assert x.shape[0] == y.shape[0], 'must be same size.'
    n = x.shape[0]

    for i in range(n):
        x[i] += alpha*(y[i] - x[i])


def ridge(A, rs, shrinkage, fit_intercept, **params):
    "Perform ridge regression. Utility method exists becase pylint is a pain."
    from sklearn.linear_model import Ridge
    clf = Ridge(alpha=shrinkage, copy_X=False, fit_intercept=fit_intercept,
                tol=0.001, **params)
    clf.fit(A, rs)
    return clf


def tosparse(gs, D):
    """
    Convert a list of ldp.math.sparse.SparseVectors into a scipy.sparse.CSR matrix.

    >>> import numpy as np
    >>> from ldp.math.sparse import SparseVector, SparseVectorHash
    >>> tosparse([SparseVector([5,6], [1,1]),
    ...           SparseVectorHash(np.array([1,6], dtype=np.int), np.array([2.0,2.0]))], 10).todense()
    matrix([[ 0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.],
            [ 0.,  2.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.]])
    """
    A = dok_matrix((len(gs), D))
    for j, g in enumerate(gs):
        try:
            keys, vals = g.to_arrays()
        except AttributeError:
            keys, vals = g.keys, g.vals
        for i in xrange(g.length):
            k = keys[i]
            v = vals[i]
            A[j, k] += v
    return A.tocsr()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
