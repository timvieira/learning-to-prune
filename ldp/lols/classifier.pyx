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
from ldp.math.sparse cimport SparseBinaryVector, V_t
from libc.math cimport exp, log
from ldp.math.util cimport _sigmoid, _logsigmoid
from numpy cimport ndarray

from ldp.math.adagrad cimport LazyRegularizedAdagrad
from ldp.prune.example cimport Example

from numpy import zeros
from numpy.random import shuffle

from scipy.optimize import fmin_l_bfgs_b

from arsenal.timer import timeit

from scipy.sparse import dok_matrix
from sklearn.svm import SVC


def glm(ndarray[double,ndim=1] theta,
        list data,
        double regularizer,
        int loss,
        int with_gradient=1):
    """
    Squared-loss muliclass classification -- argmax regression.
    """
    cdef:
        ndarray[double, ndim=1] grad
        double func
        double[:] reward
        SparseBinaryVector f
        int N, I, K
        double W, w, z, y, v
        Example e

    grad = None
    if with_gradient:
        grad = theta*0.0
    func = 0.0
    W = 0
    for e in data:
        N = e.N
        for I in xrange(N):
            for K in xrange(I+1,N+1):
                if K-I > 1 and K-I != N:
                    w = abs(e.Q[I,K,0] - e.Q[I,K,1])
                    if w == 0:
                        continue
                    f = e.features[I,K]
                    W += w
                    y = 1 if e.Q[I,K,0] < e.Q[I,K,1] else -1
                    z = f._dot(theta)
                    if loss == 0:
                        func -= w*_logsigmoid(y*z)
                        v = -y*_sigmoid(-y*z)*w
                    else:
                        func += 0.5*w*(z - y)**2
                        v = w*(z - y)
                    if with_gradient:
                        f._pluseq(grad, v)
    if W == 0:
        W = 1
    func /= W
    func += 0.5 * regularizer * theta.dot(theta)
    if with_gradient:
        grad /= W
        grad += regularizer*theta   # XXX: use a loop?
        return func, grad
    return func


cdef class Classifier:

    cdef public:
        V_t[:] _coef
        V_t _intercept
        int D

    def __init__(self, int nfeatures):
        self.D = nfeatures
        self._coef = np.zeros(self.D)
        self._intercept = 0.0

    def save(self, filename):
        np.savez(filename, coef=self._coef, intercept=self._intercept, D=self.D)

    def load(self, filename):
        f = dict(np.load(filename).iteritems())
        self._coef = f['coef']
        self._intercept = float(f['intercept'])
        self.D = int(f['D'])

    def __call__(self, Example example, long[:,:] m, tuple x, SparseBinaryVector f):
        return f._dot(self._coef) + self._intercept >= 0

    @staticmethod
    def csc(theta, b, data, normalizer=''):
        "Cost-sensitive classification objective for a linear model."
        assert False, 'broken.'
        R = []
        W = 0.0
        for (e,x),rs in data.items():
            a = rs.argmax()
            w = rs.ptp()
            W += w
            p = (e.features[x].dot(theta) + b) >= 0
            R.append(0.0 if p == a else w)
        R = np.array(R)
        if normalizer == '':
            pass
        elif normalizer == 'W':
            R /= W
        elif normalizer == 'N':
            R /= len(data)
        else:
            raise ValueError('Unrecognized normalization option %r' % normalizer)
        return np.mean(R)

    def train(self, list data, **kwargs):
        print '#csc-examples %s' % sum(len(e.nodes) for e in data)
        self._coef, self._intercept = self._train(data, **kwargs)

    def _train(self, data, **kw):
        raise NotImplementedError('_train')


cdef class GLM(Classifier):

    cdef double C
    cdef int loss

    def __init__(self, int D, double C, int loss):
        Classifier.__init__(self, D)
        self.C = C
        self.loss = loss

    def _train(self, data):
        regularizer = np.exp(self.C)
        def func(x):
            return glm(x, data=data, regularizer=regularizer, loss=self.loss)
        [w, value, info] = fmin_l_bfgs_b(func, self._coef, disp=0)
        del info['grad']
        print '%g %s' % (value, info)
        return w, 0.0


class SVM(Classifier):

    def __init__(self, int D, double C):
        Classifier.__init__(self, D)
        self.C = C

    def _train(self, data):
        n = sum(len(e.nodes) for e in data)
        with timeit('giant matrix'):
            # make training data sklearn-friendly
            y = np.zeros(n)
            w = np.zeros(n)
            W = 0.0
            X = dok_matrix((n, self.D))
            i = 0
            for e in data:
                N = e.N
                for I in xrange(N):
                    for K in xrange(I+1,N+1):
                        if K-I > 1 and K-I != N:
                            w[i] = abs(e.Q[I,K,0] - e.Q[I,K,1])
                            y[i] = 1 if e.Q[I,K,0] < e.Q[I,K,1] else -1
                            f = e.features[I,K]
                            for k in f.get_keys():
                                X[i,k] = 1
                            W += w
                            i += 1
            X = X.tocsr()
        w /= W
        with timeit('train'):
            c = SVC(degree=1, kernel='linear', verbose=1, C=np.exp(self.C), shrinking=0)
            c.fit(X, y, sample_weight=w)
        if hasattr(c.coef_, 'todense'):
            w = np.asarray(c.coef_.todense()).flatten()
        else:
            w = c.coef_.flatten()
        b = c.intercept_ * 1.0
        return (w,b)


cdef class Perceptron(Classifier):

    cdef V_t[:] avg, theta
    cdef int t

    def __init__(self, D):
        Classifier.__init__(self, D)
        self.avg = np.zeros(D)
        self.theta = np.zeros(D)

    def _train(self, list data, int passes=1):
        cdef:
            Example e
            int n, N, I, K
            double w, p, y, s, z
            SparseBinaryVector f
        n = sum(len(e.nodes) for e in data)
        for _ in xrange(passes):
            shuffle(data)
            for e in data:
                N = e.N
                for I in xrange(N):
                    for K in xrange(I+1,N+1):
                        if K-I > 1 and K-I != N:
                            w = abs(e.Q[I,K,0] - e.Q[I,K,1])
                            f = e.features[I,K]
                            y = 0 if e.Q[I,K,0] > e.Q[I,K,1] else 1
                            p = f._dot(self.theta) >= 0
                            if y != p:
                                s = +1 if y == 1 else -1
                                f._pluseq(self.theta, coeff=s*w)
                                f._pluseq(self.avg, coeff=s*w*self.t)

                            # Hinge loss
                            #z = f._dot(self.theta)
                            #y = -1 if e.Q[I,K,0] > e.Q[I,K,1] else 1
                            #if y*z < 1:
                            #    f._pluseq(self.theta, coeff=y*w)
                            #    f._pluseq(self.avg, coeff=y*w*self.t)

                            self.t += 1
        # use averaged parameters.
        for i in range(self.D):
            self._coef[i] = self.theta[i] - self.avg[i]/self.t
        self._intercept = 0.0
        return self._coef, 0.0


cdef class Adagrad(Classifier):

    cdef LazyRegularizedAdagrad u
    cdef int loss

    def __init__(self, int nfeatures, C, loss, eta = 0.1, L = 2, fudge = 1e-4):
        Classifier.__init__(self, nfeatures)
        self.loss = loss
        self.u = LazyRegularizedAdagrad(self.D, L=L, C=np.exp(C), eta = eta, fudge = fudge)

    def _train(self, list data, int passes = 1):
        """
        Train binary logistic regression classifier with importance weights.
        """
        cdef:
            double W, y, w, z
            int i, k, I, K, N
            SparseBinaryVector f
            Example e
        # compute sum of importance weights
        W = 0
        for e in data:
            N = e.N
            for I in xrange(N):
                for K in xrange(I+1,N+1):
                    if K-I > 1 and K-I != N:
                        w = abs(e.Q[I,K,0] - e.Q[I,K,1])
                        W += w
        # run training
        for _ in range(passes):
            shuffle(data)
            for e in data:
                N = e.N
                for I in xrange(N):
                    for K in xrange(I+1,N+1):
                        if K-I > 1 and K-I != N:
                            w = abs(e.Q[I,K,0] - e.Q[I,K,1])
                            if w == 0:
                                continue
                            y = 1 if e.Q[I,K,0] < e.Q[I,K,1] else -1
                            f = e.features[I,K]
                            # dot product
                            z = 0.0
                            for i in range(f.length):
                                k = f.keys[i]
                                z += self.u.catchup(k)
                            # normalize importance weight
                            w = w/W
                            # gradient magnitude (update active assumes descent
                            if self.loss == 0:    # logistic
                                v = -y*_sigmoid(-y*z)
                            elif self.loss == 1:  # squared
                                v = (z - y)
                            elif self.loss == 2:  # hinge
                                if y*z > 1:
                                    v = 0
                                else:
                                    v = -y
                            else:
                                v = 0.0
                            v = v*w
                            if v != 0:
                                # gradient update
                                for i in range(f.length):
                                    k = f.keys[i]
                                    self.u.update_active(k, v)
                            self.u.step += 1

        return self.u.finalize(), 0.0
